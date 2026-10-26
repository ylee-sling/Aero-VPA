# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
predictor.py -- Latent world model of Aero-VPA (preprint_261026.pdf, Secs. 3.2 and 3.5, Table 1).

Implements
    * ``JEPAPredictor`` -- P_omega: N_e = 5 independently initialised MLP heads that forecast the latent
      T_nu = 0.1 s ahead, conditioned on the current latent, the active primitive and the executing
      setpoint (Eq. 7):
          z_hat^(i)_{t+1} = P_omega^(i)( z_t, Embed(u_t), a_bar_t ),   z_hat_{t+1} = (1/N_e) sum_i z_hat^(i)_{t+1}
      Each head is 292 -> 512 -> 512 -> 256 with 292 = d + d_u + d_a = 256 + 32 + 4. With the primitive
      embedding Embed (6 x 32) the module has 2,720,192 parameters (Table 1: 2.72M). The heads are one
      batched product, i.e. one network evaluation on the K + 3 dependency chain (Sec. 3.7).
    * ``JEPAPredictor.ensemble_uncertainty`` -- the normalised ensemble disagreement (Eq. 22)
          sigma_{t+1} = 1 / (sqrt(d) gamma*) * ( (1/N_e) sum_i ||z_hat^(i)_{t+1} - z_hat_{t+1}||_2^2 )^(1/2),
      computed in float32 or wider whatever the network precision (FP16 TensorRT deployment, Sec. 4.2).
    * ``VICRegLoss`` -- L_JEPA (Eqs. 8-10).
    * ``LatentScaleTracker`` -- the realised latent scale gamma_bar (Eq. 11) and the latent mean mu_Z,
      frozen to (gamma*, mu_Z) at the end of stage (i).

Changes from VPA (preprint_261008)
    * Heads take Concat(z_t, Embed(u_t), a_bar_t). a_bar_t, the setpoint dispatched most recently to the
      flight controller, is known before the decision, so P_omega is action-conditioned without putting
      the solver output on its input and the dependency chain stays acyclic (Sec. 3.2).
    * One primitive embedding shared by all heads, as Eq. 7 is written (VPA had one per head).
    * a_bar_t is divided by the fixed buffer ``setpoint_scale`` (the action-box bounds) before the heads,
      so that metres per second enter at the scale of the latents. Ones (the default) is the literal
      Eq. 7; the scale must be fixed before stage (i) because it is part of the frozen model.
    * sigma is computed in at least float32 from the head outputs.
    * Eq. 11 moves here from the VPA milestone tracker, which Aero-VPA does not have.

Tensor layout
    z_t ``[B, 256]``, u_t ``[B]`` int64, a_bar_t ``[B, 4]`` (v_x, v_y, v_z, yaw rate in the heading frame
    H_t), head input ``[B, 292]``, head predictions ``[N_e, B, 256]``, z_hat ``[B, 256]``, sigma ``[B]``.

This file is self-contained (it imports nothing from the other VPA modules) and can be tested
on its own with ``python predictor.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "AERO_PREDICTOR_PARAMS",
    "AERO_PREDICTOR_HEAD_PARAMS",
    "count_parameters",
    "EnsembleLinear",
    "JEPAPredictor",
    "PredictorOutput",
    "VICRegLoss",
    "LatentScaleTracker",
]

AERO_PREDICTOR_HEAD_PARAMS: int = 2_720_000  # 5 x (292*512+512 + 512*512+512 + 512*256+256)
AERO_PREDICTOR_PARAMS: int = 2_720_192       # heads + Embed (6 x 32); Table 1: 2.72M


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters of ``module``; only those with ``requires_grad`` if ``trainable_only``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------------------------
# Ensemble building block
# ---------------------------------------------------------------------------------------------
class EnsembleLinear(nn.Module):
    """N_e independent affine maps evaluated in one batched product: y_i = x_i W_i + b_i.

    Each member's weights are drawn independently with the default ``nn.Linear`` scheme
    (U(-1/sqrt(in), 1/sqrt(in)) for weights and biases), giving the independent initialisations
    that the ensemble disagreement of Eq. 22 relies on.

    Shapes:
        input ``x``: ``[N_e, B, in_features]`` -> output ``[N_e, B, out_features]``
    """

    def __init__(self, num_members: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.num_members: int = num_members
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight = nn.Parameter(torch.empty(num_members, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_members, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """``[N_e, B, in]`` -> ``[N_e, B, out]``."""
        return torch.baddbmm(self.bias, x, self.weight)


class PredictorOutput(NamedTuple):
    """Output of ``JEPAPredictor.forward``."""

    z_hat: Tensor             # [B, d]       ensemble mean z_hat_{t+1}
    sigma: Tensor             # [B]          normalised ensemble disagreement sigma_{t+1} >= 0 (Eq. 22)
    head_predictions: Tensor  # [N_e, B, d]  per-head forecasts z_hat^(i)_{t+1}


# ---------------------------------------------------------------------------------------------
# P_omega : JEPA latent predictor with prediction-head ensemble (Eq. 7)
# ---------------------------------------------------------------------------------------------
class JEPAPredictor(nn.Module):
    """Primitive- and setpoint-conditioned latent predictor P_omega with N_e independent heads (Eq. 7).

    Every head maps the same input Concat(z_t, Embed(u_t), a_bar_t / s) in R^{d + d_u + d_a} through
    its own MLP to z_hat^(i)_{t+1} in R^d, where s is ``setpoint_scale``. The forecast horizon is
    T_nu = 0.1 s (``t+1`` abbreviates tau_t + T_nu, not the next decision).

    The defaults are the Table 1 configuration, 2,720,192 parameters:

        Embed: nn.Embedding(6, 32)                    192   (shared by all heads)
        5 x Linear(292, 512)                      750,080
        5 x Linear(512, 512)                    1,313,280
        5 x Linear(512, 256)                      656,640   (544,000 per head)

    The frozen latent scale gamma* is the buffer ``latent_scale`` (set with ``set_latent_scale`` after
    stage (i)); it is required to compute sigma.

    Args:
        latent_dim: d.
        num_primitives: N_u.
        setpoint_dim: d_a, dimension of a_bar_t.
        num_heads: N_e >= 2 (disagreement is undefined for a single head).
        primitive_embed_dim: d_u.
        hidden_dim: hidden width of each head's MLP.
        num_hidden_layers: number of hidden layers per head (>= 1).
        setpoint_scale: ``[d_a]`` positive divisor of a_bar_t (the action-box bounds
            (v_h, v_h, v_z, yaw rate)); None means ones, the literal Eq. 7.

    Shapes:
        ``head_input``:    (``z_t [B, d]``, ``u_t [B]`` int64, ``a_bar [B, d_a]``) -> ``[B, d + d_u + d_a]``
        ``forward_heads``: same inputs -> ``[N_e, B, d]``
        ``forward``:       same inputs -> ``PredictorOutput`` (z_hat ``[B, d]``, sigma ``[B]``,
                           head_predictions ``[N_e, B, d]``)
        ``ensemble_uncertainty``: ``[N_e, B, d]`` (+ optional ``z_hat [B, d]``) -> ``[B]``
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_primitives: int = 6,
        setpoint_dim: int = 4,
        num_heads: int = 5,
        primitive_embed_dim: int = 32,
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
        setpoint_scale: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if num_heads < 2:
            raise ValueError("the ensemble needs N_e >= 2 heads to measure disagreement")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        if min(latent_dim, num_primitives, setpoint_dim, primitive_embed_dim, hidden_dim) < 1:
            raise ValueError("all dimensions must be positive")
        self.latent_dim: int = latent_dim
        self.num_primitives: int = num_primitives
        self.setpoint_dim: int = setpoint_dim
        self.num_heads: int = num_heads
        self.primitive_embed_dim: int = primitive_embed_dim
        self.input_dim: int = latent_dim + primitive_embed_dim + setpoint_dim  # 292 for Table 1
        self.primitive_embedding = nn.Embedding(num_primitives, primitive_embed_dim)  # Embed (Eq. 7)
        layers = [EnsembleLinear(num_heads, self.input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden_layers - 1):
            layers += [EnsembleLinear(num_heads, hidden_dim, hidden_dim), nn.GELU()]
        layers.append(EnsembleLinear(num_heads, hidden_dim, latent_dim))
        self.heads = nn.Sequential(*layers)
        if setpoint_scale is None:
            scale = torch.ones(setpoint_dim)
        else:
            scale = torch.as_tensor([float(s) for s in setpoint_scale], dtype=torch.float32)
        if tuple(scale.shape) != (setpoint_dim,) or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError(f"setpoint_scale must be {setpoint_dim} finite positive values")
        self.register_buffer("setpoint_scale", scale.to(torch.float32))
        self.register_buffer("latent_scale", torch.tensor(float("nan")))  # gamma*

    @torch.no_grad()
    def set_latent_scale(self, gamma_bar_star: float) -> None:
        """Store the frozen latent scale gamma* used to normalise sigma (Eq. 22)."""
        if not (math.isfinite(gamma_bar_star) and gamma_bar_star > 0.0):
            raise ValueError("gamma_bar_star must be finite and positive")
        self.latent_scale.fill_(float(gamma_bar_star))

    @property
    def has_latent_scale(self) -> bool:
        return not bool(torch.isnan(self.latent_scale))

    def head_input(self, z_t: Tensor, u_t: Tensor, a_bar: Tensor) -> Tensor:
        """Concat(z_t, Embed(u_t), a_bar_t / setpoint_scale) (Eq. 7).

        Args:
            z_t:   ``[B, d]`` current latent.
            u_t:   ``[B]`` int64 primitive tokens in {0, ..., N_u - 1}.
            a_bar: ``[B, d_a]`` setpoint dispatched most recently, in the current heading frame.

        Returns:
            ``[B, d + d_u + d_a]``.
        """
        if z_t.dim() != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(f"expected z_t of shape [B, {self.latent_dim}], got {tuple(z_t.shape)}")
        b = z_t.shape[0]
        if tuple(u_t.shape) != (b,):
            raise ValueError(f"expected u_t of shape [{b}], got {tuple(u_t.shape)}")
        if tuple(a_bar.shape) != (b, self.setpoint_dim):
            raise ValueError(f"expected a_bar of shape [{b}, {self.setpoint_dim}], got {tuple(a_bar.shape)}")
        emb = self.primitive_embedding(u_t.long()).to(z_t.dtype)                        # [B, d_u]
        setpoint = a_bar.to(z_t.dtype) / self.setpoint_scale.to(z_t.dtype)              # [B, d_a]
        return torch.cat([z_t, emb, setpoint], dim=-1)                                  # [B, d + d_u + d_a]

    def forward_heads(self, z_t: Tensor, u_t: Tensor, a_bar: Tensor) -> Tensor:
        """Per-head forecasts z_hat^(i)_{t+1}: ``[N_e, B, d]``. All heads read the same input."""
        x = self.head_input(z_t, u_t, a_bar)                                            # [B, in]
        return self.heads(x.unsqueeze(0).expand(self.num_heads, -1, -1))                # [N_e, B, d]

    @staticmethod
    def _ensemble_mean(heads: Tensor) -> Tensor:
        """(1/N_e) sum_i z_hat^(i), evaluated relative to head 1: ``[N_e, B, d]`` -> ``[B, d]``.

        z_hat^(1) + (1/N_e) sum_i (z_hat^(i) - z_hat^(1)) is the same mean in exact arithmetic and
        returns z_hat bit-exactly when all heads agree, so sigma = 0 exactly for identical heads.
        """
        return heads[0] + (heads - heads[0:1]).mean(dim=0)

    def ensemble_uncertainty(self, head_predictions: Tensor, z_hat: Optional[Tensor] = None) -> Tensor:
        """Eq. 22: sigma = ( (1/N_e) sum_i ||z_hat^(i) - z_hat||_2^2 )^(1/2) / (sqrt(d) gamma*).

        Computed in float32 or wider (float16 / bfloat16 head outputs are promoted first). The mean
        over heads is the population mean (1/N_e). Without ``z_hat`` the ensemble mean is formed here
        in the promoted precision.

        Args:
            head_predictions: ``[N_e, B, d]``.
            z_hat: optional ``[B, d]`` ensemble mean.

        Returns:
            sigma: ``[B]`` (dimensionless, >= 0).
        """
        if not self.has_latent_scale:
            raise RuntimeError("sigma requires the frozen latent scale gamma*; call set_latent_scale first")
        if head_predictions.dim() != 3 or tuple(head_predictions.shape[::2]) != (self.num_heads, self.latent_dim):
            raise ValueError(
                f"expected head predictions [{self.num_heads}, B, {self.latent_dim}], got {tuple(head_predictions.shape)}"
            )
        dtype = torch.promote_types(head_predictions.dtype, torch.float32)
        heads = head_predictions.to(dtype)
        mean = self._ensemble_mean(heads) if z_hat is None else z_hat.to(dtype)
        mean_sq_dev = (heads - mean.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=0)       # [B]
        return mean_sq_dev.sqrt() / (math.sqrt(self.latent_dim) * self.latent_scale.to(dtype))

    def forward(self, z_t: Tensor, u_t: Tensor, a_bar: Tensor) -> PredictorOutput:
        """Eq. 7 and Eq. 22: (z_hat_{t+1}, sigma_{t+1}) = P_omega(z_t, Embed(u_t), a_bar_t).

        Args:
            z_t: ``[B, d]``;  u_t: ``[B]`` int64;  a_bar: ``[B, d_a]``.

        Returns:
            ``PredictorOutput(z_hat [B, d], sigma [B], head_predictions [N_e, B, d])``.
        """
        heads = self.forward_heads(z_t, u_t, a_bar)                # [N_e, B, d]
        z_hat = self._ensemble_mean(heads)                         # [B, d]
        return PredictorOutput(z_hat, self.ensemble_uncertainty(heads), heads)


# ---------------------------------------------------------------------------------------------
# VICReg-regularised JEPA objective (Eqs. 8-10)
# ---------------------------------------------------------------------------------------------
class VICRegLoss(nn.Module):
    """L_JEPA = E_t[(1/N_e) sum_i ||z_hat^(i)_{t+1} - sg(E_psi_bar(I_{t+1}))||_2^2] + lambda_v v(Z) + lambda_c c(Z)   (Eq. 8).

    ||.||_2^2 is the squared Euclidean norm summed over the d latent dimensions; E_t is the mean over
    the batch. ``head_reduction="mean"`` (default) is the 1/N_e of Eq. 8; "sum" is kept for ablations.
    v(Z) and c(Z) (Eqs. 9, 10) are evaluated on the batch Z of online latents z_t = E_psi(I_t).

    Args:
        gamma: variance margin gamma > 0 (Eq. 9).
        eps: numerical constant with 0 < eps < gamma^2 (Eq. 9; also used by Eq. 11).
        lambda_v: weight of v(Z); must be > 0.
        lambda_c: weight of c(Z); >= 0.
        head_reduction: "mean" or "sum" over ensemble heads for the invariance term.

    Shapes:
        ``variance`` / ``covariance``: ``Z [B, d]`` -> scalar
        ``invariance``: (``z_hat [N_e, B, d]`` or ``[B, d]``, ``z_target [B, d]``) -> scalar
        ``forward``: (``z_online [B, d]``, ``z_hat [N_e, B, d]`` or ``[B, d]``, ``z_target [B, d]``)
                     -> (scalar total, dict of scalar terms)
    """

    def __init__(
        self,
        lambda_v: float,
        lambda_c: float,
        gamma: float = 1.0,
        eps: float = 1e-4,
        head_reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 < eps < gamma ** 2:
            raise ValueError("eps must satisfy 0 < eps < gamma^2")
        if lambda_v <= 0.0:
            raise ValueError("lambda_v must be positive")
        if lambda_c < 0.0:
            raise ValueError("lambda_c must be non-negative")
        if head_reduction not in ("mean", "sum"):
            raise ValueError("head_reduction must be 'mean' or 'sum'")
        self.gamma: float = float(gamma)
        self.eps: float = float(eps)
        self.lambda_v: float = float(lambda_v)
        self.lambda_c: float = float(lambda_c)
        self.head_reduction: str = head_reduction

    @staticmethod
    def _check_batch(latents: Tensor) -> Tuple[int, int]:
        if latents.dim() != 2:
            raise ValueError(f"expected a latent batch of shape [B, d], got {tuple(latents.shape)}")
        b, d = latents.shape
        if b < 2:
            raise ValueError("variance/covariance estimators require B >= 2")
        return b, d

    def variance(self, latents: Tensor) -> Tensor:
        """Eq. 9: v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(Z_{.,j}) + eps)).

        Var uses the unbiased (1/(B-1)) estimator, the same estimator as C(Z) in Eq. 10.
        """
        self._check_batch(latents)
        std = torch.sqrt(latents.var(dim=0, unbiased=True) + self.eps)  # [d]
        return F.relu(self.gamma - std).mean()

    def covariance(self, latents: Tensor) -> Tensor:
        """Eq. 10: c(Z) = (1/d) sum_{i != j} [C(Z)]_{i,j}^2 with C(Z) = Zc^T Zc / (B - 1)."""
        b, d = self._check_batch(latents)
        centred = latents - latents.mean(dim=0, keepdim=True)            # [B, d]
        cov = centred.transpose(0, 1) @ centred / (b - 1)                # [d, d]
        off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
        return off_diagonal.pow(2).sum() / d

    def invariance(self, z_hat: Tensor, z_target: Tensor) -> Tensor:
        """Prediction term of Eq. 8, applied to each head.

        Args:
            z_hat: ``[N_e, B, d]`` per-head forecasts (or ``[B, d]`` for a single predictor).
            z_target: ``[B, d]`` momentum-encoder targets E_psi_bar(I_{t+1}); detached here (sg).
        """
        heads = z_hat.unsqueeze(0) if z_hat.dim() == 2 else z_hat          # [N_e, B, d]
        if heads.shape[1:] != z_target.shape:
            raise ValueError(f"shape mismatch: z_hat {tuple(z_hat.shape)} vs z_target {tuple(z_target.shape)}")
        target = z_target.detach()                                        # sg(.)
        per_head = (heads - target.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=-1)  # [N_e]
        return per_head.mean() if self.head_reduction == "mean" else per_head.sum()

    def anti_collapse(self, latents: Tensor) -> Tensor:
        """lambda_v v(Z) + lambda_c c(Z)."""
        return self.lambda_v * self.variance(latents) + self.lambda_c * self.covariance(latents)

    def forward(
        self, z_online: Tensor, z_hat: Tensor, z_target: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Full objective of Eq. 8.

        Args:
            z_online: ``[B, d]`` online latents z_t = E_psi(I_t) (the batch Z).
            z_hat:    ``[N_e, B, d]`` (or ``[B, d]``) forecasts.
            z_target: ``[B, d]`` targets E_psi_bar(I_{t+1}).

        Returns:
            total loss (scalar) and a dict with the "invariance", "variance" and "covariance" terms.
        """
        inv = self.invariance(z_hat, z_target)
        var = self.variance(z_online)
        cov = self.covariance(z_online)
        total = inv + self.lambda_v * var + self.lambda_c * cov
        return total, {"invariance": inv.detach(), "variance": var.detach(), "covariance": cov.detach()}


# ---------------------------------------------------------------------------------------------
# Realised latent scale (Eq. 11) and latent mean
# ---------------------------------------------------------------------------------------------
class LatentScaleTracker(nn.Module):
    """Realised latent scale gamma_bar (Eq. 11) and latent mean mu_Z, tracked during stage (i).

        gamma_bar_n = (1 - alpha_gamma) gamma_bar_{n-1} + alpha_gamma ( (1/d) sum_j Var(Z^(n)_{.,j}) + eps )^(1/2)

    The square root is taken of the mean variance, not the mean of the per-dimension standard
    deviations; Var is unbiased and eps is the eps of Eq. 9. The paper does not give gamma_bar_0: it is
    set to the first batch estimate. The paper gives mu_Z no tracking rule: here it is the EMA of the
    batch means with the same alpha_gamma, and ``freeze`` accepts exact dataset values instead.

    After ``freeze`` the tracker refuses updates. (gamma*, mu_Z) go to ``JEPAPredictor.set_latent_scale``
    (Eq. 22) and to the standardisation of e_t (Eq. 15).

    Args:
        latent_dim: d.
        alpha_gamma: EMA rate in (0, 1].
        eps: numerical constant > 0 (the eps of Eq. 9).

    Shapes:
        ``update``: ``Z [B, d]`` (B >= 2) -> float gamma_bar_n
        ``freeze``: -> (float gamma*, ``mu_Z [d]``)
    """

    def __init__(self, latent_dim: int, alpha_gamma: float, eps: float) -> None:
        super().__init__()
        if not 0.0 < alpha_gamma <= 1.0:
            raise ValueError("alpha_gamma must lie in (0, 1]")
        if not eps > 0.0:
            raise ValueError("eps must be positive")
        self.latent_dim: int = int(latent_dim)
        self.alpha_gamma: float = float(alpha_gamma)
        self.eps: float = float(eps)
        self.register_buffer("gamma_bar", torch.tensor(float("nan")))
        self.register_buffer("latent_mean", torch.full((self.latent_dim,), float("nan")))
        self.register_buffer("frozen", torch.tensor(False))

    @staticmethod
    def instantaneous_scale(latents: Tensor, eps: float) -> Tensor:
        """( (1/d) sum_j Var(Z_{.,j}) + eps )^(1/2) for one batch ``[B, d]`` -> scalar."""
        return torch.sqrt(latents.var(dim=0, unbiased=True).mean() + eps)

    @property
    def is_initialized(self) -> bool:
        return bool(torch.isfinite(self.gamma_bar)) and bool(torch.isfinite(self.latent_mean).all())

    @torch.no_grad()
    def update(self, latents: Tensor) -> float:
        """One Eq. 11 step on the batch of online latents ``[B, d]``; returns gamma_bar_n."""
        if bool(self.frozen):
            raise RuntimeError("the tracker is frozen; gamma* and mu_Z are fixed")
        if latents.dim() != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"expected latents [B, {self.latent_dim}], got {tuple(latents.shape)}")
        if latents.shape[0] < 2:
            raise ValueError("the variance estimate requires B >= 2")
        z = latents.detach().to(device=self.gamma_bar.device, dtype=self.gamma_bar.dtype)
        inst = self.instantaneous_scale(z, self.eps)
        batch_mean = z.mean(dim=0)
        if not self.is_initialized:
            self.gamma_bar.copy_(inst)            # gamma_bar_0 := first estimate
            self.latent_mean.copy_(batch_mean)
        else:
            a = self.alpha_gamma
            self.gamma_bar.mul_(1.0 - a).add_(a * inst)
            self.latent_mean.mul_(1.0 - a).add_(a * batch_mean)
        return float(self.gamma_bar)

    @torch.no_grad()
    def freeze(
        self, gamma_bar_star: Optional[float] = None, latent_mean: Optional[Tensor] = None
    ) -> Tuple[float, Tensor]:
        """Fix (gamma*, mu_Z), optionally overriding the EMA values with exact dataset statistics."""
        if gamma_bar_star is not None:
            if not (math.isfinite(gamma_bar_star) and gamma_bar_star > 0.0):
                raise ValueError("gamma_bar_star must be finite and positive")
            self.gamma_bar.fill_(float(gamma_bar_star))
        if latent_mean is not None:
            mean = torch.as_tensor(latent_mean, dtype=self.latent_mean.dtype, device=self.latent_mean.device)
            if tuple(mean.shape) != (self.latent_dim,) or not bool(torch.isfinite(mean).all()):
                raise ValueError(f"latent_mean must be {self.latent_dim} finite values")
            self.latent_mean.copy_(mean)
        if not self.is_initialized:
            raise RuntimeError("nothing to freeze: call update() or pass gamma_bar_star and latent_mean")
        self.frozen.fill_(True)
        return float(self.gamma_bar), self.latent_mean.clone()

    @property
    def gamma_bar_star(self) -> float:
        if not bool(self.frozen):
            raise RuntimeError("gamma* is defined only after freeze()")
        return float(self.gamma_bar)


# ---------------------------------------------------------------------------------------------
# Self-test (run: python predictor.py)
# ---------------------------------------------------------------------------------------------
_RTOL, _ATOL = 1e-5, 1e-6              # float32 module output vs float64 reference (CLAUDE.md)
_RTOL_WIDE, _ATOL_WIDE = 1e-4, 1e-5    # full Table 1 width (512-wide hidden layers)


def _close(actual, ref, rtol: float = _RTOL, atol: float = _ATOL) -> bool:
    a = torch.as_tensor(actual).detach().double()
    r = torch.as_tensor(ref).detach().double()
    return torch.allclose(a, r, rtol=rtol, atol=atol)


def _expect(exc: type, fn) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _ref_head_forward(pred: JEPAPredictor, z_t: Tensor, u_t: Tensor, a_bar: Tensor) -> Tensor:
    """Float64 per-head, per-sample reference of Eq. 7 -> ``[N_e, B, d]``."""
    ne, (b, d) = pred.num_heads, z_t.shape
    emb = pred.primitive_embedding.weight.detach().double()
    scale = pred.setpoint_scale.double()
    out = torch.zeros(ne, b, d, dtype=torch.float64)
    for i in range(ne):
        for k in range(b):
            x = torch.cat([z_t[k].double(), emb[int(u_t[k])], a_bar[k].double() / scale])  # shared Embed
            for layer in pred.heads:
                if isinstance(layer, EnsembleLinear):
                    x = x @ layer.weight[i].detach().double() + layer.bias[i, 0].detach().double()
                else:
                    assert isinstance(layer, nn.GELU) and layer.approximate == "none"
                    x = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
            out[i, k] = x
    return out


def _ref_sigma(heads: Tensor, gamma_star: float) -> Tensor:
    """sqrt((1/N_e) sum_i ||z_i - z_bar||^2) / (sqrt(d) gamma*) with explicit loops -> ``[B]``."""
    ne, b, d = heads.shape
    h = heads.detach().double()
    sig = torch.zeros(b, dtype=torch.float64)
    for k in range(b):
        mean = [sum(float(h[i, k, j]) for i in range(ne)) / ne for j in range(d)]
        sq = sum((float(h[i, k, j]) - mean[j]) ** 2 for i in range(ne) for j in range(d))
        sig[k] = math.sqrt(sq / ne) / (math.sqrt(d) * gamma_star)
    return sig


def _ref_column_var(z: Tensor) -> list:
    """Unbiased per-column variance with explicit float64 sums."""
    zd = z.detach().double()
    b, d = zd.shape
    out = []
    for j in range(d):
        col = [float(zd[k, j]) for k in range(b)]
        mean = sum(col) / b
        out.append(sum((v - mean) ** 2 for v in col) / (b - 1))
    return out


def _ref_v(z: Tensor, gamma: float, eps: float) -> float:
    """Eq. 9 with loops."""
    var = _ref_column_var(z)
    return sum(max(0.0, gamma - math.sqrt(v + eps)) for v in var) / len(var)


def _ref_cov_matrix(z: Tensor) -> Tensor:
    """C(Z) = 1/(B-1) sum_b (Z_b - Zbar)(Z_b - Zbar)^T with a triple loop (Eq. 10)."""
    zd = z.detach().double()
    b, d = zd.shape
    mean = [sum(float(zd[k, j]) for k in range(b)) / b for j in range(d)]
    c = torch.zeros(d, d, dtype=torch.float64)
    for i in range(d):
        for j in range(d):
            c[i, j] = sum((float(zd[k, i]) - mean[i]) * (float(zd[k, j]) - mean[j]) for k in range(b)) / (b - 1)
    return c


def _ref_c(z: Tensor) -> float:
    """Eq. 10 with loops."""
    c = _ref_cov_matrix(z)
    d = c.shape[0]
    return sum(float(c[i, j]) ** 2 for i in range(d) for j in range(d) if i != j) / d


def _ref_invariance(heads: Tensor, target: Tensor, reduction: str) -> float:
    """Eq. 8 prediction term per head (squared norm over d, mean over B), then mean/sum over heads."""
    h, t = heads.detach().double(), target.detach().double()
    ne, b, d = h.shape
    per_head = [sum(sum((float(h[i, k, j]) - float(t[k, j])) ** 2 for j in range(d)) for k in range(b)) / b
                for i in range(ne)]
    return sum(per_head) / ne if reduction == "mean" else sum(per_head)


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(4321)

    # ---- Table 1: P_omega = 5 heads 292 -> 512 -> 512 -> 256 plus the shared Embed (6 x 32).
    pred = JEPAPredictor()
    assert pred.input_dim == 292 == 256 + 32 + 4 and pred.num_heads == 5
    linears = [layer for layer in pred.heads if isinstance(layer, EnsembleLinear)]
    assert [(lin.num_members, lin.in_features, lin.out_features) for lin in linears] == [
        (5, 292, 512), (5, 512, 512), (5, 512, 256)]
    assert [count_parameters(lin) for lin in linears] == [750_080, 1_313_280, 656_640]
    assert count_parameters(pred.heads) == AERO_PREDICTOR_HEAD_PARAMS == 5 * 544_000
    assert count_parameters(pred.primitive_embedding) == 6 * 32
    assert count_parameters(pred) == count_parameters(pred, trainable_only=False) == AERO_PREDICTOR_PARAMS
    assert round(AERO_PREDICTOR_PARAMS / 1e6, 2) == 2.72
    assert sorted(name for name, _ in pred.named_buffers()) == ["latent_scale", "setpoint_scale"]

    # ---- Eq. 7 tensor shapes: input [B, 292] = (z_t | Embed(u_t) | a_bar), heads [5, B, 256].
    b, d = 4, 256
    z_t = torch.randn(b, d, generator=gen)
    u_t = torch.randint(0, 6, (b,), generator=gen)
    a_bar = torch.randn(b, 4, generator=gen)
    x = pred.head_input(z_t, u_t, a_bar)
    assert x.shape == (b, 292)
    assert torch.equal(x[:, :256], z_t) and torch.equal(x[:, 288:], a_bar)
    assert torch.equal(x[:, 256:288], pred.primitive_embedding.weight[u_t])
    seen = []
    hook = linears[0].register_forward_hook(lambda _m, args, _out: seen.append(tuple(args[0].shape)))
    heads = pred.forward_heads(z_t, u_t, a_bar)
    hook.remove()
    assert seen == [(5, b, 292)] and heads.shape == (5, b, d)
    ref_heads = _ref_head_forward(pred, z_t, u_t, a_bar)
    assert _close(heads, ref_heads, _RTOL_WIDE, _ATOL_WIDE)
    for i in range(5):  # independent initialisations
        for j in range(i + 1, 5):
            assert not torch.equal(linears[0].weight[i], linears[0].weight[j])
            assert float((heads[i] - heads[j]).detach().abs().max()) > 1e-3
    # the setpoint and the primitive condition every head
    heads_a = pred.forward_heads(z_t, u_t, a_bar + 1.0)
    heads_u = pred.forward_heads(z_t, (u_t + 1) % 6, a_bar)
    for i in range(5):
        assert float((heads_a[i] - heads[i]).detach().abs().max()) > 1e-4
        assert float((heads_u[i] - heads[i]).detach().abs().max()) > 1e-4
    _expect(ValueError, lambda: pred.forward_heads(z_t[:, :-1], u_t, a_bar))
    _expect(ValueError, lambda: pred.forward_heads(z_t.unsqueeze(0), u_t, a_bar))
    _expect(ValueError, lambda: pred.forward_heads(z_t, u_t[:-1], a_bar))
    _expect(ValueError, lambda: pred.forward_heads(z_t, u_t, a_bar[:, :3]))
    _expect(ValueError, lambda: pred.forward_heads(z_t, u_t, a_bar[:-1]))

    # ---- setpoint_scale: dividing inside equals dividing outside; the buffer is in the state_dict.
    scale = (7.0, 7.0, 2.0, 1.5)
    pred_s = JEPAPredictor(setpoint_scale=scale)
    pred_s.load_state_dict({k: v for k, v in pred.state_dict().items() if k != "setpoint_scale"}, strict=False)
    s = torch.tensor(scale)
    assert torch.equal(pred_s.forward_heads(z_t, u_t, a_bar), pred.forward_heads(z_t, u_t, a_bar / s))
    pred_r = JEPAPredictor()
    pred_r.load_state_dict(pred_s.state_dict())
    assert torch.equal(pred_r.setpoint_scale, s)
    for bad in ((1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, float("inf"), 1.0, 1.0)):
        _expect(ValueError, lambda sc=bad: JEPAPredictor(setpoint_scale=sc))

    # ---- Eq. 22: sigma requires gamma* and matches an explicit-loop reference.
    _expect(RuntimeError, lambda: pred(z_t, u_t, a_bar))
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        _expect(ValueError, lambda g=bad: pred.set_latent_scale(g))
    gamma_star = 1.7
    pred.set_latent_scale(gamma_star)
    out = pred(z_t, u_t, a_bar)
    assert out.z_hat.shape == (b, d) and out.sigma.shape == (b,) and out.head_predictions.shape == (5, b, d)
    assert out.sigma.dtype == torch.float32 and torch.equal(out.head_predictions, heads)
    assert _close(out.z_hat, ref_heads.mean(dim=0), _RTOL_WIDE, _ATOL_WIDE)
    assert _close(out.sigma, _ref_sigma(ref_heads, gamma_star), _RTOL_WIDE, _ATOL_WIDE)
    assert bool((out.sigma > 0).all())
    assert _close(pred.ensemble_uncertainty(heads, out.z_hat), out.sigma)    # explicit z_hat: same value
    # Reference value: heads (i - 2) * 1, i = 0..4, gamma* = 1 -> sigma = sqrt(2); gamma* = 2 -> sqrt(2)/2.
    const = torch.stack([torch.full((2, d), float(i - 2)) for i in range(5)])
    pred.set_latent_scale(1.0)
    assert torch.allclose(pred.ensemble_uncertainty(const), torch.full((2,), math.sqrt(2.0)), rtol=1e-6, atol=0.0)
    pred.set_latent_scale(2.0)
    assert torch.allclose(pred.ensemble_uncertainty(const), torch.full((2,), math.sqrt(2.0) / 2), rtol=1e-6, atol=0.0)
    # Float16 head outputs (FP16 engine): sigma is still float32 and close to the float32 value.
    pred.set_latent_scale(gamma_star)
    sigma16 = pred.ensemble_uncertainty(heads.detach().half())
    assert sigma16.dtype == torch.float32 and _close(sigma16, out.sigma, rtol=1e-2, atol=1e-4)
    _expect(ValueError, lambda: pred.ensemble_uncertainty(heads[:4]))          # wrong N_e
    _expect(ValueError, lambda: pred.ensemble_uncertainty(heads[..., :-1]))    # wrong d
    # Identical heads: zero disagreement, exactly.
    same = JEPAPredictor()
    with torch.no_grad():
        for layer in same.heads:
            if isinstance(layer, EnsembleLinear):
                layer.weight.copy_(layer.weight[0:1].expand_as(layer.weight))
                layer.bias.copy_(layer.bias[0:1].expand_as(layer.bias))
    same.set_latent_scale(gamma_star)
    out_same = same(z_t, u_t, a_bar)
    for i in range(1, 5):
        assert torch.equal(out_same.head_predictions[i], out_same.head_predictions[0])
    assert torch.equal(out_same.sigma, torch.zeros(b)), float(out_same.sigma.abs().max())

    # ---- Eqs. 9 and 10 against loop references (small d keeps the loops fast).
    bs, ds = 32, 8
    gamma, eps, lam_v, lam_c = 1.0, 1e-4, 25.0, 1.0
    crit = VICRegLoss(lambda_v=lam_v, lambda_c=lam_c, gamma=gamma, eps=eps)
    z = torch.randn(bs, ds, generator=gen) * torch.linspace(0.2, 2.0, ds)  # some columns below the margin
    var_cols = _ref_column_var(z)
    assert any(math.sqrt(v + eps) < gamma for v in var_cols) and any(math.sqrt(v + eps) > gamma for v in var_cols)
    assert _close(crit.variance(z), _ref_v(z, gamma, eps))
    assert _close(crit.covariance(z), _ref_c(z))
    z_corr = torch.randn(bs, ds, generator=gen) @ torch.randn(ds, ds, generator=gen)  # correlated columns
    assert _ref_c(z_corr) > 0.1 and _close(crit.covariance(z_corr), _ref_c(z_corr))
    assert _close(crit.variance(z_corr), _ref_v(z_corr, gamma, eps))
    z2 = torch.randn(2, ds, generator=gen) * 0.3                           # unbiased: Var = (a - b)^2 / 2 for B = 2
    v2 = sum(max(0.0, gamma - math.sqrt(float(z2[0, j] - z2[1, j]) ** 2 / 2 + eps)) for j in range(ds)) / ds
    assert _close(crit.variance(z2), v2)
    for zeta in (torch.randn(ds, generator=gen), torch.full((ds,), 3.0)):  # constant encoder
        zc = zeta.expand(bs, ds).clone()
        assert _close(crit.variance(zc), gamma - math.sqrt(eps)) and _close(crit.covariance(zc), 0.0)
        assert _close(crit.anti_collapse(zc), lam_v * (gamma - math.sqrt(eps)))

    # ---- Eq. 8: total = invariance + lambda_v v + lambda_c c; stop-gradient on the target; gradients
    #      reach every predictor parameter, including the shared Embed.
    small = JEPAPredictor(latent_dim=ds, num_primitives=4, setpoint_dim=4, num_heads=5,
                          primitive_embed_dim=6, hidden_dim=32, num_hidden_layers=2)
    small.set_latent_scale(1.0)
    u_s = torch.randint(0, 4, (bs,), generator=gen)
    a_s = torch.randn(bs, 4, generator=gen)
    z_s = torch.randn(bs, ds, generator=gen)
    assert _close(small.forward_heads(z_s, u_s, a_s), _ref_head_forward(small, z_s, u_s, a_s))
    for reduction in ("mean", "sum"):
        crit_r = VICRegLoss(lambda_v=lam_v, lambda_c=lam_c, gamma=gamma, eps=eps, head_reduction=reduction)
        small.zero_grad(set_to_none=True)
        z_on = z_s.clone().requires_grad_(True)
        target = torch.randn(bs, ds, generator=gen).requires_grad_(True)
        hp = small(z_on, u_s, a_s).head_predictions
        total, terms = crit_r(z_on, hp, target)
        inv_ref = _ref_invariance(hp, target, reduction)
        v_ref, c_ref = _ref_v(z_on, gamma, eps), _ref_c(z_on)
        assert _close(terms["invariance"], inv_ref)
        assert _close(terms["variance"], v_ref) and _close(terms["covariance"], c_ref)
        assert _close(total, inv_ref + lam_v * v_ref + lam_c * c_ref)
        assert all(t.grad_fn is None for t in terms.values())
        total.backward()
        assert target.grad is None                                         # sg(.)
        assert z_on.grad is not None
        assert all(p.grad is not None for p in small.parameters())
    assert _close(crit.invariance(hp[0], z_s), _ref_invariance(hp[:1], z_s, "mean"))  # single predictor

    # ---- Eq. 11: gamma_bar = EMA of sqrt(mean_j Var + eps); mu_Z = EMA of batch means; freeze.
    alpha = 0.3
    tracker = LatentScaleTracker(latent_dim=ds, alpha_gamma=alpha, eps=eps)
    assert not tracker.is_initialized and count_parameters(tracker, trainable_only=False) == 0
    _expect(RuntimeError, tracker.freeze)
    _expect(RuntimeError, lambda: tracker.gamma_bar_star)
    g_ref, m_ref = 0.0, [0.0] * ds
    for n in range(5):
        zb = torch.randn(bs, ds, generator=gen) * torch.linspace(0.2, 2.0, ds) * (1.0 + 0.1 * n) + float(n)
        var = _ref_column_var(zb)
        inst = math.sqrt(sum(var) / ds + eps)
        if n == 0:   # sqrt of the mean variance differs from the mean of the standard deviations
            assert abs(inst - sum(math.sqrt(v + eps) for v in var) / ds) > 1e-3
        mean = [sum(float(zb[k, j]) for k in range(bs)) / bs for j in range(ds)]
        if n == 0:
            g_ref, m_ref = inst, mean
        else:
            g_ref = (1 - alpha) * g_ref + alpha * inst
            m_ref = [(1 - alpha) * mo + alpha * mn for mo, mn in zip(m_ref, mean, strict=True)]
        got = tracker.update(zb)
        assert _close(got, g_ref) and _close(tracker.gamma_bar, g_ref) and _close(tracker.latent_mean, m_ref)
    g_star, mu_z = tracker.freeze()
    assert _close(g_star, g_ref) and _close(mu_z, m_ref) and tracker.gamma_bar_star == g_star
    _expect(RuntimeError, lambda: tracker.update(zb))
    small.set_latent_scale(g_star)                                         # gamma* feeds Eq. 22
    assert float(small.latent_scale) == g_star
    exact = LatentScaleTracker(ds, alpha, eps)
    g2, mu2 = exact.freeze(gamma_bar_star=2.5, latent_mean=torch.zeros(ds))  # exact dataset statistics
    assert g2 == 2.5 and torch.equal(mu2, torch.zeros(ds))
    for kw in (dict(alpha_gamma=0.0), dict(alpha_gamma=1.5), dict(eps=0.0)):
        args = dict(latent_dim=ds, alpha_gamma=alpha, eps=eps)
        args.update(kw)
        _expect(ValueError, lambda a=args: LatentScaleTracker(**a))
    fresh = LatentScaleTracker(ds, alpha, eps)
    _expect(ValueError, lambda: fresh.update(torch.randn(1, ds)))
    _expect(ValueError, lambda: fresh.update(torch.randn(4, ds + 1)))
    _expect(ValueError, lambda: fresh.freeze(gamma_bar_star=-1.0))
    _expect(ValueError, lambda: fresh.freeze(gamma_bar_star=1.0, latent_mean=torch.zeros(ds + 1)))

    # ---- Errors.
    for kw in (dict(eps=1.0), dict(eps=2.0), dict(eps=0.0), dict(gamma=0.0), dict(gamma=-1.0),
               dict(lambda_v=0.0), dict(lambda_v=-1.0), dict(lambda_c=-0.1), dict(head_reduction="max")):
        args = dict(lambda_v=1.0, lambda_c=1.0, gamma=1.0, eps=1e-4)
        args.update(kw)
        _expect(ValueError, lambda a=args: VICRegLoss(**a))
    one = torch.randn(1, ds)
    _expect(ValueError, lambda: crit.variance(one))
    _expect(ValueError, lambda: crit.covariance(one))
    _expect(ValueError, lambda: crit(one, one, one))
    _expect(ValueError, lambda: crit.variance(torch.randn(2, bs, ds)))
    _expect(ValueError, lambda: crit.invariance(hp, z_s[:-1]))
    _expect(ValueError, lambda: JEPAPredictor(num_heads=1))
    _expect(ValueError, lambda: JEPAPredictor(num_hidden_layers=0))
    print("predictor.py self-test passed")


if __name__ == "__main__":
    _self_test()

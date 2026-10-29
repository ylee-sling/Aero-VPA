# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
solver.py -- Action generation and safety layer of Aero-VPA
(preprint_261026.pdf, Secs. 3.4-3.6, Table 1, Propositions 3.1 and 3.2).

Implements
    * ``standardize_latents`` / ``standardize_proprioception`` -- z~ = (z - mu_Z) / gamma*  and
      S~ = (S - mu_S) / sigma_S (Sec. 3.4).
    * ``SinusoidalTimeEmbedding`` -- features of the flow time rho in [0, 1] (256 wide).
    * ``ConditionalVectorField`` -- v_theta : R^{H x d_a} x [0, 1] x R^{d_e} -> R^{H x d_a} (Table 1):
          c   = MLP_c( Concat(e_t, time(rho)) )                     conditioning token       [B, 256]
          x_k = W_a a_k + p_k + c,  k = 1..H                        knot tokens              [B, 16, 256]
          X   = PreLN-Transformer^4( Concat(c, x_1, ..., x_H) )     H + 1 = 17 tokens, no mask
          v   = W_o LN( X[:, 1:] )                                  heading-frame velocities [B, 16, 4]
      3,572,228 parameters (Table 1: 3.57M).
    * ``CFMSolver`` -- the solver head: the learned Embed(u_t), the conditioning vector (Eq. 15)
          e_t = Concat( Embed(u_t), z~_t, z^~_{t+1}, S~_t, g_t, c_text ) in R^1073,
      the flow-matching objective on the linear path (Eqs. 16, 17)
          A^(rho) = rho A_t + (1 - rho) xi,    L_CFM = E || v_theta(A^(rho), rho | e_t) - (A_t - xi) ||_F^2,
      K-step explicit Euler sampling (Eqs. 18, 19), and denormalisation plus projection onto A_0.
      With Embed (6 x 32) the module has 3,572,420 parameters, also 3.57M.
    * ``NoisePersistence`` -- Eq. 20: xi_t = xi_{t-1} if u_t = u_{t-1} and t != 0 (mod N_xi), else fresh.
    * ``chunk_coherence_bound`` -- the right-hand sides of Proposition 3.1 (Eq. 21).
    * ``SafetyFilter`` -- Secs. 3.5 and 3.6: the asymmetric filter (Eq. 23), the attenuation law
      (Eq. 24), the validity horizon (Eq. 25), the BRAKE_HOVER override, the minimum-jerk braking
      profile (Eq. 26), the safe speed (Eq. 27) and the common translational scale factor (Eq. 28).
    * ``executed_prefix_mask`` -- the synchronous executed prefix, kept for ablation A8.

Changes from VPA (preprint_261008)
    * e_t gains g_t (d_e = 1073) and the flow-time embedding is 256 wide (Table 1 count).
    * Actions are normalised by the box A_0; ``generate_chunk`` returns the denormalised chunk projected
      onto A_0. K is any positive integer (ablation A1 uses 1, 2, 4, 8).
    * Noise persistence (Eq. 20) and the safety layer (Secs. 3.5, 3.6) are new; the VPA
      ``UncertaintyHorizonFilter`` is replaced by ``SafetyFilter``.

Decisions the paper leaves open (documented where they are made)
    * alpha(sigma_bar) = alpha_min for sigma_bar >= sigma_crit (Eq. 24 is defined only below sigma_crit;
      the value is reachable when the fallback is disabled, ablation A2).
    * T_r = T_e2e^max + H_t Delta + T_trk is evaluated per decision with the current H_t.
    * Eq. 27 is evaluated in the algebraically identical form v = 2 D / (T_r + sqrt(T_r^2 + 3 D / a_max)),
      D = d - eps_d - r_s, which avoids the cancellation of -T_r + sqrt(...) in float32.
    * A blind decision (direction outside the field of view, too many invalid pixels, or a non-finite
      range) limits the speed to v_blind instead of v_safe(d_t).
    * The depth region R_t and the range d_t are computed by the caller (they need the camera model).

This file is self-contained (it imports nothing from the other VPA modules; only its self-test reads
``config.py``) and can be tested on its own with ``python solver.py``.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "AERO_FIELD_PARAMS",
    "AERO_SOLVER_PARAMS",
    "BRAKE_HOVER",
    "count_parameters",
    "standardize_latents",
    "standardize_proprioception",
    "SinusoidalTimeEmbedding",
    "ConditionalVectorField",
    "CFMSolver",
    "NoisePersistence",
    "chunk_coherence_bound",
    "SafetyAssessment",
    "SpeedLimit",
    "SafetyFilter",
    "executed_prefix_mask",
]

AERO_FIELD_PARAMS: int = 3_572_228    # v_theta (Table 1: 3.57M)
AERO_SOLVER_PARAMS: int = 3_572_420   # v_theta + Embed (6 x 32)
BRAKE_HOVER: int = 5                  # token of BRAKE_HOVER in config.PRIMITIVES (checked by the self-test)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters of ``module``; only those with ``requires_grad`` if ``trainable_only``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------------------------
# Fixed standardisation (Sec. 3.4)
# ---------------------------------------------------------------------------------------------
def standardize_latents(z: Tensor, latent_mean: Tensor, gamma_bar_star: Union[float, Tensor]) -> Tensor:
    """z~ = (z - mu_Z) / gamma*, applied to z_t and z_hat_{t+1}. ``[..., d]`` -> ``[..., d]``."""
    return (z - latent_mean) / gamma_bar_star


def standardize_proprioception(state: Tensor, proprio_mean: Tensor, proprio_std: Tensor) -> Tensor:
    """S~ = (S - mu_S) / sigma_S with per-dimension dataset statistics. ``[..., d_s]`` -> ``[..., d_s]``."""
    return (state - proprio_mean) / proprio_std


# ---------------------------------------------------------------------------------------------
# v_theta : conditional vector field (Table 1)
# ---------------------------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal features of the flow time rho in [0, 1]:

        time(rho) = [ sin(s rho f_0), ..., sin(s rho f_{n-1}),  cos(s rho f_0), ..., cos(s rho f_{n-1}) ],
        f_i = 10000^(-i / n),  n = dim / 2,  s = scale = 1000.

    Parameter-free (``freqs`` is a non-persistent buffer).

    Shapes:
        ``rho: [B]`` -> ``[B, dim]``
    """

    def __init__(self, dim: int, scale: float = 1000.0) -> None:
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError("time embedding dimension must be a positive even number")
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.dim: int = int(dim)
        self.scale: float = float(scale)

    def forward(self, rho: Tensor) -> Tensor:
        """``[B]`` -> ``[B, dim]``."""
        args = self.scale * rho.to(self.freqs.dtype).unsqueeze(-1) * self.freqs  # [B, dim/2]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionalVectorField(nn.Module):
    """v_theta(A^(rho), rho | e_t): a non-causal pre-norm transformer over H + 1 tokens (Table 1).

    The conditioning token c = MLP_c(Concat(e_t, time(rho))) is prepended to the H knot tokens
    W_a a_k + p_k + c, so every layer can attend to it; no mask is used, so the whole chunk is produced
    in one evaluation whatever H. The defaults are Table 1, 3,572,228 parameters:

        action_in   Linear(4, 256)                                        1,280
        pos_embed   [1, 16, 256] (knot tokens only)                       4,096
        cond_mlp    Linear(1073 + 256, 256) - GELU - Linear(256, 256)   406,272
        blocks      4 x pre-norm TransformerEncoderLayer(256, 4 heads,
                    FFN 1024, GELU)                                   3,159,040   (789,760 each)
        norm        LayerNorm(256)                                          512
        action_out  Linear(256, 4)                                        1,028

    Args:
        horizon: H, knots per chunk.
        action_dim: d_a.
        cond_dim: d_e.
        width, depth, num_heads, mlp_ratio: transformer hyper-parameters.
        time_embed_dim: width of the rho embedding (256 for Table 1; 128 gives 3,539,460 = 3.54M).

    Shapes:
        (``A_rho [B, H, d_a]``, ``rho`` float, 0-dim or ``[B]``, ``e_t [B, d_e]``) -> ``[B, H, d_a]``
    """

    def __init__(
        self,
        horizon: int = 16,
        action_dim: int = 4,
        cond_dim: int = 1073,
        width: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(horizon, action_dim, cond_dim, width, depth, num_heads) < 1:
            raise ValueError("all dimensions must be positive")
        if width % num_heads != 0:
            raise ValueError(f"width {width} must be divisible by num_heads {num_heads}")
        self.horizon: int = int(horizon)
        self.action_dim: int = int(action_dim)
        self.cond_dim: int = int(cond_dim)
        self.width: int = int(width)
        self.depth: int = int(depth)
        self.num_heads: int = int(num_heads)
        self.time_embed_dim: int = int(time_embed_dim)
        self.action_in = nn.Linear(action_dim, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + time_embed_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=int(round(width * mlp_ratio)),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.action_out = nn.Linear(width, action_dim)

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        """Evaluate v_theta.

        Args:
            actions: ``[B, H, d_a]`` current point A^(rho) on the flow (normalised action space).
            rho: flow time, a Python float, a 0-dim tensor or ``[B]``.
            cond: ``[B, d_e]`` conditioning vector e_t.

        Returns:
            velocity: ``[B, H, d_a]``.
        """
        if actions.dim() != 3 or tuple(actions.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError(f"expected actions [B, {self.horizon}, {self.action_dim}], got {tuple(actions.shape)}")
        b = actions.shape[0]
        if tuple(cond.shape) != (b, self.cond_dim):
            raise ValueError(f"expected e_t of shape [{b}, {self.cond_dim}], got {tuple(cond.shape)}")
        rho_t = torch.as_tensor(rho, dtype=actions.dtype, device=actions.device)
        if rho_t.dim() == 0:
            rho_t = rho_t.expand(b)
        if tuple(rho_t.shape) != (b,):
            raise ValueError(f"expected rho as a scalar or of shape [{b}], got {tuple(rho_t.shape)}")
        c = self.cond_mlp(torch.cat([cond.to(actions.dtype), self.time_embed(rho_t).to(actions.dtype)], dim=-1))
        tokens = self.action_in(actions) + self.pos_embed + c.unsqueeze(1)                 # [B, H, W]
        x = self.blocks(torch.cat([c.unsqueeze(1), tokens], dim=1))                        # [B, H + 1, W]
        return self.action_out(self.norm(x[:, 1:]))                                        # [B, H, d_a]


# ---------------------------------------------------------------------------------------------
# CFM solver head (Eqs. 15-19)
# ---------------------------------------------------------------------------------------------
class CFMSolver(nn.Module):
    """Conditional flow-matching action solver (Sec. 3.4).

    Holds the learned primitive embedding Embed : U -> R^{d_u} (trained in stage (ii), separate from the
    frozen predictor's), the fixed standardisation statistics (mu_Z, gamma*, mu_S, sigma_S; set with
    ``set_standardization_stats`` after stage (i)), the box half-widths of A_0, and v_theta.

    Action spaces:
        physical   A in A_0 = [-v_h, v_h]^2 x [-v_z, v_z] x [-yaw_max, yaw_max] (m/s, rad/s, heading frame)
        normalised A / (v_h, v_h, v_z, yaw_max), componentwise in [-1, 1]; the flow lives here.

    Args:
        num_primitives: N_u.
        primitive_embed_dim: d_u.
        latent_dim: d.
        proprio_dim: d_s.
        goal_dim: d_g.
        text_dim: d_c.
        horizon: H.
        action_dim: d_a.
        action_scale: ``[d_a]`` positive half-widths of A_0.
        num_integration_steps: K >= 1.
        width, depth, num_heads, mlp_ratio, time_embed_dim: v_theta hyper-parameters.
        proprio_mask: optional ``[d_s]`` 0/1 factors on the standardised S_t in e_t (``config.policy_state_mask``):
            a withheld component enters at its training mean. A fixed, non-persistent buffer (from the config).

    Shapes:
        ``build_conditioning``: (``u [B]``, ``z_t [B, d]``, ``z_hat [B, d]``, ``S_t [B, d_s]``, ``g_t [B, d_g]``,
                                ``c_text [B, d_c]``) -> ``e_t [B, d_e]``
        ``flow_matching_loss``: (``A_t [B, H, d_a]`` normalised, ``e_t [B, d_e]``) -> scalar
        ``integrate``:          (``e_t [B, d_e]``, ``xi [B, H, d_a]``) -> normalised ``[B, H, d_a]``
        ``generate_chunk``:     ``e_t [B, d_e]`` -> physical ``[B, H, d_a]`` in A_0
    """

    def __init__(
        self,
        num_primitives: int = 6,
        primitive_embed_dim: int = 32,
        latent_dim: int = 256,
        proprio_dim: int = 13,
        goal_dim: int = 4,
        text_dim: int = 512,
        horizon: int = 16,
        action_dim: int = 4,
        action_scale: Sequence[float] = (7.0, 7.0, 2.0, 1.5),
        num_integration_steps: int = 2,
        width: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        proprio_mask: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if int(num_integration_steps) != num_integration_steps or num_integration_steps < 1:
            raise ValueError("the number of Euler steps K must be a positive integer")
        if min(num_primitives, primitive_embed_dim, latent_dim, proprio_dim, goal_dim, text_dim) < 1:
            raise ValueError("all dimensions must be positive")
        scale = torch.as_tensor([float(s) for s in action_scale], dtype=torch.float32)
        if tuple(scale.shape) != (action_dim,) or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError(f"action_scale must be {action_dim} finite positive half-widths")
        self.num_primitives: int = int(num_primitives)
        self.primitive_embed_dim: int = int(primitive_embed_dim)
        self.latent_dim: int = int(latent_dim)
        self.proprio_dim: int = int(proprio_dim)
        self.goal_dim: int = int(goal_dim)
        self.text_dim: int = int(text_dim)
        self.horizon: int = int(horizon)
        self.action_dim: int = int(action_dim)
        self.num_integration_steps: int = int(num_integration_steps)
        self.cond_dim: int = (self.primitive_embed_dim + 2 * self.latent_dim + self.proprio_dim
                              + self.goal_dim + self.text_dim)                                 # d_e = 1073
        self.primitive_embedding = nn.Embedding(self.num_primitives, self.primitive_embed_dim)  # Embed
        self.vector_field = ConditionalVectorField(
            horizon, action_dim, self.cond_dim, width=width, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, time_embed_dim=time_embed_dim,
        )
        nan = float("nan")
        self.register_buffer("latent_mean", torch.full((self.latent_dim,), nan))
        self.register_buffer("gamma_bar_star", torch.tensor(nan))
        self.register_buffer("proprio_mean", torch.full((self.proprio_dim,), nan))
        self.register_buffer("proprio_std", torch.full((self.proprio_dim,), nan))
        self.register_buffer("action_scale", scale)
        mask = torch.ones(self.proprio_dim) if proprio_mask is None else torch.as_tensor(proprio_mask, dtype=torch.float32)
        if tuple(mask.shape) != (self.proprio_dim,) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"proprio_mask must be {self.proprio_dim} values in {{0, 1}}")
        self.register_buffer("proprio_mask", mask.clone(), persistent=False)

    @classmethod
    def from_config(cls, cfg: Any) -> "CFMSolver":
        """Build from a ``config.VPAConfig`` (read by attribute; no import needed)."""
        return cls(
            num_primitives=cfg.num_primitives, primitive_embed_dim=cfg.primitive_embed_dim,
            latent_dim=cfg.latent_dim, proprio_dim=cfg.proprio_dim, goal_dim=cfg.goal_dim, text_dim=cfg.text_dim,
            horizon=cfg.horizon, action_dim=cfg.action_dim, action_scale=cfg.action_scale,
            num_integration_steps=cfg.integration_steps, width=cfg.field_width, depth=cfg.field_depth,
            num_heads=cfg.field_heads, mlp_ratio=cfg.field_mlp_ratio, time_embed_dim=cfg.time_embed_dim,
            proprio_mask=getattr(cfg, "policy_state_mask", None),
        )

    # ---------------- fixed statistics ----------------
    @torch.no_grad()
    def set_standardization_stats(
        self, latent_mean: Tensor, gamma_bar_star: float, proprio_mean: Tensor, proprio_std: Tensor
    ) -> None:
        """Store mu_Z ``[d]``, gamma* (scalar), mu_S ``[d_s]`` and sigma_S ``[d_s]``."""
        latent_mean = torch.as_tensor(latent_mean, dtype=self.latent_mean.dtype)
        proprio_mean = torch.as_tensor(proprio_mean, dtype=self.proprio_mean.dtype)
        proprio_std = torch.as_tensor(proprio_std, dtype=self.proprio_std.dtype)
        if tuple(latent_mean.shape) != (self.latent_dim,):
            raise ValueError(f"latent_mean must have shape [{self.latent_dim}]")
        if tuple(proprio_mean.shape) != (self.proprio_dim,) or tuple(proprio_std.shape) != (self.proprio_dim,):
            raise ValueError(f"proprioception statistics must have shape [{self.proprio_dim}]")
        if not (math.isfinite(gamma_bar_star) and gamma_bar_star > 0.0):
            raise ValueError("gamma_bar_star must be finite and positive")
        if not bool((proprio_std > 0).all()):
            raise ValueError("proprio_std must be strictly positive in every dimension")
        if not (bool(torch.isfinite(latent_mean).all()) and bool(torch.isfinite(proprio_mean).all())
                and bool(torch.isfinite(proprio_std).all())):
            raise ValueError("statistics must be finite")
        self.latent_mean.copy_(latent_mean)
        self.gamma_bar_star.fill_(float(gamma_bar_star))
        self.proprio_mean.copy_(proprio_mean)
        self.proprio_std.copy_(proprio_std)

    @property
    def is_calibrated(self) -> bool:
        return bool(
            torch.isfinite(self.latent_mean).all()
            and torch.isfinite(self.gamma_bar_star)
            and torch.isfinite(self.proprio_mean).all()
            and torch.isfinite(self.proprio_std).all()
        )

    # ---------------- Eq. 15: conditioning vector ----------------
    def build_conditioning(
        self, primitive: Tensor, z_t: Tensor, z_hat: Tensor, s_t: Tensor, g_t: Tensor, c_text: Tensor
    ) -> Tensor:
        """e_t = Concat(Embed(u_t), z~_t, z^~_{t+1}, S~_t, g_t, c_text) (Eq. 15).

        Args:
            primitive: ``[B]`` int64 tokens u_t.
            z_t:       ``[B, d]`` online latent (raw; standardised here).
            z_hat:     ``[B, d]`` ensemble-mean forecast z_hat_{t+1} (raw; standardised here).
            s_t:       ``[B, d_s]`` raw proprioceptive state S_t (standardised here).
            g_t:       ``[B, d_g]`` goal feature (Eq. 3), not re-standardised.
            c_text:    ``[B, d_c]`` instruction embedding, not re-standardised.

        Returns:
            e_t: ``[B, d_e]``, d_e = d_u + 2d + d_s + d_g + d_c = 1073.
        """
        if not self.is_calibrated:
            raise RuntimeError("standardisation statistics are not set; call set_standardization_stats")
        if z_t.dim() != 2:
            raise ValueError(f"expected z_t of shape [B, {self.latent_dim}], got {tuple(z_t.shape)}")
        b = z_t.shape[0]
        expected = (
            ("z_t", z_t, (b, self.latent_dim)),
            ("z_hat", z_hat, (b, self.latent_dim)),
            ("S_t", s_t, (b, self.proprio_dim)),
            ("g_t", g_t, (b, self.goal_dim)),
            ("c_text", c_text, (b, self.text_dim)),
            ("primitive", primitive, (b,)),
        )
        for name, tensor, shape in expected:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {list(shape)}, got {list(tensor.shape)}")
        dtype = z_t.dtype
        return torch.cat(
            [
                self.primitive_embedding(primitive.long()).to(dtype),                               # [B, d_u]
                standardize_latents(z_t, self.latent_mean.to(dtype), self.gamma_bar_star.to(dtype)),  # [B, d]
                standardize_latents(z_hat.to(dtype), self.latent_mean.to(dtype), self.gamma_bar_star.to(dtype)),
                standardize_proprioception(s_t.to(dtype), self.proprio_mean.to(dtype), self.proprio_std.to(dtype))
                * self.proprio_mask.to(dtype),                                                      # [B, d_s]
                g_t.to(dtype),                                                                      # [B, d_g]
                c_text.to(dtype),                                                                   # [B, d_c]
            ],
            dim=-1,
        )

    # ---------------- action normalisation and projection onto A_0 ----------------
    def normalize_actions(self, actions: Tensor) -> Tensor:
        """Physical -> normalised: A / (v_h, v_h, v_z, yaw_max), componentwise. ``[..., d_a]``."""
        return actions / self.action_scale.to(actions.dtype)

    def denormalize_actions(self, actions: Tensor) -> Tensor:
        """Normalised -> physical: A * (v_h, v_h, v_z, yaw_max), componentwise. ``[..., d_a]``."""
        return actions * self.action_scale.to(actions.dtype)

    def project(self, actions: Tensor) -> Tensor:
        """Pi_{A_0}: componentwise projection of physical actions onto the box. ``[..., d_a]``."""
        scale = self.action_scale.to(actions.dtype)
        return torch.maximum(torch.minimum(actions, scale), -scale)

    # ---------------- Eqs. 16, 17: training objective ----------------
    def flow_matching_loss(
        self, actions_norm: Tensor, cond: Tensor, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """L_CFM = E_{t, xi, rho ~ U[0, 1]} || v_theta(A^(rho), rho | e_t) - (A_t - xi) ||_F^2  (Eq. 17).

        A^(rho) = rho A_t + (1 - rho) xi (Eq. 16). The squared Frobenius norm is summed over the
        H x d_a entries; the expectation over t is the batch mean. xi is drawn before rho.

        Args:
            actions_norm: ``[B, H, d_a]`` normalised expert chunks A_t.
            cond: ``[B, d_e]`` conditioning vectors e_t.
            generator: optional RNG for xi and rho.

        Returns:
            scalar loss.
        """
        if actions_norm.dim() != 3 or tuple(actions_norm.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError(f"expected actions [B, {self.horizon}, {self.action_dim}], got {tuple(actions_norm.shape)}")
        b = actions_norm.shape[0]
        xi = torch.randn(actions_norm.shape, generator=generator, device=actions_norm.device, dtype=actions_norm.dtype)
        rho = torch.rand(b, generator=generator, device=actions_norm.device, dtype=actions_norm.dtype)  # U[0, 1)
        rho_b = rho.view(b, 1, 1)
        a_rho = rho_b * actions_norm + (1.0 - rho_b) * xi                                     # Eq. 16
        velocity = self.vector_field(a_rho, rho, cond)
        return (velocity - (actions_norm - xi)).pow(2).sum(dim=(1, 2)).mean()                 # Eq. 17

    # ---------------- Eqs. 18, 19: K-step Euler sampling ----------------
    def integrate(self, cond: Tensor, noise: Tensor) -> Tensor:
        """Integrate dA/drho = v_theta(A, rho | e_t) from A^(0) = xi with K explicit Euler steps (Eq. 18):

            A^(rho_{k+1}) = A^(rho_k) + (1/K) v_theta(A^(rho_k), rho_k | e_t),   rho_k = k / K,

        which for K = 2 is Eq. 19. Exactly K sequential evaluations of v_theta, each over the whole chunk.

        Args:
            cond: ``[B, d_e]``.
            noise: ``[B, H, d_a]`` initial condition xi.

        Returns:
            normalised chunk ``[B, H, d_a]`` (not projected).
        """
        shape = (cond.shape[0], self.horizon, self.action_dim)
        if tuple(noise.shape) != shape:
            raise ValueError(f"noise must have shape {list(shape)}, got {list(noise.shape)}")
        k_steps = self.num_integration_steps
        dt = 1.0 / k_steps
        a = noise
        for k in range(k_steps):
            a = a + dt * self.vector_field(a, k / k_steps, cond)
        return a

    def generate_chunk(
        self, cond: Tensor, noise: Optional[Tensor] = None, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """A_hat_t = Pi_{A_0}( denormalise( Euler_K(xi; e_t) ) ): physical chunk ``[B, H, d_a]`` in A_0.

        Args:
            cond: ``[B, d_e]``.
            noise: optional ``[B, H, d_a]`` xi (e.g. from ``NoisePersistence``); drawn from N(0, I) if omitted.
            generator: optional RNG for xi.
        """
        shape = (cond.shape[0], self.horizon, self.action_dim)
        if noise is None:
            noise = torch.randn(shape, generator=generator, device=cond.device, dtype=cond.dtype)
        return self.project(self.denormalize_actions(self.integrate(cond, noise)))


# ---------------------------------------------------------------------------------------------
# Noise persistence (Eq. 20)
# ---------------------------------------------------------------------------------------------
class NoisePersistence:
    """Initial-noise reuse across decisions (Eq. 20):

        xi_t = xi_{t-1}                  if u_t = u_{t-1} and t != 0 (mod N_xi),
        xi_t ~ N(0, I) (fresh draw)      otherwise,

    so the noise is redrawn when the primitive changes and at least every N_xi decisions. t is the
    per-episode decision index of each batch element, starting at 0 (the first decision is always fresh).
    A fresh ``[B, H, d_a]`` sample is drawn at every ``step`` whether or not it is used, so the random
    stream does not depend on the refresh pattern.

    A decision that produces no chunk (the BRAKE_HOVER fallback, Algorithm 1 line 7) is recorded with
    ``skip``: it advances t and stores the primitive, so the first decision after the fallback draws
    fresh noise unless the selector returns BRAKE_HOVER again.

    Args:
        horizon: H.
        action_dim: d_a.
        refresh_period: N_xi >= 1.

    Shapes:
        ``step``: ``u_t [B]`` int64 (+ optional ``mask [B]`` bool) -> ``xi_t [B, H, d_a]``
        ``skip``: ``u_t [B]`` int64 (+ optional ``mask [B]``)
    """

    def __init__(self, horizon: int, action_dim: int, refresh_period: int) -> None:
        for name, value in (("horizon", horizon), ("action_dim", action_dim), ("refresh_period", refresh_period)):
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        self.horizon: int = int(horizon)
        self.action_dim: int = int(action_dim)
        self.refresh_period: int = int(refresh_period)
        self.noise: Optional[Tensor] = None           # [B, H, d_a]
        self.prev_primitive: Optional[Tensor] = None  # [B] int64, -1 before the first decision
        self.counter: Optional[Tensor] = None         # [B] int64, decision index t
        self.last_refreshed: Optional[Tensor] = None  # [B] bool, refreshed at the latest step

    def reset(self, batch_size: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> None:
        """Start an episode: no previous primitive, t = 0."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.noise = torch.zeros(batch_size, self.horizon, self.action_dim, device=device, dtype=dtype)
        self.prev_primitive = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        self.counter = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.last_refreshed = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def _require(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.noise is None or self.prev_primitive is None or self.counter is None:
            raise RuntimeError("call reset(batch_size) before step/skip")
        return self.noise, self.prev_primitive, self.counter

    @staticmethod
    def _check(primitive: Tensor, mask: Optional[Tensor], b: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        if tuple(primitive.shape) != (b,):
            raise ValueError(f"expected primitives of shape [{b}], got {tuple(primitive.shape)}")
        if mask is None:
            return primitive.to(device=device, dtype=torch.long), torch.ones(b, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (b,):
            raise ValueError(f"expected mask of shape [{b}], got {tuple(mask.shape)}")
        return primitive.to(device=device, dtype=torch.long), mask.to(device=device, dtype=torch.bool)

    def step(self, primitive: Tensor, generator: Optional[torch.Generator] = None,
             mask: Optional[Tensor] = None) -> Tensor:
        """Eq. 20 for the deciding elements (``mask``); returns xi_t ``[B, H, d_a]``.

        Elements outside the mask are not deciding: they keep their noise, primitive and counter.
        """
        noise, prev, t = self._require()
        u, m = self._check(primitive, mask, noise.shape[0], noise.device)
        fresh = torch.randn(noise.shape, generator=generator, device=noise.device, dtype=noise.dtype)
        refresh = ((u != prev) | (t % self.refresh_period == 0)) & m
        self.noise = torch.where(refresh.view(-1, 1, 1), fresh, noise)
        self.prev_primitive = torch.where(m, u, prev)
        self.counter = t + m.long()
        self.last_refreshed = refresh
        return self.noise.clone()

    def skip(self, primitive: Tensor, mask: Optional[Tensor] = None) -> None:
        """Record a decision without a chunk (fallback): store u_t, advance t, keep xi, draw nothing."""
        noise, prev, t = self._require()
        u, m = self._check(primitive, mask, noise.shape[0], noise.device)
        self.prev_primitive = torch.where(m, u, prev)
        self.counter = t + m.long()
        self.last_refreshed = torch.zeros_like(m)


def chunk_coherence_bound(
    lipschitz_actions: float, lipschitz_cond: float, num_steps: int, cond_distance: float, exponential: bool = False
) -> float:
    """Right-hand side of Proposition 3.1 (Eq. 21):

        ||Phi_K(xi; e) - Phi_K(xi; e')||_F <= (L_e / L_A) [ (1 + L_A / K)^K - 1 ] ||e - e'||_2
                                             <= (L_e / L_A) ( e^{L_A} - 1 ) ||e - e'||_2   (``exponential``)

    for a field that is L_A-Lipschitz in A (Frobenius norm) and L_e-Lipschitz in e (Euclidean norm) on a
    set containing all Euler iterates, with the same initial noise xi.
    """
    if not (math.isfinite(lipschitz_actions) and lipschitz_actions > 0.0):
        raise ValueError("L_A must be finite and positive")
    if not (math.isfinite(lipschitz_cond) and lipschitz_cond >= 0.0):
        raise ValueError("L_e must be finite and non-negative")
    if int(num_steps) != num_steps or num_steps < 1:
        raise ValueError("K must be a positive integer")
    if not (math.isfinite(cond_distance) and cond_distance >= 0.0):
        raise ValueError("||e - e'|| must be finite and non-negative")
    if exponential:
        growth = math.exp(lipschitz_actions) - 1.0
    else:
        growth = (1.0 + lipschitz_actions / num_steps) ** num_steps - 1.0
    return lipschitz_cond / lipschitz_actions * growth * cond_distance


# ---------------------------------------------------------------------------------------------
# Safety layer (Secs. 3.5, 3.6)
# ---------------------------------------------------------------------------------------------
class SafetyAssessment(NamedTuple):
    """Output of ``SafetyFilter.assess`` (before the solver; Algorithm 1 lines 5-8)."""

    sigma_bar: Tensor  # [B]        filtered score sigma_bar_{t+1} (Eq. 23)
    alpha: Tensor      # [B]        attenuation alpha(sigma_bar) (Eq. 24)
    horizon: Tensor    # [B] int64  validity horizon H_t (Eq. 25)
    fallback: Tensor   # [B] bool   sigma_bar >= sigma_crit: brake to hover, no chunk
    primitive: Tensor  # [B] int64  u_t, overridden to BRAKE_HOVER where fallback


class SpeedLimit(NamedTuple):
    """Output of ``SafetyFilter.limit`` (after the solver; Algorithm 1 line 15)."""

    v_safe: Tensor         # [B]           speed envelope v_safe(d_t) (Eq. 27), or v_blind
    reaction_time: Tensor  # [B]           T_r = T_e2e^max + H_t Delta + T_trk
    scale: Tensor          # [B]           s_t (Eq. 28), in [0, 1]
    chunk: Tensor          # [B, H, d_a]   (s_t v_{t,k}, yaw rate_{t,k}): commanded chunk


class SafetyFilter:
    """Uncertainty gating and braking envelope (Secs. 3.5, 3.6, Proposition 3.2).

    Before the solver (``assess``):
        sigma_bar_{t+1} = max{ sigma_{t+1}, (1 - alpha_sigma) sigma_bar_t + alpha_sigma sigma_{t+1} }      (Eq. 23)
        alpha(sigma_bar) = 1 - (1 - alpha_min) clip( (sigma_bar - sigma_lo) / (sigma_crit - sigma_lo), 0, 1 ) (Eq. 24)
        H_t = max( H_min, floor( H_max exp(-beta sigma_bar_{t+1}) ) )                                      (Eq. 25)
        sigma_bar_{t+1} >= sigma_crit  ->  u_t <- BRAKE_HOVER, fallback (no chunk)
    Eq. 24 gives alpha only below sigma_crit; the clip extends it with alpha_min above (open decision).

    After the solver (``limit``):
        T_r = T_e2e^max + H_t Delta + T_trk
        v_safe(d) = 2 D / (T_r + sqrt(T_r^2 + 3 D / a_max)),  D = d - eps_d - r_s > 0;  0 otherwise      (Eq. 27)
        s_t = min{ alpha(sigma_bar_{t+1}),  v_safe(d_t) / max(max_k ||v_{t,k}||_2, eps_v) }                 (Eq. 28)
        a^cmd_{t,k} = (s_t v_{t,k}, yaw rate_{t,k}):  translation scaled, yaw rate untouched
    The form of Eq. 27 used here equals (2 a_max / 3)(-T_r + sqrt(T_r^2 + 3 D / a_max)) exactly and is the
    positive root of (3 / (4 a_max)) v^2 + T_r v - D = 0: reaction travel plus braking distance equals D.

    Fallback (``braking_profile``): the minimum-jerk stop from v_0 with zero initial acceleration,
        v_brk(tau) = v_0 (1 - 3 (tau/T_b)^2 + 2 (tau/T_b)^3),   T_b = 3 ||v_0||_2 / (2 a_max)            (Eq. 26)
    peak deceleration a_max at T_b / 2, stopping distance 3 ||v_0||_2^2 / (4 a_max).

    Args (keyword only):
        sigma_lo, sigma_crit: calibrated thresholds, 0 <= sigma_lo < sigma_crit.
        alpha_min: attenuation floor in (0, 1].
        alpha_sigma: filter release rate in (0, 1].
        min_horizon, max_horizon: 1 <= H_min <= H_max = H.
        beta: sensitivity of Eq. 25, > 0.
        max_decel: a_max > 0 (m/s^2).
        depth_error: eps_d >= 0 (m).
        safety_radius: r_s >= r_v (m).
        blind_speed: v_blind >= 0 (m/s).
        speed_eps: eps_v > 0 (m/s).
        knot_dt: Delta > 0 (s).
        latency_bound: T_e2e^max >= 0 (s), the measured worst-case end-to-end latency (Sec. 4.4).
        tracking_lag: T_trk >= 0 (s), the velocity-tracking lag of the flight controller.

    State: ``sigma_bar [B]`` per episode, 0 after ``reset`` (so the first decision has sigma_bar = sigma).
    """

    def __init__(
        self,
        *,
        sigma_lo: float,
        sigma_crit: float,
        alpha_min: float,
        alpha_sigma: float,
        min_horizon: int,
        max_horizon: int,
        beta: float,
        max_decel: float,
        depth_error: float,
        safety_radius: float,
        blind_speed: float,
        speed_eps: float,
        knot_dt: float,
        latency_bound: float,
        tracking_lag: float,
    ) -> None:
        values = dict(sigma_lo=sigma_lo, sigma_crit=sigma_crit, alpha_min=alpha_min, alpha_sigma=alpha_sigma,
                      beta=beta, max_decel=max_decel, depth_error=depth_error, safety_radius=safety_radius,
                      blind_speed=blind_speed, speed_eps=speed_eps, knot_dt=knot_dt, latency_bound=latency_bound,
                      tracking_lag=tracking_lag)
        for name, value in values.items():
            if value is None or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value}")
        if not 0.0 <= sigma_lo < sigma_crit:
            raise ValueError("require 0 <= sigma_lo < sigma_crit (Eq. 24)")
        if not 0.0 < alpha_min <= 1.0:
            raise ValueError("alpha_min must lie in (0, 1] (Eq. 24)")
        if not 0.0 < alpha_sigma <= 1.0:
            raise ValueError("alpha_sigma must lie in (0, 1] (Eq. 23)")
        if int(min_horizon) != min_horizon or int(max_horizon) != max_horizon or not 1 <= min_horizon <= max_horizon:
            raise ValueError("require integers 1 <= H_min <= H_max (Eq. 25)")
        for name in ("beta", "max_decel", "speed_eps", "knot_dt"):
            if not values[name] > 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("depth_error", "safety_radius", "blind_speed", "latency_bound", "tracking_lag"):
            if not values[name] >= 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.sigma_lo: float = float(sigma_lo)
        self.sigma_crit: float = float(sigma_crit)
        self.alpha_min: float = float(alpha_min)
        self.alpha_sigma: float = float(alpha_sigma)
        self.min_horizon: int = int(min_horizon)
        self.max_horizon: int = int(max_horizon)
        self.beta: float = float(beta)
        self.max_decel: float = float(max_decel)
        self.depth_error: float = float(depth_error)
        self.safety_radius: float = float(safety_radius)
        self.blind_speed: float = float(blind_speed)
        self.speed_eps: float = float(speed_eps)
        self.knot_dt: float = float(knot_dt)
        self.latency_bound: float = float(latency_bound)
        self.tracking_lag: float = float(tracking_lag)
        self.sigma_bar: Optional[Tensor] = None

    @classmethod
    def from_config(cls, cfg: Any, *, latency_bound: float, tracking_lag: float) -> "SafetyFilter":
        """Build from a ``config.VPAConfig``; refuses while sigma_lo or sigma_crit is not calibrated."""
        if cfg.sigma_lo is None or cfg.sigma_crit is None:
            raise ValueError("sigma_lo and sigma_crit are not calibrated yet (Sec. 3.5)")
        return cls(
            sigma_lo=cfg.sigma_lo, sigma_crit=cfg.sigma_crit, alpha_min=cfg.alpha_min, alpha_sigma=cfg.alpha_sigma,
            min_horizon=cfg.min_horizon, max_horizon=cfg.horizon, beta=cfg.beta, max_decel=cfg.max_decel,
            depth_error=cfg.depth_error, safety_radius=cfg.safety_radius, blind_speed=cfg.blind_speed,
            speed_eps=cfg.speed_eps, knot_dt=cfg.knot_dt, latency_bound=latency_bound, tracking_lag=tracking_lag,
        )

    # ---------------- Eq. 23 ----------------
    def reset(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """Start an episode with sigma_bar = 0."""
        self.sigma_bar = torch.zeros(batch_size, device=device)

    def filter(self, sigma: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Eq. 23; commits and returns sigma_bar_{t+1} ``[B]``. Unmasked elements keep their value."""
        if sigma.dim() != 1:
            raise ValueError(f"expected sigma of shape [B], got {tuple(sigma.shape)}")
        prev = self.sigma_bar if self.sigma_bar is not None else torch.zeros_like(sigma)
        if tuple(prev.shape) != tuple(sigma.shape):
            raise ValueError(f"sigma has shape {tuple(sigma.shape)} but the filter state has {tuple(prev.shape)}")
        prev = prev.to(device=sigma.device, dtype=sigma.dtype)
        new = torch.maximum(sigma, (1.0 - self.alpha_sigma) * prev + self.alpha_sigma * sigma)
        if mask is not None:
            new = torch.where(mask.to(device=sigma.device, dtype=torch.bool), new, prev)
        self.sigma_bar = new
        return new

    # ---------------- Eqs. 24, 25 ----------------
    def attenuation(self, sigma_bar: Tensor) -> Tensor:
        """Eq. 24 (with alpha_min above sigma_crit): non-increasing, Lipschitz (1 - alpha_min)/(sigma_crit - sigma_lo)."""
        ramp = ((sigma_bar - self.sigma_lo) / (self.sigma_crit - self.sigma_lo)).clamp(0.0, 1.0)
        return 1.0 - (1.0 - self.alpha_min) * ramp

    def validity_horizon(self, sigma_bar: Tensor) -> Tensor:
        """Eq. 25: H_t = max(H_min, floor(H_max exp(-beta sigma_bar))) -> ``[B]`` int64 in [H_min, H_max]."""
        raw = torch.floor(self.max_horizon * torch.exp(-self.beta * sigma_bar)).long()
        return raw.clamp(self.min_horizon, self.max_horizon)

    def assess(self, sigma: Tensor, primitive: Tensor, mask: Optional[Tensor] = None) -> SafetyAssessment:
        """Algorithm 1 lines 5-8: filter sigma, attenuation, validity horizon and the BRAKE_HOVER override.

        Args:
            sigma: ``[B]`` raw scores sigma_{t+1} (Eq. 22).
            primitive: ``[B]`` int64 selector output u_t.
            mask: optional ``[B]`` bool of deciding elements (Eq. 23 state is kept for the others).
        """
        if tuple(primitive.shape) != tuple(sigma.shape):
            raise ValueError(f"primitive {tuple(primitive.shape)} and sigma {tuple(sigma.shape)} must both be [B]")
        sigma_bar = self.filter(sigma, mask)
        fallback = sigma_bar >= self.sigma_crit
        overridden = torch.where(fallback, torch.full_like(primitive.long(), BRAKE_HOVER), primitive.long())
        return SafetyAssessment(sigma_bar, self.attenuation(sigma_bar), self.validity_horizon(sigma_bar),
                                fallback, overridden)

    # ---------------- Eqs. 27, 28 ----------------
    def reaction_time(self, horizon: Tensor) -> Tensor:
        """T_r = T_e2e^max + H_t Delta + T_trk: ``H_t [B]`` -> ``[B]`` (s)."""
        return self.latency_bound + horizon.to(torch.float32) * self.knot_dt + self.tracking_lag

    def safe_speed(self, depth: Tensor, reaction_time: Tensor, blind: Optional[Tensor] = None) -> Tensor:
        """Eq. 27: v_safe(d_t) ``[B]``; v_blind for blind or non-finite ranges.

        Args:
            depth: ``[B]`` d_t, smallest valid range (m) in the region swept along the commanded direction.
            reaction_time: ``[B]`` T_r (s).
            blind: optional ``[B]`` bool, True when the direction is outside the field of view or the region
                has too many invalid pixels.
        """
        if depth.dim() != 1 or tuple(reaction_time.shape) != tuple(depth.shape):
            raise ValueError("depth and reaction_time must both have shape [B]")
        t_r = reaction_time.to(depth.dtype)
        valid = torch.isfinite(depth)
        if blind is not None:
            if tuple(blind.shape) != tuple(depth.shape):
                raise ValueError("blind must have shape [B]")
            valid = valid & ~blind.to(device=depth.device, dtype=torch.bool)
        margin = depth - (self.depth_error + self.safety_radius)                     # D = d - eps_d - r_s
        positive = valid & (margin > 0.0)
        d_pos = torch.where(positive, margin, torch.ones_like(margin))               # keeps the unused branch finite
        speed = 2.0 * d_pos / (t_r + torch.sqrt(t_r * t_r + 3.0 * d_pos / self.max_decel))
        speed = torch.where(positive, speed, torch.zeros_like(speed))
        return torch.where(valid, speed, torch.full_like(speed, self.blind_speed))

    def scale_factor(self, alpha: Tensor, v_safe: Tensor, chunk: Tensor) -> Tensor:
        """Eq. 28: s_t = min{alpha, v_safe / max(max_k ||v_{t,k}||_2, eps_v)} -> ``[B]`` in [0, 1]."""
        peak = torch.linalg.vector_norm(chunk[..., :3], ord=2, dim=-1).amax(dim=-1)  # max_k ||v_{t,k}||  [B]
        return torch.minimum(alpha.to(chunk.dtype), v_safe.to(chunk.dtype) / peak.clamp_min(self.speed_eps))

    @staticmethod
    def apply_scale(chunk: Tensor, scale: Tensor) -> Tensor:
        """(s_t v_{t,k}, yaw rate_{t,k}): scale the translational components, keep the yaw rate bit-exact."""
        return torch.cat([chunk[..., :3] * scale.to(chunk.dtype).view(-1, 1, 1), chunk[..., 3:]], dim=-1)

    def limit(
        self, chunk: Tensor, alpha: Tensor, depth: Tensor, horizon: Tensor, blind: Optional[Tensor] = None
    ) -> SpeedLimit:
        """Algorithm 1 line 15: T_r, v_safe(d_t), s_t and the commanded chunk.

        Args:
            chunk: ``[B, H, 4]`` physical chunk (v_x, v_y, v_z, yaw rate) in A_0 (heading frame).
            alpha: ``[B]`` from ``assess``.
            depth: ``[B]`` d_t (m).
            horizon: ``[B]`` H_t from ``assess``.
            blind: optional ``[B]`` bool.
        """
        if chunk.dim() != 3 or chunk.shape[-1] != 4:
            raise ValueError(f"expected a chunk [B, H, 4], got {tuple(chunk.shape)}")
        b = chunk.shape[0]
        for name, tensor in (("alpha", alpha), ("depth", depth), ("horizon", horizon)):
            if tuple(tensor.shape) != (b,):
                raise ValueError(f"expected {name} of shape [{b}], got {tuple(tensor.shape)}")
        t_r = self.reaction_time(horizon).to(device=chunk.device, dtype=chunk.dtype)
        v_safe = self.safe_speed(depth.to(chunk.dtype), t_r, blind)
        scale = self.scale_factor(alpha, v_safe, chunk)
        return SpeedLimit(v_safe, t_r, scale, self.apply_scale(chunk, scale))

    # ---------------- Eq. 26 ----------------
    @staticmethod
    def braking_time(v0: Tensor, max_decel: float) -> Tensor:
        """T_b = 3 ||v_0||_2 / (2 a_max): ``[B, 3]`` -> ``[B]``."""
        return 1.5 * torch.linalg.vector_norm(v0, ord=2, dim=-1) / max_decel

    @staticmethod
    def stopping_distance(v0: Tensor, max_decel: float) -> Tensor:
        """int_0^{T_b} ||v_brk|| = 3 ||v_0||_2^2 / (4 a_max): ``[B, 3]`` -> ``[B]``."""
        return 0.75 * torch.linalg.vector_norm(v0, ord=2, dim=-1).pow(2) / max_decel

    @staticmethod
    def braking_profile(v0: Tensor, tau: Tensor, max_decel: float) -> Tuple[Tensor, Tensor]:
        """Eq. 26 and its derivative at times tau since the fallback was engaged.

        v_brk(tau) = v_0 p(s), a_brk(tau) = v_0 p'(s) / T_b, p(s) = 1 - 3 s^2 + 2 s^3 = (1 - s)^2 (1 + 2 s),
        s = clip(tau / T_b, 0, 1).
        Before engagement (tau <= 0) the velocity is v_0; after T_b it is 0; the acceleration is 0 outside
        (0, T_b), so v and a are continuous. A zero initial velocity gives zeros.

        Args:
            v0: ``[B, 3]`` velocity when the fallback is engaged (m/s).
            tau: ``[N]`` or ``[B, N]`` times (s).
            max_decel: a_max > 0.

        Returns:
            velocity ``[B, N, 3]`` and acceleration feed-forward ``[B, N, 3]``.
        """
        if not max_decel > 0.0:
            raise ValueError("max_decel must be positive")
        if v0.dim() != 2 or v0.shape[1] != 3:
            raise ValueError(f"expected v0 of shape [B, 3], got {tuple(v0.shape)}")
        b = v0.shape[0]
        if tau.dim() == 1:
            tau = tau.unsqueeze(0).expand(b, -1)
        if tau.dim() != 2 or tau.shape[0] != b:
            raise ValueError(f"expected tau of shape [N] or [{b}, N], got {tuple(tau.shape)}")
        tau = tau.to(v0.dtype)
        t_b = SafetyFilter.braking_time(v0, max_decel)                               # [B]
        moving = (t_b > 0.0).unsqueeze(1)                                             # [B, 1]
        t_b_safe = torch.where(t_b > 0.0, t_b, torch.ones_like(t_b)).unsqueeze(1)    # [B, 1]
        s = torch.where(moving, (tau / t_b_safe).clamp(0.0, 1.0), torch.ones_like(tau))
        # p(s) = 1 - 3 s^2 + 2 s^3 = (1 - s)^2 (1 + 2 s): the factored form has no cancellation near s = 1,
        # where the expanded one loses ~1e-6 m/s absolute in float32 at the end of the stop.
        shape = (1.0 - s) * (1.0 - s) * (1.0 + 2.0 * s)                               # p(s)
        inside = moving & (tau > 0.0) & (tau < t_b_safe)
        slope = torch.where(inside, -6.0 * s * (1.0 - s) / t_b_safe, torch.zeros_like(s))  # p'(s) / T_b
        return v0.unsqueeze(1) * shape.unsqueeze(-1), v0.unsqueeze(1) * slope.unsqueeze(-1)


def executed_prefix_mask(executed_horizon: Tensor, horizon: int) -> Tensor:
    """Synchronous executed prefix (ablation A8): ``H_t [B]`` -> ``[B, H]``, True for knots k < H_t."""
    steps = torch.arange(horizon, device=executed_horizon.device)
    return steps.unsqueeze(0) < executed_horizon.unsqueeze(1)


# ---------------------------------------------------------------------------------------------
# Self-test (run: python solver.py)
# ---------------------------------------------------------------------------------------------
_RTOL, _ATOL = 1e-5, 1e-6              # float32 module output vs float64 reference (CLAUDE.md)
_RTOL_WIDE, _ATOL_WIDE = 1e-4, 1e-5    # full Table 1 transformer (4 layers of width 256, 1329-wide input)


def _close(actual: Any, ref: Any, rtol: float = _RTOL, atol: float = _ATOL) -> bool:
    a = torch.as_tensor(actual).detach().double()
    r = torch.as_tensor(ref).detach().double()
    return torch.allclose(a, r, rtol=rtol, atol=atol)


def _expect(exc: type, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _config_module() -> Any:
    """The sibling ``config.py`` (package member or flat file); used only by the self-test."""
    import importlib

    if __package__:
        return importlib.import_module(".config", __package__)
    return importlib.import_module("config")


class _CountingField(nn.Module):
    """Wraps v_theta and records the flow time of every evaluation (self-test only)."""

    def __init__(self, field: nn.Module) -> None:
        super().__init__()
        self.field = field
        self.calls: List[float] = []

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        # Record rho exactly as passed: torch.as_tensor(float) would round to float32.
        self.calls.append(rho if isinstance(rho, float) else float(rho.reshape(-1)[0]))
        return self.field(actions, rho, cond)


class _ZeroField(nn.Module):
    """Stand-in v_theta == 0 (self-test only)."""

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        return torch.zeros_like(actions)


class _AffineField(nn.Module):
    """Stand-in v(A, rho, e) = a A + b rho + reshape(e W): L_A = a, L_e = ||W||_2 (self-test only).

    Evaluates in the dtype of its inputs (float64 for the Proposition 3.1 check).
    """

    def __init__(self, horizon: int, action_dim: int, cond_dim: int, generator: torch.Generator) -> None:
        super().__init__()
        self.a, self.b = 0.7, 0.3
        self.register_buffer("w", 0.1 * torch.randn(cond_dim, horizon * action_dim, generator=generator))

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        bsz = actions.shape[0]
        rho_t = torch.as_tensor(rho, dtype=actions.dtype)
        rho_t = rho_t.expand(bsz) if rho_t.dim() == 0 else rho_t
        return self.a * actions + self.b * rho_t.view(bsz, 1, 1) + (cond @ self.w.to(cond.dtype)).view(actions.shape)


class _RecordingField(nn.Module):
    """Wraps v_theta and records its last input and output (self-test only)."""

    def __init__(self, field: nn.Module) -> None:
        super().__init__()
        self.field = field
        self.record: Optional[Tuple[Tensor, Tensor, Tensor]] = None

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        out = self.field(actions, rho, cond)
        self.record = (actions.detach().clone(), torch.as_tensor(rho).detach().clone(), out)
        return out


def _ref_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """LayerNorm over the last axis, written out: (x - mean) / sqrt(biased var + eps) * w + b."""
    mean = x.sum(-1, keepdim=True) / x.shape[-1]
    var = ((x - mean) ** 2).sum(-1, keepdim=True) / x.shape[-1]
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def _ref_gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _ref_time_embedding(rho: Tensor, dim: int, scale: float = 1000.0) -> Tensor:
    """Float64 sinusoidal features with exact frequencies 10000^(-i/n): ``[B]`` -> ``[B, dim]``."""
    half = dim // 2
    freqs = torch.tensor([math.exp(-math.log(10000.0) * i / half) for i in range(half)], dtype=torch.float64)
    args = scale * rho.double().unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def _ref_vector_field(field: ConditionalVectorField, actions: Tensor, cond: Tensor, time_features: Tensor) -> Tensor:
    """Independent float64 reference of v_theta from its definition (Table 1 architecture), with
    per-head softmax attention, exact GELU and hand-written LayerNorms. ``time_features [B, 256]`` is
    supplied (the sinusoidal embedding is checked separately). Returns ``[B, H, d_a]`` float64."""
    sd = {k: v.detach().double() for k, v in field.state_dict().items()}

    def lin(x: Tensor, name: str) -> Tensor:
        return x @ sd[name + ".weight"].T + sd[name + ".bias"]

    c = lin(_ref_gelu(lin(torch.cat([cond.double(), time_features.double()], dim=-1), "cond_mlp.0")), "cond_mlp.2")
    tokens = lin(actions.double(), "action_in") + sd["pos_embed"] + c.unsqueeze(1)
    x = torch.cat([c.unsqueeze(1), tokens], dim=1)                                            # [B, H + 1, W]
    width, heads = field.width, field.num_heads
    dh = width // heads
    for n, layer in enumerate(field.blocks.layers):
        p = f"blocks.layers.{n}."
        hn = _ref_layer_norm(x, sd[p + "norm1.weight"], sd[p + "norm1.bias"], layer.norm1.eps)
        w_in, b_in = sd[p + "self_attn.in_proj_weight"], sd[p + "self_attn.in_proj_bias"]
        q = hn @ w_in[:width].T + b_in[:width]
        k = hn @ w_in[width:2 * width].T + b_in[width:2 * width]
        v = hn @ w_in[2 * width:].T + b_in[2 * width:]
        outs = []
        for a in range(heads):
            sl = slice(a * dh, (a + 1) * dh)
            scores = q[..., sl] @ k[..., sl].transpose(1, 2) / math.sqrt(dh)                # no mask
            e = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
            outs.append((e / e.sum(dim=-1, keepdim=True)) @ v[..., sl])
        x = x + lin(torch.cat(outs, dim=-1), p + "self_attn.out_proj")
        hn = _ref_layer_norm(x, sd[p + "norm2.weight"], sd[p + "norm2.bias"], layer.norm2.eps)
        x = x + lin(_ref_gelu(lin(hn, p + "linear1")), p + "linear2")
    return lin(_ref_layer_norm(x[:, 1:], sd["norm.weight"], sd["norm.bias"], field.norm.eps), "action_out")


def _ref_conditioning(solver: CFMSolver, u: Tensor, z_t: Tensor, z_hat: Tensor, s_t: Tensor, g_t: Tensor,
                      c: Tensor) -> Tensor:
    """Float64 reference of Eq. 15 with the standardisation, written slot by slot."""
    du, d, ds, dg, dc = solver.primitive_embed_dim, solver.latent_dim, solver.proprio_dim, solver.goal_dim, solver.text_dim
    bsz = z_t.shape[0]
    emb = solver.primitive_embedding.weight.detach().double()
    mu_z, gam = solver.latent_mean.double(), float(solver.gamma_bar_star)
    mu_s, sd_s = solver.proprio_mean.double(), solver.proprio_std.double()
    e = torch.zeros(bsz, du + 2 * d + ds + dg + dc, dtype=torch.float64)
    for k in range(bsz):
        for j in range(du):
            e[k, j] = emb[int(u[k]), j]
        for j in range(d):
            e[k, du + j] = (float(z_t[k, j]) - float(mu_z[j])) / gam
            e[k, du + d + j] = (float(z_hat[k, j]) - float(mu_z[j])) / gam
        for j in range(ds):
            e[k, du + 2 * d + j] = (float(s_t[k, j]) - float(mu_s[j])) / float(sd_s[j])
        for j in range(dg):
            e[k, du + 2 * d + ds + j] = float(g_t[k, j])
        for j in range(dc):
            e[k, du + 2 * d + ds + dg + j] = float(c[k, j])
    return e


def _ref_safe_speed(depth: float, reaction_time: float, clearance: float, max_decel: float) -> float:
    """Eq. 27 as printed: (2 a / 3)(-T_r + sqrt(T_r^2 + 3 D / a)) for D = d - eps_d - r_s > 0, else 0."""
    margin = depth - clearance
    if margin <= 0.0:
        return 0.0
    return 2.0 * max_decel / 3.0 * (-reaction_time + math.sqrt(reaction_time ** 2 + 3.0 * margin / max_decel))


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(2610)
    config = _config_module()
    cfg = config.VPAConfig()
    assert config.PRIMITIVES.index("BRAKE_HOVER") == BRAKE_HOVER
    b, h, da, de = 3, cfg.horizon, cfg.action_dim, cfg.cond_dim

    # ---- Table 1: v_theta has exactly 3,572,228 parameters; the solver with Embed has 3,572,420 (both 3.57M).
    solver = CFMSolver.from_config(cfg).eval()
    field = solver.vector_field
    assert (h, da, de) == (16, 4, 1073) and solver.cond_dim == de == field.cond_dim
    groups: dict = {}
    for name, p in field.named_parameters():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
    assert groups == {"action_in": 1_280, "pos_embed": 4_096, "cond_mlp": 406_272, "blocks": 3_159_040,
                      "norm": 512, "action_out": 1_028}, groups
    assert len(field.blocks.layers) == 4 and all(count_parameters(layer) == 789_760 for layer in field.blocks.layers)
    layer0 = field.blocks.layers[0]
    assert layer0.norm_first and layer0.self_attn.num_heads == 4 and layer0.linear1.out_features == 1024
    assert count_parameters(field) == AERO_FIELD_PARAMS == 3_572_228 == config._solver_params(cfg)
    assert count_parameters(solver) == count_parameters(solver, trainable_only=False) == AERO_SOLVER_PARAMS
    assert AERO_SOLVER_PARAMS == AERO_FIELD_PARAMS + cfg.num_primitives * cfg.primitive_embed_dim
    assert round(AERO_FIELD_PARAMS / 1e6, 2) == round(AERO_SOLVER_PARAMS / 1e6, 2) == config.TABLE1_PARAMS_M["solver"]
    as_config = ConditionalVectorField(cfg.horizon, cfg.action_dim, cfg.cond_dim, width=cfg.field_width,
                                       depth=cfg.field_depth, num_heads=cfg.field_heads, mlp_ratio=cfg.field_mlp_ratio,
                                       time_embed_dim=cfg.time_embed_dim)                  # the call config.py makes
    assert count_parameters(as_config) == count_parameters(ConditionalVectorField()) == AERO_FIELD_PARAMS
    assert count_parameters(ConditionalVectorField(time_embed_dim=128)) == 3_539_460      # why Table 1 needs 256

    # ---- H + 1 = 17 tokens of width 256; output [B, 16, 4]; float64 reference of the architecture.
    seen = []
    pre = field.blocks.register_forward_pre_hook(lambda _m, args: seen.append(tuple(args[0].shape)))
    actions = torch.randn(b, h, da, generator=gen)
    e_rand = torch.randn(b, de, generator=gen)
    with torch.no_grad():
        v_out = field(actions, 0.5, e_rand)
    pre.remove()
    assert seen == [(b, 17, 256)] and v_out.shape == (b, h, da) and v_out.dtype == torch.float32
    with torch.no_grad():
        temb = field.time_embed(torch.full((b,), 0.5))
    assert _close(v_out, _ref_vector_field(field, actions, e_rand, temb), _RTOL_WIDE, _ATOL_WIDE)
    rho_grid = torch.tensor([0.0, 0.25, 0.5, 1.0])
    temb_grid = field.time_embed(rho_grid)
    assert temb_grid.shape == (4, 256)
    assert torch.equal(temb_grid[0], torch.cat([torch.zeros(128), torch.ones(128)]))       # rho = 0
    # arguments reach 1000 rad, where float32 argument rounding is ~3e-5 rad
    assert _close(temb_grid, _ref_time_embedding(rho_grid, 256), rtol=0.0, atol=2e-4)
    with torch.no_grad():
        v_vec = field(actions, torch.full((b,), 0.5), e_rand)
    assert torch.equal(v_vec, v_out)                                                         # scalar rho == [B] rho
    _expect(ValueError, lambda: field(actions[:, :-1], 0.5, e_rand))
    _expect(ValueError, lambda: field(actions, 0.5, e_rand[:, :-1]))
    _expect(ValueError, lambda: field(actions, torch.zeros(b + 1), e_rand))
    _expect(ValueError, lambda: ConditionalVectorField(width=250, num_heads=4))

    # ---- Eq. 15: e_t layout and standardisation, slot by slot against float64.
    u = torch.randint(0, cfg.num_primitives, (b,), generator=gen)
    z_t, z_hat = torch.randn(b, 256, generator=gen), torch.randn(b, 256, generator=gen)
    s_t, g_t = torch.randn(b, 13, generator=gen), torch.randn(b, 4, generator=gen)
    c_text = torch.randn(b, 512, generator=gen)
    mu_z, gamma_star = torch.randn(256, generator=gen), 1.7
    mu_s, sd_s = torch.randn(13, generator=gen), torch.rand(13, generator=gen) + 0.5
    _expect(RuntimeError, lambda: solver.build_conditioning(u, z_t, z_hat, s_t, g_t, c_text))
    bad_sd = sd_s.clone()
    bad_sd[3] = 0.0
    nan_mu = mu_z.clone()
    nan_mu[0] = float("nan")
    for args in ((mu_z, gamma_star, mu_s, bad_sd), (mu_z, 0.0, mu_s, sd_s), (mu_z, float("inf"), mu_s, sd_s),
                 (mu_z[:-1], gamma_star, mu_s, sd_s), (mu_z, gamma_star, mu_s[:-1], sd_s), (nan_mu, gamma_star, mu_s, sd_s)):
        _expect(ValueError, lambda a=args: solver.set_standardization_stats(*a))
    assert not solver.is_calibrated                                                          # nothing was installed
    solver.set_standardization_stats(mu_z, gamma_star, mu_s, sd_s)
    e_t = solver.build_conditioning(u, z_t, z_hat, s_t, g_t, c_text)
    assert e_t.shape == (b, de)
    sl = cfg.cond_slices
    ref_e = _ref_conditioning(solver, u, z_t, z_hat, s_t, g_t, c_text)
    assert torch.equal(e_t[:, sl["primitive"]], solver.primitive_embedding.weight[u])        # Embed(u_t)
    assert _close(e_t[:, sl["z_t"]], ref_e[:, sl["z_t"]]) and _close(e_t[:, sl["z_hat"]], ref_e[:, sl["z_hat"]])
    assert _close(e_t[:, sl["proprio"]], ref_e[:, sl["proprio"]])
    assert torch.equal(e_t[:, sl["goal"]], g_t) and torch.equal(e_t[:, sl["text"]], c_text)  # not re-standardised
    assert _close(e_t, ref_e)
    for args in ((u, z_t[:, :-1], z_hat, s_t, g_t, c_text), (u, z_t, z_hat[:-1], s_t, g_t, c_text),
                 (u, z_t, z_hat, s_t[:, :-1], g_t, c_text), (u, z_t, z_hat, s_t, g_t[:, :3], c_text),
                 (u, z_t, z_hat, s_t, g_t, c_text[:, :-1]), (u[:-1], z_t, z_hat, s_t, g_t, c_text),
                 (u, z_t[0], z_hat, s_t, g_t, c_text)):
        _expect(ValueError, lambda a=args: solver.build_conditioning(*a))
    e_t = e_t.detach()
    # withheld groups of S_t (config.policy_state_mask): zero in e_t, everything else unchanged; not saved
    from config import VPAConfig as _Config
    masked = CFMSolver.from_config(_Config(policy_state_inputs=("velocity", "attitude", "rates")))
    masked.load_state_dict(solver.state_dict())
    assert "proprio_mask" not in masked.state_dict() and torch.equal(solver.proprio_mask, torch.ones(13))
    e_masked = masked.build_conditioning(u, z_t, z_hat, s_t, g_t, c_text)
    proprio = sl["proprio"].start
    assert torch.equal(e_masked[:, proprio + 9:proprio + 13], torch.zeros(b, 4))
    assert torch.equal(e_masked[:, :proprio + 9], e_t[:, :proprio + 9]) and torch.equal(e_masked[:, proprio + 13:],
                                                                                          e_t[:, proprio + 13:])
    _expect(ValueError, lambda: CFMSolver(proprio_mask=(1.0,) * 12))

    # ---- Eqs. 16, 17: flow-matching loss with a fixed generator (xi drawn before rho).
    a_true = torch.rand(b, h, da, generator=gen) * 2.0 - 1.0                                  # normalised chunk

    def replay(seed: int) -> Tuple[Tensor, Tensor]:
        g = torch.Generator().manual_seed(seed)
        xi_r = torch.randn(a_true.shape, generator=g)
        return xi_r, torch.rand(b, generator=g)

    real_field = solver.vector_field
    solver.vector_field = _ZeroField()
    loss0 = solver.flow_matching_loss(a_true, e_t, generator=torch.Generator().manual_seed(7))
    xi_r, rho_r = replay(7)
    ref0 = sum(float(a_true[k, i, j] - xi_r[k, i, j]) ** 2 for k in range(b) for i in range(h) for j in range(da)) / b
    assert _close(loss0, ref0)
    rec = _RecordingField(real_field)
    solver.vector_field = rec
    loss = solver.flow_matching_loss(a_true, e_t, generator=torch.Generator().manual_seed(7))
    a_in, rho_in, vel = rec.record
    vel_ref = vel.detach()
    assert torch.equal(rho_in, rho_r) and bool(((rho_r >= 0) & (rho_r < 1)).all())
    a_rho_ref = torch.zeros(b, h, da, dtype=torch.float64)
    for k in range(b):
        r = float(rho_r[k])
        a_rho_ref[k] = r * a_true[k].double() + (1.0 - r) * xi_r[k].double()                 # Eq. 16
    assert _close(a_in, a_rho_ref)
    ref_loss = sum((float(vel_ref[k, i, j]) - (float(a_true[k, i, j]) - float(xi_r[k, i, j]))) ** 2
                   for k in range(b) for i in range(h) for j in range(da)) / b                  # Eq. 17
    assert _close(loss, ref_loss)
    loss.backward()
    assert all(p.grad is not None for p in real_field.parameters())
    solver.vector_field = real_field
    _expect(ValueError, lambda: solver.flow_matching_loss(a_true[:, :-1], e_t))

    # ---- Eq. 18 for K = 1, 2, 4, 8 (ablation A1) with an affine field against a float64 Euler loop.
    xi = torch.randn(b, h, da, generator=gen)
    for k in (1, 2, 4, 8):
        solver.num_integration_steps = k
        affine = _AffineField(h, da, de, gen)
        counting = _CountingField(affine)
        solver.vector_field = counting
        with torch.no_grad():
            a_hat = solver.integrate(e_t, xi)
        assert counting.calls == [j / k for j in range(k)]                                     # rho_k = k / K
        # The conditioning term is the stand-in's own float32 product e_t W: its 1073-long dot products carry
        # ~5e-6 absolute rounding, which is a property of the stand-in, not of Eq. 18. Taking it as given makes
        # the float64 loop check the Euler recursion itself at rtol = 1e-5, atol = 1e-6.
        with torch.no_grad():
            bias = (e_t @ affine.w).double().view(b, h, da)
        a_ref = xi.double()
        for j in range(k):
            a_ref = a_ref + (1.0 / k) * (affine.a * a_ref + affine.b * (j / k) + bias)
        assert a_hat.shape == (b, h, da) and _close(a_hat, a_ref, rtol=1e-5, atol=1e-6)
    solver.num_integration_steps = cfg.integration_steps
    _expect(ValueError, lambda: solver.integrate(e_t, xi[:, :-1]))

    # ---- Eq. 19 (K = 2) with the real v_theta: two evaluations at rho = 0 and 1/2, the closed form, and
    #      the float64 reference of the architecture.
    counting = _CountingField(real_field)
    solver.vector_field = counting
    with torch.no_grad():
        a_hat = solver.integrate(e_t, xi)
        v0 = real_field(xi, 0.0, e_t)
        eq19 = xi + 0.5 * v0 + 0.5 * real_field(xi + 0.5 * v0, 0.5, e_t)
        ref_a = xi.double()
        for j in range(2):
            temb_j = real_field.time_embed(torch.full((b,), j / 2))
            ref_a = ref_a + 0.5 * _ref_vector_field(real_field, ref_a, e_t, temb_j)
    solver.vector_field = real_field
    assert counting.calls == [0.0, 0.5] and torch.equal(a_hat, eq19)
    assert _close(a_hat, ref_a, _RTOL_WIDE, _ATOL_WIDE)

    # ---- generate_chunk = Pi_{A_0}(denormalise(Euler)); explicit noise == generator noise.
    scale = torch.tensor(cfg.action_scale)
    with torch.no_grad():
        big = 10.0 * xi                                                                       # pushes knots out of the box
        chunk_big = solver.generate_chunk(e_t, noise=big)
        norm_big = solver.integrate(e_t, big)
        c1 = solver.generate_chunk(e_t, generator=torch.Generator().manual_seed(11))
        c2 = solver.generate_chunk(e_t, noise=torch.randn(b, h, da, generator=torch.Generator().manual_seed(11)))
    assert torch.equal(chunk_big, torch.maximum(torch.minimum(norm_big * scale, scale), -scale))
    assert bool((chunk_big.abs() <= scale).all()) and bool((chunk_big.abs() == scale).any())
    assert torch.equal(c1, c2) and c1.shape == (b, h, da)
    raw = torch.tensor([[[8.0, -9.0, 1.0, 0.2], [-3.0, 2.0, -5.0, -2.0]]])
    assert torch.equal(solver.project(raw), torch.tensor([[[7.0, -7.0, 1.0, 0.2], [-3.0, 2.0, -2.0, -1.5]]]))
    assert _close(solver.normalize_actions(solver.denormalize_actions(a_true)), a_true)
    assert _close(solver.normalize_actions(raw), raw.double() / scale.double())
    for kw in (dict(num_integration_steps=0), dict(action_scale=(7.0, 7.0, 2.0)), dict(action_scale=(7.0, 0.0, 2.0, 1.5))):
        _expect(ValueError, lambda k=kw: CFMSolver(**k))

    # ---- Eq. 20: noise persistence against a per-element reference driven by the same random stream.
    n_xi = 3
    pers = NoisePersistence(h, da, refresh_period=n_xi)
    _expect(RuntimeError, lambda: pers.step(torch.zeros(2, dtype=torch.long)))
    pers.reset(2)
    g_mod, g_ref = torch.Generator().manual_seed(31), torch.Generator().manual_seed(31)
    script = [
        ("step", [1, 2], [True, True]),    # t = 0: fresh (first decision)
        ("step", [1, 3], [True, True]),    # t = 1: element 0 reuses; element 1 changed primitive -> fresh
        ("step", [1, 3], [True, True]),    # t = 2: both reuse
        ("step", [1, 3], [True, True]),    # t = 3 = N_xi: both fresh
        ("step", [1, 3], [True, False]),   # element 1 not deciding: keeps noise, primitive and counter
        ("step", [1, 3], [True, True]),    # element 0 at t = 5, element 1 at t = 4: both reuse
        ("skip", [BRAKE_HOVER, BRAKE_HOVER], [True, True]),  # fallback decision: no chunk, no draw
        ("step", [1, 3], [True, True]),    # previous primitive was BRAKE_HOVER -> fresh for both
    ]
    noise_ref: List[Optional[Tensor]] = [None, None]
    prev_ref, t_ref, log = [-1, -1], [0, 0], []
    for kind, prims, mask in script:
        u_s, m_s = torch.tensor(prims), torch.tensor(mask)
        if kind == "skip":
            pers.skip(u_s, mask=m_s)
            for i in range(2):
                if mask[i]:
                    prev_ref[i], t_ref[i] = prims[i], t_ref[i] + 1
            continue
        out = pers.step(u_s, generator=g_mod, mask=m_s)
        fresh = torch.randn(2, h, da, generator=g_ref)
        flags = []
        for i in range(2):
            refresh = mask[i] and (prims[i] != prev_ref[i] or t_ref[i] % n_xi == 0)
            if refresh:
                noise_ref[i] = fresh[i].clone()
            if mask[i]:
                prev_ref[i], t_ref[i] = prims[i], t_ref[i] + 1
            flags.append(refresh)
            assert torch.equal(out[i], noise_ref[i])
        assert pers.last_refreshed.tolist() == flags and out.shape == (2, h, da)
        log.append(flags)
    assert log == [[True, True], [False, True], [False, False], [True, True], [False, False], [False, False],
                   [True, True]]
    assert pers.counter.tolist() == t_ref == [8, 7] and pers.prev_primitive.tolist() == [1, 3]
    _expect(ValueError, lambda: pers.step(torch.zeros(3, dtype=torch.long)))
    _expect(ValueError, lambda: pers.step(torch.zeros(2, dtype=torch.long), mask=torch.ones(3, dtype=torch.bool)))
    _expect(ValueError, lambda: NoisePersistence(h, da, refresh_period=0))

    # ---- Proposition 3.1 with an affine field (L_A = a, L_e = ||W||_2), float64: with a shared xi the chunk
    #      change is exactly linear in e - e' and obeys Eq. 21, with equality along the top singular direction.
    affine = _AffineField(h, da, de, gen)
    w64 = affine.w.double()
    lip_e = float(torch.linalg.matrix_norm(w64, ord=2))
    top = torch.linalg.svd(w64, full_matrices=False).U[:, 0]                                # ||top||_2 = 1
    xi64, e64 = xi.double(), e_t.double()
    solver.vector_field = affine
    for k in (1, 2, 4, 8):
        solver.num_integration_steps = k
        base = solver.integrate(e64, xi64)
        bound = chunk_coherence_bound(affine.a, lip_e, k, 1.0)
        assert bound <= chunk_coherence_bound(affine.a, lip_e, k, 1.0, exponential=True)
        for raw_dir, tight in ((top, True), (torch.randn(de, generator=gen, dtype=torch.float64), False)):
            unit_dir = raw_dir / raw_dir.norm()
            diffs = {lam: solver.integrate(e64 + lam * unit_dir, xi64) - base for lam in (0.5, 1.0, 2.0)}
            assert _close(2.0 * diffs[0.5], diffs[1.0], rtol=1e-10, atol=1e-12)               # proportional to ||e - e'||
            assert _close(0.5 * diffs[2.0], diffs[1.0], rtol=1e-10, atol=1e-12)
            change = diffs[1.0].flatten(1).norm(dim=-1)                                       # ||e - e'||_2 = 1 per row
            assert bool((change <= bound * (1.0 + 1e-10)).all())
            if tight:
                assert _close(change, torch.full((b,), bound, dtype=torch.float64), rtol=1e-9, atol=0.0)
            else:
                assert bool((change < bound).all())
    solver.vector_field = real_field
    solver.num_integration_steps = cfg.integration_steps
    for bad in ((0.0, 1.0, 2, 1.0), (0.7, -1.0, 2, 1.0), (0.7, 1.0, 0, 1.0), (0.7, 1.0, 2, -1.0)):
        _expect(ValueError, lambda a=bad: chunk_coherence_bound(*a))

    # ---- Proposition 3.1 mechanism with the real v_theta (float64 copy): with a shared xi the chunk change
    #      vanishes linearly as e' -> e (bounded difference quotient); with independent draws it does not.
    solver64 = copy.deepcopy(solver).double()
    direction = torch.randn(de, generator=gen, dtype=torch.float64)
    direction = direction / direction.norm()
    with torch.no_grad():
        base = solver64.integrate(e64, xi64)
        quotients = [(solver64.integrate(e64 + lam * direction, xi64) - base).flatten(1).norm(dim=-1) / lam
                     for lam in (1e-4, 1e-5)]
        xi_other = torch.randn(b, h, da, generator=gen, dtype=torch.float64)
        independent = (solver64.integrate(e64 + 1e-5 * direction, xi_other) - base).flatten(1).norm(dim=-1)
    assert bool((quotients[0] > 0).all()) and _close(quotients[0], quotients[1], rtol=1e-2, atol=0.0)
    assert bool((independent > 100.0 * 1e-5 * quotients[1]).all())

    # ---- Safety filter configuration (clearance eps_d + r_s = 0.75 m is exact in float32).
    sf_args = dict(sigma_lo=0.2, sigma_crit=0.6, alpha_min=0.25, alpha_sigma=0.25, min_horizon=4, max_horizon=16,
                   beta=1.0, max_decel=5.0, depth_error=0.25, safety_radius=0.5, blind_speed=1.0, speed_eps=1e-3,
                   knot_dt=0.02, latency_bound=0.05, tracking_lag=0.03)
    sf = SafetyFilter(**sf_args)
    clearance, a_max = 0.75, 5.0

    # ---- Eq. 23: immediate rise, gradual decay, masked elements frozen; alpha_sigma = 1 is no filtering.
    seq = [0.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.1, 0.0, 0.0, 0.6]
    mask1 = [True, True, False, True, False, True, True, False, True, True, False, True]
    sf.reset(3)
    assert torch.equal(sf.sigma_bar, torch.zeros(3))
    ref_prev, prev_mod = [0.0, 0.0, 0.0], sf.sigma_bar.clone()
    rises, decays = [], []
    for t, s in enumerate(seq):
        sig = [s, s, 0.5 * s]
        mask = [True, mask1[t], True]
        sb = sf.filter(torch.tensor(sig), mask=torch.tensor(mask))
        ref = [max(sig[i], (1 - 0.25) * ref_prev[i] + 0.25 * sig[i]) if mask[i] else ref_prev[i] for i in range(3)]
        assert _close(sb, torch.tensor(ref, dtype=torch.float64))
        if t == 0:
            assert float(sb[0]) == float(torch.tensor(s))                                     # sigma_bar_0 = 0
        if sig[0] > ref_prev[0]:                                                              # rise: immediate, exact
            rises.append(t)
            assert float(sb[0]) == float(torch.tensor(sig[0]))
        if sig[0] < ref_prev[0]:                                                              # decay: gradual, above sigma
            decays.append(t)
            assert sig[0] < float(sb[0]) < float(prev_mod[0])
        if not mask[1]:
            assert float(sb[1]) == float(prev_mod[1])
        ref_prev, prev_mod = ref, sb.clone()
    assert {1, 2, 11} <= set(rises) and {6, 7, 8, 9, 10} <= set(decays)
    instant = SafetyFilter(**dict(sf_args, alpha_sigma=1.0))
    instant.reset(1)
    for s in seq:
        assert torch.equal(instant.filter(torch.tensor([s])), torch.tensor([s]))
    _expect(ValueError, lambda: sf.filter(torch.zeros(4)))                                    # state is [3]

    # ---- Eq. 24: 1 below sigma_lo, linear ramp, alpha_min from sigma_crit on; monotone and Lipschitz.
    grid = torch.linspace(0.0, 1.0, 101)
    alpha = sf.attenuation(grid)
    ref_alpha = [1.0 if v <= 0.2 else (0.25 if v >= 0.6 else 1.0 - 0.75 * (v - 0.2) / 0.4) for v in grid.double().tolist()]
    assert _close(alpha, torch.tensor(ref_alpha, dtype=torch.float64))
    assert bool((alpha[grid < 0.19] == 1.0).all()) and bool((alpha[grid > 0.61] == 0.25).all())
    assert bool((alpha[1:] <= alpha[:-1]).all())
    lipschitz = (1.0 - 0.25) / (0.6 - 0.2)
    assert bool(((alpha[1:] - alpha[:-1]).abs() <= lipschitz * (grid[1:] - grid[:-1]) * (1 + 1e-5) + 1e-7).all())
    assert _close(sf.attenuation(torch.tensor([0.4])), torch.tensor([0.625]))

    # ---- Eq. 25: H_t = max(H_min, floor(H_max exp(-beta sigma_bar))).
    assert sf.validity_horizon(torch.tensor([0.0, 0.5, 50.0])).tolist() == [16, 9, 4]
    grid2 = torch.linspace(0.0, 3.0, 31)[1:]
    for v in grid2.tolist():
        x = 16 * math.exp(-v)
        assert abs(x - round(x)) > 1e-4, v                                                    # floor robust to float32
    ht = sf.validity_horizon(grid2)
    assert ht.dtype == torch.long and ht.tolist() == [max(4, math.floor(16 * math.exp(-v))) for v in grid2.tolist()]
    assert bool((ht[1:] <= ht[:-1]).all())

    # ---- Override: sigma_bar >= sigma_crit (inclusive) -> BRAKE_HOVER and fallback; others untouched.
    instant.reset(3)
    assessment = instant.assess(torch.tensor([0.1, 0.6, 0.9]), torch.tensor([0, 1, 2]))
    assert assessment.fallback.tolist() == [False, True, True]
    assert assessment.primitive.tolist() == [0, BRAKE_HOVER, BRAKE_HOVER] and assessment.primitive.dtype == torch.long
    assert torch.equal(assessment.alpha, instant.attenuation(assessment.sigma_bar))
    assert torch.equal(assessment.horizon, instant.validity_horizon(assessment.sigma_bar))
    assert float(assessment.alpha[0]) == 1.0 and float(assessment.alpha[2]) == 0.25
    held = instant.assess(torch.tensor([0.9, 0.0, 0.0]), torch.tensor([3, 4, 1]), mask=torch.tensor([False, True, True]))
    assert held.primitive.tolist() == [3, 4, 1] and held.fallback.tolist() == [False, False, False]
    _expect(ValueError, lambda: instant.assess(torch.zeros(3), torch.zeros(2, dtype=torch.long)))

    # ---- Eq. 26: minimum-jerk stop. v(0) = v_0, v(T_b) = 0, a(0) = a(T_b) = 0, peak deceleration a_max at
    #      T_b / 2, stopping distance 3 ||v_0||^2 / (4 a_max) by quadrature; zero speed gives zeros.
    v0 = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 2.0, 2.0]])                 # speeds 5, 0, 3
    t_b = SafetyFilter.braking_time(v0, a_max)
    assert _close(t_b, torch.tensor([1.5, 0.0, 0.9])) and _close(SafetyFilter.stopping_distance(v0, a_max),
                                                                 torch.tensor([3.75, 0.0, 1.35]))
    v_end, a_end = SafetyFilter.braking_profile(v0, t_b.unsqueeze(1), a_max)
    assert torch.equal(v_end, torch.zeros(3, 1, 3)) and torch.equal(a_end, torch.zeros(3, 1, 3))
    v_start, a_start = SafetyFilter.braking_profile(v0, torch.zeros(1), a_max)
    assert torch.equal(v_start[:, 0], v0) and torch.equal(a_start, torch.zeros(3, 1, 3))
    _, a_mid = SafetyFilter.braking_profile(v0, (0.5 * t_b).unsqueeze(1), a_max)
    assert _close(a_mid[:, 0].norm(dim=-1), torch.tensor([a_max, 0.0, a_max]))
    tau = torch.linspace(0.0, 2.0, 20001)
    vel, acc = SafetyFilter.braking_profile(v0, tau, a_max)
    assert vel.shape == (3, 20001, 3) and acc.shape == (3, 20001, 3)
    assert bool(torch.isfinite(vel).all()) and bool(torch.isfinite(acc).all())
    assert bool((acc.norm(dim=-1) <= a_max * (1.0 + 1e-5)).all())                            # peak deceleration <= a_max
    assert torch.equal(vel[1], torch.zeros(20001, 3)) and torch.equal(vel[0, tau >= 1.5], torch.zeros(int((tau >= 1.5).sum()), 3))
    s_ref = (tau.double().unsqueeze(0) / torch.tensor([1.5, 1.0, 0.9], dtype=torch.float64).unsqueeze(1)).clamp(0.0, 1.0)
    p_ref = 1.0 - 3.0 * s_ref ** 2 + 2.0 * s_ref ** 3
    p_ref[1] = 0.0
    assert _close(vel, v0.double().unsqueeze(1) * p_ref.unsqueeze(-1))
    speed0 = vel[0].norm(dim=-1).double()
    assert abs(float(torch.trapezoid(speed0, tau.double())) - 3.75) < 1e-6 * 3.75 + 1e-6
    unit = v0[0] / v0[0].norm()
    assert _close(vel[0] - (vel[0] @ unit).unsqueeze(-1) * unit, torch.zeros(20001, 3))       # parallel to v_0
    # Eq. 26 is the minimum-jerk profile: adding an admissible variation phi(s) = s^2 (1 - s)^2 (zero value
    # and slope at both ends) leaves the first variation zero and increases int p''^2 (float64 quadrature).
    s64 = torch.linspace(0.0, 1.0, 200001, dtype=torch.float64)
    p2, phi2 = -6.0 + 12.0 * s64, 2.0 - 12.0 * s64 + 12.0 * s64 ** 2
    jerk0 = float(torch.trapezoid(p2 ** 2, s64))
    assert abs(jerk0 - 12.0) < 1e-6 and abs(float(torch.trapezoid(p2 * phi2, s64))) < 1e-8
    for eps in (-0.5, -0.1, 0.1, 0.5):
        assert float(torch.trapezoid((p2 + eps * phi2) ** 2, s64)) > jerk0
    _expect(ValueError, lambda: SafetyFilter.braking_profile(v0[:, :2], tau, a_max))
    _expect(ValueError, lambda: SafetyFilter.braking_profile(v0, torch.zeros(2, 5), a_max))
    _expect(ValueError, lambda: SafetyFilter.braking_profile(v0, tau, 0.0))

    # ---- Eq. 27: against the printed formula and the quadratic root; Remark 3.1; boundary; blind ranges.
    t_r = sf.reaction_time(torch.tensor([16, 9, 4]))
    assert _close(t_r, torch.tensor([0.05 + 16 * 0.02 + 0.03, 0.05 + 9 * 0.02 + 0.03, 0.05 + 4 * 0.02 + 0.03]))
    assert abs(float(t_r[0]) - 0.4) < 1e-7                                                    # Remark 3.1 uses T_r = 0.4 s
    depths = torch.tensor([0.0, 0.5, 0.75, 0.76, 1.0, 3.0, 6.75, 20.0, 100.0])
    for t_val in (0.0, 0.16, 0.4, 2.0):
        t_vec = torch.full_like(depths, t_val)
        v_safe = sf.safe_speed(depths, t_vec)
        ref_v = [_ref_safe_speed(float(d), float(t_vec[0]), clearance, a_max) for d in depths]
        assert _close(v_safe, torch.tensor(ref_v, dtype=torch.float64), rtol=1e-6, atol=1e-7)
        assert v_safe[:3].tolist() == [0.0, 0.0, 0.0]                                         # d <= eps_d + r_s
        assert bool((v_safe[3:] > 0).all()) and bool((v_safe[1:] >= v_safe[:-1]).all())       # increasing in d
        for d_val, v_val in zip(depths.tolist(), v_safe.tolist(), strict=True):
            margin = d_val - clearance
            if margin > 0:                                                                     # root of (3/4a) v^2 + T v - D
                residual = 0.75 / a_max * v_val ** 2 + float(t_vec[0]) * v_val - margin
                assert abs(residual) <= 1e-5 * margin + 1e-7, (d_val, t_val, residual)
    remark = sf.safe_speed(torch.tensor([6.75]), torch.tensor([0.4]))
    assert abs(float(remark) - 5.130239809888439) < 1e-5                                    # Remark 3.1: 5.1 m/s
    v_blind = sf.safe_speed(torch.tensor([10.0, 10.0, float("inf"), float("nan")]), torch.full((4,), 0.4),
                            blind=torch.tensor([False, True, False, False]))
    assert v_blind[1:].tolist() == [1.0, 1.0, 1.0] and float(v_blind[0]) > 1.0
    assert float(sf.safe_speed(torch.tensor([6.75]), torch.tensor([0.8]))) < float(remark)    # decreasing in T_r
    _expect(ValueError, lambda: sf.safe_speed(depths, torch.zeros(3)))

    # ---- Proposition 3.2: at v = v_safe(d), reaction travel v T_r plus the quadrature of the Eq. 26 stop
    #      equals d - eps_d - r_s; any slower speed stops strictly inside the measured free space.
    for d_val, t_val in ((6.75, 0.4), (3.0, 0.16), (20.0, 0.25), (0.9, 0.1), (1.5, 0.0)):
        margin = d_val - clearance
        v_val = float(sf.safe_speed(torch.tensor([d_val]), torch.tensor([t_val])))
        for factor, tight in ((1.0, True), (0.9, False)):
            speed = factor * v_val
            stop_t = 1.5 * speed / a_max
            tau_s = torch.linspace(0.0, stop_t, 20001)
            vel_s, _ = SafetyFilter.braking_profile(torch.tensor([[speed, 0.0, 0.0]]), tau_s, a_max)
            braking = float(torch.trapezoid(vel_s[0, :, 0].double(), tau_s.double()))
            assert abs(braking - 0.75 * speed ** 2 / a_max) <= 1e-6 * margin + 1e-7
            travel = speed * t_val + braking
            if tight:
                assert abs(travel - margin) <= 1e-5 * margin, (d_val, t_val, travel, margin)
            else:
                assert travel < margin

    # ---- Eq. 28: common factor, translation scaled, yaw rate untouched, every knot inside the envelope.
    chunk = (torch.rand(b, h, da, generator=gen) * 2.0 - 1.0) * scale                        # physical, inside A_0
    alpha_b = torch.tensor([1.0, 0.625, 1.0])
    depth_b = torch.tensor([30.0, 30.0, 3.0])
    horizon_b = torch.tensor([16, 9, 4])
    lim = sf.limit(chunk, alpha_b, depth_b, horizon_b)
    t_ref = [0.05 + int(hh) * 0.02 + 0.03 for hh in horizon_b]
    v_ref = [_ref_safe_speed(float(d), float(tt), clearance, a_max)
             for d, tt in zip(depth_b, torch.tensor(t_ref), strict=True)]
    peak = chunk[..., :3].double().norm(dim=-1).amax(dim=-1)
    s_ref = [min(float(alpha_b[i]), v_ref[i] / max(float(peak[i]), 1e-3)) for i in range(b)]
    assert _close(lim.reaction_time, torch.tensor(t_ref, dtype=torch.float64))
    assert _close(lim.v_safe, torch.tensor(v_ref, dtype=torch.float64))
    assert _close(lim.scale, torch.tensor(s_ref, dtype=torch.float64))
    assert torch.equal(lim.chunk[..., 3], chunk[..., 3])                                     # yaw rate bit-exact
    assert _close(lim.chunk[..., :3], chunk[..., :3].double() * torch.tensor(s_ref, dtype=torch.float64).view(-1, 1, 1))
    assert bool((lim.scale <= alpha_b).all()) and bool((lim.scale >= 0).all())
    commanded_peak = lim.chunk[..., :3].norm(dim=-1).amax(dim=-1)
    assert bool((commanded_peak <= lim.v_safe * (1.0 + 1e-5) + 1e-6).all())                  # inside the envelope
    assert float(lim.scale[0]) == 1.0 and abs(float(lim.scale[1]) - 0.625) < 1e-7          # alpha binds
    assert float(lim.scale[2]) < 1.0 and abs(float(commanded_peak[2]) - float(lim.v_safe[2])) < 1e-4  # envelope binds
    cosine = (lim.chunk[2, :, :3] * chunk[2, :, :3]).sum(-1) / (lim.chunk[2, :, :3].norm(dim=-1) * chunk[2, :, :3].norm(dim=-1))
    assert _close(cosine, torch.ones(h))                                                     # directions preserved

    # ---- Proposition 3.2 corner case: d_t <= eps_d + r_s gives s_t = 0, so translation stops exactly while
    #      the yaw rate keeps its (non-zero) values for scanning.
    lim0 = sf.limit(chunk, torch.ones(b), torch.tensor([0.75, 0.5, 0.0]), horizon_b)
    assert torch.equal(lim0.v_safe, torch.zeros(b)) and torch.equal(lim0.scale, torch.zeros(b))
    assert torch.equal(lim0.chunk[..., :3], torch.zeros(b, h, 3))
    assert torch.equal(lim0.chunk[..., 3], chunk[..., 3]) and bool((lim0.chunk[..., 3] != 0).all())

    # ---- Zero-speed guard: max_k ||v_{t,k}|| = 0 divides by eps_v, never by zero.
    still = chunk.clone()
    still[..., :3] = 0.0
    lim_s = sf.limit(still, torch.tensor([1.0, 0.625, 0.25]), torch.tensor([30.0, 0.75, 3.0]), horizon_b)
    assert bool(torch.isfinite(lim_s.scale).all()) and bool(torch.isfinite(lim_s.chunk).all())
    assert _close(lim_s.scale, torch.tensor([1.0, 0.0, 0.25])) and torch.equal(lim_s.chunk, still)
    blind_lim = sf.limit(chunk, torch.ones(b), depth_b, horizon_b, blind=torch.tensor([True, False, False]))
    assert float(blind_lim.v_safe[0]) == 1.0 and float(blind_lim.chunk[0, :, :3].norm(dim=-1).max()) <= 1.0 + 1e-6
    _expect(ValueError, lambda: sf.limit(chunk[..., :3], alpha_b, depth_b, horizon_b))
    _expect(ValueError, lambda: sf.limit(chunk, alpha_b[:-1], depth_b, horizon_b))

    # ---- Configuration and argument checks.
    _expect(ValueError, lambda: SafetyFilter.from_config(cfg, latency_bound=0.05, tracking_lag=0.03))  # not calibrated
    sf_cfg = SafetyFilter.from_config(config.VPAConfig(sigma_lo=0.2, sigma_crit=0.6), latency_bound=0.05, tracking_lag=0.03)
    assert (sf_cfg.sigma_lo, sf_cfg.sigma_crit, sf_cfg.max_horizon, sf_cfg.min_horizon) == (0.2, 0.6, 16, cfg.min_horizon)
    assert (sf_cfg.max_decel, sf_cfg.knot_dt, sf_cfg.alpha_min) == (cfg.max_decel, cfg.knot_dt, cfg.alpha_min)
    for kw in (dict(sigma_lo=0.6), dict(sigma_lo=-0.1), dict(sigma_crit=float("nan")), dict(alpha_min=0.0),
               dict(alpha_sigma=1.5), dict(min_horizon=0), dict(min_horizon=17), dict(beta=0.0), dict(max_decel=-1.0),
               dict(speed_eps=0.0), dict(knot_dt=0.0), dict(depth_error=-0.1), dict(latency_bound=-0.01)):
        _expect(ValueError, lambda k=kw: SafetyFilter(**dict(sf_args, **k)))
    mask_prefix = executed_prefix_mask(torch.tensor([4, 9, 16]), h)
    assert mask_prefix.shape == (3, h) and mask_prefix.sum(dim=1).tolist() == [4, 9, 16]
    print("solver.py self-test passed")


if __name__ == "__main__":
    _self_test()

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
selector.py -- Primitive selection and waypoint progression of Aero-VPA
(preprint_261026.pdf, Secs. 3.1 and 3.3, Table 1).

Implements
    * ``Primitive`` / ``PRIMITIVES`` -- the flight-primitive dictionary U in the order of Eq. 12; the
      enum value is the token u_t:
          CRUISE 0, LATERAL_AVOID 1, VERTICAL_AVOID 2, YAW_SCAN 3, APPROACH 4, BRAKE_HOVER 5.
      The names are exactly ``config.PRIMITIVES`` (checked by the self-test).
    * ``PrimitiveSelector`` -- pi_phi^h, an MLP over the concatenated decision inputs (Eq. 12):
          u_t = argmax_{u in U} pi_phi^h(u | z_t, g_t, S_t, c_text),
          input Concat(z_t [256], g_t [4], S_t [13], c_text [512]) in R^785,
      785 -> 512 -> 512 -> 6 with GELU: 668,166 parameters (Table 1: 0.67M). Trained with
          L_CE(phi) = -E_t[ log pi_phi^h(u*_t | z_t, g_t, S_t, c_text) ]                        (Eq. 13)
    * ``primitive_labels`` -- the fixed priority rules of Sec. 3.3 that derive the label u*_t from an
      expert command (a stand-in for annotated segmentations; not part of the method).
    * ``world_to_heading`` / ``heading_to_world`` -- x^H = R_z(psi)^T x^W and its inverse (Sec. 3.1).
    * ``goal_feature`` -- the goal feature of Eq. 3:
          r_t = R_z(psi_t)^T (w^(m_t) - p_t),
          g_t = [ r_t / max(||r_t||_2, eps_r) ;  min(||r_t||_2, D_max) / D_max ]  in R^4.
    * ``WaypointTracker`` -- the symbolic waypoint pointer m_t (Eq. 14):
          m_{t+1} = m_t + 1   if ||p_t - w^(m_t)||_2 < tau_w and m_t < M,   otherwise m_t,
      with a sticky task-complete flag once the test holds at m_t = M. Following Algorithm 1 (line 2),
      one ``step`` first applies Eq. 14 and then computes g_t for the (possibly advanced) pointer.

Changes from VPA (preprint_261008)
    * Milestone goal images and latent distances are replaced by geometric waypoints and the state
      estimate. The acceptance radius tau_w is a fixed length, not a variance-scaled latent threshold:
      if ||p_hat_t - p_t||_2 <= eps_p, a transition implies ||p_t - w^(m_t)||_2 < tau_w + eps_p by the
      triangle inequality, with no assumption on E_psi (Sec. 3.3). The gamma_bar EMA moved to
      ``predictor.LatentScaleTracker``.
    * The selector reads Concat(z_t, g_t, S_t, c_text) (785) instead of Concat(z_t, z_g, c_text).
    * Optional fixed input standardisation of z_t and S_t (identity by default, i.e. the literal Eq. 12);
      g_t and c_text are never rescaled.

Frames and units
    W: local NED world frame (m). H_t: heading frame, W rotated about its down axis by the estimated yaw
    psi_t (gravity-aligned forward-right-down). R_z(psi) = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]].
    Positions ``[B, 3]`` in W, yaw ``[B]`` in rad, waypoints ``[B, M, 3]`` in W. The pointer m_t is
    1-based, as in the paper, and converted to a 0-based index only when gathering w^(m_t).

This file is self-contained (it imports nothing from the other VPA modules; only its self-test reads
``config.py``) and can be tested on its own with ``python selector.py``.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "AERO_SELECTOR_PARAMS",
    "Primitive",
    "PRIMITIVES",
    "count_parameters",
    "PrimitiveSelector",
    "primitive_labels",
    "world_to_heading",
    "heading_to_world",
    "goal_feature",
    "WaypointUpdate",
    "WaypointTracker",
]

AERO_SELECTOR_PARAMS: int = 668_166  # (785*512 + 512) + (512*512 + 512) + (512*6 + 6); Table 1: 0.67M


class Primitive(IntEnum):
    """Flight primitives U (Sec. 3.3). The value is the token u_t; the order is that of Eq. 12."""

    CRUISE = 0
    LATERAL_AVOID = 1
    VERTICAL_AVOID = 2
    YAW_SCAN = 3
    APPROACH = 4
    BRAKE_HOVER = 5  # also the target of the safety override (Sec. 3.5)


PRIMITIVES: Tuple[str, ...] = tuple(p.name for p in Primitive)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters of ``module``; only those with ``requires_grad`` if ``trainable_only``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------------------------
# pi_phi^h : primitive selector (Eqs. 12, 13)
# ---------------------------------------------------------------------------------------------
class PrimitiveSelector(nn.Module):
    """MLP classifier pi_phi^h over the flight primitives (Sec. 3.3, Eq. 12, Table 1).

        x_t   = Concat( (z_t - mu_Z) / gamma*,  g_t,  (S_t - mu_S) / sigma_S,  c_text )   in R^785
        pi_phi^h(. | z_t, g_t, S_t, c_text) = softmax( MLP(x_t) ),   u_t = argmax_u pi_phi^h(u | ...)

    The standardisation statistics are fixed buffers, identity by default (mu = 0, scale = 1), which
    is the literal Eq. 12. ``set_input_standardization`` installs the frozen (mu_Z, gamma*) of stage (i)
    and the dataset (mu_S, sigma_S), the same statistics that standardise e_t (Eq. 15). They must be
    fixed before stage (ii) training, because the trained weights depend on them.

    The defaults are the Table 1 configuration, 668,166 parameters:

        Linear(785, 512)   402,432
        GELU
        Linear(512, 512)   262,656
        GELU
        Linear(512, 6)       3,078

    The per-decision cost is one MLP evaluation. ``select`` calls the module through ``__call__``, so
    forward hooks (the pipeline's sequential-depth audit) observe exactly one evaluation.

    Args:
        latent_dim: d.
        goal_dim: d_g (4).
        proprio_dim: d_s (13).
        text_dim: d_c (512).
        num_primitives: N_u = |U| (6).
        hidden_dims: widths of the hidden layers.

    Shapes:
        ``input_features``:          (``z_t [B, d]``, ``g_t [B, d_g]``, ``S_t [B, d_s]``, ``c_text [B, d_c]``)
                                     -> ``[B, d + d_g + d_s + d_c]``
        ``forward`` / ``probabilities``: same inputs -> ``[B, N_u]``
        ``select``:                  same inputs -> ``[B]`` int64
        ``loss``:                    (``logits [B, N_u]``, ``targets [B]`` int64) -> scalar
    """

    def __init__(
        self,
        latent_dim: int = 256,
        goal_dim: int = 4,
        proprio_dim: int = 13,
        text_dim: int = 512,
        num_primitives: int = len(Primitive),
        hidden_dims: Sequence[int] = (512, 512),
    ) -> None:
        super().__init__()
        if num_primitives < 2:
            raise ValueError("the primitive dictionary must contain at least two primitives")
        if min(latent_dim, goal_dim, proprio_dim, text_dim) < 1:
            raise ValueError("all input dimensions must be positive")
        if len(hidden_dims) == 0 or min(int(w) for w in hidden_dims) < 1:
            raise ValueError("hidden_dims must be a non-empty sequence of positive widths")
        self.latent_dim: int = int(latent_dim)
        self.goal_dim: int = int(goal_dim)
        self.proprio_dim: int = int(proprio_dim)
        self.text_dim: int = int(text_dim)
        self.num_primitives: int = int(num_primitives)
        self.input_dim: int = self.latent_dim + self.goal_dim + self.proprio_dim + self.text_dim  # 785
        layers = []
        in_features = self.input_dim
        for width in hidden_dims:
            layers += [nn.Linear(in_features, int(width)), nn.GELU()]
            in_features = int(width)
        layers.append(nn.Linear(in_features, self.num_primitives))
        self.mlp = nn.Sequential(*layers)
        # Fixed input standardisation (identity = literal Eq. 12). Float32: MPS has no float64.
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_scale", torch.ones(()))
        self.register_buffer("proprio_mean", torch.zeros(self.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(self.proprio_dim))

    @torch.no_grad()
    def set_input_standardization(
        self,
        latent_mean: Tensor,
        gamma_bar_star: float,
        proprio_mean: Tensor,
        proprio_std: Tensor,
    ) -> None:
        """Install mu_Z ``[d]``, gamma* (scalar), mu_S ``[d_s]`` and sigma_S ``[d_s]``."""
        def as_buffer(value: Any, like: Tensor, name: str) -> Tensor:
            t = torch.as_tensor(value, dtype=like.dtype, device=like.device)
            if tuple(t.shape) != tuple(like.shape) or not bool(torch.isfinite(t).all()):
                raise ValueError(f"{name} must be finite with shape {list(like.shape)}")
            return t

        mu_z = as_buffer(latent_mean, self.latent_mean, "latent_mean")
        mu_s = as_buffer(proprio_mean, self.proprio_mean, "proprio_mean")
        sd_s = as_buffer(proprio_std, self.proprio_std, "proprio_std")
        if not (math.isfinite(gamma_bar_star) and gamma_bar_star > 0.0):
            raise ValueError("gamma_bar_star must be finite and positive")
        if not bool((sd_s > 0).all()):
            raise ValueError("proprio_std must be strictly positive")
        self.latent_mean.copy_(mu_z)
        self.latent_scale.fill_(float(gamma_bar_star))
        self.proprio_mean.copy_(mu_s)
        self.proprio_std.copy_(sd_s)

    def input_features(self, z_t: Tensor, g_t: Tensor, s_t: Tensor, c_text: Tensor) -> Tensor:
        """x_t of Eq. 12: ``[B, d + d_g + d_s + d_c]`` = ``[B, 785]`` for Table 1.

        Args:
            z_t:    ``[B, d]``   online latent E_psi(I_t).
            g_t:    ``[B, d_g]`` goal feature (Eq. 3).
            s_t:    ``[B, d_s]`` proprioceptive state (Eq. 2): v^H, n^B, omega^B, a_bar.
            c_text: ``[B, d_c]`` instruction embedding.
        """
        if z_t.dim() != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(f"expected z_t of shape [B, {self.latent_dim}], got {tuple(z_t.shape)}")
        b = z_t.shape[0]
        for name, tensor, width in (("g_t", g_t, self.goal_dim), ("S_t", s_t, self.proprio_dim),
                                    ("c_text", c_text, self.text_dim)):
            if tuple(tensor.shape) != (b, width):
                raise ValueError(f"expected {name} of shape [{b}, {width}], got {tuple(tensor.shape)}")
        dtype = z_t.dtype
        z = (z_t - self.latent_mean.to(dtype)) / self.latent_scale.to(dtype)                    # [B, d]
        s = (s_t.to(dtype) - self.proprio_mean.to(dtype)) / self.proprio_std.to(dtype)          # [B, d_s]
        return torch.cat([z, g_t.to(dtype), s, c_text.to(dtype)], dim=-1)                       # [B, 785]

    def forward(self, z_t: Tensor, g_t: Tensor, s_t: Tensor, c_text: Tensor) -> Tensor:
        """Unnormalised scores of pi_phi^h(u | z_t, g_t, S_t, c_text): ``[B, N_u]``."""
        return self.mlp(self.input_features(z_t, g_t, s_t, c_text))

    def probabilities(self, z_t: Tensor, g_t: Tensor, s_t: Tensor, c_text: Tensor) -> Tensor:
        """Categorical distribution pi_phi^h(. | z_t, g_t, S_t, c_text): ``[B, N_u]``."""
        return F.softmax(self(z_t, g_t, s_t, c_text), dim=-1)

    def select(self, z_t: Tensor, g_t: Tensor, s_t: Tensor, c_text: Tensor) -> Tensor:
        """Eq. 12: u_t = argmax_u pi_phi^h(u | ...). Returns int64 tokens ``[B]``.

        Softmax is strictly monotone, so the argmax of the logits is the argmax of pi_phi^h.
        """
        return self(z_t, g_t, s_t, c_text).argmax(dim=-1)

    @staticmethod
    def loss(logits: Tensor, targets: Tensor) -> Tensor:
        """Eq. 13: L_CE = -E_t[log pi_phi^h(u*_t | ...)], the mean cross-entropy over the batch.

        Args:
            logits: ``[B, N_u]``.
            targets: ``[B]`` integer labels u*_t in {0, ..., N_u - 1}.

        Returns:
            scalar loss.
        """
        if logits.dim() != 2:
            raise ValueError(f"expected logits of shape [B, N_u], got {tuple(logits.shape)}")
        if tuple(targets.shape) != (logits.shape[0],):
            raise ValueError(f"expected targets of shape [{logits.shape[0]}], got {tuple(targets.shape)}")
        if targets.is_floating_point() or targets.dtype == torch.bool:
            raise ValueError("targets must be integer primitive tokens")
        if targets.numel() > 0 and (int(targets.min()) < 0 or int(targets.max()) >= logits.shape[1]):
            raise ValueError(f"targets must lie in {{0, ..., {logits.shape[1] - 1}}}")
        return F.cross_entropy(logits, targets.long())


# ---------------------------------------------------------------------------------------------
# Rule-based primitive labels u*_t (Sec. 3.3)
# ---------------------------------------------------------------------------------------------
def primitive_labels(
    command: Tensor,
    distance_to_final: Tensor,
    *,
    hover_speed: float,
    approach_distance: float,
    vertical_threshold: float,
    lateral_threshold: float,
    yaw_rate_threshold: float,
) -> Tensor:
    """Labels u*_t from an expert command by the fixed rules of Sec. 3.3, in priority order:

        1. BRAKE_HOVER     if ||(v_x, v_y, v_z)||_2 < hover_speed
        2. APPROACH        if the distance to the final waypoint w^(M) < approach_distance
        3. VERTICAL_AVOID  if |v_z| > vertical_threshold
        4. LATERAL_AVOID   if |v_y| > lateral_threshold
        5. YAW_SCAN        if |yaw rate| > yaw_rate_threshold
        6. CRUISE          otherwise

    The paper lists the three avoidance rules in this order and gives no thresholds; "below" and
    "within" are strict here, as in Eq. 14. Which expert command represents the decision (the first
    knot of the expert chunk, or its mean) is the caller's choice.

    Args:
        command: ``[B, 4]`` expert command (v_x, v_y, v_z, yaw rate) in the heading frame H_t.
        distance_to_final: ``[B]`` ||p_t - w^(M)||_2.
        hover_speed, approach_distance, vertical_threshold, lateral_threshold, yaw_rate_threshold:
            non-negative thresholds (m/s, m, m/s, m/s, rad/s).

    Returns:
        ``[B]`` int64 tokens.
    """
    if command.dim() != 2 or command.shape[1] != 4:
        raise ValueError(f"expected command of shape [B, 4], got {tuple(command.shape)}")
    if tuple(distance_to_final.shape) != (command.shape[0],):
        raise ValueError(f"expected distance_to_final of shape [{command.shape[0]}], got {tuple(distance_to_final.shape)}")
    thresholds = (hover_speed, approach_distance, vertical_threshold, lateral_threshold, yaw_rate_threshold)
    if not all(math.isfinite(t) and t >= 0.0 for t in thresholds):
        raise ValueError("thresholds must be finite and non-negative")
    speed = torch.linalg.vector_norm(command[:, :3], ord=2, dim=-1)                        # [B]
    labels = torch.full((command.shape[0],), int(Primitive.CRUISE), dtype=torch.long, device=command.device)
    rules = (  # lowest priority first, so that higher priorities overwrite
        (command[:, 3].abs() > yaw_rate_threshold, Primitive.YAW_SCAN),
        (command[:, 1].abs() > lateral_threshold, Primitive.LATERAL_AVOID),
        (command[:, 2].abs() > vertical_threshold, Primitive.VERTICAL_AVOID),
        (distance_to_final.to(command.dtype) < approach_distance, Primitive.APPROACH),
        (speed < hover_speed, Primitive.BRAKE_HOVER),
    )
    for condition, primitive in rules:
        labels = torch.where(condition, torch.full_like(labels, int(primitive)), labels)
    return labels


# ---------------------------------------------------------------------------------------------
# Heading frame and goal feature (Sec. 3.1, Eq. 3)
# ---------------------------------------------------------------------------------------------
def _check_vectors(vec: Tensor, psi: Tensor, name: str) -> None:
    if vec.dim() != 2 or vec.shape[1] != 3:
        raise ValueError(f"expected {name} of shape [B, 3], got {tuple(vec.shape)}")
    if tuple(psi.shape) != (vec.shape[0],):
        raise ValueError(f"expected yaw of shape [{vec.shape[0]}], got {tuple(psi.shape)}")


def world_to_heading(vec_w: Tensor, psi: Tensor) -> Tensor:
    """x^H = R_z(psi)^T x^W: NED -> heading frame. ``[B, 3]``, ``[B]`` -> ``[B, 3]``."""
    _check_vectors(vec_w, psi, "vec_w")
    psi = psi.to(vec_w.dtype)
    c, s = torch.cos(psi), torch.sin(psi)
    x, y, z = vec_w.unbind(dim=-1)
    return torch.stack([c * x + s * y, -s * x + c * y, z], dim=-1)


def heading_to_world(vec_h: Tensor, psi: Tensor) -> Tensor:
    """x^W = R_z(psi) x^H: heading frame -> NED (Eq. 32). ``[B, 3]``, ``[B]`` -> ``[B, 3]``."""
    _check_vectors(vec_h, psi, "vec_h")
    psi = psi.to(vec_h.dtype)
    c, s = torch.cos(psi), torch.sin(psi)
    x, y, z = vec_h.unbind(dim=-1)
    return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)


def goal_feature(
    waypoint: Tensor, p_hat: Tensor, psi_hat: Tensor, saturation_distance: float, eps: float
) -> Tensor:
    """Eq. 3: g_t = [ r_t / max(||r_t||_2, eps_r) ;  min(||r_t||_2, D_max) / D_max ].

    r_t = R_z(psi_t)^T (w^(m_t) - p_t) is the offset to the active waypoint in the heading frame
    (x forward, y right, z down). For ||r_t|| < eps_r the direction part is r_t / eps_r, so it shrinks
    continuously to 0 at r_t = 0 instead of dividing by zero.

    Args:
        waypoint: ``[B, 3]`` active waypoint w^(m_t) in W.
        p_hat: ``[B, 3]`` estimated position in W.
        psi_hat: ``[B]`` estimated yaw (rad).
        saturation_distance: D_max > 0 (m).
        eps: eps_r > 0 (m).

    Returns:
        g_t: ``[B, 4]``; the first three entries have norm <= 1, the last lies in [0, 1].
    """
    if not (math.isfinite(saturation_distance) and saturation_distance > 0.0):
        raise ValueError("saturation_distance (D_max) must be finite and positive")
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError("eps (eps_r) must be finite and positive")
    if tuple(waypoint.shape) != tuple(p_hat.shape):
        raise ValueError(f"waypoint {tuple(waypoint.shape)} and p_hat {tuple(p_hat.shape)} must have the same shape")
    r = world_to_heading(waypoint - p_hat, psi_hat)                                           # [B, 3]
    dist = torch.linalg.vector_norm(r, ord=2, dim=-1)                                         # [B]
    direction = r / dist.clamp_min(eps).unsqueeze(-1)                                         # [B, 3]
    saturated = dist.clamp_max(saturation_distance) / saturation_distance                     # [B]
    return torch.cat([direction, saturated.unsqueeze(-1)], dim=-1)                            # [B, 4]


# ---------------------------------------------------------------------------------------------
# Waypoint pointer m_t (Eq. 14)
# ---------------------------------------------------------------------------------------------
class WaypointUpdate(NamedTuple):
    """Result of ``WaypointTracker.step`` (leading dimension B)."""

    goal: Tensor           # [B, 4]  g_t (Eq. 3) for the active waypoint after the Eq. 14 update
    m_t: Tensor            # [B]     int64, active waypoint index (1-based) used for g_t
    task_complete: Tensor  # [B]     bool, sticky: Eq. 14's test has held with m_t = M


class WaypointTracker:
    """Symbolic finite-state machine over the mission waypoints (Sec. 3.3, Eqs. 3 and 14).

    Per decision step (Algorithm 1, line 2):
        1. reached_t = ||p_hat_t - w^(m_t)||_2 < tau_w                          (strict, Eq. 14)
        2. m_t <- m_t + 1 if reached_t and m_t < M; task_complete <- True if reached_t and m_t = M
        3. g_t from Eq. 3 with the updated pointer

    The pointer moves by at most one waypoint per decision and never decreases; the task-complete flag
    is sticky. With ``update_mask`` only the selected elements are updated (the others keep their
    pointer and flag, but still receive g_t for their current waypoint).

    The tracker has no learned parameters. Its episode state (``waypoints``, ``m``, ``task_complete``)
    lives on the waypoints' device and dtype; positions and yaw are converted to them.

    Args:
        acceptance_radius: tau_w > 0 (m).
        saturation_distance: D_max > 0 (m) of Eq. 3.
        eps: eps_r > 0 (m) of Eq. 3.

    Shapes:
        ``reset``: ``waypoints [B, M, 3]``
        ``step``: (``p_hat [B, 3]``, ``psi_hat [B]``) -> ``WaypointUpdate(goal [B, 4], m_t [B], task_complete [B])``
        ``active_waypoint``: -> ``[B, 3]``;  ``distance`` / ``distance_to_final``: ``p_hat [B, 3]`` -> ``[B]``
    """

    def __init__(self, acceptance_radius: float, saturation_distance: float, eps: float) -> None:
        for name, value in (("acceptance_radius", acceptance_radius), ("saturation_distance", saturation_distance),
                            ("eps", eps)):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        self.acceptance_radius: float = float(acceptance_radius)
        self.saturation_distance: float = float(saturation_distance)
        self.eps: float = float(eps)
        self.waypoints: Optional[Tensor] = None
        self.m: Optional[Tensor] = None
        self.task_complete: Optional[Tensor] = None

    @property
    def num_waypoints(self) -> Optional[int]:
        """M, or None before ``reset``."""
        return None if self.waypoints is None else int(self.waypoints.shape[1])

    def reset(self, waypoints: Tensor) -> None:
        """Start a mission: store w^(1..M) ``[B, M, 3]`` (NED, m), set m_0 = 1 and clear the flag."""
        if waypoints.dim() != 3 or waypoints.shape[2] != 3:
            raise ValueError(f"expected waypoints of shape [B, M, 3], got {tuple(waypoints.shape)}")
        if waypoints.shape[0] < 1 or waypoints.shape[1] < 1:
            raise ValueError("a mission needs B >= 1 and at least one waypoint (M >= 1)")
        if not waypoints.is_floating_point():
            raise ValueError("waypoints must be a floating-point tensor")
        if not bool(torch.isfinite(waypoints).all()):
            raise ValueError("waypoints must be finite")
        self.waypoints = waypoints.detach().clone()
        b = waypoints.shape[0]
        self.m = torch.ones(b, dtype=torch.long, device=waypoints.device)
        self.task_complete = torch.zeros(b, dtype=torch.bool, device=waypoints.device)

    def _require_episode(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.waypoints is None or self.m is None or self.task_complete is None:
            raise RuntimeError("call reset(waypoints) before using the waypoint pointer")
        return self.waypoints, self.m, self.task_complete

    def _positions(self, p_hat: Tensor, waypoints: Tensor) -> Tensor:
        if tuple(p_hat.shape) != (waypoints.shape[0], 3):
            raise ValueError(f"expected p_hat of shape [{waypoints.shape[0]}, 3], got {tuple(p_hat.shape)}")
        return p_hat.to(dtype=waypoints.dtype, device=waypoints.device)

    def active_waypoint(self) -> Tensor:
        """w^(m_t): ``[B, 3]``."""
        waypoints, m, _ = self._require_episode()
        index = (m - 1).view(-1, 1, 1).expand(-1, 1, 3)                                    # [B, 1, 3]
        return waypoints.gather(1, index).squeeze(1)                                        # [B, 3]

    def distance(self, p_hat: Tensor) -> Tensor:
        """||p_hat_t - w^(m_t)||_2 for the current pointer: ``[B, 3]`` -> ``[B]``."""
        waypoints, _, _ = self._require_episode()
        p = self._positions(p_hat, waypoints)
        return torch.linalg.vector_norm(p - self.active_waypoint(), ord=2, dim=-1)

    def distance_to_final(self, p_hat: Tensor) -> Tensor:
        """||p_hat_t - w^(M)||_2 (used by the APPROACH rule of ``primitive_labels``): ``[B]``."""
        waypoints, _, _ = self._require_episode()
        p = self._positions(p_hat, waypoints)
        return torch.linalg.vector_norm(p - waypoints[:, -1], ord=2, dim=-1)

    @torch.no_grad()
    def step(self, p_hat: Tensor, psi_hat: Tensor, update_mask: Optional[Tensor] = None) -> WaypointUpdate:
        """Apply Eq. 14, commit the new pointer, and return g_t (Eq. 3) for it.

        Args:
            p_hat: ``[B, 3]`` estimated position in W (m).
            psi_hat: ``[B]`` estimated yaw (rad).
            update_mask: optional ``[B]`` bool; unselected elements keep their pointer and flag.

        Returns:
            ``WaypointUpdate(goal [B, 4], m_t [B] int64, task_complete [B] bool)``.
        """
        waypoints, m, done = self._require_episode()
        b, num_waypoints = waypoints.shape[0], waypoints.shape[1]
        p = self._positions(p_hat, waypoints)
        if tuple(psi_hat.shape) != (b,):
            raise ValueError(f"expected psi_hat of shape [{b}], got {tuple(psi_hat.shape)}")
        psi = psi_hat.to(dtype=waypoints.dtype, device=waypoints.device)
        reached = self.distance(p) < self.acceptance_radius                                  # strict (Eq. 14)
        advance = reached & (m < num_waypoints)
        finished = reached & (m == num_waypoints)
        if update_mask is not None:
            if tuple(update_mask.shape) != (b,):
                raise ValueError(f"expected update_mask of shape [{b}], got {tuple(update_mask.shape)}")
            mask = update_mask.to(device=m.device, dtype=torch.bool)
            advance = advance & mask
            finished = finished & mask
        self.m = m + advance.long()
        self.task_complete = done | finished
        goal = goal_feature(self.active_waypoint(), p, psi, self.saturation_distance, self.eps)
        return WaypointUpdate(goal, self.m.clone(), self.task_complete.clone())


# ---------------------------------------------------------------------------------------------
# Self-test (run: python selector.py)
# ---------------------------------------------------------------------------------------------
_RTOL, _ATOL = 1e-5, 1e-6              # float32 module output vs float64 reference (CLAUDE.md)
_RTOL_WIDE, _ATOL_WIDE = 1e-4, 1e-5    # full Table 1 width (785- and 512-wide layers)


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


def _ref_mlp_forward(sel: PrimitiveSelector, x: Tensor) -> Tensor:
    """Float64 reference of the selector MLP: Linear and exact erf-GELU written out. ``[B, in]`` -> ``[B, N_u]``."""
    x = x.double()
    for layer in sel.mlp:
        if isinstance(layer, nn.Linear):
            x = x @ layer.weight.detach().double().T + layer.bias.detach().double()
        else:
            assert isinstance(layer, nn.GELU) and layer.approximate == "none"
            x = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x


def _ref_selector_logits(sel: PrimitiveSelector, z: Tensor, g: Tensor, s: Tensor, c: Tensor) -> Tensor:
    """Float64 reference of Eq. 12: input slots written by hand (with the selector's statistics)."""
    b, d, dg, ds, dc = z.shape[0], sel.latent_dim, sel.goal_dim, sel.proprio_dim, sel.text_dim
    x = torch.zeros(b, d + dg + ds + dc, dtype=torch.float64)
    x[:, :d] = (z.double() - sel.latent_mean.double()) / float(sel.latent_scale)
    x[:, d:d + dg] = g.double()
    x[:, d + dg:d + dg + ds] = (s.double() - sel.proprio_mean.double()) / sel.proprio_std.double()
    x[:, d + dg + ds:] = c.double()
    return _ref_mlp_forward(sel, x)


def _ref_softmax(s: Tensor) -> Tensor:
    e = torch.exp(s - s.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def _ref_goal_feature(waypoint: Tensor, p_hat: Tensor, psi: Tensor, d_max: float, eps: float) -> Tensor:
    """Float64 reference of Eq. 3 with an explicit rotation matrix R_z(psi): ``[B, 4]``."""
    b = waypoint.shape[0]
    out = torch.zeros(b, 4, dtype=torch.float64)
    for k in range(b):
        c, s = math.cos(float(psi[k])), math.sin(float(psi[k]))
        rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        r = rz.T @ (waypoint[k].double() - p_hat[k].double())
        n = math.sqrt(float((r ** 2).sum()))
        out[k, :3] = r / max(n, eps)
        out[k, 3] = min(n, d_max) / d_max
    return out


def _ref_tracker_step(m, done, p, psi, waypoints, tau, mask, d_max, eps):
    """Per-element float64 reference of Algorithm 1 line 2: Eq. 14 on the current pointer, then Eq. 3
    on the updated one. Lists in, (m_next, done_next, goals [B, 4], distances) out."""
    num = waypoints.shape[1]
    m_next, done_next, goals, dist = [], [], [], []
    for i in range(len(m)):
        w = waypoints[i, m[i] - 1].double()
        d_i = math.sqrt(float(((p[i].double() - w) ** 2).sum()))
        hit = d_i < tau
        m_i, done_i = m[i], done[i]
        if mask[i] and hit:
            if m_i < num:
                m_i += 1
            else:
                done_i = True
        m_next.append(m_i)
        done_next.append(done_i)
        dist.append(d_i)
        goals.append(_ref_goal_feature(waypoints[i, m_i - 1].unsqueeze(0), p[i].unsqueeze(0),
                                       psi[i].unsqueeze(0), d_max, eps)[0])
    return m_next, done_next, torch.stack(goals), dist


def _ref_labels(command: Tensor, dist_final: Tensor, hover_speed: float, approach_distance: float,
                vertical_threshold: float, lateral_threshold: float, yaw_rate_threshold: float) -> list:
    """Sec. 3.3 rules as an if/elif chain in float64."""
    out = []
    for k in range(command.shape[0]):
        vx, vy, vz, yaw_rate = (float(v) for v in command[k])
        if math.sqrt(vx * vx + vy * vy + vz * vz) < hover_speed:
            out.append(int(Primitive.BRAKE_HOVER))
        elif float(dist_final[k]) < approach_distance:
            out.append(int(Primitive.APPROACH))
        elif abs(vz) > vertical_threshold:
            out.append(int(Primitive.VERTICAL_AVOID))
        elif abs(vy) > lateral_threshold:
            out.append(int(Primitive.LATERAL_AVOID))
        elif abs(yaw_rate) > yaw_rate_threshold:
            out.append(int(Primitive.YAW_SCAN))
        else:
            out.append(int(Primitive.CRUISE))
    return out


def _mlp_param_count(widths: Tuple[int, ...]) -> int:
    """Parameters of a Linear stack with biases: sum_i (w_i w_{i+1} + w_{i+1})."""
    return sum(i * o + o for i, o in zip(widths[:-1], widths[1:], strict=True))


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(2610)
    config = _config_module()
    cfg = config.VPAConfig()

    # ---- U (Sec. 3.3): tokens 0..5 and names exactly config.PRIMITIVES.
    assert PRIMITIVES == tuple(config.PRIMITIVES) and len(PRIMITIVES) == cfg.num_primitives == 6
    assert [int(p) for p in Primitive] == list(range(6))
    assert Primitive.CRUISE == 0 and Primitive.LATERAL_AVOID == 1 and Primitive.VERTICAL_AVOID == 2
    assert Primitive.YAW_SCAN == 3 and Primitive.APPROACH == 4 and Primitive.BRAKE_HOVER == 5

    # ---- Table 1: 785 -> 512 -> 512 -> 6 with GELU, exactly 668,166 parameters.
    sel = PrimitiveSelector(latent_dim=cfg.latent_dim, goal_dim=cfg.goal_dim, proprio_dim=cfg.proprio_dim,
                            text_dim=cfg.text_dim, num_primitives=cfg.num_primitives,
                            hidden_dims=cfg.selector_hidden)
    assert sel.input_dim == cfg.selector_in_dim == 785 == 256 + 4 + 13 + 512
    linears = [layer for layer in sel.mlp if isinstance(layer, nn.Linear)]
    assert [(lin.in_features, lin.out_features) for lin in linears] == [(785, 512), (512, 512), (512, 6)]
    assert [count_parameters(lin) for lin in linears] == [402_432, 262_656, 3_078]
    assert len(sel.mlp) == 5 and sum(isinstance(layer, nn.GELU) for layer in sel.mlp) == 2
    assert all(layer.approximate == "none" for layer in sel.mlp if isinstance(layer, nn.GELU))
    assert count_parameters(sel) == count_parameters(sel, trainable_only=False) == AERO_SELECTOR_PARAMS == 668_166
    assert round(AERO_SELECTOR_PARAMS / 1e6, 2) == config.TABLE1_PARAMS_M["selector"]
    assert count_parameters(PrimitiveSelector()) == 668_166                       # the defaults are Table 1
    assert sorted(name for name, _ in sel.named_buffers()) == [
        "latent_mean", "latent_scale", "proprio_mean", "proprio_std"]

    # ---- Eq. 12 input layout: [z_t 0:256 | g_t 256:260 | S_t 260:273 | c_text 273:785].
    b = 8
    z_t = torch.randn(b, 256, generator=gen)
    g_t = torch.randn(b, 4, generator=gen)
    s_t = torch.randn(b, 13, generator=gen)
    c_text = torch.randn(b, 512, generator=gen)
    captured = []
    pre = sel.mlp.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
    logits = sel(z_t, g_t, s_t, c_text)
    pre.remove()
    x_in = captured[0]
    assert x_in.shape == (b, 785) and logits.shape == (b, 6) and logits.dtype == torch.float32
    assert torch.equal(x_in[:, :256], z_t) and torch.equal(x_in[:, 256:260], g_t)
    assert torch.equal(x_in[:, 260:273], s_t) and torch.equal(x_in[:, 273:], c_text)  # identity by default
    assert torch.equal(sel.input_features(z_t, g_t, s_t, c_text), x_in)

    # ---- Eq. 12 against float64 references: full Table 1 width, and a small selector at the repo tolerance.
    ref_logits = _ref_selector_logits(sel, z_t, g_t, s_t, c_text)
    assert _close(logits, ref_logits, _RTOL_WIDE, _ATOL_WIDE)
    assert _close(_ref_mlp_forward(sel, x_in), ref_logits)                       # hand-built slots == module input
    small = PrimitiveSelector(latent_dim=8, goal_dim=4, proprio_dim=5, text_dim=6, hidden_dims=(32, 32))
    zs, gs, ss, cs = (torch.randn(b, n, generator=gen) for n in (8, 4, 5, 6))
    assert small.input_dim == 23 and count_parameters(small) == _mlp_param_count((23, 32, 32, 6))
    assert _close(small(zs, gs, ss, cs), _ref_selector_logits(small, zs, gs, ss, cs))

    # ---- Distribution and argmax (one evaluation through __call__ for the depth audit).
    probs = sel.probabilities(z_t, g_t, s_t, c_text)
    ref_probs = _ref_softmax(ref_logits)
    assert _close(probs, ref_probs, _RTOL_WIDE, _ATOL_WIDE) and _close(probs.sum(dim=-1), torch.ones(b))
    calls = []
    hook = sel.register_forward_hook(lambda *_: calls.append(1))
    u_t = sel.select(z_t, g_t, s_t, c_text)
    hook.remove()
    assert len(calls) == 1 and u_t.shape == (b,) and u_t.dtype == torch.long
    assert torch.equal(u_t, logits.argmax(dim=-1))
    top2 = ref_logits.topk(2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > 1e-4                                       # unambiguous argmax rows
    assert bool(clear.any())
    assert torch.equal(u_t[clear], ref_logits.argmax(dim=-1)[clear])
    assert torch.equal(probs.argmax(dim=-1)[clear], u_t[clear])

    # ---- Eq. 13: L_CE = -mean_t log pi(u*_t | ...), against a float64 log-softmax; gradients reach every parameter.
    labels = torch.randint(0, 6, (b,), generator=gen)
    loss = PrimitiveSelector.loss(logits, labels)
    ref_loss = -float(torch.log(ref_probs[torch.arange(b), labels]).mean())
    assert loss.dim() == 0 and _close(loss, ref_loss, _RTOL_WIDE, _ATOL_WIDE)
    sel.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None for p in sel.parameters())
    lg = logits.detach()
    for bad_logits, bad_labels in ((lg, labels[:-1]), (lg, labels.float()), (lg, labels + 6), (lg, labels - 7),
                                   (lg[0], labels), (lg, labels.bool())):
        _expect(ValueError, lambda x=bad_logits, y=bad_labels: PrimitiveSelector.loss(x, y))

    # ---- Optional input standardisation: z_t and S_t only; the float64 reference uses the same statistics.
    mu_z, gamma_star = torch.randn(256, generator=gen), 1.7
    mu_s, sd_s = torch.randn(13, generator=gen), torch.rand(13, generator=gen) + 0.5
    std_sel = PrimitiveSelector()
    std_sel.load_state_dict(sel.state_dict())
    std_sel.set_input_standardization(mu_z, gamma_star, mu_s, sd_s)
    assert torch.equal(std_sel.latent_mean, mu_z) and torch.equal(std_sel.proprio_std, sd_s)
    with torch.no_grad():
        got = std_sel(z_t, g_t, s_t, c_text)
        x_std = std_sel.input_features(z_t, g_t, s_t, c_text)
        want = sel((z_t - std_sel.latent_mean) / std_sel.latent_scale, g_t,
                   (s_t - std_sel.proprio_mean) / std_sel.proprio_std, c_text)
    assert torch.equal(got, want)
    assert torch.equal(x_std[:, 256:260], g_t) and torch.equal(x_std[:, 273:], c_text)  # never rescaled
    assert _close(got, _ref_selector_logits(std_sel, z_t, g_t, s_t, c_text), _RTOL_WIDE, _ATOL_WIDE)
    reloaded = PrimitiveSelector()
    reloaded.load_state_dict(std_sel.state_dict())                                # statistics live in the state_dict
    assert float(reloaded.latent_scale) == float(std_sel.latent_scale)
    good = dict(latent_mean=mu_z, gamma_bar_star=gamma_star, proprio_mean=mu_s, proprio_std=sd_s)
    for kw in (dict(latent_mean=torch.zeros(255)), dict(gamma_bar_star=0.0), dict(gamma_bar_star=float("nan")),
               dict(proprio_std=torch.zeros(13)), dict(proprio_mean=torch.zeros(12)),
               dict(proprio_mean=torch.full((13,), float("inf")))):
        args = dict(good)
        args.update(kw)
        _expect(ValueError, lambda a=args: PrimitiveSelector().set_input_standardization(**a))

    # ---- Input dimension and batch mismatches raise ValueError (forward and select).
    for args in ((z_t[:, :-1], g_t, s_t, c_text), (z_t, g_t[:, :3], s_t, c_text), (z_t, g_t, s_t[:, :12], c_text),
                 (z_t, g_t, s_t, c_text[:, :511]), (z_t, g_t[:-1], s_t, c_text), (z_t, g_t, s_t, c_text[:-1]),
                 (z_t, g_t, s_t[:-1], c_text), (z_t[0], g_t[0], s_t[0], c_text[0]),
                 (z_t.unsqueeze(0), g_t, s_t, c_text)):
        _expect(ValueError, lambda a=args: sel(*a))
        _expect(ValueError, lambda a=args: sel.select(*a))
    for kw in (dict(num_primitives=1), dict(hidden_dims=()), dict(hidden_dims=(512, 0)), dict(goal_dim=0)):
        _expect(ValueError, lambda k=kw: PrimitiveSelector(**k))

    # ---- Sec. 3.3 labels u*_t: priority order of the rules.
    th = dict(hover_speed=0.3, approach_distance=3.0, vertical_threshold=0.5, lateral_threshold=0.8,
              yaw_rate_threshold=0.6)
    command = torch.tensor([
        [0.1, 0.0, 0.0, 0.0],     # slow                                   -> BRAKE_HOVER
        [0.1, 0.0, 0.0, 1.0],     # slow beats yaw                         -> BRAKE_HOVER
        [3.0, 0.0, 0.0, 0.0],     # near the final waypoint                -> APPROACH
        [3.0, 2.0, 1.5, 1.0],     # near the final waypoint beats avoiding -> APPROACH
        [3.0, 2.0, 1.5, 1.0],     # far: vertical beats lateral and yaw    -> VERTICAL_AVOID
        [3.0, -2.0, 0.2, 1.0],    # lateral beats yaw                      -> LATERAL_AVOID
        [3.0, 0.1, -0.2, -0.9],   # yaw only                               -> YAW_SCAN
        [3.0, 0.1, 0.1, 0.1],     # nothing else                           -> CRUISE
        [0.0, 0.0, -0.7, 0.0],    # climbing in place (0.7 > hover speed)  -> VERTICAL_AVOID
    ])
    dist_final = torch.tensor([10.0, 10.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    expected = [5, 5, 4, 4, 2, 1, 3, 0, 2]
    rule_labels = primitive_labels(command, dist_final, **th)
    assert rule_labels.dtype == torch.long and rule_labels.tolist() == expected
    assert _ref_labels(command, dist_final, **th) == expected
    quantised = torch.randint(-12, 13, (64, 4), generator=gen).float() * 0.25        # no value near a threshold
    dist_q = torch.randint(0, 40, (64,), generator=gen).float() * 0.25
    assert primitive_labels(quantised, dist_q, **th).tolist() == _ref_labels(quantised, dist_q, **th)
    _expect(ValueError, lambda: primitive_labels(command[:, :3], dist_final, **th))
    _expect(ValueError, lambda: primitive_labels(command, dist_final[:-1], **th))
    _expect(ValueError, lambda: primitive_labels(command, dist_final, **dict(th, hover_speed=-0.1)))

    # ---- Heading frame (Sec. 3.1): R_z(psi)^T and R_z(psi) are inverse and preserve length.
    d_max, eps_r = 20.0, 1e-3
    n = 64
    psi = (torch.rand(n, generator=gen) * 2.0 - 1.0) * math.pi
    v = torch.randn(n, 3, generator=gen)
    assert _close(heading_to_world(world_to_heading(v, psi), psi), v)
    assert _close(world_to_heading(v, psi).norm(dim=-1), v.norm(dim=-1))
    _expect(ValueError, lambda: world_to_heading(v[:, :2], psi))
    _expect(ValueError, lambda: heading_to_world(v, psi[:-1]))

    # ---- Eq. 3 against the float64 reference (explicit R_z(psi)^T), inside and beyond D_max.
    wp = torch.randn(n, 3, generator=gen) * 15.0
    pos = torch.randn(n, 3, generator=gen) * 15.0
    g = goal_feature(wp, pos, psi, d_max, eps_r)
    assert g.shape == (n, 4) and g.dtype == torch.float32
    assert _close(g, _ref_goal_feature(wp, pos, psi, d_max, eps_r))
    dist = torch.linalg.vector_norm(wp - pos, dim=-1)
    assert bool((dist > d_max + 1e-3).any()) and bool((dist < d_max - 1e-3).any())
    assert torch.allclose(g[:, :3].norm(dim=-1), torch.ones(n), atol=1e-6)
    assert bool((g[:, 3] >= 0.0).all()) and bool((g[:, 3] <= 1.0).all())
    assert bool((g[dist > d_max + 1e-3, 3] == 1.0).all())                         # saturation
    origin = torch.zeros(1, 3)
    for wp_c, psi_c, unit_c in (                                                    # (waypoint NED, yaw) -> direction in H
        ((10.0, 0.0, 0.0), 0.0, (1.0, 0.0, 0.0)),                                  # north, facing north: ahead
        ((0.0, 10.0, 0.0), 0.0, (0.0, 1.0, 0.0)),                                  # east, facing north: right
        ((0.0, 10.0, 0.0), math.pi / 2, (1.0, 0.0, 0.0)),                          # east, facing east: ahead
        ((10.0, 0.0, 0.0), math.pi / 2, (0.0, -1.0, 0.0)),                         # north, facing east: left
        ((0.0, 0.0, 10.0), 1.234, (0.0, 0.0, 1.0)),                                # below, any yaw: down
    ):
        g_c = goal_feature(torch.tensor([wp_c]), origin, torch.tensor([psi_c]), d_max, eps_r)
        assert torch.allclose(g_c[0, :3], torch.tensor(unit_c), atol=1e-6) and abs(float(g_c[0, 3]) - 0.5) < 1e-6

    # ---- Eq. 3 singularity: r_t = 0 gives g_t = 0 (eps_r guards the division); ||r_t|| < eps_r is continuous.
    w0 = torch.tensor([[1.0, -2.0, 0.5], [-3.0, 4.0, 1.0]])
    g0 = goal_feature(w0, w0.clone(), torch.tensor([0.7, -2.9]), d_max, eps_r)
    assert bool(torch.isfinite(g0).all()) and torch.equal(g0, torch.zeros(2, 4))
    tiny = goal_feature(torch.tensor([[0.5 * eps_r, 0.0, 0.0]]), origin, torch.tensor([0.0]), d_max, eps_r)
    assert torch.allclose(tiny, torch.tensor([[0.5, 0.0, 0.0, 0.5 * eps_r / d_max]]), rtol=1e-5, atol=1e-9)
    for kw in (dict(saturation_distance=0.0), dict(eps=0.0), dict(eps=float("nan"))):
        args = dict(waypoint=wp, p_hat=pos, psi_hat=psi, saturation_distance=d_max, eps=eps_r)
        args.update(kw)
        _expect(ValueError, lambda a=args: goal_feature(**a))
    _expect(ValueError, lambda: goal_feature(wp, pos[:-1], psi, d_max, eps_r))
    _expect(ValueError, lambda: goal_feature(wp, pos, psi[:-1], d_max, eps_r))

    # ---- Eq. 14 strict boundary with exact float32 distances (offsets along x, waypoint x = 0):
    #      ||p - w|| = tau_w does not advance; the next float32 below tau_w does.
    tau = 1.5
    below = float(torch.nextafter(torch.tensor(tau), torch.tensor(0.0)))
    for num in (2, 1):  # M = 2: advance at m < M;  M = 1: completion at m = M
        tracker = WaypointTracker(acceptance_radius=tau, saturation_distance=d_max, eps=eps_r)
        wps = torch.randn(2, num, 3, generator=gen) * 5.0
        wps[..., 0] = 0.0
        tracker.reset(wps)
        p = wps[:, 0].clone()
        p[0, 0], p[1, 0] = tau, below
        assert tracker.distance(p).tolist() == [tau, below]
        yaw = torch.tensor([0.3, -1.1])
        out = tracker.step(p, yaw)
        if num == 2:
            assert out.m_t.tolist() == [1, 2] and out.task_complete.tolist() == [False, False]
        else:  # single-waypoint mission: completes without moving the pointer
            assert out.m_t.tolist() == [1, 1] and out.task_complete.tolist() == [False, True]
        active = torch.stack([wps[i, int(out.m_t[i]) - 1] for i in range(2)])
        assert _close(out.goal, _ref_goal_feature(active, p, yaw, d_max, eps_r))  # g_t uses the updated pointer

    # ---- Completion exactly at the final waypoint: reached, sticky, and g_t = 0 without NaN.
    tracker = WaypointTracker(tau, d_max, eps_r)
    wps1 = torch.randn(3, 1, 3, generator=gen)
    tracker.reset(wps1)
    for _ in range(2):
        out = tracker.step(wps1[:, 0].clone(), torch.randn(3, generator=gen))
        assert out.m_t.tolist() == [1, 1, 1] and bool(out.task_complete.all())
        assert bool(torch.isfinite(out.goal).all()) and torch.equal(out.goal, torch.zeros(3, 4))
    far = wps1[:, 0] + torch.tensor([50.0, 0.0, 0.0])
    assert bool(tracker.step(far, torch.zeros(3)).task_complete.all())             # leaving does not clear it

    # ---- Eq. 14 randomised rollout against the per-element reference: monotone pointer, at most one
    #      waypoint per decision, sticky completion, masked elements held, g_t for the updated pointer.
    b_r, num_wp = 4, 3
    tracker = WaypointTracker(tau, d_max, eps_r)
    wps = torch.randn(b_r, num_wp, 3, generator=gen) * 8.0
    wps[..., 0] = 0.0
    tracker.reset(wps)
    assert tracker.num_waypoints == num_wp and tracker.m.tolist() == [1] * b_r
    m_ref, done_ref = [1] * b_r, [False] * b_r
    seen = {"advance": 0, "hold": 0, "complete": 0, "masked_reach": 0, "boundary": 0}
    for _ in range(60):
        kinds = torch.randint(0, 3, (b_r,), generator=gen).tolist()                # 0 inside, 1 outside, 2 boundary
        mask = (torch.rand(b_r, generator=gen) < 0.7).tolist()
        yaw = (torch.rand(b_r, generator=gen) * 2.0 - 1.0) * math.pi
        p = torch.empty(b_r, 3)
        for i in range(b_r):
            w = wps[i, m_ref[i] - 1]
            if kinds[i] == 2:
                p[i] = w
                p[i, 0] = tau
            else:
                direction = torch.randn(3, generator=gen)
                p[i] = w + (0.7 if kinds[i] == 0 else 4.0) * direction / direction.norm()
        assert torch.equal(tracker.active_waypoint(), torch.stack([wps[i, m_ref[i] - 1] for i in range(b_r)]))
        m_next, done_next, goal_ref, dist_ref = _ref_tracker_step(
            m_ref, done_ref, p, yaw, wps, tau, mask, d_max, eps_r)
        assert _close(tracker.distance(p), torch.tensor(dist_ref, dtype=torch.float64))
        final_ref = [math.sqrt(float(((p[i].double() - wps[i, -1].double()) ** 2).sum())) for i in range(b_r)]
        assert _close(tracker.distance_to_final(p), torch.tensor(final_ref, dtype=torch.float64))
        out = tracker.step(p, yaw, update_mask=torch.tensor(mask))
        goal_out, m_out, done_out = out                                               # a plain 3-tuple as well
        assert m_out.tolist() == m_next and done_out.tolist() == done_next
        assert goal_out.shape == (b_r, 4) and _close(goal_out, goal_ref)
        assert torch.equal(tracker.m, m_out) and torch.equal(tracker.task_complete, done_out)
        for i in range(b_r):
            reached = dist_ref[i] < tau
            assert m_ref[i] <= m_next[i] <= num_wp and m_next[i] - m_ref[i] <= 1      # monotone, one step at most
            assert not done_ref[i] or done_next[i]                                     # sticky completion
            if not mask[i]:
                assert m_next[i] == m_ref[i] and done_next[i] == done_ref[i]
                seen["masked_reach"] += int(reached)
            seen["advance"] += int(m_next[i] > m_ref[i])
            seen["hold"] += int(mask[i] and not reached)
            seen["complete"] += int(done_next[i] and not done_ref[i])
            seen["boundary"] += int(kinds[i] == 2 and not reached)
        m_ref, done_ref = m_next, done_next
    assert all(count > 0 for count in seen.values()), seen
    assert all(done_ref) and tracker.m.tolist() == [num_wp] * b_r

    # ---- Configuration mapping and errors.
    from_cfg = WaypointTracker(cfg.waypoint_radius, cfg.goal_saturation_distance, cfg.goal_eps)
    assert (from_cfg.acceptance_radius, from_cfg.saturation_distance, from_cfg.eps) == (1.0, 20.0, 1e-3)
    fresh = WaypointTracker(tau, d_max, eps_r)
    assert fresh.num_waypoints is None
    _expect(RuntimeError, lambda: fresh.step(torch.zeros(2, 3), torch.zeros(2)))
    _expect(RuntimeError, fresh.active_waypoint)
    _expect(RuntimeError, lambda: fresh.distance(torch.zeros(2, 3)))
    for bad in (torch.zeros(2, 3), torch.zeros(2, 0, 3), torch.zeros(2, 3, 2), torch.zeros(0, 2, 3),
                torch.full((1, 2, 3), float("nan")), torch.zeros(1, 2, 3, dtype=torch.long)):
        _expect(ValueError, lambda w=bad: fresh.reset(w))
    fresh.reset(torch.zeros(2, 3, 3))
    for p_bad, yaw_bad in ((torch.zeros(3, 3), torch.zeros(3)), (torch.zeros(2, 2), torch.zeros(2)),
                           (torch.zeros(2, 3), torch.zeros(3)), (torch.zeros(2, 3), torch.zeros(2, 1))):
        _expect(ValueError, lambda a=p_bad, y=yaw_bad: fresh.step(a, y))
    _expect(ValueError, lambda: fresh.step(torch.zeros(2, 3), torch.zeros(2), update_mask=torch.ones(3, dtype=torch.bool)))
    for kw in (dict(acceptance_radius=0.0), dict(acceptance_radius=float("inf")), dict(saturation_distance=-1.0),
               dict(eps=0.0)):
        args = dict(acceptance_radius=tau, saturation_distance=d_max, eps=eps_r)
        args.update(kw)
        _expect(ValueError, lambda a=args: WaypointTracker(**a))
    print("selector.py self-test passed")


if __name__ == "__main__":
    _self_test()

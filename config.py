# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
config.py -- Hyper-parameters of Aero-VPA (preprint_261026.pdf, Secs. 3-4, Table 1).

``VPAConfig`` holds every dimension of Section 3 and Table 1, plus the constants the paper leaves
symbolic. Tags in the field comments:

    [P]   stated in the paper
    [D]   forced by a Table 1 parameter count
    [I]   placeholder chosen here: set it per airframe, sensor or experiment
    None  calibrated from validation flights (Sec. 3.5); code that needs it must refuse to run while
          it is None

Derived dimensions (properties):

    num_patches        N_p = (128/16)(160/16) = 80
    selector_in_dim    d + d_g + d_s + d_c        = 256 + 4 + 13 + 512        = 785   (Eq. 12)
    predictor_in_dim   d + d_u + d_a              = 256 + 32 + 4              = 292   (Eq. 7)
    cond_dim           d_e = d_u + 2d + d_s + d_g + d_c = 32 + 512 + 13 + 4 + 512 = 1073  (Eq. 15)
    sequential_depth   K + 3 = 5

``VPAConfig`` moved here from ``pipeline.py`` so that it can be imported and tested before the
selector, solver and pipeline are migrated to Aero-VPA; ``pipeline.py`` re-exports it. This file
needs neither torch nor the other modules at import time. The self-test (``python config.py``)
builds the modules from the config and checks the Table 1 budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__ = ["VPAConfig", "TABLE1_PARAMS_M", "PRIMITIVES", "STATE_GROUPS"]

# Table 1 (millions, as printed).
TABLE1_PARAMS_M: Dict[str, float] = {
    "encoder": 11.17, "selector": 0.67, "predictor": 2.72, "solver": 3.57, "total": 18.13,
}

# Flight primitives U in the order of Eq. 12; the index is the token u_t.
PRIMITIVES: Tuple[str, ...] = (
    "CRUISE", "LATERAL_AVOID", "VERTICAL_AVOID", "YAW_SCAN", "APPROACH", "BRAKE_HOVER",
)

# The four groups of S_t (Eq. 2) as [start, stop) indices: v^H, n^B, omega^B and a_bar (the last dispatched setpoint).
STATE_GROUPS: Dict[str, Tuple[int, int]] = {
    "velocity": (0, 3), "attitude": (3, 6), "rates": (6, 9), "last_command": (9, 13),
}


@dataclass
class VPAConfig:
    """Hyper-parameters of the Aero-VPA pipeline (defaults: Table 1 and Section 3)."""

    # ---- Perception (Sec. 3.2, Table 1) ----
    image_size: Tuple[int, int] = (128, 160)    # [P] (H_img, W_img)
    in_channels: int = 4                        # [P] R, G, B, D
    patch_size: int = 16                        # [P]
    vit_width: int = 384                        # [P]
    vit_depth: int = 6                          # [P]
    vit_heads: int = 6                          # [P]
    vit_mlp_ratio: float = 4.0                  # [D] 11.17M
    latent_dim: int = 256                       # [P] d
    depth_range: Tuple[float, float] = (0.1, 10.0)  # [I] depth sensor range (m): clip and normalise
    clip_model_name: str = "openai/clip-vit-base-patch32"  # [P] frozen text tower
    text_dim: int = 512                         # [P] d_c

    # ---- Observation and mission (Sec. 3.1) ----
    proprio_dim: int = 13                       # [P] d_s = v^H (3) + n^B (3) + omega^B (3) + a_bar (4)
    goal_dim: int = 4                           # [P] d_g: unit direction (3) + saturated distance (1)
    goal_saturation_distance: float = 20.0      # [I] D_max (m)
    goal_eps: float = 1e-3                      # [I] eps_r (m)
    waypoint_radius: float = 1.0                # [P] tau_w (m), Sec. 5.4

    # ---- Primitive selector (Sec. 3.3) ----
    num_primitives: int = 6                     # [P] N_u, see PRIMITIVES
    selector_hidden: Tuple[int, ...] = (512, 512)  # [P] Table 1
    # [I] groups of S_t (STATE_GROUPS) that pi_phi^h (Eq. 12) and e_t (Eq. 15) see; the others enter at their
    # training mean (0 after standardisation). P_omega always receives a_bar (Eq. 7). All four: Eq. 2 as written.
    policy_state_inputs: Tuple[str, ...] = ("velocity", "attitude", "rates", "last_command")

    # ---- JEPA predictor (Sec. 3.2) ----
    ensemble_heads: int = 5                     # [P] N_e
    primitive_embed_dim: int = 32               # [P] d_u (shared by predictor and solver input)
    predictor_hidden: int = 512                 # [P] Table 1
    predictor_hidden_layers: int = 2            # [P] Table 1
    predictor_horizon_s: float = 0.1            # [P] T_nu: "t+1" is 0.1 s ahead, not the next decision
    predictor_residual: bool = False            # [I] residual heads: z_hat^(i) = z_t + f^(i)(...) (same parameters)
    predictor_residual_init: float = 1.0        # [I] factor on the default init of the residual output layer

    # ---- JEPA training (Eqs. 8-11) ----
    vicreg_gamma: float = 1.0                   # [I] gamma
    vicreg_eps: float = 1e-4                    # [I] eps, shared by Eqs. 9 and 11
    lambda_v: float = 25.0                      # [I]
    lambda_c: float = 1.0                       # [I]
    ema_momentum: float = 0.996                 # [I] mu of E_psi_bar
    alpha_gamma: float = 0.01                   # [I] alpha_gamma of Eq. 11

    # ---- Action chunk and CFM solver (Secs. 3.1, 3.4, Table 1) ----
    action_dim: int = 4                         # [P] d_a = (v_x, v_y, v_z, yaw rate), heading frame
    horizon: int = 16                           # [P] H
    knot_dt: float = 0.02                       # [P] Delta (s); a chunk spans H * Delta = 0.32 s
    integration_steps: int = 2                  # [P] K
    field_width: int = 256                      # [P]
    field_depth: int = 4                        # [P]
    field_heads: int = 4                        # [P]
    field_mlp_ratio: float = 4.0                # [D] 3.57M
    time_embed_dim: int = 256                   # [D] 3.57M (128 gives 3.54M)
    lambda_h: float = 1.0                       # [I] weight of L_CE in Eq. 30
    noise_refresh_period: int = 30              # [I] N_xi of Eq. 20 (ablation A5: 10 or 30)

    # ---- Action box A_0 (Sec. 3.1): also the normalisation of actions and of a_bar ----
    max_speed_h: float = 7.0                    # [I] v_h (m/s): largest Sec. 5.1 limit, fixed scale
    max_speed_z: float = 2.0                    # [I] v_z (m/s)
    max_yaw_rate: float = 1.5                   # [I] yaw-rate bound (rad/s)

    # ---- Uncertainty and safety layer (Secs. 3.5, 3.6) ----
    alpha_sigma: float = 0.1                    # [I] alpha_sigma of Eq. 23
    sigma_lo: Optional[float] = None            # calibrated: 0.9-quantile of sigma_bar
    sigma_crit: Optional[float] = None          # calibrated: (1 - p_fa)-quantile of sigma_bar
    alpha_min: float = 0.5                      # [I] attenuation floor (ablation A2: 0.25, 0.5, 1)
    min_horizon: int = 4                        # [I] H_min of Eq. 25
    beta: float = 1.0                           # [I] beta of Eq. 25
    max_decel: float = 5.0                      # [P] a_max (m/s^2), Remark 3.1 example
    depth_error: float = 0.2                    # [I] eps_d (m)
    safety_radius: float = 0.5                  # [I] r_s >= r_v (m)
    blind_speed: float = 1.0                    # [I] v_blind (m/s)
    speed_eps: float = 1e-3                     # [I] eps_v (m/s)

    # ---- Executor (Secs. 3.7, 4.2) ----
    executor_rate_hz: float = 100.0             # [P] f_c
    blend_time_s: float = 0.04                  # [I] T_bl of Eq. 35

    def __post_init__(self) -> None:
        h, w = self.image_size
        positive_ints = {
            "in_channels": self.in_channels, "patch_size": self.patch_size, "vit_width": self.vit_width,
            "vit_depth": self.vit_depth, "vit_heads": self.vit_heads, "latent_dim": self.latent_dim,
            "text_dim": self.text_dim, "proprio_dim": self.proprio_dim, "goal_dim": self.goal_dim,
            "primitive_embed_dim": self.primitive_embed_dim, "predictor_hidden": self.predictor_hidden,
            "action_dim": self.action_dim, "horizon": self.horizon, "integration_steps": self.integration_steps,
            "field_width": self.field_width, "field_depth": self.field_depth, "field_heads": self.field_heads,
            "noise_refresh_period": self.noise_refresh_period,
        }
        for name, value in positive_ints.items():
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f"image_size {self.image_size} is not divisible by patch_size {self.patch_size}")
        if self.vit_width % self.vit_heads or self.field_width % self.field_heads:
            raise ValueError("transformer widths must be divisible by their head counts")
        if self.time_embed_dim < 2 or self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be a positive even number (sin and cos halves)")
        if self.num_primitives != len(PRIMITIVES):
            raise ValueError(f"num_primitives must be {len(PRIMITIVES)} (the primitives of Eq. 12)")
        if self.ensemble_heads < 2:
            raise ValueError("ensemble_heads must be >= 2 (Eq. 22 needs disagreement)")
        if self.predictor_hidden_layers < 1 or not self.selector_hidden or min(self.selector_hidden) < 1:
            raise ValueError("hidden layer specifications must be non-empty and positive")
        unknown = [g for g in self.policy_state_inputs if g not in STATE_GROUPS]
        if unknown or len(set(self.policy_state_inputs)) != len(self.policy_state_inputs):
            raise ValueError(f"policy_state_inputs must be distinct groups of {list(STATE_GROUPS)}, got "
                             f"{list(self.policy_state_inputs)}")
        if self.proprio_dim != 13 and set(self.policy_state_inputs) != set(STATE_GROUPS):
            raise ValueError("policy_state_inputs needs the 13-dimensional S_t of Eq. 2")
        if not 1 <= self.min_horizon <= self.horizon:
            raise ValueError("require 1 <= min_horizon <= horizon (Eq. 25)")
        if not 0.0 < self.depth_range[0] < self.depth_range[1]:
            raise ValueError("depth_range must satisfy 0 < d_min < d_max")
        if self.sigma_lo is not None and self.sigma_crit is not None and not 0.0 <= self.sigma_lo < self.sigma_crit:
            raise ValueError("require 0 <= sigma_lo < sigma_crit (Eq. 24)")
        if not 0.0 < self.alpha_min <= 1.0:
            raise ValueError("alpha_min must lie in (0, 1] (Eq. 24)")
        for name in ("alpha_gamma", "alpha_sigma"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not 0.0 < self.vicreg_eps < self.vicreg_gamma ** 2:
            raise ValueError("require 0 < vicreg_eps < vicreg_gamma^2 (Eq. 9)")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must lie in [0, 1)")
        if not isinstance(self.predictor_residual, bool):
            raise ValueError("predictor_residual must be a bool")
        if not (math.isfinite(self.predictor_residual_init) and self.predictor_residual_init >= 0.0):
            raise ValueError("predictor_residual_init must be finite and non-negative")
        for name in ("max_speed_h", "max_speed_z", "max_yaw_rate", "max_decel", "knot_dt", "predictor_horizon_s",
                     "goal_saturation_distance", "goal_eps", "waypoint_radius", "executor_rate_hz", "speed_eps",
                     "beta", "lambda_v"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        for name in ("depth_error", "safety_radius", "blind_speed", "lambda_c", "lambda_h", "blend_time_s"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be non-negative")

    # ------------------------------------------------------------------ derived dimensions
    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """``(C, H_img, W_img)`` of one observation I_t."""
        return (self.in_channels, *self.image_size)

    @property
    def patch_grid(self) -> Tuple[int, int]:
        return (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)

    @property
    def num_patches(self) -> int:
        """N_p = (H_img / P)(W_img / P) = 80."""
        return self.patch_grid[0] * self.patch_grid[1]

    @property
    def selector_in_dim(self) -> int:
        """Eq. 12 input Concat(z_t, g_t, S_t, c_text): d + d_g + d_s + d_c = 785."""
        return self.latent_dim + self.goal_dim + self.proprio_dim + self.text_dim

    @property
    def policy_state_mask(self) -> Tuple[float, ...]:
        """``[d_s]`` 1.0 where pi_phi^h and e_t see S_t, 0.0 where a group is withheld (``policy_state_inputs``)."""
        if set(self.policy_state_inputs) == set(STATE_GROUPS):
            return (1.0,) * self.proprio_dim
        mask = [0.0] * self.proprio_dim
        for group in self.policy_state_inputs:
            start, stop = STATE_GROUPS[group]
            mask[start:stop] = [1.0] * (stop - start)
        return tuple(mask)

    @property
    def predictor_in_dim(self) -> int:
        """Eq. 7 head input Concat(z_t, Embed(u_t), a_bar_t): d + d_u + d_a = 292."""
        return self.latent_dim + self.primitive_embed_dim + self.action_dim

    @property
    def cond_layout(self) -> Dict[str, int]:
        """Eq. 15 blocks of e_t, in order, with their widths."""
        return {
            "primitive": self.primitive_embed_dim,   # Embed(u_t)
            "z_t": self.latent_dim,                  # (z_t - mu_Z) / gamma*
            "z_hat": self.latent_dim,                # (z_hat_{t+1} - mu_Z) / gamma*
            "proprio": self.proprio_dim,             # (S_t - mu_S) / sigma_S
            "goal": self.goal_dim,                   # g_t (not re-standardised)
            "text": self.text_dim,                   # c_text (not re-standardised)
        }

    @property
    def cond_slices(self) -> Dict[str, slice]:
        """Slice of e_t occupied by each block of ``cond_layout``."""
        out, start = {}, 0
        for name, width in self.cond_layout.items():
            out[name] = slice(start, start + width)
            start += width
        return out

    @property
    def cond_dim(self) -> int:
        """d_e = d_u + 2d + d_s + d_g + d_c = 1073 (Eq. 15)."""
        return sum(self.cond_layout.values())

    @property
    def action_scale(self) -> Tuple[float, float, float, float]:
        """Half-widths of the symmetric box A_0: (v_h, v_h, v_z, yaw rate). Normalises actions and a_bar."""
        return (self.max_speed_h, self.max_speed_h, self.max_speed_z, self.max_yaw_rate)

    @property
    def chunk_duration_s(self) -> float:
        """H * Delta = 0.32 s."""
        return self.horizon * self.knot_dt

    @property
    def sequential_depth(self) -> int:
        """K + 3 = 5 sequential network evaluations per decision (Sec. 3.7)."""
        return self.integration_steps + 3

    # ------------------------------------------------------------------ builders
    def build_rgbd_normalizer(self) -> Any:
        """``perception.RGBDNormalizer`` for the configured depth range."""
        return _module("perception").RGBDNormalizer(depth_range=self.depth_range)

    def build_vision_encoder(self) -> Any:
        """E_psi (Eq. 6): ``perception.VisionEncoder``, 11,171,200 parameters with the defaults."""
        return _module("perception").VisionEncoder(
            image_size=self.image_size, patch_size=self.patch_size, in_channels=self.in_channels,
            width=self.vit_width, depth=self.vit_depth, num_heads=self.vit_heads,
            latent_dim=self.latent_dim, mlp_ratio=self.vit_mlp_ratio,
        )

    def build_momentum_encoder(self, online_encoder: Any) -> Any:
        """E_psi_bar (Eq. 8) for ``online_encoder``."""
        return _module("perception").MomentumEncoder(online_encoder, momentum=self.ema_momentum)

    def build_predictor(self) -> Any:
        """P_omega (Eq. 7): ``predictor.JEPAPredictor`` with a_bar normalised by the action box."""
        return _module("predictor").JEPAPredictor(
            latent_dim=self.latent_dim, num_primitives=self.num_primitives, setpoint_dim=self.action_dim,
            num_heads=self.ensemble_heads, primitive_embed_dim=self.primitive_embed_dim,
            hidden_dim=self.predictor_hidden, num_hidden_layers=self.predictor_hidden_layers,
            setpoint_scale=self.action_scale, residual=self.predictor_residual,
            residual_init_scale=self.predictor_residual_init,
        )

    def build_vicreg_loss(self) -> Any:
        """L_JEPA (Eqs. 8-10)."""
        return _module("predictor").VICRegLoss(
            lambda_v=self.lambda_v, lambda_c=self.lambda_c, gamma=self.vicreg_gamma, eps=self.vicreg_eps,
        )

    def build_latent_scale_tracker(self) -> Any:
        """gamma_bar and mu_Z (Eq. 11), with the eps of Eq. 9."""
        return _module("predictor").LatentScaleTracker(
            latent_dim=self.latent_dim, alpha_gamma=self.alpha_gamma, eps=self.vicreg_eps,
        )


def _module(name: str) -> Any:
    """Import a sibling module lazily, as a package member or as a flat file."""
    import importlib

    if __package__:
        return importlib.import_module(f".{name}", __package__)
    return importlib.import_module(name)


# ---------------------------------------------------------------------------------------------
# Self-test (run: python config.py)
# ---------------------------------------------------------------------------------------------
def _mlp_params(widths: Tuple[int, ...]) -> int:
    return sum(i * o + o for i, o in zip(widths[:-1], widths[1:], strict=True))


def _solver_params(cfg: VPAConfig) -> int:
    """Analytic size of the repo's ``solver.ConditionalVectorField`` (sinusoidal time, cond MLP, H knot
    tokens + 1 conditioning token, pre-norm encoder layers)."""
    w, ff = cfg.field_width, int(round(cfg.field_width * cfg.field_mlp_ratio))
    layer = 4 * w + (3 * w * w + 3 * w) + (w * w + w) + (w * ff + ff) + (ff * w + w)
    return ((cfg.action_dim * w + w) + cfg.horizon * w + ((cfg.cond_dim + cfg.time_embed_dim) * w + w)
            + (w * w + w) + cfg.field_depth * layer + 2 * w + (w * cfg.action_dim + cfg.action_dim))


def _expect_value_error(fn: Any) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _self_test() -> None:
    import torch
    import torch.nn as nn

    perception, predictor = _module("perception"), _module("predictor")

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    cfg = VPAConfig()

    # ---- Derived dimensions of Section 3.
    assert cfg.frame_shape == (4, 128, 160) and cfg.patch_grid == (8, 10) and cfg.num_patches == 80
    assert cfg.selector_in_dim == 785 and cfg.predictor_in_dim == 292
    assert cfg.cond_dim == 1073 == 32 + 2 * 256 + 13 + 4 + 512
    assert list(cfg.cond_layout) == ["primitive", "z_t", "z_hat", "proprio", "goal", "text"]
    sl = cfg.cond_slices
    assert (sl["primitive"], sl["z_t"], sl["z_hat"]) == (slice(0, 32), slice(32, 288), slice(288, 544))
    assert (sl["proprio"], sl["goal"], sl["text"]) == (slice(544, 557), slice(557, 561), slice(561, 1073))
    assert cfg.sequential_depth == 5 and math.isclose(cfg.chunk_duration_s, 0.32)
    assert cfg.action_scale == (7.0, 7.0, 2.0, 1.5) and len(PRIMITIVES) == cfg.num_primitives == 6
    assert cfg.policy_state_mask == (1.0,) * 13                                     # Eq. 2 as written
    no_command = VPAConfig(policy_state_inputs=("velocity", "attitude", "rates")).policy_state_mask
    assert no_command == (1.0,) * 9 + (0.0,) * 4
    assert VPAConfig(policy_state_inputs=("attitude",)).policy_state_mask == (0.0,) * 3 + (1.0,) * 3 + (0.0,) * 7
    for bad in (("speed",), ("velocity", "velocity")):
        try:
            VPAConfig(policy_state_inputs=bad)
            raise AssertionError(f"accepted policy_state_inputs={bad}")
        except ValueError:
            pass
    assert (cfg.action_dim, cfg.horizon, cfg.integration_steps, cfg.latent_dim) == (4, 16, 2, 256)

    # ---- E_psi from the config: [B, 4, 128, 160] -> [B, 256], 11,171,200 parameters.
    gen = torch.Generator().manual_seed(7)
    b = 2
    enc = cfg.build_vision_encoder().eval()
    assert count(enc) == perception.AERO_ENCODER_PARAMS == 11_171_200
    norm = cfg.build_rgbd_normalizer()
    rgb = torch.randint(0, 256, (b, 3, *cfg.image_size), generator=gen, dtype=torch.uint8)
    depth = torch.rand(b, *cfg.image_size, generator=gen) * 12.0
    frames = norm(rgb, depth).image
    assert frames.shape == (b, *cfg.frame_shape)
    with torch.no_grad():
        z_t = enc(frames)
    assert z_t.shape == (b, cfg.latent_dim)
    target = cfg.build_momentum_encoder(enc)
    assert count(target) == 0 and target(frames).shape == (b, cfg.latent_dim)

    # ---- P_omega from the config: a_bar normalised by the action box, 2,720,192 parameters.
    pred = cfg.build_predictor()
    assert count(pred) == predictor.AERO_PREDICTOR_PARAMS == 2_720_192
    assert torch.equal(pred.setpoint_scale, torch.tensor(cfg.action_scale))
    assert not pred.residual
    res_cfg = VPAConfig(predictor_residual=True, predictor_residual_init=0.5)
    res_pred = res_cfg.build_predictor()
    assert res_pred.residual and res_pred.residual_init_scale == 0.5 and count(res_pred) == count(pred)
    for bad in (dict(predictor_residual_init=-1.0), dict(predictor_residual_init=float("nan")),
                dict(predictor_residual=1)):
        try:
            VPAConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    tracker = cfg.build_latent_scale_tracker()
    tracker.update(torch.randn(64, cfg.latent_dim, generator=gen))
    gamma_star, mu_z = tracker.freeze()
    pred.set_latent_scale(gamma_star)
    u_t = torch.randint(0, cfg.num_primitives, (b,), generator=gen)
    a_bar = torch.randn(b, cfg.action_dim, generator=gen)
    with torch.no_grad():
        out = pred(z_t, u_t, a_bar)
    assert pred.head_input(z_t, u_t, a_bar).shape == (b, cfg.predictor_in_dim)
    assert out.head_predictions.shape == (cfg.ensemble_heads, b, cfg.latent_dim)
    assert out.z_hat.shape == (b, cfg.latent_dim) and out.sigma.shape == (b,) and mu_z.shape == (cfg.latent_dim,)
    loss_fn = cfg.build_vicreg_loss()
    assert loss_fn.eps == tracker.eps == cfg.vicreg_eps                     # one eps for Eqs. 9 and 11

    # ---- e_t (Eq. 15) assembled from the config layout: [B, 1073].
    blocks = {
        "primitive": torch.randn(b, cfg.primitive_embed_dim), "z_t": (z_t - mu_z) / gamma_star,
        "z_hat": (out.z_hat - mu_z) / gamma_star, "proprio": torch.randn(b, cfg.proprio_dim),
        "goal": torch.randn(b, cfg.goal_dim), "text": torch.randn(b, cfg.text_dim),
    }
    e_t = torch.cat([blocks[name] for name in cfg.cond_layout], dim=-1)
    assert e_t.shape == (b, cfg.cond_dim)
    assert all(torch.equal(e_t[:, sl[name]], blocks[name]) for name in cfg.cond_layout)

    # ---- Table 1 rows not migrated yet, pinned so the selector and solver refactor has fixed targets.
    selector_ref = nn.Sequential(
        nn.Linear(cfg.selector_in_dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, cfg.num_primitives)
    )
    selector_params = count(selector_ref)
    assert selector_params == _mlp_params((785, 512, 512, 6)) == 668_166
    solver_params = _solver_params(cfg)
    assert solver_params == 3_572_228
    assert _solver_params(VPAConfig(time_embed_dim=128)) == 3_539_460      # rounds to 3.54M, not Table 1
    try:
        field_cls = _module("solver").ConditionalVectorField
    except ImportError:
        field_cls = None
    if field_cls is not None:  # the current repo vector field, at the Aero-VPA dimensions
        field = field_cls(cfg.horizon, cfg.action_dim, cfg.cond_dim, width=cfg.field_width, depth=cfg.field_depth,
                          num_heads=cfg.field_heads, mlp_ratio=cfg.field_mlp_ratio, time_embed_dim=cfg.time_embed_dim)
        assert count(field) == solver_params
        with torch.no_grad():
            v = field(torch.randn(b, cfg.horizon, cfg.action_dim, generator=gen), 0.0, e_t)
        assert v.shape == (b, cfg.horizon, cfg.action_dim)

    # ---- Table 1 totals, FP16 weights (36.3 MB) and the bytes streamed per decision (Eq. 31: 43.4 MB).
    rows = {"encoder": count(enc), "selector": selector_params, "predictor": count(pred), "solver": solver_params}
    for name, params in rows.items():
        assert round(params / 1e6, 2) == TABLE1_PARAMS_M[name], (name, params)
    total = sum(rows.values())
    assert round(total / 1e6, 2) == TABLE1_PARAMS_M["total"] and total == 18_131_786
    assert round(2 * total / 1e6, 1) == 36.3
    k = cfg.integration_steps
    m_dec = 2 * (rows["encoder"] + rows["selector"] + rows["predictor"] + k * rows["solver"])
    assert round(m_dec / 1e6, 1) == 43.4

    # ---- Validation.
    for kw in (dict(image_size=(130, 160)), dict(vit_heads=5), dict(min_horizon=17), dict(ensemble_heads=1),
               dict(integration_steps=0), dict(sigma_lo=0.5, sigma_crit=0.4), dict(depth_range=(5.0, 1.0)),
               dict(max_speed_h=0.0), dict(num_primitives=5), dict(time_embed_dim=255), dict(vicreg_eps=2.0),
               dict(alpha_min=0.0)):
        _expect_value_error(lambda k=kw: VPAConfig(**k))
    print("config.py self-test passed")


if __name__ == "__main__":
    _self_test()

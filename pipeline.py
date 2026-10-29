# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
pipeline.py -- Closed-loop Aero-VPA (preprint_261026.pdf, Algorithm 1, Secs. 3.7, 4.2, 4.3).

``VPAInferencePipeline`` binds the Aero-VPA modules and runs the two loops of Algorithm 1.

Inference loop, ``step`` (DECIDE, event-driven, one decision per camera frame):
    1. z_t = E_psi(I_t)                                                         (Eq. 6)
    2. Eq. 14 waypoint update on p_hat_t, then g_t (Eq. 3)                      (WaypointTracker)
    3. u_t = argmax pi_phi^h(u | z_t, g_t, S_t, c_text)                          (Eq. 12)
    4. (z_hat_{t+1}, sigma_{t+1}) = P_omega(z_t, Embed(u_t), a_bar_t),  a_bar_t = S_t[:, 9:13]   (Eqs. 7, 22)
    5. sigma_bar_{t+1}, alpha, H_t (Eqs. 23-25); sigma_bar >= sigma_crit -> u_t <- BRAKE_HOVER,
       request the fallback and return without a chunk (lines 6-7)
    6. xi_t by noise persistence (Eq. 20); e_t (Eq. 15)
    7. A_hat_t = Pi_{A_0}(denormalise(Euler_K(xi_t; e_t)))                      (Eqs. 18, 19)
    8. d_t (from the caller, possibly as a function of A_hat_t), T_r, v_safe, s_t (Eqs. 27, 28)
    9. rate limit of consecutive knots (Eq. 34), then publish C_t = (tau_t, psi_t, knots, H_t) to the
       single-slot chunk buffer
   The networks form the dependency chain E_psi -> pi_phi^h -> P_omega -> v_theta^K: K + 3 sequential
   evaluations per decision whatever H (Sec. 3.7), or 3 when every deciding element falls back. Every
   ``step`` audits the chain with forward hooks and raises if the evaluations differ in count or order.

Executor, ``tick`` (TICK, fixed rate f_c = 100 Hz):
    * newest chunk C, age = tau - tau_C (latency compensation: knots are indexed by time since capture)
    * if a fallback is requested, there is no chunk, the chunk is stale (age > H_C Delta) or odometry is
      stale: the minimum-jerk braking profile (Eq. 26) from the last dispatched velocity, then hold
    * otherwise: piecewise-linear interpolation of the knots with exactly consistent acceleration
      feed-forward (Eq. 33), cross-fade from the previous reference over T_bl (Eq. 35), and rotation from
      the heading frame frozen at capture time to NED (Eq. 32)
    * the dispatched setpoint is stored; ``proprioception`` re-expresses it in the current heading frame
      as a_bar_t of the next decision (Eq. 2, Algorithm 1 line 27)

``act`` runs both loops in one thread for simulation and tests: with ``execution_mode="async"`` (the paper)
every new observation triggers a decision; with ``"sync"`` (ablation A8) a decision is triggered only when
the cursor floor(age / Delta) reaches H_t, there is no chunk, or a fallback is pending.

Decisions the paper leaves open, as implemented here
    * The rate limit of Eq. 34 is applied once when the chunk is committed (it depends only on the chunk);
      it moves each knot towards the next one, so every knot stays inside the speed envelope of Eq. 28.
    * The braking profile starts from the last dispatched NED velocity setpoint with zero acceleration; the
      yaw-rate setpoint is 0 while braking and holding. Leaving the fallback cross-fades from the last
      dispatched setpoint.
    * The depth range d_t of Eq. 27 needs the camera model; ``clearance`` is either a ``[B]`` tensor or a
      callable that receives the projected chunk ``[B, H, 4]`` and returns d_t or (d_t, blind).
    * Times are seconds; the executor stores them relative to the mission start (``reset``) in float32.

This file imports the other Aero-VPA modules; ``python pipeline.py`` runs its self-test.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

try:  # package import (e.g. `from aero_vpa.pipeline import ...`)
    from .config import VPAConfig
    from .perception import TextEncoderWrapper, VisionEncoder, count_parameters
    from .predictor import JEPAPredictor
    from .selector import PrimitiveSelector, WaypointTracker, heading_to_world, world_to_heading
    from .solver import BRAKE_HOVER, CFMSolver, NoisePersistence, SafetyFilter
except ImportError:  # flat import (files side by side, `python pipeline.py`)
    from config import VPAConfig
    from perception import TextEncoderWrapper, VisionEncoder, count_parameters
    from predictor import JEPAPredictor
    from selector import PrimitiveSelector, WaypointTracker, heading_to_world, world_to_heading
    from solver import BRAKE_HOVER, CFMSolver, NoisePersistence, SafetyFilter

__all__ = [
    "VPAConfig",
    "MODE_TRACK",
    "MODE_BRAKE",
    "MODE_HOLD",
    "Observation",
    "StepOutput",
    "ActOutput",
    "VPAInferencePipeline",
]

MODE_TRACK: int = 0  # following the newest valid chunk
MODE_BRAKE: int = 1  # minimum-jerk stop in progress (Eq. 26)
MODE_HOLD: int = 2   # stopped: hold position (zero velocity setpoint)

Clearance = Union[Tensor, Callable[[Tensor], Union[Tensor, Tuple[Tensor, Optional[Tensor]]]]]


class Observation(NamedTuple):
    """One camera frame and the matching state estimate (input of ``act``)."""

    image: Tensor          # [B, 4, 128, 160] normalised RGB-D frame I_t
    proprioception: Tensor  # [B, 13] S_t = (v^H, n^B, omega^B, a_bar) (Eq. 2)
    position: Tensor       # [B, 3] p_hat_t in NED (m)
    yaw: Tensor            # [B] psi_hat_t (rad)
    clearance: Any         # [B] d_t (m), or a callable of the projected chunk (see ``Clearance``)
    capture_time: float    # tau_t (s)


class StepOutput(NamedTuple):
    """Everything produced by one decision step (leading dimension B)."""

    z_t: Tensor                      # [B, d]
    goal: Tensor                     # [B, 4]       g_t (Eq. 3)
    waypoint_index: Tensor           # [B] int64    m_t used for g_t (after the Eq. 14 update, 1-based)
    task_complete: Tensor            # [B] bool     sticky
    selected_primitive: Tensor       # [B] int64    argmax of pi_phi^h
    primitive: Tensor                # [B] int64    after the BRAKE_HOVER override
    z_hat: Tensor                    # [B, d]       z_hat_{t+1}
    sigma: Tensor                    # [B]          sigma_{t+1} (Eq. 22)
    sigma_bar: Tensor                # [B]          sigma_bar_{t+1} (Eq. 23)
    alpha: Tensor                    # [B]          alpha(sigma_bar) (Eq. 24)
    validity_horizon: Tensor         # [B] int64    H_t (Eq. 25)
    fallback: Tensor                 # [B] bool     deciding elements that requested the fallback
    conditioning: Optional[Tensor]   # [B, d_e]     e_t (None if every deciding element fell back)
    noise: Optional[Tensor]          # [B, H, d_a]  xi_t
    chunk_raw: Optional[Tensor]      # [B, H, d_a]  Pi_{A_0}(denormalised Euler output), heading frame
    chunk: Optional[Tensor]          # [B, H, d_a]  committed: scaled by s_t (Eq. 28) and rate-limited (Eq. 34)
    v_safe: Optional[Tensor]         # [B]          v_safe(d_t) (Eq. 27) or v_blind
    scale: Optional[Tensor]          # [B]          s_t (Eq. 28)
    reaction_time: Optional[Tensor]  # [B]          T_r
    committed: Tensor                # [B] bool     a new chunk was written to the buffer
    decision_mask: Tensor            # [B] bool     elements that decided at this step
    num_sequential_evaluations: int  # K + 3, or 3 when every deciding element fell back


class ActOutput(NamedTuple):
    """Executor output for one tick (leading dimension B): the PX4 TrajectorySetpoint fields."""

    velocity: Tensor                   # [B, 3] NED velocity setpoint (m/s)
    acceleration: Tensor               # [B, 3] NED acceleration feed-forward (m/s^2)
    yaw_rate: Tensor                   # [B]    yaw-rate setpoint (rad/s)
    mode: Tensor                       # [B] int64: MODE_TRACK, MODE_BRAKE or MODE_HOLD
    chunk_age: Tensor                  # [B] tau - tau_C (s); inf without a chunk
    knot_index: Tensor                 # [B] int64 cursor floor(age / Delta); -1 without a chunk
    replanned: Tensor                  # [B] bool, elements that decided during this call
    step_output: Optional[StepOutput]  # set when a decision ran during this call


class VPAInferencePipeline(nn.Module):
    """Aero-VPA closed loop: the inference loop and the executor of Algorithm 1.

    Args:
        vision_encoder: E_psi (frozen by ``calibrate``).
        text_encoder: frozen CLIP text tower with the learned null embedding (c_text).
        selector: pi_phi^h.
        predictor: P_omega (frozen by ``calibrate``).
        solver: CFM solver with Embed and v_theta.
        tracker: waypoint pointer (Eqs. 3, 14).
        safety: uncertainty gating and braking envelope (Eqs. 23-28).
        noise: noise persistence (Eq. 20).
        blend_time: T_bl of Eq. 35 (s), >= 0.
        execution_mode: "async" (Algorithm 1) or "sync" (ablation A8).

    Episode state (``reset``): c_text ``[B, d_c]``, the tracker, the filter state, the noise state and the
    executor state (chunk buffer, previous chunk for blending, braking state, last dispatched setpoint).
    """

    CHAIN_NAMES: Tuple[str, str, str, str] = ("E_psi", "pi_phi_h", "P_omega", "v_theta")

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        text_encoder: TextEncoderWrapper,
        selector: PrimitiveSelector,
        predictor: JEPAPredictor,
        solver: CFMSolver,
        tracker: WaypointTracker,
        safety: SafetyFilter,
        noise: NoisePersistence,
        blend_time: float = 0.04,
        execution_mode: str = "async",
    ) -> None:
        super().__init__()
        if not (math.isfinite(blend_time) and blend_time >= 0.0):
            raise ValueError("blend_time must be finite and non-negative")
        if execution_mode not in ("async", "sync"):
            raise ValueError("execution_mode must be 'async' or 'sync'")
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.selector = selector
        self.predictor = predictor
        self.solver = solver
        self.tracker = tracker
        self.safety = safety
        self.noise = noise
        self.blend_time: float = float(blend_time)
        self.execution_mode: str = execution_mode
        self._validate_dimensions()
        self._recording: bool = False
        self._chain_log: List[str] = []
        for name, module in zip(self.CHAIN_NAMES, (vision_encoder, selector, predictor, solver.vector_field),
                                strict=True):
            module.register_forward_hook(self._make_hook(name))
        self.c_text: Optional[Tensor] = None
        self._epoch: float = 0.0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls,
        config: VPAConfig,
        text_encoder: Optional[TextEncoderWrapper] = None,
        *,
        latency_bound: float,
        tracking_lag: float,
        execution_mode: str = "async",
    ) -> "VPAInferencePipeline":
        """Instantiate every module from ``config``; sigma_lo and sigma_crit must be calibrated.

        Args:
            config: ``VPAConfig``.
            text_encoder: optional injected text encoder (default: the CLIP model of the config).
            latency_bound: T_e2e^max (s), the measured worst-case end-to-end latency (Sec. 4.4).
            tracking_lag: T_trk (s), the velocity-tracking lag of the flight controller.
            execution_mode: "async" or "sync".
        """
        if text_encoder is None:
            text_encoder = TextEncoderWrapper(config.clip_model_name)
        if text_encoder.embed_dim != config.text_dim:
            raise ValueError(f"text encoder gives d_c = {text_encoder.embed_dim}, config expects {config.text_dim}")
        selector = PrimitiveSelector(
            latent_dim=config.latent_dim, goal_dim=config.goal_dim, proprio_dim=config.proprio_dim,
            text_dim=config.text_dim, num_primitives=config.num_primitives, hidden_dims=config.selector_hidden,
            proprio_mask=config.policy_state_mask,
        )
        return cls(
            vision_encoder=config.build_vision_encoder(),
            text_encoder=text_encoder,
            selector=selector,
            predictor=config.build_predictor(),
            solver=CFMSolver.from_config(config),
            tracker=WaypointTracker(config.waypoint_radius, config.goal_saturation_distance, config.goal_eps),
            safety=SafetyFilter.from_config(config, latency_bound=latency_bound, tracking_lag=tracking_lag),
            noise=NoisePersistence(config.horizon, config.action_dim, config.noise_refresh_period),
            blend_time=config.blend_time_s,
            execution_mode=execution_mode,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        latency_bound: float,
        tracking_lag: float,
        text_encoder: Union[str, TextEncoderWrapper] = "auto",
        execution_mode: str = "async",
        map_location: Any = "cpu",
        **config_overrides: Any,
    ) -> "VPAInferencePipeline":
        """Rebuild the closed loop from a "policy" checkpoint (checkpoint.py), calibrated and frozen.

        Args:
            path: checkpoint written by train_stage2.py (and calibrate_sigma.py for sigma_lo, sigma_crit).
            latency_bound, tracking_lag, execution_mode: as in ``from_config``.
            text_encoder: "clip" (the frozen CLIP tower of the config), "null" (no text tower: missions without
                instructions only), "auto" ("null" if the training data had no instructions, else "clip"), or an
                injected ``TextEncoderWrapper``. The learned null embedding is loaded into it in every case.
            map_location: device for loading.
            **config_overrides: ``VPAConfig`` fields to override, e.g. sigma_lo and sigma_crit before calibration.
        """
        try:
            from .checkpoint import config_from_dict, load_checkpoint, null_text_encoder
        except ImportError:
            from checkpoint import config_from_dict, load_checkpoint, null_text_encoder
        ckpt = load_checkpoint(path, kind="policy", map_location=map_location)
        config = config_from_dict(ckpt["config"], **config_overrides)
        if config.sigma_lo is None or config.sigma_crit is None:
            raise ValueError(f"{path} has no calibrated sigma_lo / sigma_crit: run calibrate_sigma.py, or pass "
                             "sigma_lo=... and sigma_crit=... to from_checkpoint")
        if isinstance(text_encoder, str):
            choice = text_encoder
            if choice == "auto":
                choice = "clip" if ckpt["provenance"].get("instructions") else "null"
            if choice == "null":
                text_encoder = null_text_encoder(config.text_dim)
            elif choice == "clip":
                text_encoder = TextEncoderWrapper(config.clip_model_name)
            else:
                raise ValueError("text_encoder must be 'auto', 'null', 'clip' or a TextEncoderWrapper")
        pipe = cls.from_config(config, text_encoder, latency_bound=latency_bound, tracking_lag=tracking_lag,
                               execution_mode=execution_mode)
        states = ckpt["state_dict"]
        pipe.vision_encoder.load_state_dict(states["vision_encoder"])
        pipe.predictor.load_state_dict(states["predictor"])
        pipe.selector.load_state_dict(states["selector"])
        pipe.solver.load_state_dict(states["solver"])
        with torch.no_grad():
            pipe.text_encoder.null_embedding.copy_(states["text_null_embedding"])
        cal = ckpt["calibration"]
        pipe.to(map_location if isinstance(map_location, (str, torch.device)) else "cpu")
        pipe.calibrate(float(cal["gamma_bar_star"]), cal["latent_mean"], cal["proprio_mean"], cal["proprio_std"])
        pipe.text_encoder.requires_grad_(False)
        return pipe

    def _validate_dimensions(self) -> None:
        d, d_c = self.vision_encoder.latent_dim, self.text_encoder.embed_dim
        checks = [
            ("selector.latent_dim", self.selector.latent_dim, d),
            ("predictor.latent_dim", self.predictor.latent_dim, d),
            ("solver.latent_dim", self.solver.latent_dim, d),
            ("selector.text_dim", self.selector.text_dim, d_c),
            ("solver.text_dim", self.solver.text_dim, d_c),
            ("predictor.num_primitives", self.predictor.num_primitives, self.selector.num_primitives),
            ("solver.num_primitives", self.solver.num_primitives, self.selector.num_primitives),
            ("solver.proprio_dim", self.solver.proprio_dim, self.selector.proprio_dim),
            ("solver.goal_dim", self.solver.goal_dim, self.selector.goal_dim),
            ("selector.goal_dim", self.selector.goal_dim, 4),
            ("solver.action_dim", self.solver.action_dim, 4),
            ("predictor.setpoint_dim", self.predictor.setpoint_dim, self.solver.action_dim),
            ("safety.max_horizon", self.safety.max_horizon, self.solver.horizon),
            ("noise.horizon", self.noise.horizon, self.solver.horizon),
            ("noise.action_dim", self.noise.action_dim, self.solver.action_dim),
        ]
        for name, got, want in checks:
            if got != want:
                raise ValueError(f"dimension mismatch: {name} = {got}, expected {want}")
        if self.selector.proprio_dim < self.solver.action_dim:
            raise ValueError("S_t must end with a_bar_t (Eq. 2)")
        if not BRAKE_HOVER < self.selector.num_primitives:
            raise ValueError("BRAKE_HOVER must be a valid primitive token")
        if not torch.allclose(self.predictor.setpoint_scale.cpu(), self.solver.action_scale.cpu()):
            raise ValueError("a_bar (predictor) and actions (solver) must be normalised by the same box A_0")

    def _make_hook(self, name: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if self._recording:
                self._chain_log.append(name)

        return hook

    def _device(self) -> torch.device:
        return next(self.vision_encoder.parameters()).device

    def parameter_budget(self) -> Dict[str, int]:
        """Table 1 rows (all parameters, frozen or not; the CLIP text tower is excluded)."""
        rows = {
            "encoder": count_parameters(self.vision_encoder, trainable_only=False),
            "selector": count_parameters(self.selector, trainable_only=False),
            "predictor": count_parameters(self.predictor, trainable_only=False),
            "solver": count_parameters(self.solver, trainable_only=False),
        }
        rows["total"] = sum(rows.values())
        return rows

    # ------------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, gamma_bar_star: float, latent_mean: Tensor, proprio_mean: Tensor, proprio_std: Tensor) -> None:
        """Install the frozen statistics of stage (i) and the dataset, and freeze E_psi and P_omega.

        gamma* normalises sigma (Eq. 22) and the latents of e_t (Eq. 15), so predictor and solver receive
        the same value; mu_Z, mu_S and sigma_S standardise e_t.
        """
        self.predictor.set_latent_scale(float(gamma_bar_star))
        self.solver.set_standardization_stats(latent_mean, float(gamma_bar_star), proprio_mean, proprio_std)
        self.vision_encoder.requires_grad_(False)
        self.predictor.requires_grad_(False)
        self.eval()

    def _check_calibrated(self) -> None:
        if not (self.predictor.has_latent_scale and self.solver.is_calibrated):
            raise RuntimeError("pipeline is not calibrated; call calibrate(gamma*, mu_Z, mu_S, sigma_S)")
        scales = (float(self.predictor.latent_scale), float(self.solver.gamma_bar_star))
        if not math.isclose(scales[0], scales[1], rel_tol=1e-6):
            raise RuntimeError(f"inconsistent gamma* between predictor and solver: {scales}")

    # ------------------------------------------------------------------ mission lifecycle
    @torch.no_grad()
    def reset(self, instructions: Union[None, str, Sequence[Optional[str]]], waypoints: Tensor,
              mission_start: float = 0.0) -> None:
        """Begin a mission: encode c_text once, set the waypoints, and reset every per-episode state.

        Args:
            instructions: None, one string (shared by the batch) or B entries (str or None).
            waypoints: ``[B, M, 3]`` NED waypoints (m).
            mission_start: clock time (s) of the mission start; executor times are stored relative to it.
        """
        self._check_calibrated()
        if waypoints.dim() != 3 or waypoints.shape[2] != 3:
            raise ValueError(f"expected waypoints of shape [B, M, 3], got {tuple(waypoints.shape)}")
        device = self._device()
        b = waypoints.shape[0]
        c_text = self.text_encoder(instructions).to(device=device, dtype=torch.float32)          # [B or 1, d_c]
        if c_text.shape[0] == 1 and b > 1:
            c_text = c_text.expand(b, -1)
        if c_text.shape[0] != b:
            raise ValueError(f"got {c_text.shape[0]} instructions for a batch of {b}")
        self.c_text = c_text.contiguous()
        self.tracker.reset(waypoints.to(device=device, dtype=torch.float32))
        self.safety.reset(b, device)
        self.noise.reset(b, device)
        self._epoch = float(mission_start)
        h, f32 = self.solver.horizon, torch.float32
        zeros = torch.zeros(b, device=device, dtype=f32)
        false = torch.zeros(b, device=device, dtype=torch.bool)
        self._chunk = torch.zeros(b, h, 4, device=device, dtype=f32)
        self._chunk_capture = zeros.clone()
        self._chunk_yaw = zeros.clone()
        self._chunk_validity = torch.zeros(b, device=device, dtype=torch.long)
        self._has_chunk = false.clone()
        self._version = torch.zeros(b, device=device, dtype=torch.long)
        self._prev_chunk = torch.zeros(b, h, 4, device=device, dtype=f32)
        self._prev_capture = zeros.clone()
        self._prev_yaw = zeros.clone()
        self._has_prev = false.clone()
        self._fallback_requested = false.clone()
        self._seen_version = torch.zeros(b, device=device, dtype=torch.long)
        self._switch_time = zeros.clone()
        self._blend_from_hold = torch.ones(b, device=device, dtype=torch.bool)
        self._hold_setpoint = torch.zeros(b, 4, device=device, dtype=f32)
        self._braking = false.clone()
        self._brake_start = zeros.clone()
        self._brake_v0 = torch.zeros(b, 3, device=device, dtype=f32)
        self._last_setpoint = torch.zeros(b, 4, device=device, dtype=f32)   # NED velocity (3) + yaw rate
        self._last_mode = torch.full((b,), MODE_HOLD, device=device, dtype=torch.long)

    def _require_episode(self) -> Tensor:
        if self.c_text is None:
            raise RuntimeError("call reset(instructions, waypoints) before step/tick/act")
        return self.c_text

    def _relative_time(self, value: Union[float, Tensor], b: int) -> Tensor:
        """Clock time(s) -> ``[B]`` float32 seconds since the mission start."""
        device = self._device()
        if isinstance(value, Tensor):
            if value.numel() == 1:
                value = float(value.reshape(()))
            else:
                if tuple(value.shape) != (b,):
                    raise ValueError(f"expected times of shape [{b}], got {tuple(value.shape)}")
                rel = [float(v) - self._epoch for v in value.detach().cpu().tolist()]
                return torch.tensor(rel, device=device, dtype=torch.float32)
        return torch.full((b,), float(value) - self._epoch, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------ state estimate helper
    def proprioception(self, velocity_ned: Tensor, gravity_body: Tensor, body_rates: Tensor, yaw: Tensor) -> Tensor:
        """S_t (Eq. 2) from the EKF2 estimate and the last dispatched setpoint.

        S_t = ( R_z(psi)^T v^W,  n^B,  omega^B,  a_bar_t ),  a_bar_t = ( R_z(psi)^T v_ref^W,  yaw rate_ref ):
        the dispatched NED setpoint re-expressed in the current heading frame.

        Args:
            velocity_ned: ``[B, 3]``;  gravity_body: ``[B, 3]`` unit;  body_rates: ``[B, 3]``;  yaw: ``[B]``.

        Returns:
            ``[B, 13]``.
        """
        self._require_episode()
        b = self._last_setpoint.shape[0]
        for name, tensor in (("velocity_ned", velocity_ned), ("gravity_body", gravity_body), ("body_rates", body_rates)):
            if tuple(tensor.shape) != (b, 3):
                raise ValueError(f"expected {name} of shape [{b}, 3], got {tuple(tensor.shape)}")
        device = self._device()
        yaw = yaw.to(device=device, dtype=torch.float32)
        v_h = world_to_heading(velocity_ned.to(device=device, dtype=torch.float32), yaw)
        a_bar = world_to_heading(self._last_setpoint[:, :3], yaw)
        return torch.cat([v_h, gravity_body.to(device=device, dtype=torch.float32),
                          body_rates.to(device=device, dtype=torch.float32), a_bar, self._last_setpoint[:, 3:]], dim=-1)

    # ------------------------------------------------------------------ inference loop (DECIDE)
    def _resolve_clearance(self, clearance: Clearance, chunk: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        result = clearance(chunk) if callable(clearance) else clearance
        depth, blind = (result if isinstance(result, tuple) else (result, None))
        b = chunk.shape[0]
        depth = torch.as_tensor(depth, dtype=chunk.dtype, device=chunk.device)
        if depth.dim() == 0:
            depth = depth.expand(b)
        if tuple(depth.shape) != (b,):
            raise ValueError(f"expected d_t of shape [{b}], got {tuple(depth.shape)}")
        if blind is not None:
            blind = torch.as_tensor(blind, dtype=torch.bool, device=chunk.device)
            if tuple(blind.shape) != (b,):
                raise ValueError(f"expected blind of shape [{b}], got {tuple(blind.shape)}")
        return depth, blind

    def _rate_limit(self, chunk: Tensor) -> Tensor:
        """Eq. 34: v_{k+1} <- v_k + clip_{a_max Delta}(v_{k+1} - v_k), sequentially over the knots; yaw untouched."""
        max_step = self.safety.max_decel * self.safety.knot_dt
        velocities = [chunk[:, 0, :3]]
        for k in range(1, chunk.shape[1]):
            delta = chunk[:, k, :3] - velocities[-1]
            norm = torch.linalg.vector_norm(delta, ord=2, dim=-1, keepdim=True)
            factor = torch.where(norm > max_step, max_step / norm.clamp_min(1e-12), torch.ones_like(norm))
            velocities.append(velocities[-1] + delta * factor)
        return torch.cat([torch.stack(velocities, dim=1), chunk[..., 3:]], dim=-1)

    def _commit(self, chunk: Tensor, capture: Tensor, yaw: Tensor, validity: Tensor, mask: Tensor) -> None:
        """Single-slot buffer write for the masked elements; the replaced chunk is kept for blending (Eq. 35)."""
        m3 = mask.view(-1, 1, 1)
        move = mask & self._has_chunk
        self._prev_chunk = torch.where(move.view(-1, 1, 1), self._chunk, self._prev_chunk)
        self._prev_capture = torch.where(move, self._chunk_capture, self._prev_capture)
        self._prev_yaw = torch.where(move, self._chunk_yaw, self._prev_yaw)
        self._has_prev = self._has_prev | move
        self._chunk = torch.where(m3, chunk, self._chunk)
        self._chunk_capture = torch.where(mask, capture, self._chunk_capture)
        self._chunk_yaw = torch.where(mask, yaw, self._chunk_yaw)
        self._chunk_validity = torch.where(mask, validity, self._chunk_validity)
        self._has_chunk = self._has_chunk | mask
        self._version = self._version + mask.long()

    @torch.no_grad()
    def step(
        self,
        image: Tensor,
        proprioception: Tensor,
        position: Tensor,
        yaw: Tensor,
        clearance: Clearance,
        capture_time: Union[float, Tensor],
        decision_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> StepOutput:
        """One decision of Algorithm 1 (DECIDE) for the elements in ``decision_mask``.

        The networks run for the whole batch (one evaluation each on the chain), but the tracker, the
        filter, the noise state and the chunk buffer are updated only for deciding elements.

        Args:
            image: ``[B, 4, 128, 160]`` normalised RGB-D frame I_t.
            proprioception: ``[B, 13]`` S_t; its last four entries are a_bar_t (see ``proprioception``).
            position: ``[B, 3]`` p_hat_t (NED, m).
            yaw: ``[B]`` psi_hat_t (rad).
            clearance: ``[B]`` d_t, or a callable of the projected chunk returning d_t or (d_t, blind).
            capture_time: tau_t (s), scalar or ``[B]``.
            decision_mask: optional ``[B]`` bool (default: all elements decide).
            generator: optional RNG for the fresh noise draws.

        Returns:
            ``StepOutput``.
        """
        self._check_calibrated()
        c_text = self._require_episode()
        b, device = c_text.shape[0], c_text.device
        if image.dim() != 4 or image.shape[0] != b:
            raise ValueError(f"expected frames [{b}, C, H, W], got {tuple(image.shape)}")
        if tuple(proprioception.shape) != (b, self.selector.proprio_dim):
            raise ValueError(f"expected S_t of shape [{b}, {self.selector.proprio_dim}], got {tuple(proprioception.shape)}")
        if tuple(position.shape) != (b, 3) or tuple(yaw.shape) != (b,):
            raise ValueError(f"expected position [{b}, 3] and yaw [{b}]")
        mask = (torch.ones(b, dtype=torch.bool, device=device) if decision_mask is None
                else decision_mask.to(device=device, dtype=torch.bool))
        if tuple(mask.shape) != (b,):
            raise ValueError(f"expected decision_mask of shape [{b}]")
        image = image.to(device=device, dtype=torch.float32)
        s_t = proprioception.to(device=device, dtype=torch.float32)
        yaw = yaw.to(device=device, dtype=torch.float32)
        a_bar = s_t[:, -self.solver.action_dim:]                                    # a_bar_t, last entries of S_t
        capture = self._relative_time(capture_time, b)

        e_t: Optional[Tensor] = None
        xi: Optional[Tensor] = None
        chunk_raw: Optional[Tensor] = None
        self._chain_log = []
        self._recording = True
        try:
            z_t = self.vision_encoder(image)                                         # E_psi          [B, d]
            waypoint = self.tracker.step(position.to(device), yaw, update_mask=mask)  # Eq. 14, Eq. 3 [B, 4]
            u_sel = self.selector.select(z_t, waypoint.goal, s_t, c_text)            # pi_phi^h       [B]
            prediction = self.predictor(z_t, u_sel, a_bar)                           # P_omega        [B, d], [B]
            assessment = self.safety.assess(prediction.sigma, u_sel, mask=mask)       # Eqs. 23-25 + override
            fallback = assessment.fallback & mask
            solve = mask & ~assessment.fallback
            self.noise.skip(assessment.primitive, mask=fallback)                     # fallback decisions: no chunk
            if bool(solve.any()):
                xi = self.noise.step(assessment.primitive, generator=generator, mask=solve)  # Eq. 20
                e_t = self.solver.build_conditioning(assessment.primitive, z_t, prediction.z_hat, s_t,
                                                     waypoint.goal, c_text)          # Eq. 15        [B, 1073]
                chunk_raw = self.solver.generate_chunk(e_t, noise=xi)                # v_theta x K   [B, H, 4]
        finally:
            self._recording = False
        expected = list(self.CHAIN_NAMES[:3]) + ([self.CHAIN_NAMES[3]] * self.solver.num_integration_steps
                                                 if chunk_raw is not None else [])
        if self._chain_log != expected:
            raise RuntimeError(f"sequential-depth violation: evaluated {self._chain_log}, expected {expected}")

        v_safe = scale = reaction_time = committed_chunk = None
        if chunk_raw is not None:
            depth, blind = self._resolve_clearance(clearance, chunk_raw)
            limited = self.safety.limit(chunk_raw, assessment.alpha, depth, assessment.horizon, blind)  # Eqs. 27, 28
            committed_chunk = self._rate_limit(limited.chunk)                                          # Eq. 34
            v_safe, scale, reaction_time = limited.v_safe, limited.scale, limited.reaction_time
            self._commit(committed_chunk, capture, yaw, assessment.horizon, solve)
        self._fallback_requested = torch.where(fallback, torch.ones_like(fallback),
                                               torch.where(solve, torch.zeros_like(solve), self._fallback_requested))
        return StepOutput(
            z_t=z_t, goal=waypoint.goal, waypoint_index=waypoint.m_t, task_complete=waypoint.task_complete,
            selected_primitive=u_sel, primitive=assessment.primitive, z_hat=prediction.z_hat,
            sigma=prediction.sigma, sigma_bar=assessment.sigma_bar, alpha=assessment.alpha,
            validity_horizon=assessment.horizon, fallback=fallback, conditioning=e_t, noise=xi,
            chunk_raw=chunk_raw, chunk=committed_chunk, v_safe=v_safe, scale=scale, reaction_time=reaction_time,
            committed=solve if chunk_raw is not None else torch.zeros_like(solve), decision_mask=mask,
            num_sequential_evaluations=len(self._chain_log),
        )

    # ------------------------------------------------------------------ executor (TICK)
    def _evaluate(self, knots: Tensor, age: Tensor, yaw: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Eq. 33 at chunk-local time ``age`` and rotation to NED (Eq. 32).

        Between knots k and k+1 the reference is linear and its acceleration is the constant slope;
        from the last knot on, the last knot is held with zero acceleration.

        Returns:
            NED velocity ``[B, 3]``, NED acceleration ``[B, 3]``, yaw rate ``[B]``.
        """
        dt = self.safety.knot_dt
        b, h = knots.shape[0], knots.shape[1]
        rows = torch.arange(b, device=knots.device)
        position = (age / dt).clamp(min=0.0)
        if h == 1:
            value, slope = knots[:, 0], torch.zeros_like(knots[:, 0])
        else:
            k = position.floor().long().clamp(max=h - 2)
            frac = (position - k.to(position.dtype)).clamp(0.0, 1.0).unsqueeze(-1)
            lower, upper = knots[rows, k], knots[rows, k + 1]
            beyond = (position >= h - 1).unsqueeze(-1)
            value = torch.where(beyond, knots[:, -1], lower + frac * (upper - lower))
            slope = torch.where(beyond, torch.zeros_like(lower), (upper - lower) / dt)
        return heading_to_world(value[:, :3], yaw), heading_to_world(slope[:, :3], yaw), value[:, 3]

    @torch.no_grad()
    def tick(self, now: float, odometry_valid: Optional[Tensor] = None) -> ActOutput:
        """One executor tick of Algorithm 1 (TICK) at clock time ``now`` (s).

        Args:
            now: clock time (s), same clock as the capture times.
            odometry_valid: optional ``[B]`` bool; False engages the fallback (stale odometry).

        Returns:
            ``ActOutput`` with ``replanned`` all False and no step output.
        """
        self._require_episode()
        b = self._chunk.shape[0]
        dt, h = self.safety.knot_dt, self._chunk.shape[1]
        t = self._relative_time(float(now), b)                                         # [B]
        age = t - self._chunk_capture
        valid = (self._has_chunk & ~self._fallback_requested
                 & (age <= self._chunk_validity.to(age.dtype) * dt))                  # deadline: age <= H_C Delta
        if odometry_valid is not None:
            if tuple(odometry_valid.shape) != (b,):
                raise ValueError(f"expected odometry_valid of shape [{b}]")
            valid = valid & odometry_valid.to(device=valid.device, dtype=torch.bool)

        # a newly published chunk starts the Eq. 35 cross-fade
        switched = self._version != self._seen_version
        from_hold = (self._last_mode != MODE_TRACK) | ~self._has_prev
        self._switch_time = torch.where(switched, t, self._switch_time)
        self._blend_from_hold = torch.where(switched, from_hold, self._blend_from_hold)
        self._hold_setpoint = torch.where(switched.unsqueeze(-1), self._last_setpoint, self._hold_setpoint)
        self._seen_version = self._version.clone()

        # tracking reference: Eq. 33 on the newest chunk, Eq. 35 against the previous reference
        v_new, a_new, r_new = self._evaluate(self._chunk, age, self._chunk_yaw)
        age_prev = (t - self._prev_capture).clamp(max=(h - 1) * dt)                   # an expired chunk holds its last knot
        v_prev, a_prev, r_prev = self._evaluate(self._prev_chunk, age_prev, self._prev_yaw)
        hold = self._blend_from_hold.unsqueeze(-1)
        v_old = torch.where(hold, self._hold_setpoint[:, :3], v_prev)
        a_old = torch.where(hold, torch.zeros_like(a_prev), a_prev)
        r_old = torch.where(self._blend_from_hold, self._hold_setpoint[:, 3], r_prev)
        if self.blend_time > 0.0:
            lam = ((t - self._switch_time) / self.blend_time).clamp(0.0, 1.0)
            ramp = torch.where(lam < 1.0, torch.full_like(lam, 1.0 / self.blend_time), torch.zeros_like(lam))
        else:
            lam, ramp = torch.ones_like(t), torch.zeros_like(t)
        lam3, ramp3 = lam.unsqueeze(-1), ramp.unsqueeze(-1)
        v_track = lam3 * v_new + (1.0 - lam3) * v_old
        a_track = lam3 * a_new + (1.0 - lam3) * a_old + ramp3 * (v_new - v_old)
        r_track = lam * r_new + (1.0 - lam) * r_old

        # fallback: minimum-jerk stop from the last dispatched velocity (Eq. 26), then hold
        entering = ~valid & ~self._braking
        self._brake_start = torch.where(entering, t, self._brake_start)
        self._brake_v0 = torch.where(entering.unsqueeze(-1), self._last_setpoint[:, :3], self._brake_v0)
        elapsed = (t - self._brake_start).clamp(min=0.0)
        v_brk, a_brk = SafetyFilter.braking_profile(self._brake_v0, elapsed.unsqueeze(1), self.safety.max_decel)
        stop_time = SafetyFilter.braking_time(self._brake_v0, self.safety.max_decel)
        fallback_mode = torch.where(elapsed < stop_time, torch.full_like(self._last_mode, MODE_BRAKE),
                                    torch.full_like(self._last_mode, MODE_HOLD))
        self._braking = ~valid

        valid3 = valid.unsqueeze(-1)
        velocity = torch.where(valid3, v_track, v_brk[:, 0])
        acceleration = torch.where(valid3, a_track, a_brk[:, 0])
        yaw_rate = torch.where(valid, r_track, torch.zeros_like(r_track))
        mode = torch.where(valid, torch.full_like(fallback_mode, MODE_TRACK), fallback_mode)
        self._last_setpoint = torch.cat([velocity, yaw_rate.unsqueeze(-1)], dim=-1)
        self._last_mode = mode
        chunk_age = torch.where(self._has_chunk, age, torch.full_like(age, float("inf")))
        knot_index = torch.where(self._has_chunk, (age.clamp(min=0.0) / dt).floor().long(),
                                 torch.full_like(self._version, -1))
        return ActOutput(velocity, acceleration, yaw_rate, mode, chunk_age, knot_index,
                         torch.zeros(b, dtype=torch.bool, device=velocity.device), None)

    @torch.no_grad()
    def act(
        self,
        now: float,
        observation: Optional[Observation] = None,
        odometry_valid: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> ActOutput:
        """Both loops in one thread: an optional decision on a new observation, then one executor tick.

        "async": every observation triggers a decision for all elements. "sync" (ablation A8): a decision
        runs only for elements whose cursor floor((now - tau_C) / Delta) has reached H_t, that have no
        chunk, or that have a pending fallback. Without an observation no decision runs, and the executor
        brakes once the chunk becomes stale (age > H_t Delta).
        """
        c_text = self._require_episode()
        b = c_text.shape[0]
        replanned = torch.zeros(b, dtype=torch.bool, device=c_text.device)
        step_output: Optional[StepOutput] = None
        if observation is not None:
            if self.execution_mode == "async":
                need = torch.ones_like(replanned)
            else:
                t = self._relative_time(float(now), b)
                cursor = ((t - self._chunk_capture).clamp(min=0.0) / self.safety.knot_dt).floor().long()
                need = ~self._has_chunk | self._fallback_requested | (cursor >= self._chunk_validity)
            if bool(need.any()):
                step_output = self.step(observation.image, observation.proprioception, observation.position,
                                        observation.yaw, observation.clearance, observation.capture_time,
                                        decision_mask=need, generator=generator)
                replanned = need
        out = self.tick(now, odometry_valid)
        return out._replace(replanned=replanned, step_output=step_output)

    @torch.no_grad()
    def external_setpoint(self, velocity_ned: Tensor, yaw_rate: Tensor) -> None:
        """Record a setpoint that another controller dispatched instead of the executor (an expert takeover
        while collecting DAgger data, Sec. 3.8). It becomes a_bar of the next decision (``proprioception``) and
        the hold point from which the executor cross-fades (Eq. 35) when it tracks a chunk again; the executor
        is marked as holding, not braking. Call it for every dispatched setpoint and do not ``tick`` meanwhile.

        Args:
            velocity_ned: ``[B, 3]`` NED velocity setpoint (m/s).
            yaw_rate: ``[B]`` yaw-rate setpoint (rad/s).
        """
        self._require_episode()
        b, device = self._last_setpoint.shape[0], self._last_setpoint.device
        v = torch.as_tensor(velocity_ned, dtype=torch.float32, device=device)
        r = torch.as_tensor(yaw_rate, dtype=torch.float32, device=device)
        if tuple(v.shape) != (b, 3) or tuple(r.shape) != (b,):
            raise ValueError(f"expected velocity_ned [{b}, 3] and yaw_rate [{b}], got {tuple(v.shape)}, {tuple(r.shape)}")
        self._last_setpoint = torch.cat([v, r.unsqueeze(-1)], dim=-1)
        self._last_mode = torch.full_like(self._last_mode, MODE_HOLD)
        self._braking = torch.zeros_like(self._braking)

    # ------------------------------------------------------------------ state accessors
    @property
    def waypoint_index(self) -> Optional[Tensor]:
        """Active waypoint m_t ``[B]`` (1-based)."""
        return self.tracker.m

    @property
    def task_complete(self) -> Optional[Tensor]:
        """``[B]`` bool: the final waypoint has been reached (sticky)."""
        return self.tracker.task_complete

    @property
    def fallback_requested(self) -> Optional[Tensor]:
        """``[B]`` bool: the latest decision requested the fallback and no chunk has replaced it."""
        return None if self.c_text is None else self._fallback_requested


# ---------------------------------------------------------------------------------------------
# Self-test (run: python pipeline.py)
# ---------------------------------------------------------------------------------------------
_RTOL, _ATOL = 1e-5, 1e-6  # float32 module output vs float64 reference (CLAUDE.md)


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


def _ref_interpolate(knots: Tensor, age: float, dt: float, yaw: float) -> Tuple[Tensor, Tensor, float]:
    """Float64 reference of Eq. 33 + Eq. 32 for one element: ``knots [H, 4]`` -> (v^W [3], a^W [3], yaw rate)."""
    k64 = knots.double()
    h = k64.shape[0]
    pos = max(age / dt, 0.0)
    if pos >= h - 1:
        value, slope = k64[-1], torch.zeros(4, dtype=torch.float64)
    else:
        k = int(math.floor(pos))
        frac = pos - k
        value = k64[k] + frac * (k64[k + 1] - k64[k])
        slope = (k64[k + 1] - k64[k]) / dt
    c, s = math.cos(yaw), math.sin(yaw)
    rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return rz @ value[:3], rz @ slope[:3], float(value[3])


def _ref_goal_feature(waypoint: Tensor, p_hat: Tensor, yaw: float, d_max: float, eps: float) -> Tensor:
    """Float64 reference of Eq. 3 for one element with an explicit R_z(psi): ``[4]``."""
    c, s = math.cos(yaw), math.sin(yaw)
    rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    r = rz.T @ (waypoint.double() - p_hat.double())
    n = float(r.norm())
    return torch.cat([r / max(n, eps), torch.tensor([min(n, d_max) / d_max], dtype=torch.float64)])


def _test_pipeline(execution_mode: str = "async", **config_overrides: Any) -> VPAInferencePipeline:
    """Full Table 1 pipeline with an offline 512-d text stand-in, calibrated."""
    try:
        from .perception import _ToyTextModel, _toy_tokenizer
    except ImportError:
        from perception import _ToyTextModel, _toy_tokenizer

    # beta = 1e-6 keeps H_t = 15 for the untrained networks, so the executor timings below are fixed.
    settings: Dict[str, Any] = dict(sigma_lo=50.0, sigma_crit=100.0, beta=1e-6)
    settings.update(config_overrides)
    cfg = VPAConfig(**settings)
    text = TextEncoderWrapper(text_model=_ToyTextModel(projection_dim=512), tokenizer=_toy_tokenizer)
    pipe = VPAInferencePipeline.from_config(cfg, text_encoder=text, latency_bound=0.05, tracking_lag=0.03,
                                            execution_mode=execution_mode)
    gen = torch.Generator().manual_seed(17)
    pipe.calibrate(1.0, torch.randn(cfg.latent_dim, generator=gen) * 0.1, torch.zeros(cfg.proprio_dim),
                   torch.ones(cfg.proprio_dim))
    return pipe


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(2610)
    b, h, dt = 2, 16, 0.02
    cfg = VPAConfig()

    def frame(n: int = b) -> Tensor:
        return torch.randn(n, *cfg.frame_shape, generator=gen)

    def state(n: int = b) -> Tensor:
        return torch.randn(n, cfg.proprio_dim, generator=gen) * 0.3

    # ---- Table 1 budget of the assembled pipeline: 18.13M (text tower excluded; null embedding 512 aside).
    pipe = _test_pipeline()
    budget = pipe.parameter_budget()
    assert budget == {"encoder": 11_171_200, "selector": 668_166, "predictor": 2_720_192, "solver": 3_572_420,
                      "total": 18_131_978}, budget
    table1 = {"encoder": 11.17, "selector": 0.67, "predictor": 2.72, "solver": 3.57, "total": 18.13}
    assert all(round(budget[k] / 1e6, 2) == table1[k] for k in table1)
    assert count_parameters(pipe.text_encoder) == 512                                   # only the null embedding
    assert not any(p.requires_grad for p in pipe.vision_encoder.parameters())           # frozen by calibrate
    assert not any(p.requires_grad for p in pipe.predictor.parameters()) and not pipe.training

    # ---- Calibration and lifecycle errors.
    raw = VPAInferencePipeline.from_config(
        VPAConfig(sigma_lo=50.0, sigma_crit=100.0), text_encoder=pipe.text_encoder, latency_bound=0.05,
        tracking_lag=0.03)
    waypoints = torch.tensor([[[0.0, 0.0, -2.0], [10.0, 0.0, -2.0], [10.0, 10.0, -3.0]],
                              [[0.0, 5.0, -1.0], [-8.0, 5.0, -1.0], [-8.0, -5.0, -2.0]]])
    _expect(RuntimeError, lambda: raw.reset("fly", waypoints))                          # not calibrated
    _expect(RuntimeError, lambda: pipe.tick(0.0))                                       # no reset yet
    _expect(ValueError, lambda: VPAInferencePipeline.from_config(VPAConfig(), text_encoder=pipe.text_encoder,
                                                                 latency_bound=0.05, tracking_lag=0.03))
    raw.calibrate(1.0, torch.zeros(256), torch.zeros(13), torch.ones(13))
    raw.predictor.set_latent_scale(2.0)                                                 # gamma* no longer shared
    _expect(RuntimeError, lambda: raw.reset("fly", waypoints))

    # ---- reset: c_text once per mission ([B, 512]; None gets the learned null embedding), m_0 = 1.
    pipe.reset(["fly through the gate", None], waypoints, mission_start=0.0)
    assert pipe.c_text.shape == (b, 512)
    assert torch.equal(pipe.c_text[1], pipe.text_encoder.null_embedding.detach())
    assert pipe.waypoint_index.tolist() == [1, 1] and pipe.task_complete.tolist() == [False, False]
    _expect(ValueError, lambda: pipe.reset("fly", waypoints[..., :2]))
    _expect(ValueError, lambda: pipe.reset(["a", "b", "c"], waypoints))

    # ---- Algorithm 1 DECIDE: K + 3 = 5 evaluations in chain order, and every stage against a manual run.
    log: List[str] = []
    for name, module in (("E_psi", pipe.vision_encoder), ("pi_phi_h", pipe.selector),
                         ("P_omega", pipe.predictor), ("v_theta", pipe.solver.vector_field)):
        module.register_forward_hook(lambda *_args, n=name: log.append(n))
    seen_a_bar: List[Tensor] = []
    pre = pipe.predictor.register_forward_pre_hook(lambda _m, args: seen_a_bar.append(args[2].clone()))
    img, s_t = frame(), state()
    pos = torch.tensor([[3.0, 1.0, -2.0], [1.0, 4.0, -1.0]])
    yaw = torch.tensor([0.3, -1.2])
    seen_chunks: List[Tensor] = []

    def clearance(chunk: Tensor) -> Tensor:                                              # d_t as a function of A_hat_t
        seen_chunks.append(chunk.clone())
        return torch.full((chunk.shape[0],), 30.0)

    log.clear()
    out = pipe.step(img, s_t, pos, yaw, clearance, capture_time=1.0, generator=torch.Generator().manual_seed(5))
    pre.remove()
    assert log == ["E_psi", "pi_phi_h", "P_omega", "v_theta", "v_theta"], log
    assert out.num_sequential_evaluations == 5 == cfg.sequential_depth
    assert out.z_t.shape == (b, 256) and out.goal.shape == (b, 4) and out.z_hat.shape == (b, 256)
    assert out.sigma.shape == (b,) and out.conditioning.shape == (b, 1073)
    assert out.noise.shape == out.chunk_raw.shape == out.chunk.shape == (b, h, 4)
    assert out.validity_horizon.tolist() == [15, 15] and out.committed.tolist() == [True, True]
    assert not bool(out.fallback.any()) and torch.equal(out.primitive, out.selected_primitive)
    assert torch.equal(seen_a_bar[0], s_t[:, 9:13])                                     # a_bar_t = last entries of S_t
    assert len(seen_chunks) == 1 and torch.equal(seen_chunks[0], out.chunk_raw)          # clearance sees A_hat_t
    with torch.no_grad():
        z_ref = pipe.vision_encoder(img)
        u_ref = pipe.selector.select(z_ref, out.goal, s_t, pipe.c_text)
        pred_ref = pipe.predictor(z_ref, u_ref, s_t[:, 9:13])
        e_ref = pipe.solver.build_conditioning(u_ref, z_ref, pred_ref.z_hat, s_t, out.goal, pipe.c_text)
        xi_ref = torch.randn(b, h, 4, generator=torch.Generator().manual_seed(5))       # t = 0: fresh draw
        chunk_ref = pipe.solver.generate_chunk(e_ref, noise=xi_ref)
    assert torch.equal(out.z_t, z_ref) and torch.equal(out.selected_primitive, u_ref)
    assert torch.equal(out.z_hat, pred_ref.z_hat) and torch.equal(out.sigma, pred_ref.sigma)
    assert torch.equal(out.conditioning, e_ref) and torch.equal(out.noise, xi_ref) and torch.equal(out.chunk_raw, chunk_ref)
    goal_ref = torch.stack([_ref_goal_feature(waypoints[i, 0], pos[i], float(yaw[i]), 20.0, 1e-3) for i in range(b)])
    assert _close(out.goal, goal_ref)                                                    # Eq. 3 on w^(1)
    # Eqs. 27, 28, 34: scaled translation, yaw untouched, knots inside the envelope and rate-limited.
    assert torch.equal(out.chunk[..., 3], out.chunk_raw[..., 3])
    peak = out.chunk[..., :3].norm(dim=-1).amax(dim=-1)
    assert bool((peak <= out.v_safe * (1 + 1e-5) + 1e-6).all())
    steps = (out.chunk[:, 1:, :3] - out.chunk[:, :-1, :3]).norm(dim=-1)
    assert bool((steps <= cfg.max_decel * dt * (1 + 1e-5) + 1e-6).all())
    assert torch.equal(out.chunk[:, 0, :3], out.chunk_raw[:, 0, :3] * out.scale.view(-1, 1))
    assert _close(out.reaction_time, torch.full((b,), 0.05 + 15 * dt + 0.03))

    # ---- Executor: Eq. 35 starts from the hold setpoint, Eq. 33 interpolation with consistent acceleration,
    #      Eq. 32 rotation with the capture-time yaw, then the deadline (age > H_t Delta) engages Eq. 26.
    first = pipe.tick(1.0)
    assert first.mode.tolist() == [MODE_TRACK] * b and torch.equal(first.velocity, torch.zeros(b, 3))  # lambda = 0
    half = pipe.tick(1.02)
    with torch.no_grad():
        v_mid = torch.stack([_ref_interpolate(out.chunk[i], 0.02, dt, float(yaw[i]))[0] for i in range(b)])
    assert _close(half.velocity, 0.5 * v_mid, rtol=1e-4, atol=1e-5)                    # lambda = 1/2 from zero
    for now in (1.05, 1.07, 1.11, 1.19, 1.25, 1.29):
        tick = pipe.tick(now)
        age = float(torch.tensor(now, dtype=torch.float32) - torch.tensor(1.0, dtype=torch.float32))
        assert tick.mode.tolist() == [MODE_TRACK] * b and tick.knot_index.tolist() == [math.floor(age / dt)] * b
        for i in range(b):
            v_ref, a_ref, r_ref = _ref_interpolate(out.chunk[i], age, dt, float(yaw[i]))
            assert _close(tick.velocity[i], v_ref, rtol=1e-4, atol=1e-5)
            assert _close(tick.acceleration[i], a_ref, rtol=1e-4, atol=1e-4)
            assert abs(float(tick.yaw_rate[i]) - r_ref) < 1e-5
    last_track = tick
    stale = pipe.tick(1.0 + 15 * dt + 0.005)                                            # just past the deadline
    assert stale.mode.tolist() == [MODE_BRAKE] * b and torch.equal(stale.velocity, last_track.velocity)
    assert torch.equal(stale.yaw_rate, torch.zeros(b))
    later = pipe.tick(1.0 + 15 * dt + 0.105)
    v_brk, a_brk = SafetyFilter.braking_profile(last_track.velocity, torch.full((b, 1), 0.1), cfg.max_decel)
    assert _close(later.velocity, v_brk[:, 0], rtol=1e-4, atol=1e-5) and _close(later.acceleration, a_brk[:, 0], rtol=1e-4, atol=1e-4)
    stop = float(SafetyFilter.braking_time(last_track.velocity, cfg.max_decel).max())
    held = pipe.tick(1.0 + 15 * dt + 0.005 + stop + 0.02)
    assert held.mode.tolist() == [MODE_HOLD] * b and torch.equal(held.velocity, torch.zeros(b, 3))

    # ---- a_bar for the next decision: the dispatched NED setpoint in the current heading frame (Eq. 2).
    s_next = pipe.proprioception(torch.randn(b, 3, generator=gen), torch.tensor([[0.0, 0.0, 1.0]] * b),
                                 torch.zeros(b, 3), yaw)
    assert s_next.shape == (b, 13) and _close(s_next[:, 9:12], world_to_heading(held.velocity, yaw))

    # ---- Eq. 35 between chunks: continuity at the switch, convex mix inside T_bl, new chunk after.
    pipe.reset(["fly through the gate", None], waypoints, mission_start=0.0)
    out_a = pipe.step(frame(), state(), pos, yaw, torch.full((b,), 30.0), capture_time=2.0,
                      generator=torch.Generator().manual_seed(6))
    for now in (2.0, 2.05, 2.10):
        before = pipe.tick(now)
    assert before.mode.tolist() == [MODE_TRACK] * b
    out_b = pipe.step(frame(), state(), pos, yaw, torch.full((b,), 30.0), capture_time=2.10,
                      generator=torch.Generator().manual_seed(7))
    switch32 = torch.tensor(2.12, dtype=torch.float32)                                  # first tick that sees chunk B
    for now in (2.12, 2.14, 2.17):
        tick = pipe.tick(now)
        t32 = torch.tensor(now, dtype=torch.float32)
        lam = min(1.0, max(0.0, float((t32 - switch32) / 0.04)))
        assert (now, lam) != (2.14, 0.0) and (lam == 0.0) == (now == 2.12) and (lam == 1.0) == (now == 2.17)
        age_a = float(t32 - torch.tensor(2.0, dtype=torch.float32))
        age_b = float(t32 - torch.tensor(2.10, dtype=torch.float32))
        for i in range(b):
            va, aa, _ = _ref_interpolate(out_a.chunk[i], age_a, dt, float(yaw[i]))
            vb, ab, _ = _ref_interpolate(out_b.chunk[i], age_b, dt, float(yaw[i]))
            assert _close(tick.velocity[i], lam * vb + (1 - lam) * va, rtol=1e-4, atol=1e-5)
            ramp = (vb - va) / 0.04 if lam < 1.0 else torch.zeros(3, dtype=torch.float64)
            assert _close(tick.acceleration[i], lam * ab + (1 - lam) * aa + ramp, rtol=1e-3, atol=1e-3)

    # ---- Waypoint progression m_t = 1 -> 2 -> 3 and sticky completion (one waypoint per decision at most).
    pipe.reset("inspect", waypoints, mission_start=0.0)
    indices, complete = [], []
    for k, p in enumerate([waypoints[:, 0], waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], waypoints[:, 2] + 30.0]):
        o = pipe.step(frame(), state(), p, yaw, torch.full((b,), 30.0), capture_time=3.0 + 0.04 * k)
        indices.append(o.waypoint_index.tolist())
        complete.append(o.task_complete.tolist())
    assert indices == [[2, 2], [2, 2], [3, 3], [3, 3], [3, 3]], indices                    # 1 -> 2 -> 3, one per decision
    assert complete == [[False, False], [False, False], [False, False], [True, True], [True, True]], complete

    # ---- Dynamic obstacle: an approaching obstacle shrinks d_t, v_safe(d_t) and the commanded speed; at
    #      d_t <= eps_d + r_s translation stops exactly while the yaw rate is kept.
    pipe.reset("hold the corridor", waypoints, mission_start=0.0)
    distances = [20.0, 10.0, 5.0, 2.0, 1.0, 0.7, 0.5]
    peaks, v_limits = [], []
    for k, d in enumerate(distances):
        o = pipe.step(frame(), state(), pos, yaw, torch.full((b,), d), capture_time=4.0 + 0.04 * k)
        assert _close(o.v_safe, pipe.safety.safe_speed(torch.full((b,), d), o.reaction_time))
        peak = o.chunk[..., :3].norm(dim=-1).amax(dim=-1)
        assert bool((peak <= o.v_safe * (1 + 1e-5) + 1e-6).all())
        peaks.append(peak)
        v_limits.append(o.v_safe)
        if d <= cfg.depth_error + cfg.safety_radius + 1e-9:
            assert torch.equal(o.scale, torch.zeros(b)) and torch.equal(o.chunk[..., :3], torch.zeros(b, h, 3))
            assert torch.equal(o.chunk[..., 3], o.chunk_raw[..., 3]) and bool((o.chunk[..., 3] != 0).any())
    fixed_tr = torch.full((len(distances),), 0.4)
    v_curve = pipe.safety.safe_speed(torch.tensor(distances), fixed_tr)
    assert bool((v_curve[1:] <= v_curve[:-1]).all()) and float(v_curve[-1]) == 0.0
    assert all(bool((v_limits[k + 1] <= v_limits[k] + 1e-6).all()) for k in range(len(distances) - 1))

    # ---- Fallback: sigma_bar >= sigma_crit -> BRAKE_HOVER, no chunk, K + 3 -> 3 evaluations, executor brakes;
    #      a later confident decision releases it with a cross-fade from the hold setpoint.
    pipe.reset("hold the corridor", waypoints, mission_start=0.0)
    pipe.step(frame(), state(), pos, yaw, torch.full((b,), 30.0), capture_time=5.0)
    pipe.tick(5.05)
    moving = pipe.tick(5.06)
    base_args = dict(alpha_min=0.5, alpha_sigma=1.0, min_horizon=4, max_horizon=16, beta=1e-6, max_decel=5.0,
                     depth_error=0.2, safety_radius=0.5, blind_speed=1.0, speed_eps=1e-3, knot_dt=dt,
                     latency_bound=0.05, tracking_lag=0.03)
    confident = pipe.safety
    pipe.safety = SafetyFilter(sigma_lo=0.0, sigma_crit=1e-9, **base_args)
    pipe.safety.reset(b)
    log.clear()
    fb = pipe.step(frame(), state(), pos, yaw, torch.full((b,), 30.0), capture_time=5.07)
    assert log == ["E_psi", "pi_phi_h", "P_omega"] and fb.num_sequential_evaluations == 3
    assert fb.fallback.tolist() == [True, True] and fb.primitive.tolist() == [BRAKE_HOVER] * b
    assert fb.chunk is None and fb.conditioning is None and fb.committed.tolist() == [False, False]
    assert pipe.noise.prev_primitive.tolist() == [BRAKE_HOVER] * b and pipe.fallback_requested.tolist() == [True, True]
    braking = pipe.tick(5.07)
    assert braking.mode.tolist() == [MODE_BRAKE if float(moving.velocity[i].norm()) > 0 else MODE_HOLD for i in range(b)]
    assert torch.equal(braking.velocity, moving.velocity)                                # Eq. 26 starts at v_0
    pipe.safety = confident
    release = pipe.step(frame(), state(), pos, yaw, torch.full((b,), 30.0), capture_time=5.10)
    assert release.committed.tolist() == [True, True] and pipe.fallback_requested.tolist() == [False, False]
    resumed = pipe.tick(5.10)
    assert resumed.mode.tolist() == [MODE_TRACK] * b
    assert _close(resumed.velocity, braking.velocity)                                    # lambda = 0: continuous release

    # ---- Mixed batch: one element falls back, the other commits; the solver still runs once (K + 3).
    pipe.reset("hold the corridor", waypoints, mission_start=0.0)
    img_m, s_m = frame(), state()
    with torch.no_grad():
        z_m = pipe.vision_encoder(img_m)
        goal_m = pipe.tracker.step(pos, yaw).goal
        pipe.tracker.reset(waypoints)
        sig = pipe.predictor(z_m, pipe.selector.select(z_m, goal_m, s_m, pipe.c_text), s_m[:, 9:13]).sigma
    lo_i, hi_i = int(sig.argmin()), int(sig.argmax())
    pipe.safety = SafetyFilter(sigma_lo=0.0, sigma_crit=float(sig.mean()), **base_args)
    pipe.safety.reset(b)
    log.clear()
    mixed = pipe.step(img_m, s_m, pos, yaw, torch.full((b,), 30.0), capture_time=6.0)
    assert log == ["E_psi", "pi_phi_h", "P_omega", "v_theta", "v_theta"]
    assert mixed.fallback[hi_i] and not mixed.fallback[lo_i]
    assert mixed.committed[lo_i] and not mixed.committed[hi_i]
    assert int(mixed.primitive[hi_i]) == BRAKE_HOVER
    pipe.safety = confident

    # ---- Odometry loss engages the fallback even with a fresh chunk.
    pipe.reset("hold the corridor", waypoints, mission_start=0.0)
    pipe.step(frame(), state(), pos, yaw, torch.full((b,), 30.0), capture_time=7.0)
    assert pipe.tick(7.05).mode.tolist() == [MODE_TRACK] * b
    lost = pipe.tick(7.06, odometry_valid=torch.tensor([True, False]))
    assert lost.mode.tolist()[0] == MODE_TRACK and lost.mode.tolist()[1] in (MODE_BRAKE, MODE_HOLD)

    # ---- act, async: every observation triggers a decision for all elements; without one, the deadline holds.
    pipe.reset("hold the corridor", waypoints, mission_start=0.0)
    obs = Observation(frame(), state(), pos, yaw, torch.full((b,), 30.0), 8.0)
    res = pipe.act(8.0, obs)
    assert res.replanned.tolist() == [True, True] and res.step_output is not None
    assert res.step_output.num_sequential_evaluations == 5
    res = pipe.act(8.01)
    assert res.replanned.tolist() == [False, False] and res.step_output is None and res.mode.tolist() == [MODE_TRACK] * b
    late = pipe.act(8.0 + 15 * dt + 0.01)                                                 # no new frame: stale chunk
    assert late.mode.tolist() == [MODE_BRAKE] * b and late.replanned.tolist() == [False, False]

    # ---- act, sync (ablation A8): a decision runs only when the cursor reaches H_t (or no chunk / fallback).
    sync = _test_pipeline("sync")
    sync.reset("hold the corridor", waypoints, mission_start=0.0)
    decisions = []
    now = 9.0
    capture = None
    for _ in range(40):
        obs = Observation(frame(), state(), pos, yaw, torch.full((b,), 30.0), now)
        res = sync.act(now, obs)
        if bool(res.replanned.any()):
            decisions.append(round(now, 2))
            capture = now
        else:
            age32 = torch.tensor(now, dtype=torch.float32) - torch.tensor(capture, dtype=torch.float32)
            assert int((age32 / dt).floor()) < 15 and res.step_output is None
        assert res.mode.tolist() == [MODE_TRACK] * b
        now = round(now + 0.01, 2)
    assert decisions[0] == 9.0 and len(decisions) >= 2
    gaps = [round(decisions[i + 1] - decisions[i], 2) for i in range(len(decisions) - 1)]
    assert all(0.29 <= g <= 0.31 for g in gaps), gaps                                     # H_t Delta = 0.30 s

    # ---- External setpoints (an expert takeover): a_bar of the next decision, then a cross-fade from them.
    pipe.reset("hold the corridor", waypoints, mission_start=0.0)
    pipe.step(frame(), state(), pos, yaw, torch.full((b,), 30.0), capture_time=10.0)
    pipe.tick(10.0)
    v_ext, r_ext = torch.tensor([[1.0, -0.5, 0.2], [0.0, 2.0, 0.0]]), torch.tensor([0.3, -0.1])
    pipe.external_setpoint(v_ext, r_ext)
    s_ext = pipe.proprioception(torch.zeros(b, 3), torch.tensor([[0.0, 0.0, 1.0]] * b), torch.zeros(b, 3), yaw)
    assert _close(s_ext[:, 9:12], world_to_heading(v_ext, yaw)) and _close(s_ext[:, 12], r_ext)
    pipe.step(frame(), s_ext, pos, yaw, torch.full((b,), 30.0), capture_time=10.5)
    resumed = pipe.tick(10.5)                                                       # first tick of the new chunk
    assert resumed.mode.tolist() == [MODE_TRACK] * b
    assert _close(resumed.velocity, v_ext) and _close(resumed.yaw_rate, r_ext)        # lambda = 0: the expert's setpoint
    _expect(ValueError, lambda: pipe.external_setpoint(v_ext[:, :2], r_ext))

    # ---- Argument checks.
    pipe.reset("x", waypoints)
    _expect(ValueError, lambda: pipe.step(frame(3), state(), pos, yaw, torch.full((b,), 30.0), 0.0))
    _expect(ValueError, lambda: pipe.step(frame(), state()[:, :12], pos, yaw, torch.full((b,), 30.0), 0.0))
    _expect(ValueError, lambda: pipe.step(frame(), state(), pos[:, :2], yaw, torch.full((b,), 30.0), 0.0))
    _expect(ValueError, lambda: pipe.step(frame(), state(), pos, yaw, torch.full((b + 1,), 30.0), 0.0))
    _expect(ValueError, lambda: pipe.tick(0.1, odometry_valid=torch.ones(b + 1, dtype=torch.bool)))
    _expect(ValueError, lambda: VPAInferencePipeline.from_config(VPAConfig(sigma_lo=0.1, sigma_crit=0.2),
                                                                 text_encoder=pipe.text_encoder, latency_bound=0.05,
                                                                 tracking_lag=0.03, execution_mode="batch"))
    print("pipeline.py self-test passed")


if __name__ == "__main__":
    _self_test()

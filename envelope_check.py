# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
envelope_check.py -- The braking envelope and the blind rule on recorded expert flights, and the policy's
decisions on the same frames without the closed loop (preprint_261026.pdf, Secs. 3.3-3.6).

    python3 envelope_check.py --data 'data_val/*.hdf5' --frames 5000
    python3 envelope_check.py --data 'data_val/*.hdf5' --checkpoint checkpoints/policy_cal_ep.pt \\
        checkpoints/policy_res_cal_ep.pt --json logs/envelope.json

Expert. For every sampled frame, the expert's chunk A*_t with the recorded depth image and attitude (n^B from S_t
and the yaw psi_hat_t) gives d_t and the blind flag exactly as in closed loop (``clearance.DepthRange``: direction
of the first knot, r_s cylinder, field of view), then v_safe(d_t) and s_t = min(1, v_safe / max_k ||v_k||) with
T_r = T_e2e + H Delta + T_trk (alpha = 1, H = 16): how often the deployed envelope would have slowed the expert,
and why (behind the camera, outside the image left/right, above/below, or both).

Policy (--checkpoint). On the same frames, the deployed decision without its own history: z_t = E_psi(I_t), u_t
from the selector on the recorded g_t and S_t, sigma from P_omega (unfiltered, so sigma_bar = sigma), and the CFM
chunk with fresh noise, conditioned on u_t (as deployed) and on the expert's u*_t (as the stage-2 validation).
The same envelope is applied (alpha and H_t from sigma). Against the expert: selector agreement, first-knot
velocity and direction errors, speed ratio, yaw-rate error, lateral and vertical shares, the horizontal angle to
the active waypoint, and the blind and throttling shares. If the policy matches the expert here but not in
flight, the gap is compounding error (its own states), not the networks on expert states.

Frames are drawn uniformly from the complete, successful flights of --data (held-out worlds: data_val).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from clearance import DepthRange, rotation_from_gravity  # noqa: E402
from config import PRIMITIVES, VPAConfig  # noqa: E402
from dataset import expand_paths  # noqa: E402
from solver import SafetyFilter  # noqa: E402

OUTSIDE = ("behind", "horizontal", "vertical", "corner", "invalid")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data_val/*.hdf5"], help="demonstration files or glob patterns")
    p.add_argument("--checkpoint", nargs="*", default=[], help="calibrated policy checkpoints (open-loop comparison)")
    p.add_argument("--frames", type=int, default=5000, help="frames sampled uniformly from all flights")
    p.add_argument("--latency_bound", type=float, default=0.06, help="T_e2e^max (s), as fly_policy.py")
    p.add_argument("--tracking_lag", type=float, default=0.15, help="T_trk (s), as fly_policy.py")
    p.add_argument("--max_invalid_fraction", type=float, default=1.0, help="as fly_policy.py")
    p.add_argument("--moving", type=float, default=0.3, help="first knots slower than this (m/s) get no direction error")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="write every number to this file")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if args.frames < 1:
        p.error("--frames must be >= 1")
    return args


# ---------------------------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------------------------
class Frames(NamedTuple):
    """Sampled frames of one flight."""

    flight: str
    rgb: np.ndarray        # [N, H, W, 3] uint8
    depth: np.ndarray      # [N, H, W] float32 m, 0 = no return
    state: np.ndarray      # [N, 13] S_t
    goal: np.ndarray       # [N, 4] g_t
    actions: np.ndarray    # [N, 16, 4] A*_t (heading frame, physical)
    primitive: np.ndarray  # [N] u*_t
    yaw: np.ndarray        # [N] psi_hat_t


def flights(paths: Sequence[str]) -> List[Tuple[str, str, int]]:
    """(file, episode, frames) of the complete, successful flights."""
    out = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for name in sorted(k for k in f if k.startswith("episode_")):
                ep = f[name]
                if not bool(ep.attrs.get("complete", True)) or not bool(ep.attrs.get("success", True)):
                    continue
                if "privileged/yaw" not in ep:
                    continue
                n = int(ep["timestamps"].shape[0])
                if n:
                    out.append((path, name, n))
    if not out:
        raise SystemExit(f"no complete, successful flights in {list(paths)}")
    return out


def sample_frames(items: List[Tuple[str, str, int]], count: int, seed: int) -> Iterator[Frames]:
    """``count`` frames drawn uniformly without replacement, one ``Frames`` per flight (in file order)."""
    sizes = np.array([n for _, _, n in items])
    total = int(sizes.sum())
    pick = np.sort(np.random.default_rng(seed).choice(total, size=min(count, total), replace=False))
    starts = np.concatenate([[0], np.cumsum(sizes)])
    for i, (path, name, _) in enumerate(items):
        rows = pick[(pick >= starts[i]) & (pick < starts[i + 1])] - starts[i]
        if rows.size == 0:
            continue
        rows = [int(r) for r in rows]
        with h5py.File(path, "r") as f:
            ep = f[name]
            scale = float(f.attrs.get("depth_scale_m", 1e-3))
            images = np.stack([ep["observations/images"][r] for r in rows])
            yield Frames(
                flight=f"{os.path.basename(path)}:{name}",
                rgb=images[..., :3].astype(np.uint8),
                depth=images[..., 3].astype(np.float32) * np.float32(scale),
                state=np.asarray(ep["observations/state"][rows], dtype=np.float32),
                goal=np.asarray(ep["observations/goal"][rows], dtype=np.float32),
                actions=np.asarray(ep["actions"][rows], dtype=np.float32),
                primitive=np.asarray(ep["primitives"][rows], dtype=np.int64),
                yaw=np.asarray(ep["privileged/yaw"][rows], dtype=np.float64),
            )


# ---------------------------------------------------------------------------------------------
# Envelope (Eqs. 27, 28) on recorded frames
# ---------------------------------------------------------------------------------------------
def envelope(dr: DepthRange, safety: SafetyFilter, fr: Frames, chunks: np.ndarray, alpha: torch.Tensor,
             horizon: torch.Tensor) -> Dict[str, np.ndarray]:
    """d_t, blind, reason, direction, v_safe, s_t and the peak chunk speed for ``chunks [N, H, 4]``."""
    readings = [dr.evaluate_detail(fr.depth[i], rotation_from_gravity(fr.state[i, 3:6], float(fr.yaw[i])),
                                   float(fr.yaw[i]), chunks[i]) for i in range(len(chunks))]
    rng = torch.tensor([r.range for r in readings], dtype=torch.float32)
    blind = torch.tensor([r.blind for r in readings], dtype=torch.bool)
    chunk = torch.as_tensor(chunks, dtype=torch.float32)
    v_safe = safety.safe_speed(rng, safety.reaction_time(horizon.cpu()), blind)
    scale = safety.scale_factor(alpha.cpu().float(), v_safe, chunk)
    return {"range": rng.numpy(), "blind": blind.numpy(), "reason": np.array([r.reason for r in readings]),
            "azimuth": np.array([r.azimuth for r in readings]), "elevation": np.array([r.elevation for r in readings]),
            "v_safe": v_safe.numpy(), "scale": scale.numpy(),
            "peak": torch.linalg.vector_norm(chunk[..., :3], dim=-1).amax(dim=-1).numpy()}


def _q(values: np.ndarray, q: float) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else None


def _share(mask: np.ndarray, within: Optional[np.ndarray] = None) -> Optional[float]:
    mask = np.asarray(mask, dtype=bool)
    if within is not None:
        mask = mask[np.asarray(within, dtype=bool)]
    return float(mask.mean()) if mask.size else None


def envelope_summary(env: Dict[str, np.ndarray], labels: np.ndarray, v_blind: float) -> Dict[str, Any]:
    blind, scale, peak = env["blind"], env["scale"], env["peak"]
    reasons = Counter(env["reason"].tolist())
    n = len(blind)
    seen = ~blind & (env["reason"] != "still")
    hit = env["reason"] == "obstacle"
    capped = blind & (peak > v_blind + 1e-6)
    out = {
        "frames": n,
        "reasons": {k: reasons.get(k, 0) / n for k in ("still", "clear", "obstacle", *OUTSIDE)},
        "blind": float(blind.mean()),
        "blind_capped": float(capped.mean()),
        "obstacle_in_region": float(hit.mean()),
        "range_median": _q(env["range"][hit], 0.5), "range_p10": _q(env["range"][hit], 0.1),
        "v_safe_median": _q(env["v_safe"][seen], 0.5), "v_safe_p10": _q(env["v_safe"][seen], 0.1),
        "peak_median": _q(peak, 0.5),
        "throttled": float((scale < 0.999).mean()), "throttled_half": float((scale < 0.5).mean()),
        "scale_median": _q(scale, 0.5), "scale_p10": _q(scale, 0.1),
        "speed_after_median": _q(np.minimum(peak, peak * scale), 0.5),
        "elevation_abs_median": _q(np.abs(env["elevation"]), 0.5),
        "azimuth_abs_median": _q(np.abs(env["azimuth"]), 0.5),
        "by_primitive": {},
    }
    for k, name in enumerate(PRIMITIVES):
        sel = labels == k
        if not sel.any():
            continue
        out["by_primitive"][name] = {"share": float(sel.mean()), "blind": _share(blind, sel), "capped": _share(capped, sel),
                                     "throttled": _share(scale < 0.999, sel), "scale_median": _q(scale[sel], 0.5),
                                     "peak_median": _q(peak[sel], 0.5)}
    return out


# ---------------------------------------------------------------------------------------------
# Policy on recorded frames (no closed loop)
# ---------------------------------------------------------------------------------------------
class OpenLoop:
    """One calibrated checkpoint: the decision of ``pipeline.step`` on a recorded frame, without history."""

    def __init__(self, path: str, args: argparse.Namespace, device: torch.device) -> None:
        from checkpoint import config_from_dict, load_checkpoint
        from pipeline import VPAInferencePipeline

        self.name = os.path.basename(path)
        ckpt = load_checkpoint(path, kind="policy")
        self.checkpoint_id = str(ckpt["checkpoint_id"])
        self.cfg = config_from_dict(ckpt["config"])
        if self.cfg.sigma_lo is None or self.cfg.sigma_crit is None:
            raise SystemExit(f"{path} is not calibrated: run calibrate_sigma.py first")
        self.device = device
        self.pipe = VPAInferencePipeline.from_checkpoint(path, latency_bound=args.latency_bound,
                                                         tracking_lag=args.tracking_lag, text_encoder="auto",
                                                         map_location=device)
        self.pipe.eval()
        self.normalizer = self.cfg.build_rgbd_normalizer().to(device)
        self.c_text = self.pipe.text_encoder(None).to(device=device, dtype=torch.float32)       # [1, d_c]
        self.generator = torch.Generator(device=device).manual_seed(args.seed)
        self.val_metrics = ckpt["provenance"].get("val_metrics", {})

    @torch.no_grad()
    def decide(self, fr: Frames, start: int, stop: int) -> Dict[str, torch.Tensor]:
        p, dev = self.pipe, self.device
        rgb = torch.from_numpy(fr.rgb[start:stop]).permute(0, 3, 1, 2).to(dev)
        image = self.normalizer(rgb, torch.from_numpy(fr.depth[start:stop]).to(dev)).image
        s = torch.from_numpy(fr.state[start:stop]).to(dev)
        g = torch.from_numpy(fr.goal[start:stop]).to(dev)
        u_star = torch.from_numpy(fr.primitive[start:stop]).to(dev)
        c = self.c_text.expand(s.shape[0], -1)
        a_bar = s[:, -p.solver.action_dim:]
        z = p.vision_encoder(image)
        u = p.selector.select(z, g, s, c)
        pred = p.predictor(z, u, a_bar)
        chunk = p.solver.generate_chunk(p.solver.build_conditioning(u, z, pred.z_hat, s, g, c), generator=self.generator)
        pred_star = p.predictor(z, u_star, a_bar)
        chunk_star = p.solver.generate_chunk(p.solver.build_conditioning(u_star, z, pred_star.z_hat, s, g, c),
                                             generator=self.generator)
        sigma = pred.sigma
        return {"u": u, "sigma": sigma, "alpha": p.safety.attenuation(sigma), "horizon": p.safety.validity_horizon(sigma),
                "fallback": sigma >= p.safety.sigma_crit, "chunk": chunk, "chunk_star": chunk_star}


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def chunk_agreement(chunk: np.ndarray, expert: np.ndarray, moving: float) -> Dict[str, Any]:
    """First-knot errors and whole-chunk RMSE of ``chunk`` against the expert's ``[N, H, 4]``."""
    v, ve = chunk[:, 0, :3], expert[:, 0, :3]
    sp, spe = np.linalg.norm(v, axis=1), np.linalg.norm(ve, axis=1)
    both = (sp > moving) & (spe > moving)
    angle = _angle_deg(v[both], ve[both])
    rmse = np.sqrt(np.mean((chunk - expert) ** 2, axis=(0, 1)))
    return {
        "knot0_error_median": _q(np.linalg.norm(v - ve, axis=1), 0.5), "knot0_error_p90": _q(np.linalg.norm(v - ve, axis=1), 0.9),
        "direction_error_median": _q(angle, 0.5), "direction_error_p90": _q(angle, 0.9),
        "direction_error_over_45": float((angle > 45.0).mean()) if angle.size else None, "direction_frames": int(both.sum()),
        "speed_ratio_median": _q(sp[spe > moving] / spe[spe > moving], 0.5),
        "policy_still_when_expert_moves": _share(sp <= moving, spe > moving),
        "yaw_rate_error_median": _q(np.abs(chunk[:, 0, 3] - expert[:, 0, 3]), 0.5),
        "rmse": dict(zip(("vx", "vy", "vz", "yaw_rate"), rmse.tolist(), strict=True)),
    }


def motion_shares(chunk: np.ndarray, goal: np.ndarray, moving: float) -> Dict[str, Any]:
    """Lateral and vertical first knots and the horizontal angle between the first knot and the active waypoint."""
    v = chunk[:, 0, :3]
    h = np.linalg.norm(v[:, :2], axis=1)
    gh = np.linalg.norm(goal[:, :2], axis=1)
    ok = (h > moving) & (gh > 0.1)
    angle = _angle_deg(v[ok, :2], goal[ok, :2])
    return {"lateral": float((np.abs(v[:, 1]) > 0.5).mean()), "vertical": float((np.abs(v[:, 2]) > 0.5).mean()),
            "first_knot_speed_median": _q(np.linalg.norm(v, axis=1), 0.5),
            "goal_angle_median": _q(angle, 0.5), "goal_angle_over_90": float((angle > 90.0).mean()) if angle.size else None}


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100.0 * x:.1f}%"


def _num(x: Optional[float], unit: str = "", digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}{unit}"


def print_report(report: Dict[str, Any]) -> None:
    actors = ["expert", *report["policies"]]
    env = {a: (report["expert"]["envelope"] if a == "expert" else report["policies"][a]["envelope"]) for a in actors}
    width = max(16, *(len(a) + 2 for a in actors))
    head = f"{'':44s}" + "".join(f"{a:>{width}s}" for a in actors)

    def row(label: str, cells: Sequence[str]) -> None:
        print(f"{label:44s}" + "".join(f"{c:>{width}s}" for c in cells))

    cfg = report["settings"]
    print(f"{report['frames']} frames of {report['flights']} flights ({', '.join(report['data'])}); T_r = "
          f"{cfg['latency_bound']:.2f} + H x {cfg['knot_dt']:.2f} + {cfg['tracking_lag']:.2f} s, v_blind "
          f"{cfg['blind_speed']:.1f} m/s, r_s {cfg['safety_radius']:.2f} m, eps_d {cfg['depth_error']:.2f} m, "
          f"a_max {cfg['max_decel']:.1f} m/s^2")
    print("\nenvelope on the first knot of each chunk (closed-loop rule; policy: its deployed chunk)")
    print(head)
    row("blind (speed limited to v_blind)", [_pct(env[a]["blind"]) for a in actors])
    for reason in OUTSIDE:
        if any(env[a]["reasons"][reason] > 0 for a in actors):
            row(f"  {reason}", [_pct(env[a]["reasons"][reason]) for a in actors])
    row("  ... and faster than v_blind (capped)", [_pct(env[a]["blind_capped"]) for a in actors])
    row("no direction (chunk below 0.1 m/s)", [_pct(env[a]["reasons"]["still"]) for a in actors])
    row("obstacle in the swept region", [_pct(env[a]["obstacle_in_region"]) for a in actors])
    row("  d_t median / p10 (m)", [f"{_num(env[a]['range_median'], '', 1)} / {_num(env[a]['range_p10'], '', 1)}"
                                   for a in actors])
    row("v_safe median / p10, in view (m/s)", [f"{_num(env[a]['v_safe_median'], '', 1)} / {_num(env[a]['v_safe_p10'], '', 1)}"
                                               for a in actors])
    row("peak chunk speed median (m/s)", [_num(env[a]["peak_median"]) for a in actors])
    row("throttled (s_t < 1) / below half", [f"{_pct(env[a]['throttled'])} / {_pct(env[a]['throttled_half'])}" for a in actors])
    row("s_t median / p10", [f"{_num(env[a]['scale_median'])} / {_num(env[a]['scale_p10'])}" for a in actors])
    row("|azimuth| / |elevation| of u_t median (deg)", [f"{_num(env[a]['azimuth_abs_median'], '', 0)} / "
                                                         f"{_num(env[a]['elevation_abs_median'], '', 0)}" for a in actors])
    motion = {a: (report["expert"]["motion"] if a == "expert" else report["policies"][a]["motion"]) for a in actors}
    print("\nfirst knot")
    print(head)
    row("speed median (m/s)", [_num(motion[a]["first_knot_speed_median"]) for a in actors])
    row("lateral (|v_y| > 0.5 m/s)", [_pct(motion[a]["lateral"]) for a in actors])
    row("vertical (|v_z| > 0.5 m/s)", [_pct(motion[a]["vertical"]) for a in actors])
    row("angle to the active waypoint, median (deg)", [_num(motion[a]["goal_angle_median"], "", 0) for a in actors])
    row("  away from it (> 90 deg)", [_pct(motion[a]["goal_angle_over_90"]) for a in actors])
    if report["policies"]:
        pols = list(report["policies"])
        pw = max(16, *(len(a) + 2 for a in pols))
        print("\npolicy against the expert on the same frames")
        print(f"{'':44s}" + "".join(f"{a:>{pw}s}" for a in pols))

        def prow(label: str, fn: Any) -> None:
            print(f"{label:44s}" + "".join(f"{fn(report['policies'][a]):>{pw}s}" for a in pols))

        prow("selector agrees with u*_t", lambda r: _pct(r["selector_accuracy"]))
        prow("sigma median / p95", lambda r: f"{_num(r['sigma_median'], '', 3)} / {_num(r['sigma_p95'], '', 3)}")
        prow("sigma >= sigma_crit (fallback) / > sigma_lo", lambda r: f"{_pct(r['fallback'])} / {_pct(r['above_sigma_lo'])}")
        for key, label in (("deployed", "with the selected u_t"), ("teacher", "with the expert's u*_t")):
            print(f"  {label}:")
            prow("    |v0 - v0*| median / p90 (m/s)",
                 lambda r, k=key: f"{_num(r[k]['knot0_error_median'])} / {_num(r[k]['knot0_error_p90'])}")
            prow("    direction error median / p90 (deg)",
                 lambda r, k=key: f"{_num(r[k]['direction_error_median'], '', 0)} / {_num(r[k]['direction_error_p90'], '', 0)}")
            prow("    direction error > 45 deg", lambda r, k=key: _pct(r[k]["direction_error_over_45"]))
            prow("    speed ratio median", lambda r, k=key: _num(r[k]["speed_ratio_median"]))
            prow("    still while the expert moves", lambda r, k=key: _pct(r[k]["policy_still_when_expert_moves"]))
            prow("    yaw-rate error median (rad/s)", lambda r, k=key: _num(r[k]["yaw_rate_error_median"]))
            prow("    chunk RMSE vx/vy/vz (m/s)", lambda r, k=key: "/".join(f"{r[k]['rmse'][c]:.2f}" for c in ("vx", "vy", "vz")))
        print("\nselector: rows u*_t (expert), columns u_t (selected), share of the row")
        for a in pols:
            conf = np.asarray(report["policies"][a]["confusion"], dtype=np.float64)
            print(f"  {a}")
            print("    " + f"{'':15s}" + "".join(f"{n[:9]:>10s}" for n in PRIMITIVES) + f"{'frames':>9s}")
            for k, name in enumerate(PRIMITIVES):
                total = conf[k].sum()
                if total == 0:
                    continue
                print("    " + f"{name:15s}" + "".join(f"{100 * conf[k, j] / total:9.0f}%" for j in range(len(PRIMITIVES)))
                      + f"{int(total):9d}")
    print("\nby primitive (expert: its label u*_t; policy: its selected u_t): share, blind, capped, throttled, median s_t")
    for a in actors:
        print(f"  {a}")
        for name, r in env[a]["by_primitive"].items():
            print(f"    {name:15s} {_pct(r['share']):>7s}  blind {_pct(r['blind']):>6s}  capped {_pct(r['capped']):>6s}  "
                  f"throttled {_pct(r['throttled']):>6s}  s_t {_num(r['scale_median'])}  peak {_num(r['peak_median'])} m/s")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    paths = expand_paths(args.data)
    items = flights(paths)
    policies = [OpenLoop(path, args, device) for path in args.checkpoint]
    cfg = policies[0].cfg if policies else VPAConfig()
    safety = SafetyFilter(sigma_lo=0.0, sigma_crit=1.0, alpha_min=cfg.alpha_min, alpha_sigma=cfg.alpha_sigma,
                          min_horizon=cfg.min_horizon, max_horizon=cfg.horizon, beta=cfg.beta, max_decel=cfg.max_decel,
                          depth_error=cfg.depth_error, safety_radius=cfg.safety_radius, blind_speed=cfg.blind_speed,
                          speed_eps=cfg.speed_eps, knot_dt=cfg.knot_dt, latency_bound=args.latency_bound,
                          tracking_lag=args.tracking_lag)
    for pol in policies:
        for name in ("max_decel", "depth_error", "safety_radius", "blind_speed", "knot_dt"):
            if not math.isclose(getattr(pol.pipe.safety, name), getattr(safety, name)):
                print(f"[envelope] note: {pol.name} has {name} = {getattr(pol.pipe.safety, name)} "
                      f"(the expert column uses {getattr(safety, name)})", flush=True)
    dr = DepthRange(safety_radius=cfg.safety_radius, max_invalid_fraction=args.max_invalid_fraction)
    keys = ("range", "blind", "reason", "azimuth", "elevation", "v_safe", "scale", "peak")
    expert_env: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    labels, goals, actions = [], [], []
    pol_env = {p.name: {k: [] for k in keys} for p in policies}
    pol_out: Dict[str, Dict[str, List[np.ndarray]]] = {p.name: {k: [] for k in ("u", "sigma", "fallback", "chunk", "chunk_star")}
                                                       for p in policies}
    n_flights = 0
    for fr in sample_frames(items, args.frames, args.seed):
        n_flights += 1
        n = len(fr.primitive)
        ones = torch.ones(n)
        e = envelope(dr, safety, fr, fr.actions, ones, torch.full((n,), cfg.horizon, dtype=torch.long))
        for k in keys:
            expert_env[k].append(e[k])
        labels.append(fr.primitive)
        goals.append(fr.goal)
        actions.append(fr.actions)
        for pol in policies:
            outs = [pol.decide(fr, s, min(n, s + args.batch_size)) for s in range(0, n, args.batch_size)]
            cat = {k: torch.cat([o[k] for o in outs]).cpu() for k in outs[0]}
            chunk = cat["chunk"].numpy().astype(np.float64)
            pe = envelope(dr, pol.pipe.safety, fr, chunk, cat["alpha"], cat["horizon"])
            for k in keys:
                pol_env[pol.name][k].append(pe[k])
            for k in pol_out[pol.name]:
                pol_out[pol.name][k].append(cat[k].numpy())
        if n_flights % 20 == 0:
            print(f"[envelope] {n_flights} flights done", file=sys.stderr, flush=True)
    labels_all, goals_all, actions_all = np.concatenate(labels), np.concatenate(goals), np.concatenate(actions).astype(np.float64)
    e_all = {k: np.concatenate(v) for k, v in expert_env.items()}
    report: Dict[str, Any] = {
        "data": list(args.data), "frames": int(len(labels_all)), "flights": n_flights,
        "settings": {"latency_bound": args.latency_bound, "tracking_lag": args.tracking_lag, "knot_dt": safety.knot_dt,
                     "horizon": cfg.horizon, "blind_speed": safety.blind_speed, "safety_radius": safety.safety_radius,
                     "depth_error": safety.depth_error, "max_decel": safety.max_decel,
                     "max_invalid_fraction": args.max_invalid_fraction, "moving": args.moving, "seed": args.seed},
        "expert": {"envelope": envelope_summary(e_all, labels_all, safety.blind_speed),
                   "motion": motion_shares(actions_all, goals_all, args.moving)},
        "policies": {},
    }
    for pol in policies:
        o = {k: np.concatenate(v) for k, v in pol_out[pol.name].items()}
        pe_all = {k: np.concatenate(v) for k, v in pol_env[pol.name].items()}
        u = o["u"].astype(np.int64)
        confusion = np.zeros((len(PRIMITIVES), len(PRIMITIVES)), dtype=np.int64)
        np.add.at(confusion, (labels_all, u), 1)
        chunk, chunk_star = o["chunk"].astype(np.float64), o["chunk_star"].astype(np.float64)
        report["policies"][pol.name] = {
            "checkpoint_id": pol.checkpoint_id, "sigma_lo": pol.cfg.sigma_lo, "sigma_crit": pol.cfg.sigma_crit,
            "selector_accuracy": float((u == labels_all).mean()), "confusion": confusion.tolist(),
            "sigma_median": _q(o["sigma"], 0.5), "sigma_p95": _q(o["sigma"], 0.95),
            "fallback": float(o["fallback"].mean()), "above_sigma_lo": float((o["sigma"] > pol.cfg.sigma_lo).mean()),
            "envelope": envelope_summary(pe_all, u, pol.pipe.safety.blind_speed),
            "motion": motion_shares(chunk, goals_all, args.moving),
            "deployed": chunk_agreement(chunk, actions_all, args.moving),
            "teacher": chunk_agreement(chunk_star, actions_all, args.moving),
            "stage2_val_metrics": pol.val_metrics,
        }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    report = run(args)
    print_report(report)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


# ---------------------------------------------------------------------------------------------
# Self-test: dry-run flights and an untrained, calibrated policy (plumbing only)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile

    sys.path.insert(0, os.path.join(HERE, "sim"))
    import expert as E
    from fly_policy import _demo_policy

    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "aero_forest_dense_s0001.hdf5")
        world = E.synthetic_forest(seed=2, half_size=25.0)
        E.dry_run(data, world, E.DryRunConfig(num_episodes=2, seed=3, max_frames=60, spacing=(8.0, 10.0),
                                              num_waypoints=(2, 2), speed_limits=(4.0,)))
        with h5py.File(data, "r+") as f:
            for k in f:
                if k.startswith("episode_"):
                    f[k].attrs["success"] = True
        # the attitude rebuilt from n^B and the yaw is the recorded attitude (the dry run flies level)
        with h5py.File(data, "r") as f:
            st, yaw = f["episode_0000/observations/state"][10], float(f["episode_0000/privileged/yaw"][10])
        rot = rotation_from_gravity(st[3:6], yaw)
        assert np.allclose(rot[2], st[3:6] / np.linalg.norm(st[3:6]), atol=1e-6)
        assert abs(math.atan2(rot[1, 0], rot[0, 0]) - yaw) < 1e-6
        ckpt = os.path.join(tmp, "policy.pt")
        _demo_policy(ckpt)
        out = os.path.join(tmp, "envelope.json")
        assert main(["--data", data, "--frames", "50", "--device", "cpu", "--json", out]) == 0
        expert_only = json.load(open(out, encoding="utf-8"))
        assert expert_only["frames"] == 50 and not expert_only["policies"]
        env = expert_only["expert"]["envelope"]
        assert abs(sum(env["reasons"].values()) - 1.0) < 1e-9 and 0.0 <= env["blind"] <= 1.0
        assert main(["--data", data, "--frames", "40", "--device", "cpu", "--json", out, "--checkpoint", ckpt,
                     "--batch_size", "16"]) == 0
        report = json.load(open(out, encoding="utf-8"))
        r = report["policies"]["policy.pt"]
        assert report["frames"] == 40 and sum(map(sum, r["confusion"])) == 40 and 0.0 <= r["selector_accuracy"] <= 1.0
        assert r["deployed"]["rmse"]["vx"] >= 0.0 and r["teacher"]["knot0_error_median"] is not None
        assert abs(sum(r["envelope"]["reasons"].values()) - 1.0) < 1e-9
        # a policy that copies the expert agrees with it exactly
        perfect = chunk_agreement(np.ones((5, 16, 4)), np.ones((5, 16, 4)), 0.3)
        assert perfect["knot0_error_median"] == 0.0 and perfect["direction_error_median"] == 0.0
        assert perfect["speed_ratio_median"] == 1.0
    print("envelope_check.py self-test passed")


if __name__ == "__main__":
    sys.exit(main())

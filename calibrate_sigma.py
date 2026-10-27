# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
calibrate_sigma.py -- Calibrate the uncertainty thresholds sigma_lo and sigma_crit (preprint_261026.pdf, Sec. 3.5).

On in-distribution validation flights (validation worlds whose seeds are disjoint from the training worlds)
the frozen networks are run frame by frame exactly as the closed loop runs them (Algorithm 1, lines 1-5):

    z_t = E_psi(I_t),   u_t = argmax_u pi_phi^h(u | z_t, g_t, S_t, c_text),
    sigma_{t+1} = P_omega(z_t, Embed(u_t), a_bar_t)  (Eq. 22),
    sigma_bar_{t+1} = max{ sigma_{t+1}, (1 - alpha_sigma) sigma_bar_t + alpha_sigma sigma_{t+1} }   (Eq. 23, from 0)

and the thresholds are set to the empirical quantiles of sigma_bar over all decisions:

    sigma_lo   = 0.9-quantile
    sigma_crit = (1 - p_fa)-quantile,   p_fa = n_fa / (f_dec T_mis)

with n_fa = 0.05 spurious fallbacks per mission of T_mis = 60 s and f_dec the decision rate (default: the
frame rate of the data, since every camera frame is a decision). They are written into the checkpoint's
config, with the full report under provenance["sigma_calibration"].

    python calibrate_sigma.py checkpoints/policy.pt "data_val/*.hdf5"                 # updates the checkpoint
    python calibrate_sigma.py checkpoints/policy.pt "data_val/*.hdf5" --out policy_cal.pt
    python calibrate_sigma.py checkpoints/policy.pt "data_val/*.hdf5" --dry_run       # report only
    python calibrate_sigma.py --self-test

Consecutive sigma_bar are strongly correlated, so the tail quantile needs a large validation set (Sec. 3.5).
The report estimates the effective number of independent decisions from the lag-1 autocorrelation and warns
when fewer than 10 exceedances of sigma_crit are expected: sigma_crit is then not resolved by the data.
"""

from __future__ import annotations

import argparse
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cache_latents import FrameChunks, chunk_items
from checkpoint import build_policy_modules, config_from_dict, load_checkpoint, load_stage1_modules, save_checkpoint
from dataset import expand_paths
from solver import SafetyFilter
from train_stage1 import autocast, data_summary, pick_device
from train_stage2 import InstructionTable, clip_embeddings

__all__ = ["episode_sigmas", "filtered_sigma", "thresholds", "calibrate", "parse_args"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate sigma_lo and sigma_crit on validation flights (Sec. 3.5).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="policy checkpoint (train_stage2.py)")
    p.add_argument("data", nargs="*", help="validation flights: files or glob patterns")
    p.add_argument("--out", default=None, help="output checkpoint (default: update the input in place)")
    p.add_argument("--q_lo", type=float, default=0.9, help="quantile for sigma_lo")
    p.add_argument("--n_fa", type=float, default=0.05, help="tolerated spurious fallbacks per mission")
    p.add_argument("--mission_s", type=float, default=60.0, help="mission duration T_mis (s)")
    p.add_argument("--decision_hz", type=float, default=0.0, help="f_dec (0: the frame rate of the data)")
    p.add_argument("--primitive", choices=["selector", "expert"], default="selector",
                   help="primitive conditioning P_omega: the selector's choice (as deployed) or the expert label")
    p.add_argument("--include_failed", action="store_true")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=min(8, max(0, (os.cpu_count() or 2) - 2)))
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="print the report, do not write a checkpoint")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if not args.self_test and (not args.checkpoint or not args.data):
        p.error("give the policy checkpoint and validation files (or --self-test)")
    return args


def _success_filter(include_failed: bool) -> Any:
    return None if include_failed else (lambda _p, _n, attrs: bool(attrs.get("success", True)))


@torch.no_grad()
def episode_sigmas(ckpt: Dict[str, Any], paths: Sequence[str], args: argparse.Namespace
                   ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[float]]:
    """Per episode (in time order): sigma_{t+1} and the primitive used; plus all frame intervals (s)."""
    device = pick_device(args.device)
    cfg, encoder, predictor, _, _ = load_stage1_modules(ckpt, device)
    selector, _ = build_policy_modules(cfg)
    selector.load_state_dict(ckpt["state_dict"]["selector"])
    selector.to(device).eval()
    normalizer = cfg.build_rgbd_normalizer().to(device)
    episode_filter = _success_filter(args.include_failed)
    items = chunk_items(paths, args.chunk, episode_filter)
    instructions: Dict[str, str] = {}
    dts: List[float] = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for name in f:
                if name.startswith("episode_") and "timestamps" in f[name]:
                    text = f[name].attrs.get("instruction", "")
                    instructions[f"{path}:{name}"] = (text.decode() if isinstance(text, bytes) else str(text)).strip()
                    dts += np.diff(f[name]["timestamps"][...]).tolist()
    distinct = sorted({t for t in instructions.values() if t})
    frozen = clip_embeddings(cfg.clip_model_name, distinct) if distinct else None
    table = InstructionTable(cfg.text_dim, frozen).to(device)
    with torch.no_grad():
        table.null_embedding.copy_(ckpt["state_dict"]["text_null_embedding"])
    ids = {t: k + 1 for k, t in enumerate(distinct)}
    sigmas: Dict[str, List[np.ndarray]] = {}
    prims: Dict[str, List[np.ndarray]] = {}
    loader = DataLoader(FrameChunks(items, cfg.action_dim), batch_size=None, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        prefetch_factor=2 if args.num_workers > 0 else None)
    for chunk in loader:
        key = f"{chunk['path']}:{chunk['episode']}"
        n = int(chunk["stop"]) - int(chunk["start"])
        image = normalizer(chunk["rgb"].to(device, non_blocking=True), chunk["depth"].to(device, non_blocking=True)).image
        state, goal = chunk["state"].to(device), chunk["goal"].to(device)
        c_text = table(torch.full((n,), ids.get(instructions.get(key, ""), 0), dtype=torch.long, device=device))
        with autocast(device, args.amp):
            z = encoder(image)
            u = selector.select(z, goal, state, c_text) if args.primitive == "selector" else chunk["primitive"].to(device)
            sigma = predictor(z, u, chunk["a_bar"].to(device)).sigma
        sigmas.setdefault(key, []).append(sigma.float().cpu().numpy())
        prims.setdefault(key, []).append(u.cpu().numpy())
    return ({k: np.concatenate(v) for k, v in sigmas.items()}, {k: np.concatenate(v) for k, v in prims.items()}, dts)


def filtered_sigma(sigmas: Sequence[np.ndarray], alpha_sigma: float) -> List[np.ndarray]:
    """Eq. 23 per episode, with the ``SafetyFilter`` of the closed loop (all episodes stepped together)."""
    if not sigmas:
        return []
    lengths = np.array([len(s) for s in sigmas])
    padded = np.zeros((len(sigmas), int(lengths.max())), dtype=np.float32)
    for i, s in enumerate(sigmas):
        padded[i, :len(s)] = s
    filt = SafetyFilter(sigma_lo=0.0, sigma_crit=1.0, alpha_min=0.5, alpha_sigma=alpha_sigma, min_horizon=1,
                        max_horizon=1, beta=1.0, max_decel=1.0, depth_error=0.0, safety_radius=0.0, blind_speed=0.0,
                        speed_eps=1e-3, knot_dt=0.02, latency_bound=0.0, tracking_lag=0.0)   # only Eq. 23 is used
    filt.reset(len(sigmas))
    sigma_t = torch.from_numpy(padded)
    out = torch.zeros_like(sigma_t)
    for t in range(sigma_t.shape[1]):
        out[:, t] = filt.filter(sigma_t[:, t], mask=torch.from_numpy(lengths > t))
    return [out[i, :n].numpy() for i, n in enumerate(lengths)]


def thresholds(sigma_bar: Sequence[np.ndarray], q_lo: float, p_fa: float) -> Dict[str, Any]:
    """Quantile thresholds and the sample-size diagnostics of Sec. 3.5."""
    allv = np.concatenate(sigma_bar).astype(np.float64)
    pairs_x = np.concatenate([s[:-1] for s in sigma_bar if len(s) > 1]).astype(np.float64)
    pairs_y = np.concatenate([s[1:] for s in sigma_bar if len(s) > 1]).astype(np.float64)
    rho = float(np.corrcoef(pairs_x, pairs_y)[0, 1]) if pairs_x.size > 2 and pairs_x.std() > 0 and pairs_y.std() > 0 else 0.0
    rho = min(max(rho, 0.0), 0.999999)
    n = int(allv.size)
    n_eff = n * (1.0 - rho) / (1.0 + rho)
    sigma_lo = float(np.quantile(allv, q_lo))
    sigma_crit = float(np.quantile(allv, 1.0 - p_fa))
    return {
        "sigma_lo": sigma_lo, "sigma_crit": sigma_crit, "q_lo": q_lo, "p_fa": p_fa, "decisions": n,
        "lag1_autocorrelation": rho, "effective_decisions": n_eff, "expected_exceedances": n_eff * p_fa,
        "resolved": bool(n_eff * p_fa >= 10.0),
        "sigma_bar_mean": float(allv.mean()), "sigma_bar_median": float(np.median(allv)),
        "sigma_bar_p99": float(np.quantile(allv, 0.99)), "sigma_bar_max": float(allv.max()),
        "fraction_above_lo": float((allv >= sigma_lo).mean()), "fraction_above_crit": float((allv >= sigma_crit).mean()),
        "episodes_reaching_crit": int(sum(bool((s >= sigma_crit).any()) for s in sigma_bar)),
        "episodes": len(sigma_bar),
    }


def calibrate(args: argparse.Namespace) -> Dict[str, Any]:
    """Compute the thresholds; write them unless --dry_run. Returns the report."""
    ckpt = load_checkpoint(args.checkpoint, kind="policy")
    cfg = config_from_dict(ckpt["config"])
    paths = expand_paths(args.data)
    sigmas, prims, dts = episode_sigmas(ckpt, paths, args)
    if not sigmas:
        raise ValueError("no usable validation episodes")
    keys = sorted(sigmas)
    sigma_bar = filtered_sigma([sigmas[k] for k in keys], cfg.alpha_sigma)
    f_dec = args.decision_hz if args.decision_hz > 0 else 1.0 / float(np.median(dts))
    p_fa = args.n_fa / (f_dec * args.mission_s)
    report = thresholds(sigma_bar, args.q_lo, p_fa)
    if not report["sigma_lo"] < report["sigma_crit"]:
        raise ValueError(f"degenerate sigma_bar distribution: sigma_lo {report['sigma_lo']} >= sigma_crit "
                         f"{report['sigma_crit']}; use more (and longer) validation flights")
    counts = np.bincount(np.concatenate([prims[k] for k in keys]), minlength=cfg.num_primitives)
    report.update({"decision_hz": f_dec, "n_fa": args.n_fa, "mission_s": args.mission_s, "alpha_sigma": cfg.alpha_sigma,
                   "primitive_source": args.primitive, "primitive_counts": counts.tolist(),
                   "files": data_summary(paths), "checkpoint_in": ckpt["checkpoint_id"]})
    print(f"[calibrate] {report['episodes']} episodes, {report['decisions']} decisions at {f_dec:.1f} Hz; "
          f"p_fa = {p_fa:.2e}")
    print(f"[calibrate] sigma_bar: median {report['sigma_bar_median']:.4f}, p99 {report['sigma_bar_p99']:.4f}, "
          f"max {report['sigma_bar_max']:.4f}; lag-1 autocorrelation {report['lag1_autocorrelation']:.4f}")
    print(f"[calibrate] sigma_lo = {report['sigma_lo']:.4f} (q {args.q_lo}), sigma_crit = {report['sigma_crit']:.4f} "
          f"(q {1 - p_fa:.6f}); {report['episodes_reaching_crit']} of {report['episodes']} episodes reach sigma_crit")
    if not report["resolved"]:
        need = 10.0 / p_fa * (1 + report["lag1_autocorrelation"]) / max(1e-6, 1 - report["lag1_autocorrelation"])
        print(f"[calibrate] WARNING: only {report['expected_exceedances']:.2f} exceedances of sigma_crit are expected "
              f"among {report['effective_decisions']:.0f} effective decisions: sigma_crit is not resolved by this "
              f"validation set (about {need:.3g} decisions would give 10). Use more validation flights before relying "
              f"on the fallback threshold (Sec. 3.5, ablation A2).")
    if args.dry_run:
        return report
    ckpt["config"]["sigma_lo"] = report["sigma_lo"]
    ckpt["config"]["sigma_crit"] = report["sigma_crit"]
    ckpt["provenance"]["sigma_calibration"] = report
    ckpt["checkpoint_id"] = uuid.uuid4().hex
    out = args.out or args.checkpoint
    save_checkpoint(out, ckpt)
    print(f"[calibrate] wrote {out} (id {ckpt['checkpoint_id'][:8]})")
    report["checkpoint_out"] = ckpt["checkpoint_id"]
    return report


# ---------------------------------------------------------------------------------------------
# Self-test (run: python calibrate_sigma.py --self-test): the whole training chain on dry-run data
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile

    from cache_latents import cache_latents, parse_args as cache_args
    from pipeline import VPAInferencePipeline
    from train_stage1 import make_demo_data, parse_args as stage1_args, train_stage1
    from train_stage2 import parse_args as stage2_args, train_stage2

    with tempfile.TemporaryDirectory() as tmp:
        train_file, val_file = make_demo_data(tmp, {"train.hdf5": 2, "val.hdf5": 2}, frames=30)
        stage1 = os.path.join(tmp, "stage1.pt")
        policy = os.path.join(tmp, "policy.pt")
        latents = os.path.join(tmp, "latents")
        train_stage1(stage1_args([train_file, "--out", stage1, "--batch_size", "8", "--num_workers", "0",
                                  "--device", "cpu", "--val_fraction", "0", "--steps", "2", "--log_every", "100"]))
        cache_latents(cache_args([stage1, train_file, "--out_dir", latents, "--num_workers", "0", "--device", "cpu"]))
        train_stage2(stage2_args([stage1, train_file, "--latents", latents, "--out", policy, "--val_fraction", "0",
                                  "--batch_size", "16", "--device", "cpu", "--steps", "2", "--log_every", "100"]))
        before = load_checkpoint(policy)
        args = parse_args([policy, val_file, "--num_workers", "1", "--chunk", "7", "--device", "cpu"])
        report = calibrate(args)
        after = load_checkpoint(policy, kind="policy")
        assert after["config"]["sigma_lo"] == report["sigma_lo"] < report["sigma_crit"] == after["config"]["sigma_crit"]
        assert after["checkpoint_id"] != before["checkpoint_id"]
        assert after["provenance"]["sigma_calibration"]["decisions"] == 60 and report["episodes"] == 2
        assert math.isclose(report["decision_hz"], 50.0, rel_tol=1e-3) and not report["resolved"]
        assert math.isclose(report["p_fa"], 0.05 / (50.0 * 60.0), rel_tol=1e-3)

        # independent reference: direct module calls and the Eq. 23 recursion in float64
        cfg, encoder, predictor, _, _ = load_stage1_modules(after)
        selector, _ = build_policy_modules(cfg)
        selector.load_state_dict(after["state_dict"]["selector"])
        selector.eval()
        normalizer = cfg.build_rgbd_normalizer()
        null = after["state_dict"]["text_null_embedding"]
        ref_bar = []
        with h5py.File(val_file, "r") as f, torch.no_grad():
            for name in sorted(k for k in f if k.startswith("episode_")):
                ep = f[name]
                raw = ep["observations/images"][...]
                img = normalizer(torch.from_numpy(raw[..., :3].astype(np.uint8)).permute(0, 3, 1, 2),
                                 torch.from_numpy(raw[..., 3].astype(np.float32) * 1e-3)).image
                state = torch.from_numpy(ep["observations/state"][...])
                goal = torch.from_numpy(ep["observations/goal"][...])
                z = encoder(img)
                u = selector.select(z, goal, state, null.expand(len(z), -1))
                sig = predictor(z, u, state[:, -4:]).sigma.double().numpy()
                bar, prev = np.zeros_like(sig), 0.0
                for t, s in enumerate(sig):
                    prev = max(s, (1 - cfg.alpha_sigma) * prev + cfg.alpha_sigma * s)
                    bar[t] = prev
                ref_bar.append(bar)
        allv = np.concatenate(ref_bar)
        assert np.isclose(report["sigma_lo"], np.quantile(allv, 0.9), rtol=1e-4)
        assert np.isclose(report["sigma_crit"], np.quantile(allv, 1 - report["p_fa"]), rtol=1e-4)

        # the closed loop reproduces the calibration sigma_bar decision by decision (same frames, poses, waypoints)
        pipe = VPAInferencePipeline.from_checkpoint(policy, latency_bound=0.05, tracking_lag=0.03)
        with h5py.File(val_file, "r") as f:
            ep = f["episode_0000"]
            pipe.reset(None, torch.from_numpy(np.asarray(ep.attrs["waypoints_ned"], dtype=np.float32))[None])
            raw = ep["observations/images"][...]
            img = normalizer(torch.from_numpy(raw[..., :3].astype(np.uint8)).permute(0, 3, 1, 2),
                             torch.from_numpy(raw[..., 3].astype(np.float32) * 1e-3)).image
            loop_bar = []
            for t in range(raw.shape[0]):
                out = pipe.step(img[t:t + 1], torch.from_numpy(ep["observations/state"][t:t + 1]),
                                torch.from_numpy(ep["privileged/position_ned"][t:t + 1]),
                                torch.from_numpy(ep["privileged/yaw"][t:t + 1]), torch.tensor([30.0]),
                                float(ep["timestamps"][t]))
                assert np.allclose(out.goal.numpy()[0], ep["observations/goal"][t], atol=1e-5)
                loop_bar.append(float(out.sigma_bar[0]))
        assert np.allclose(loop_bar, ref_bar[0], rtol=1e-4, atol=1e-6)
        dry = calibrate(parse_args([policy, val_file, "--dry_run", "--num_workers", "0", "--device", "cpu",
                                    "--primitive", "expert"]))
        assert load_checkpoint(policy)["checkpoint_id"] == after["checkpoint_id"] and dry["primitive_source"] == "expert"
    print("calibrate_sigma.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        calibrate(parsed)

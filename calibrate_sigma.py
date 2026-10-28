# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
calibrate_sigma.py -- Calibrate the uncertainty thresholds sigma_lo and sigma_crit (preprint_261026.pdf, Sec. 3.5).

On in-distribution validation flights (validation worlds whose seeds are disjoint from the training worlds)
the frozen networks are run frame by frame exactly as the closed loop runs them (Algorithm 1, lines 1-5):

    z_t = E_psi(I_t),   u_t = argmax_u pi_phi^h(u | z_t, g_t, S_t, c_text),
    sigma_{t+1} = P_omega(z_t, Embed(u_t), a_bar_t)  (Eq. 22),
    sigma_bar_{t+1} = max{ sigma_{t+1}, (1 - alpha_sigma) sigma_bar_t + alpha_sigma sigma_{t+1} }   (Eq. 23, from 0)

and sigma_lo is set to the 0.9-quantile of sigma_bar over all decisions. sigma_crit bounds the spurious
fallbacks (sigma_bar >= sigma_crit: brake to hover) at n_fa = 0.05 per in-distribution mission of T_mis = 60 s,
estimated in one of two ways (both are always reported; --method chooses the one written):

    frame        sigma_crit = (1 - p_fa)-quantile of sigma_bar over all decisions,  p_fa = n_fa / (f_dec T_mis)
                 (Sec. 3.5; f_dec is the decision rate, by default the frame rate of the data). This treats the
                 f_dec T_mis decisions of a mission as independent.
    episode_max  sigma_crit from the maximum of sigma_bar over each validation flight. A fallback episode starts
                 when sigma_bar first reaches sigma_crit, so the budget is a statement about the per-flight
                 maximum M_i. With flights of different durations L_i, onsets are modelled as a Poisson process
                 of rate lambda(c): P(M_i >= c) = 1 - exp(-lambda(c) L_i); lambda is the maximum-likelihood
                 estimate from the flights that do and do not reach c, and sigma_crit is the lowest level with
                 P(mission of T_mis falls back) = 1 - exp(-lambda T_mis) <= n_fa, placed midway between
                 consecutive flight maxima. With every L_i = T_mis this is the (1 - n_fa)-quantile of the M_i.
                 A bootstrap over flights gives a confidence interval.

sigma_bar is strongly correlated from one decision to the next (lag-1 autocorrelation near 1 at 50 Hz), so
the frame quantile needs on the order of 10 / p_fa * (1 + rho) / (1 - rho) decisions, far more than any
validation set; the per-flight maximum needs about 10 / n_fa flights (200 for 10 exceedances at n_fa = 0.05).
The thresholds are written into the checkpoint's config, with the full report (both methods) under
provenance["sigma_calibration"].

    python calibrate_sigma.py checkpoints/policy.pt "data_val/*.hdf5"                 # updates the checkpoint
    python calibrate_sigma.py checkpoints/policy.pt "data_val/*.hdf5" --out policy_cal.pt
    python calibrate_sigma.py checkpoints/policy.pt "data_val/*.hdf5" --dry_run       # report only
    python calibrate_sigma.py checkpoints/policy.pt "data_val/*.hdf5" --method episode_max --out policy_cal.pt
    python calibrate_sigma.py --self-test

The report estimates the effective number of independent decisions from the lag-1 autocorrelation and warns
when fewer than 10 exceedances (frame) or fewer than 10 flights above sigma_crit (episode_max) are expected:
sigma_crit is then not resolved by the data.
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

__all__ = ["episode_sigmas", "filtered_sigma", "thresholds", "onset_rate_mle", "episode_max_threshold",
           "mission_fallback_probability", "calibrate", "parse_args"]


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
    p.add_argument("--method", choices=["frame", "episode_max"], default="frame",
                   help="sigma_crit written to the checkpoint: frame quantile or per-flight maximum (both reported)")
    p.add_argument("--bootstrap", type=int, default=1000, help="resamples of flights for the episode_max interval")
    p.add_argument("--seed", type=int, default=0, help="seed of the bootstrap")
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


def onset_rate_mle(reached: np.ndarray, durations: np.ndarray) -> float:
    """Maximum-likelihood onset rate lambda (1/s) from per-flight first-passage outcomes and flight durations.

    Flight i of duration L_i reaches the level with probability 1 - exp(-lambda L_i) (onsets of sigma_bar >= level
    form a Poisson process). The log-likelihood sum_reached log(1 - exp(-lambda L_i)) - lambda sum_others L_i is
    concave; its score sum_reached L_i / (exp(lambda L_i) - 1) - sum_others L_i decreases in lambda and is solved
    by bisection in log lambda. 0 when no flight reaches the level, inf when every flight does.
    """
    reached = np.asarray(reached, dtype=bool)
    durations = np.asarray(durations, dtype=np.float64)
    if reached.ndim != 1 or reached.shape != durations.shape:
        raise ValueError("reached and durations must be 1-D arrays of the same length")
    if not bool(np.all(durations > 0.0)):
        raise ValueError("flight durations must be positive")
    if not reached.any():
        return 0.0
    rest = float(durations[~reached].sum())
    if rest == 0.0:
        return math.inf
    hit = durations[reached]
    lo = math.log(1e-12 / float(hit.max()))          # score > 0 here: about k / lambda
    hi = math.log(50.0 / float(hit.min()))           # score ~ -rest here
    with np.errstate(over="ignore"):
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(np.sum(hit / np.expm1(math.exp(mid) * hit))) > rest:
                lo = mid
            else:
                hi = mid
    return math.exp(0.5 * (lo + hi))


def mission_fallback_probability(rate: float, mission_s: float) -> float:
    """P(an in-distribution mission of T_mis reaches the level) = 1 - exp(-lambda T_mis)."""
    return 1.0 if math.isinf(rate) else -math.expm1(-rate * mission_s)


def _crit_from_maxima(maxima: np.ndarray, durations: np.ndarray, n_fa: float, mission_s: float
                      ) -> Tuple[float, int, float]:
    """(sigma_crit, flights above it, P(mission falls back)): the lowest level, midway between consecutive flight
    maxima, whose estimated mission fallback probability is at most n_fa."""
    n = maxima.size
    order = np.argsort(-maxima, kind="stable")
    desc = maxima[order]

    def prob(k: int) -> float:                       # the k flights with the largest maxima reach the level
        reached = np.zeros(n, dtype=bool)
        reached[order[:k]] = True
        return mission_fallback_probability(onset_rate_mle(reached, durations), mission_s)

    lo, hi = 0, n                                    # prob(0) = 0 <= n_fa < 1 = prob(n); prob is non-decreasing
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if prob(mid) <= n_fa:
            lo = mid
        else:
            hi = mid
    k = lo
    while 0 < k < n and desc[k - 1] == desc[k]:      # a threshold cannot split tied maxima
        k -= 1
    if k == 0:                                       # not even one flight above sigma_crit fits the budget
        return float(np.nextafter(desc[0], np.inf)), 0, 0.0
    return 0.5 * float(desc[k - 1] + desc[k]), k, prob(k)


def episode_max_threshold(sigma_bar: Sequence[np.ndarray], durations: Sequence[float], n_fa: float,
                          mission_s: float, bootstrap: int = 0, seed: int = 0) -> Dict[str, Any]:
    """sigma_crit from the per-flight maxima of sigma_bar (method episode_max), with a bootstrap over flights."""
    if not 0.0 < n_fa < 1.0:
        raise ValueError("n_fa must lie in (0, 1)")
    if not mission_s > 0.0:
        raise ValueError("mission_s must be positive")
    maxima = np.array([float(np.max(s)) for s in sigma_bar], dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if maxima.size == 0 or durations.shape != maxima.shape:
        raise ValueError("need one duration per flight and at least one flight")
    crit, above, prob = _crit_from_maxima(maxima, durations, n_fa, mission_s)
    rate = onset_rate_mle(maxima >= crit, durations)
    target = -math.log1p(-n_fa) / mission_s          # onset rate that spends the budget exactly
    per_flight = -math.expm1(-target * float(durations.mean()))
    out: Dict[str, Any] = {
        "sigma_crit": crit, "flights": int(maxima.size), "flights_above": above, "resolved": bool(above >= 10),
        "mission_fallback_probability": prob, "onset_rate_per_hour": 3600.0 * rate, "n_fa": n_fa,
        "mission_s": mission_s, "flights_for_10_above": 10.0 / per_flight,
        "flight_s_mean": float(durations.mean()), "flight_s_min": float(durations.min()),
        "flight_s_max": float(durations.max()), "flight_max_median": float(np.median(maxima)),
        "flight_max_p95": float(np.quantile(maxima, 0.95)), "flight_max_max": float(maxima.max()),
    }
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = np.empty(bootstrap)
        for b in range(bootstrap):
            idx = rng.integers(0, maxima.size, maxima.size)
            draws[b] = _crit_from_maxima(maxima[idx], durations[idx], n_fa, mission_s)[0]
        out.update(bootstrap=int(bootstrap), ci_low=float(np.quantile(draws, 0.025)),
                   ci_high=float(np.quantile(draws, 0.975)))
    return out


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
    report = thresholds(sigma_bar, args.q_lo, p_fa)                       # sigma_lo and the frame method
    durations = np.array([len(sb) for sb in sigma_bar], dtype=np.float64) / f_dec
    maxima = np.array([float(sb.max()) for sb in sigma_bar])
    frame = {k: report[k] for k in ("sigma_crit", "p_fa", "effective_decisions", "expected_exceedances", "resolved",
                                    "fraction_above_crit", "episodes_reaching_crit")}
    frame["mission_fallback_probability"] = mission_fallback_probability(
        onset_rate_mle(maxima >= frame["sigma_crit"], durations), args.mission_s)
    episode = episode_max_threshold(sigma_bar, durations, args.n_fa, args.mission_s, args.bootstrap, args.seed)
    chosen = frame if args.method == "frame" else episode
    allv = np.concatenate(sigma_bar)
    report.update({"method": args.method, "frame": frame, "episode_max": episode, "sigma_crit": chosen["sigma_crit"],
                   "resolved": chosen["resolved"], "fraction_above_crit": float((allv >= chosen["sigma_crit"]).mean()),
                   "episodes_reaching_crit": int((maxima >= chosen["sigma_crit"]).sum())})
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
    n_flights, mis = report["episodes"], f"P(fallback in a {args.mission_s:g} s mission)"
    print(f"[calibrate] sigma_lo = {report['sigma_lo']:.4f} (q {args.q_lo})")
    print(f"[calibrate] frame:       sigma_crit = {frame['sigma_crit']:.4f} (q {1 - p_fa:.6f}); "
          f"{frame['episodes_reaching_crit']} of {n_flights} flights reach it; {mis} = "
          f"{frame['mission_fallback_probability']:.4f}{'' if frame['resolved'] else ' (not resolved)'}")
    ci = f" [95% CI {episode['ci_low']:.4f}, {episode['ci_high']:.4f}]" if "ci_low" in episode else ""
    print(f"[calibrate] episode_max: sigma_crit = {episode['sigma_crit']:.4f}{ci}; {episode['flights_above']} of "
          f"{n_flights} flights reach it; {mis} = {episode['mission_fallback_probability']:.4f} (n_fa {args.n_fa:g}); "
          f"flights {episode['flight_s_mean']:.1f} s ({episode['flight_s_min']:.1f}-{episode['flight_s_max']:.1f} s)"
          f"{'' if episode['resolved'] else ' (not resolved)'}")
    if args.method == "frame" and not frame["resolved"]:
        need = 10.0 / p_fa * (1 + report["lag1_autocorrelation"]) / max(1e-6, 1 - report["lag1_autocorrelation"])
        print(f"[calibrate] WARNING: only {frame['expected_exceedances']:.2f} exceedances of sigma_crit are expected "
              f"among {frame['effective_decisions']:.0f} effective decisions: sigma_crit is not resolved by this "
              f"validation set (about {need:.3g} decisions would give 10). Use more validation flights, or "
              f"--method episode_max, before relying on the fallback threshold (Sec. 3.5, ablation A2).")
    if args.method == "episode_max" and not episode["resolved"]:
        print(f"[calibrate] WARNING: only {episode['flights_above']} validation flights lie above sigma_crit (fewer "
              f"than 10): sigma_crit is not resolved (about {episode['flights_for_10_above']:.0f} flights of this "
              f"length would give 10).")
    if args.dry_run:
        return report
    ckpt["config"]["sigma_lo"] = report["sigma_lo"]
    ckpt["config"]["sigma_crit"] = report["sigma_crit"]
    ckpt["provenance"]["sigma_calibration"] = report
    ckpt["checkpoint_id"] = uuid.uuid4().hex
    out = args.out or args.checkpoint
    save_checkpoint(out, ckpt)
    print(f"[calibrate] wrote {out} (id {ckpt['checkpoint_id'][:8]}) with sigma_crit from method {args.method}")
    report["checkpoint_out"] = ckpt["checkpoint_id"]
    return report


# ---------------------------------------------------------------------------------------------
# Self-test (run: python calibrate_sigma.py --self-test): the whole training chain on dry-run data
# ---------------------------------------------------------------------------------------------
def _episode_max_unit_tests() -> None:
    rng = np.random.default_rng(0)
    # onset-rate MLE: closed form for equal durations; a stationary point of the concave likelihood otherwise
    reached = np.zeros(50, dtype=bool)
    reached[:10] = True
    assert math.isclose(onset_rate_mle(reached, np.full(50, 30.0)), math.log(50 / 40) / 30.0, rel_tol=1e-9)
    durations = rng.uniform(5.0, 80.0, 40)
    hit = rng.random(40) < 0.3
    lam = onset_rate_mle(hit, durations)
    loglik = lambda r: float(np.sum(np.log(-np.expm1(-r * durations[hit]))) - r * durations[~hit].sum())  # noqa: E731
    assert loglik(lam) >= max(loglik(lam * 0.999), loglik(lam * 1.001))
    assert onset_rate_mle(np.zeros(5, bool), np.ones(5)) == 0.0 and math.isinf(onset_rate_mle(np.ones(5, bool), np.ones(5)))
    assert mission_fallback_probability(math.inf, 60.0) == 1.0 and mission_fallback_probability(0.0, 60.0) == 0.0
    # flights of exactly T_mis: the (1 - n_fa)-quantile of the maxima, midway between order statistics
    maxima = rng.permutation(np.arange(1.0, 201.0))
    out = episode_max_threshold([np.array([0.0, m, 0.5 * m]) for m in maxima], np.full(200, 60.0), 0.0525, 60.0)
    assert out["sigma_crit"] == 190.5 and out["flights_above"] == 10 and out["resolved"]
    assert math.isclose(out["mission_fallback_probability"], 0.05, rel_tol=1e-6)
    # tied maxima are not split; a budget that no single flight fits gives a level above every maximum
    tied_flights = [np.array([v]) for v in [5.0] * 3 + [1.0] * 17]
    tied = episode_max_threshold(tied_flights, np.full(20, 60.0), 0.16, 60.0)        # 3 of 20 allowed: all three
    assert tied["sigma_crit"] == 3.0 and tied["flights_above"] == 3
    tied = episode_max_threshold(tied_flights, np.full(20, 60.0), 0.12, 60.0)        # 2 allowed: cannot split 3 ties
    assert tied["flights_above"] == 0 and tied["sigma_crit"] > 5.0
    few = episode_max_threshold([np.array([v]) for v in (1.0, 2.0, 3.0)], np.full(3, 60.0), 0.05, 60.0)
    assert few["flights_above"] == 0 and few["sigma_crit"] > 3.0 and not few["resolved"]
    # flights shorter than the mission: the Poisson length correction recovers the true level, the plain
    # quantile of the maxima does not. Truth: lambda(c) = exp(-c) per second, so lambda(c*) T_mis = -log(1 - n_fa).
    n, n_fa, t_mis = 20000, 0.05, 60.0
    lengths = rng.uniform(5.0, 30.0, n)
    v = rng.random(n)                                       # P(M >= m) = 1 - exp(-exp(-m) L)  ->  inverse transform
    flight_max = -np.log(-np.log1p(-v) / lengths)
    truth = -math.log(-math.log1p(-n_fa) / t_mis)
    est = episode_max_threshold([np.array([m]) for m in flight_max], lengths, n_fa, t_mis, bootstrap=50, seed=1)
    assert abs(est["sigma_crit"] - truth) < 0.1, (est["sigma_crit"], truth)
    assert abs(float(np.quantile(flight_max, 1 - n_fa)) - truth) > 0.5
    assert est["ci_low"] <= est["sigma_crit"] <= est["ci_high"] and est["ci_high"] - est["ci_low"] < 0.5
    for bad in (dict(n_fa=0.0), dict(n_fa=1.0), dict(mission_s=0.0)):
        kwargs = dict(n_fa=0.05, mission_s=60.0)
        kwargs.update(bad)
        try:
            episode_max_threshold([np.ones(3)], [1.0], **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def _self_test() -> None:
    import tempfile

    _episode_max_unit_tests()

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
        # method episode_max: written to the checkpoint; frame value and sigma_lo unchanged in the report
        assert report["method"] == "frame" and report["frame"]["sigma_crit"] == report["sigma_crit"]
        out_ep = os.path.join(tmp, "policy_ep.pt")
        ep = calibrate(parse_args([policy, val_file, "--method", "episode_max", "--bootstrap", "20", "--out", out_ep,
                                   "--num_workers", "0", "--device", "cpu"]))
        written = load_checkpoint(out_ep, kind="policy")
        assert ep["method"] == "episode_max" and written["config"]["sigma_crit"] == ep["episode_max"]["sigma_crit"]
        assert written["config"]["sigma_lo"] == ep["sigma_lo"] == report["sigma_lo"]
        assert ep["frame"]["sigma_crit"] == report["sigma_crit"] and ep["episode_max"]["flights"] == 2
        maxima = [float(b.max()) for b in ref_bar]
        assert ep["episode_max"]["sigma_crit"] > max(maxima) and not ep["resolved"]       # 2 flights: unresolved
        assert written["provenance"]["sigma_calibration"]["episode_max"]["bootstrap"] == 20
    print("calibrate_sigma.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        calibrate(parsed)

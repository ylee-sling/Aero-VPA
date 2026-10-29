# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
input_sensitivity.py -- Which inputs does the policy's decision depend on? Input swaps on held-out expert frames.

    python3 input_sensitivity.py --data 'data_val/*.hdf5' --checkpoint checkpoints/policy_cal_ep.pt \\
        checkpoints/policy_res_cal_ep.pt --json logs/sensitivity.json

For each calibrated checkpoint, the deployed decision (selector u_t, P_omega, CFM chunk; as envelope_check.py) is
computed on frames sampled from held-out expert flights, then again with one input replaced by the same input of
another frame (one fixed derangement of the frames, so donors mostly come from other flights): the image (z_t),
the goal g_t, the velocity v^H, the attitude n^B, the body rates omega^B, the last dispatched command a_bar_t
(S_t[9:13]; also set to zero, as at the start of a mission), or all of S_t. a_bar_t reaches the decision twice,
through S_t (selector and CFM conditioning) and through P_omega's forecast z_hat (in the CFM conditioning), so it
is also swapped in each path alone. The CFM noise of every frame is the same in every run, so a change comes
from the swapped input alone.

Per swap: the change of the first knot (median and p90, m/s, and of its direction), the share of frames whose
primitive changes, and the first-knot error against the frame's own expert label and against the label of the
frame the input came from. "label spread" is the median difference between the two frames' labels: an input
that alone determines the decision moves the output by about that much when swapped, an ignored input by ~0.
A policy that copies its last command follows the swapped a_bar_t (error to the donor's label near 0) and
hardly moves when the image or the goal is swapped: then the closed loop keeps doing whatever it was doing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import expand_paths  # noqa: E402
from envelope_check import OpenLoop, _angle_deg, _q, flights, sample_frames  # noqa: E402

# (label, input, slice of S_t): "zero" sets the slice to 0 instead of swapping it
SWAPS: Tuple[Tuple[str, str, Optional[slice]], ...] = (
    ("image (z_t)", "z", None),
    ("goal g_t", "g", None),
    ("velocity v^H", "s", slice(0, 3)),
    ("attitude n^B", "s", slice(3, 6)),
    ("body rates", "s", slice(6, 9)),
    ("last command a_bar", "s", slice(9, 13)),
    ("  a_bar in S_t only", "s_only", slice(9, 13)),
    ("  a_bar to P_omega only", "p_only", slice(9, 13)),
    ("last command = 0", "zero", slice(9, 13)),
    ("all of S_t", "s", slice(0, 13)),
)
# decide(z_t, S_t, g_t, noise, a_bar for P_omega or None = S_t[:, 9:13]) -> (u_t, chunk)
Decide = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                  Tuple[torch.Tensor, torch.Tensor]]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data_val/*.hdf5"], help="demonstration files or glob patterns")
    p.add_argument("--checkpoint", nargs="*", default=[], help="calibrated policy checkpoints")
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--moving", type=float, default=0.3, help="first knots slower than this (m/s) get no direction change")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if not args.self_test and not args.checkpoint:
        p.error("give at least one checkpoint (or --self-test)")
    args.latency_bound, args.tracking_lag = 0.06, 0.15        # unused here; OpenLoop builds the full pipeline
    return args


def derangement(n: int, seed: int) -> np.ndarray:
    """A permutation without fixed points: frame i takes its input from frame perm[i] != i."""
    order = np.random.default_rng(seed).permutation(n)
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.roll(order, -1)
    return perm


def policy_decide(pol: OpenLoop) -> Decide:
    """(u_t, chunk) of the deployed decision for latents, S_t, g_t and fixed CFM noise (no closed-loop state)."""
    pipe = pol.pipe

    @torch.no_grad()
    def decide(z: torch.Tensor, s: torch.Tensor, g: torch.Tensor, noise: torch.Tensor,
               a_bar: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        c = pol.c_text.expand(z.shape[0], -1)
        u = pipe.selector.select(z, g, s, c)
        pred = pipe.predictor(z, u, s[:, -pipe.solver.action_dim:] if a_bar is None else a_bar)
        chunk = pipe.solver.generate_chunk(pipe.solver.build_conditioning(u, z, pred.z_hat, s, g, c), noise=noise)
        return u, chunk

    return decide


def run_batches(decide: Decide, z: torch.Tensor, s: torch.Tensor, g: torch.Tensor, noise: torch.Tensor, batch: int,
                device: torch.device, a_bar: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
    us, chunks = [], []
    for i in range(0, z.shape[0], batch):
        u, ch = decide(z[i:i + batch].to(device), s[i:i + batch].to(device), g[i:i + batch].to(device),
                       noise[i:i + batch].to(device), None if a_bar is None else a_bar[i:i + batch].to(device))
        us.append(u.cpu().numpy())
        chunks.append(ch.float().cpu().numpy())
    return np.concatenate(us), np.concatenate(chunks).astype(np.float64)


def swap_metrics(u0: np.ndarray, ch0: np.ndarray, u1: np.ndarray, ch1: np.ndarray, own: np.ndarray, donor: np.ndarray,
                 moving: float) -> Dict[str, Any]:
    """Change of the first knot from (u0, ch0) to (u1, ch1), and its error to the own and the donor's labels."""
    v0, v1 = ch0[:, 0, :3], ch1[:, 0, :3]
    change = np.linalg.norm(v1 - v0, axis=1)
    both = (np.linalg.norm(v0, axis=1) > moving) & (np.linalg.norm(v1, axis=1) > moving)
    return {
        "change_median": _q(change, 0.5), "change_p90": _q(change, 0.9),
        "direction_change_median": _q(_angle_deg(v1[both], v0[both]), 0.5) if both.any() else None,
        "yaw_rate_change_median": _q(np.abs(ch1[:, 0, 3] - ch0[:, 0, 3]), 0.5),
        "primitive_changed": float((u1 != u0).mean()),
        "error_own_median": _q(np.linalg.norm(v1 - own[:, :3], axis=1), 0.5),
        "error_donor_median": _q(np.linalg.norm(v1 - donor[:, :3], axis=1), 0.5),
    }


def sensitivity(decide: Decide, z: torch.Tensor, s: torch.Tensor, g: torch.Tensor, label: np.ndarray, perm: np.ndarray,
                noise: torch.Tensor, batch: int, device: torch.device, moving: float) -> Dict[str, Any]:
    """Every swap of ``SWAPS`` against the unswapped decision. ``label``: expert first knots ``[N, 4]``."""
    idx = torch.as_tensor(perm)
    u0, ch0 = run_batches(decide, z, s, g, noise, batch, device)
    donor = label[perm]
    out: Dict[str, Any] = {"frames": int(z.shape[0]),
                           "label_spread_median": _q(np.linalg.norm(label[:, :3] - donor[:, :3], axis=1), 0.5),
                           "swaps": {"none": swap_metrics(u0, ch0, u0, ch0, label, donor, moving)}}
    for name, key, part in SWAPS:
        z1, s1, g1, a1 = z, s, g, None
        if key == "z":
            z1 = z[idx]
        elif key == "g":
            g1 = g[idx]
        elif key == "p_only":
            a1 = s[idx][:, part]
        else:
            s1 = s.clone()
            s1[:, part] = 0.0 if key == "zero" else s[idx][:, part]
            if key == "s_only":
                a1 = s[:, part]
        u1, ch1 = run_batches(decide, z1, s1, g1, noise, batch, device, a1)
        out["swaps"][name] = swap_metrics(u0, ch0, u1, ch1, label, donor if key != "zero" else label, moving)
        if key == "zero":
            out["swaps"][name]["error_donor_median"] = None
            out["swaps"][name]["speed_median"] = _q(np.linalg.norm(ch1[:, 0, :3], axis=1), 0.5)
    return out


def print_report(report: Dict[str, Any]) -> None:
    def f(x: Optional[float], d: int = 2) -> str:
        return "-" if x is None else f"{x:.{d}f}"

    for name, r in report["policies"].items():
        seen = r.get("policy_state_inputs")
        print(f"\n{name}: {r['frames']} frames; label spread (median |v0* - v0* of the donor frame|) "
              f"{f(r['label_spread_median'])} m/s" + (f"; S_t groups seen: {', '.join(seen) or 'none'}" if seen else ""))
        print(f"  {'swapped input':26s} {'change of v0 (m/s)':>19s} {'direction':>10s} {'yaw rate':>9s} {'primitive':>10s} "
              f"{'error to':>9s} {'error to':>9s}")
        print(f"  {'':26s} {'median / p90':>19s} {'(deg)':>10s} {'(rad/s)':>9s} {'changed':>10s} {'own label':>9s} "
              f"{'donor':>9s}")
        for swap, m in r["swaps"].items():
            extra = f"   (speed then {f(m['speed_median'])} m/s)" if "speed_median" in m else ""
            print(f"  {swap:26s} {f(m['change_median']) + ' / ' + f(m['change_p90']):>19s} "
                  f"{f(m['direction_change_median'], 0):>10s} {f(m['yaw_rate_change_median']):>9s} "
                  f"{100.0 * m['primitive_changed']:9.1f}% {f(m['error_own_median']):>9s} {f(m['error_donor_median']):>9s}{extra}")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    items = flights(expand_paths(args.data))
    pols = [OpenLoop(path, args, device) for path in args.checkpoint]
    zs: Dict[str, List[torch.Tensor]] = {p.name: [] for p in pols}
    states, goals, labels = [], [], []
    for fr in sample_frames(items, args.frames, args.seed):
        rgb = torch.from_numpy(fr.rgb).permute(0, 3, 1, 2)
        depth = torch.from_numpy(fr.depth)
        for p in pols:
            with torch.no_grad():
                for i in range(0, rgb.shape[0], args.batch_size):
                    image = p.normalizer(rgb[i:i + args.batch_size].to(device), depth[i:i + args.batch_size].to(device)).image
                    zs[p.name].append(p.pipe.vision_encoder(image).float().cpu())
        states.append(fr.state)
        goals.append(fr.goal)
        labels.append(fr.actions[:, 0, :])
    s = torch.from_numpy(np.concatenate(states))
    g = torch.from_numpy(np.concatenate(goals))
    label = np.concatenate(labels).astype(np.float64)
    n = int(s.shape[0])
    perm = derangement(n, args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    report: Dict[str, Any] = {"data": list(args.data), "frames": n, "seed": args.seed, "policies": {}}
    for p in pols:
        noise = torch.randn((n, p.pipe.solver.horizon, p.pipe.solver.action_dim), generator=gen)
        r = sensitivity(policy_decide(p), torch.cat(zs[p.name]), s, g, label, perm, noise, args.batch_size, device, args.moving)
        r["checkpoint_id"] = p.checkpoint_id
        r["policy_state_inputs"] = list(p.cfg.policy_state_inputs)
        report["policies"][p.name] = r
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    report = run(args)
    print(f"{report['frames']} frames of held-out expert flights ({', '.join(report['data'])}); one input swapped at a time")
    print_report(report)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


def _self_test() -> None:
    import tempfile

    import h5py

    # metrics: a policy that copies its last command follows a swapped a_bar and ignores the image
    rng = np.random.default_rng(0)
    n, h = 64, 16
    s = torch.as_tensor(rng.normal(size=(n, 13)), dtype=torch.float32)
    z = torch.as_tensor(rng.normal(size=(n, 8)), dtype=torch.float32)
    g = torch.as_tensor(rng.normal(size=(n, 4)), dtype=torch.float32)
    label = s[:, 9:13].double().numpy()                                     # the label equals the last command

    def copycat(z_: torch.Tensor, s_: torch.Tensor, g_: torch.Tensor, xi: torch.Tensor,
                a_bar: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(z_.shape[0], dtype=torch.long), s_[:, 9:13].unsqueeze(1).expand(-1, h, -1).clone()

    perm = derangement(n, 1)
    assert not np.any(perm == np.arange(n)) and sorted(perm.tolist()) == list(range(n))
    r = sensitivity(copycat, z, s, g, label, perm, torch.zeros(n, h, 4), 16, torch.device("cpu"), 0.3)
    sw = r["swaps"]
    assert sw["none"]["change_median"] == 0.0 and sw["image (z_t)"]["change_median"] == 0.0
    assert sw["goal g_t"]["change_median"] == 0.0 and sw["velocity v^H"]["primitive_changed"] == 0.0
    assert sw["last command a_bar"]["error_donor_median"] < 1e-6
    assert abs(sw["last command a_bar"]["change_median"] - r["label_spread_median"]) < 1e-6
    assert sw["last command = 0"]["speed_median"] == 0.0 and sw["all of S_t"]["error_donor_median"] < 1e-6
    assert sw["  a_bar in S_t only"]["error_donor_median"] < 1e-6 and sw["  a_bar to P_omega only"]["change_median"] == 0.0

    # plumbing on dry-run flights with an untrained, calibrated policy
    sys.path.insert(0, os.path.join(HERE, "sim"))
    import expert as E
    from fly_policy import _demo_policy

    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "aero_forest_sparse_s0001.hdf5")
        E.dry_run(data, E.synthetic_forest(seed=2, half_size=25.0),
                  E.DryRunConfig(num_episodes=2, seed=3, max_frames=50, spacing=(8.0, 10.0), num_waypoints=(2, 2),
                                 speed_limits=(4.0,)))
        with h5py.File(data, "r+") as f:
            for k in f:
                if k.startswith("episode_"):
                    f[k].attrs["success"] = True
        ckpt = os.path.join(tmp, "policy.pt")
        _demo_policy(ckpt)
        out = os.path.join(tmp, "s.json")
        assert main(["--data", data, "--frames", "40", "--device", "cpu", "--checkpoint", ckpt, "--json", out,
                     "--batch_size", "16"]) == 0
        rep = json.load(open(out, encoding="utf-8"))
        swaps = rep["policies"]["policy.pt"]["swaps"]
        assert rep["frames"] == 40 and set(swaps) == {"none", *(name for name, _, _ in SWAPS)}
        assert swaps["none"]["change_median"] == 0.0 and swaps["none"]["primitive_changed"] == 0.0
    print("input_sensitivity.py self-test passed")


if __name__ == "__main__":
    sys.exit(main())

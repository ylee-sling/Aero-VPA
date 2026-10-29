# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
probe_depth.py -- What obstacle geometry does z_t = E_psi(I_t) keep? Linear probes on held-out flights.

For one or more checkpoints ("stage1" or "policy", e.g. the Table 1 and the residual stage-(i) runs) and a
randomly initialised encoder of the same architecture as a control, the frozen encoder embeds the same frames,
and ridge regressions fitted on training flights predict, on held-out flights:

    forward depth       log of the 5th percentile of the depth in the central third of the image
    depth grid          mean inverse depth in each cell of a 4 x 5 grid over the image (20 outputs)
    nearest obstacle    log of the distance from the vehicle to the closest obstacle of the privileged map
                        (also behind and beside the camera, so only partly observable)

Depth is read as the encoder sees it: clipped to the configured range (0.1-10 m by default); pixels without a
return (beyond the sensor range, mostly sky) count as the far end of the range. Every target is scored with
held-out R^2 (pooled over the 20 cells for the grid) and a 95% interval from resampling whole held-out flights.
The ridge penalty is chosen on a slice of the training flights. The flights are split by the episode hash of
the training scripts (``--val_fraction``, ``--split_seed``, defaults as in training), so held-out flights were
never seen by any encoder.

    python probe_depth.py checkpoints/stage1.pt checkpoints/stage1_res.pt
    python probe_depth.py --self-test
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from checkpoint import config_from_dict, load_checkpoint
from config import VPAConfig
from dataset import AeroVPADataset, episode_split
from train_stage1 import autocast, pick_device

GRID = (4, 5)
TARGETS = ("forward_depth", "depth_grid", "nearest_obstacle")
LABELS = {"forward_depth": "forward depth (log, 5th pct)", "depth_grid": "depth grid (20 cells, 1/d)",
          "nearest_obstacle": "nearest obstacle (log, privileged)"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probes of obstacle geometry in z_t on held-out flights.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoints", nargs="*", help="stage1 or policy checkpoints")
    p.add_argument("--data", nargs="*", default=["data/*.hdf5"], help="demonstration files")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--train_frames", type=int, default=30000)
    p.add_argument("--val_frames", type=int, default=10000)
    p.add_argument("--no_random_baseline", action="store_true", help="skip the randomly initialised encoder")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=min(8, max(0, (os.cpu_count() or 2) - 2)))
    p.add_argument("--device", default="auto")
    p.add_argument("--fp32", action="store_true", help="no bf16 autocast on CUDA")
    p.add_argument("--bootstrap", type=int, default=2000, help="resamples of held-out flights")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if not args.self_test and not args.checkpoints:
        p.error("give at least one checkpoint (or --self-test)")
    return args


# ---------------------------------------------------------------------------------------------
# Frames and targets
# ---------------------------------------------------------------------------------------------
class ProbeFrames(Dataset):
    """The current frame (no look-ahead frame), the privileged position and the flight of sampled steps."""

    def __init__(self, base: AeroVPADataset, indices: np.ndarray) -> None:
        self.base, self.indices = base, np.asarray(indices)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, k: int) -> Dict[str, Any]:
        b = self.base
        i = int(self.indices[k])
        episode, t = int(b._owner[i]), int(b._frame[i])
        ep = b._group(episode)
        file_id, _ = b.episodes[episode]
        item = b._image(ep, t, b.depth_scale[file_id])                     # rgb [3, H, W] uint8, depth [H, W] m
        pos = ep["privileged/position_ned"][t] if "privileged/position_ned" in ep else np.full(3, np.nan)
        item.update(position=torch.as_tensor(np.asarray(pos, dtype=np.float64)), episode=episode)
        return item


def depth_targets(depth: np.ndarray, depth_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
    """Forward depth and the depth grid from ``depth [N, H, W]`` (m, 0 or non-finite = no return)."""
    near, far = float(depth_range[0]), float(depth_range[1])
    d = np.where(np.isfinite(depth) & (depth > 0.0), depth, far)
    d = np.clip(d, near, far).astype(np.float64)
    n, h, w = d.shape
    window = d[:, h // 3: h - h // 3, w // 3: w - w // 3].reshape(n, -1)
    forward = np.log(np.percentile(window, 5, axis=1))
    gh, gw = GRID
    if h % gh or w % gw:
        raise ValueError(f"image {h}x{w} does not divide into a {gh}x{gw} grid")
    cells = (1.0 / d).reshape(n, gh, h // gh, gw, w // gw).mean(axis=(2, 4)).reshape(n, gh * gw)
    return {"forward_depth": forward[:, None], "depth_grid": cells}


def nearest_obstacle(base: AeroVPADataset, episodes: np.ndarray, positions: np.ndarray) -> Optional[np.ndarray]:
    """log distance to the closest obstacle of each flight's privileged map; None if a map is missing."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim"))
    import expert as E

    out = np.full(len(episodes), np.nan)
    maps: Dict[str, Any] = {}
    for episode in np.unique(episodes):
        file_id, name = base.episodes[int(episode)]
        with h5py.File(base.paths[file_id], "r") as f:
            world = f[name].attrs.get("world_json", "")
        world = world.decode() if isinstance(world, bytes) else str(world)
        if not world or not os.path.isfile(world):
            return None
        if world not in maps:
            maps[world] = E.load_world(world)
        sel = episodes == episode
        out[sel] = maps[world].distance(positions[sel])
    if not np.all(np.isfinite(out)):
        return None
    return np.log(np.maximum(out, 0.05))[:, None]


@torch.no_grad()
def embed(encoder: torch.nn.Module, normalizer: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool,
          depth_range: Tuple[float, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """(z [N, d], depth targets, positions [N, 3], episodes [N]) in loader order (targets per batch: no image copy)."""
    zs, targets, positions, episodes = [], [], [], []
    for batch in loader:
        image = normalizer(batch["rgb"].to(device), batch["depth"].to(device)).image
        with autocast(device, amp):
            z = encoder(image)
        zs.append(z.float().cpu().numpy())
        targets.append(depth_targets(batch["depth"].numpy(), depth_range))
        positions.append(batch["position"].numpy())
        episodes.append(batch["episode"].numpy())
    merged = {k: np.concatenate([t[k] for t in targets]) for k in targets[0]}
    return np.concatenate(zs), merged, np.concatenate(positions), np.concatenate(episodes)


# ---------------------------------------------------------------------------------------------
# Ridge probe and held-out R^2
# ---------------------------------------------------------------------------------------------
def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xtx = x.T @ x
    reg = lam * float(np.mean(np.diag(xtx))) * np.eye(xtx.shape[0])
    reg[-1, -1] = 0.0                                                    # no penalty on the intercept
    return np.linalg.solve(xtx + reg, x.T @ y)


def _design(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.concatenate([(z - mean) / std, np.ones((z.shape[0], 1))], axis=1)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Pooled R^2 over all outputs: 1 - SSE / SST around the per-output mean of ``y``."""
    sst = float(((y - y.mean(axis=0)) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / max(sst, 1e-300)


def fit_probe(z_tr: np.ndarray, y_tr: np.ndarray, ep_tr: np.ndarray, lambdas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
              select_fraction: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Standardisation and ridge weights; the penalty is chosen on a held-back slice of training flights."""
    mean, std = z_tr.mean(axis=0), z_tr.std(axis=0) + 1e-8
    x = _design(z_tr, mean, std)
    flights = np.unique(ep_tr)
    rng = np.random.default_rng(seed)
    pick = rng.choice(flights, size=max(1, int(round(select_fraction * flights.size))), replace=False)
    sel = np.isin(ep_tr, pick)
    if sel.all() or not sel.any():                                       # too few flights to hold some back
        best = lambdas[len(lambdas) // 2]
    else:
        scores = [r2(y_tr[sel], x[sel] @ _ridge(x[~sel], y_tr[~sel], lam)) for lam in lambdas]
        best = lambdas[int(np.argmax(scores))]
    return mean, std, _ridge(x, y_tr, best), best


def r2_interval(y: np.ndarray, pred: np.ndarray, episodes: np.ndarray, draws: int, seed: int) -> Tuple[float, float]:
    """95% interval of the pooled R^2, resampling whole flights."""
    _, inv = np.unique(episodes, return_inverse=True)
    k = inv.max() + 1
    n = np.bincount(inv).astype(np.float64)                              # frames per flight
    sy = np.stack([np.bincount(inv, weights=y[:, j]) for j in range(y.shape[1])], 1)          # [k, m]
    syy = np.stack([np.bincount(inv, weights=y[:, j] ** 2) for j in range(y.shape[1])], 1)
    sse = np.bincount(inv, weights=((y - pred) ** 2).sum(axis=1))
    counts = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=draws).astype(np.float64)
    tot_n = counts @ n                                                   # [draws]
    tot_y, tot_yy = counts @ sy, counts @ syy                            # [draws, m]
    sst = (tot_yy - tot_y ** 2 / tot_n[:, None]).sum(axis=1)
    values = 1.0 - (counts @ sse) / np.maximum(sst, 1e-300)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy) of the normalised eigenvalues of the covariance of z."""
    ev = np.clip(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0.0, None)
    p = ev / max(ev.sum(), 1e-300)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def _encoders(args: argparse.Namespace) -> List[Tuple[str, VPAConfig, torch.nn.Module]]:
    out = []
    for path in args.checkpoints:
        ckpt = load_checkpoint(path)
        if ckpt["kind"] not in ("stage1", "policy"):
            raise ValueError(f"{path}: expected a stage1 or policy checkpoint, got {ckpt['kind']!r}")
        cfg = config_from_dict(ckpt["config"])
        encoder = cfg.build_vision_encoder()
        encoder.load_state_dict(ckpt["state_dict"]["vision_encoder"])
        out.append((os.path.basename(path), cfg, encoder))
    if not args.no_random_baseline:
        cfg = out[0][1] if out else VPAConfig()
        torch.manual_seed(args.seed)
        out.append(("random init", cfg, cfg.build_vision_encoder()))
    return out


def probe(args: argparse.Namespace) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.fp32
    encoders = _encoders(args)
    cfg = encoders[0][1]
    train_f, val_f = episode_split(args.val_fraction, args.split_seed)
    splits = {"train": AeroVPADataset(args.data, 1, cfg, episode_filter=train_f, normalize=False),
              "val": AeroVPADataset(args.data, 1, cfg, episode_filter=val_f, normalize=False)}
    rng = np.random.default_rng(args.seed)
    sizes = {"train": args.train_frames, "val": args.val_frames}
    loaders = {}
    for name, ds in splits.items():
        idx = np.sort(rng.choice(len(ds), size=min(sizes[name], len(ds)), replace=False))
        loaders[name] = DataLoader(ProbeFrames(ds, idx), batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=device.type == "cuda")
    normalizer = cfg.build_rgbd_normalizer().to(device)
    targets: Dict[str, Dict[str, np.ndarray]] = {}
    episodes: Dict[str, np.ndarray] = {}
    latents: Dict[str, Dict[str, np.ndarray]] = {}
    for label, _, encoder in encoders:
        encoder.to(device).eval()
        latents[label] = {}
        for name, loader in loaders.items():
            z, depth_t, pos, eps = embed(encoder, normalizer, loader, device, amp, cfg.depth_range)
            latents[label][name] = z
            if name not in targets:                                       # the frames are the same for every encoder
                targets[name] = depth_t
                near = nearest_obstacle(splits[name], eps, pos)
                if near is not None:
                    targets[name]["nearest_obstacle"] = near
                episodes[name] = eps
        encoder.cpu()
    names = [t for t in TARGETS if t in targets["train"] and t in targets["val"]]
    results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for label, _, _ in encoders:
        results[label] = {}
        for t in names:
            mean, std, w, _ = fit_probe(latents[label]["train"], targets["train"][t], episodes["train"], seed=args.seed)
            pred = _design(latents[label]["val"], mean, std) @ w
            value = r2(targets["val"][t], pred)
            lo, hi = r2_interval(targets["val"][t], pred, episodes["val"], args.bootstrap, args.seed) \
                if args.bootstrap > 0 else (math.nan, math.nan)
            results[label][t] = (value, lo, hi)
        results[label]["effective_rank"] = (effective_rank(latents[label]["val"]), math.nan, math.nan)
    n_val_flights = np.unique(episodes["val"]).size
    print(f"[probe] {len(episodes['train'])} training frames, {len(episodes['val'])} held-out frames from "
          f"{n_val_flights} held-out flights; ridge on standardised z_t; held-out R^2 with 95% intervals over flights")
    width = max(22, *(len(label) + 2 for label, _, _ in encoders))
    print(f"[probe] {'target':36s}" + "".join(f"{label:>{width}s}" for label, _, _ in encoders))
    for t in names:
        cells = "".join(f"{f'{results[label][t][0]:.3f} [{results[label][t][1]:.2f}, {results[label][t][2]:.2f}]':>{width}s}"
                        for label, _, _ in encoders)
        print(f"[probe] {LABELS[t]:36s}{cells}")
    print(f"[probe] {'effective rank of z (held-out)':36s}" +
          "".join(f"{results[label]['effective_rank'][0]:>{width}.1f}" for label, _, _ in encoders))
    if "nearest_obstacle" not in names:
        print("[probe] nearest obstacle skipped: the privileged maps (episode attribute world_json) were not found; "
              "run from the repository root")
    return results


def _self_test() -> None:
    import tempfile

    from train_stage1 import make_demo_data, parse_args as stage1_args, train_stage1

    # targets on a synthetic frame: a wall at 2 m in the left half, no return in the right half
    depth = np.zeros((1, 128, 160), dtype=np.float32)
    depth[0, :, :80] = 2.0
    t = depth_targets(depth, (0.1, 10.0))
    assert math.isclose(float(np.exp(t["forward_depth"][0, 0])), 2.0, rel_tol=1e-9)
    grid = t["depth_grid"].reshape(GRID)
    assert np.allclose(grid[:, :2], 0.5) and np.allclose(grid[:, 3:], 0.1) and np.allclose(grid[:, 2], 0.3)
    # R^2: perfect prediction 1, the mean 0; the interval of a perfect fit is degenerate at 1
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 3))
    assert r2(y, y) == 1.0 and abs(r2(y, np.broadcast_to(y.mean(0), y.shape))) < 1e-12
    lo, hi = r2_interval(y, y, np.arange(200) // 10, 100, 0)
    assert lo == hi == 1.0
    # the ridge probe recovers a linear target from latents and fails on noise
    z = rng.normal(size=(600, 16))
    eps = np.arange(600) // 20
    y_lin = (z @ rng.normal(size=(16, 2))) + 0.01 * rng.normal(size=(600, 2))
    mean, std, w, _ = fit_probe(z[:400], y_lin[:400], eps[:400])
    assert r2(y_lin[400:], _design(z[400:], mean, std) @ w) > 0.99
    noise = rng.normal(size=(600, 1))
    mean, std, w, _ = fit_probe(z[:400], noise[:400], eps[:400])
    assert r2(noise[400:], _design(z[400:], mean, std) @ w) < 0.1
    assert 15.0 < effective_rank(z) <= 16.0 and effective_rank(np.outer(rng.normal(size=600), np.ones(16))) < 1.01
    # end to end on dry-run flights: a trained and a random encoder, all three targets present
    with tempfile.TemporaryDirectory() as tmp:
        (train_file,) = make_demo_data(tmp, {"train.hdf5": 4}, frames=40)
        out = os.path.join(tmp, "stage1.pt")
        train_stage1(stage1_args([train_file, "--out", out, "--batch_size", "8", "--num_workers", "0", "--device", "cpu",
                                  "--steps", "2", "--val_fraction", "0", "--log_every", "100"]))
        res = probe(parse_args([out, "--data", train_file, "--val_fraction", "0.5", "--split_seed", "0",
                                "--train_frames", "80", "--val_frames", "60", "--num_workers", "0", "--device", "cpu",
                                "--bootstrap", "50"]))
        assert set(res) == {"stage1.pt", "random init"}
        for label in res:
            assert set(res[label]) == {"forward_depth", "depth_grid", "effective_rank"}   # dry run: no world_json
            assert all(math.isfinite(v[0]) for v in res[label].values())
    print("probe_depth.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        probe(parsed)

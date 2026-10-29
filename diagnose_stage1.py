# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
diagnose_stage1.py -- Where does the stage-(i) prediction error come from? (preprint_261026.pdf, Eq. 8)

The predictor P_omega reads the online latent z_t = E(I_t) and is trained towards the EMA target E_bar(I_t+1),
so its error contains two things: what it fails to forecast about the next 0.1 s, and the lag between E and
E_bar. A "no change" baseline computed entirely in the target space, ||E_bar(I_t) - E_bar(I_t+1)||^2, contains
only the first. This tool separates them. It reads a training state (``<out>.state.pt``, written every
--save_minutes while train_stage1.py runs, so it also works during training) and reports on held-out and
training frames:

    no change, target space     ||E_bar(I_t) - E_bar(I_t+1)||^2     persistence in the target space
    no change, online input     ||E(I_t)     - E_bar(I_t+1)||^2     what copying z_t achieves
    online/target gap           ||E(I_t)     - E_bar(I_t)  ||^2     lag of the EMA target encoder
    change in online space      ||E(I_t)     - E(I_t+1)    ||^2
    predictor, Eq. 8 term       (1/N_e) sum_i ||P^(i)(z_t, u_t, a_bar_t) - E_bar(I_t+1)||^2
    predictor, ensemble mean    ||z_hat_t+1 - E_bar(I_t+1)||^2
    predictor fed E_bar(I_t)    the ensemble mean with the target latent of I_t as input (no lag)
      ... u, a_bar shuffled     the same with (u_t, a_bar_t) of other frames: does P_omega use the action?
    linear map from E(I_t)      ridge regression [z_t, onehot(u_t), a_bar_t, 1] -> E_bar(I_t+1), fitted on training frames
    linear map from E_bar(I_t)  the same from the target latent
      ... without u, a_bar      [E_bar(I_t), 1] only
    cos(pred., true change)     mean cosine of P(E_bar(I_t), u_t, a_bar_t) - E_bar(I_t) and E_bar(I_t+1) - E_bar(I_t)

All distances are squared Euclidean norms summed over the d latent dimensions and averaged over frames, the
units of Eq. 8. The episode split (data, --val/--val_fraction, --split_seed) is the one stored in the state.
The held-out ratios come with 95% intervals from resampling whole episodes (--bootstrap), since the frames of
one flight are strongly correlated.

    python diagnose_stage1.py checkpoints/stage1.pt.state.pt
    python diagnose_stage1.py --self-test
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from checkpoint import build_stage1_modules, config_from_dict
from config import VPAConfig
from train_stage1 import _batch_images, _datasets, autocast, pick_device

ROWS = (
    ("copy_target", "no change, target space"),
    ("copy_online", "no change, online input"),
    ("gap", "online/target gap"),
    ("change_online", "change in online space"),
    ("pred_heads", "predictor, Eq. 8 term (log 'inv')"),
    ("pred_mean", "predictor, ensemble mean"),
    ("pred_from_target", "predictor fed E_bar(I_t)"),
    ("pred_from_target_shuffled", "  ... u, a_bar shuffled"),
    ("linear_online", "linear map from E(I_t)"),
    ("linear_target", "linear map from E_bar(I_t)"),
    ("linear_target_noact", "  ... without u, a_bar"),
    ("cos_change", "cos(pred., true change), fed E_bar(I_t)"),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decompose the stage-(i) prediction error of a training state.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("state", nargs="?", default="checkpoints/stage1.pt.state.pt", help="train_stage1.py state")
    p.add_argument("--data", nargs="*", default=None, help="demonstration files (default: those of the state)")
    p.add_argument("--val_frames", type=int, default=4000)
    p.add_argument("--train_frames", type=int, default=8000, help="also the fitting set of the linear maps")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--fp32", action="store_true", help="no bf16 autocast on CUDA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=2000, help="resamples of held-out episodes for the ratio intervals")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def collect(modules: Dict[str, torch.nn.Module], dataset, indices: Sequence[int], args: argparse.Namespace,
            device: torch.device, amp: bool) -> Dict[str, Tensor]:
    """Latents of I_t and I_t+1 under both encoders, and the predictor outputs, for ``indices``."""
    encoder, target, predictor, normalizer = (modules[k] for k in ("encoder", "target", "predictor", "normalizer"))
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    out: Dict[str, List[Tensor]] = {k: [] for k in ("z_on", "z_on_next", "z_tg", "z_tg_next", "z_hat", "z_hat_tg",
                                                    "z_hat_tg_shuffled", "head_err", "u", "a_bar")}
    for batch in loader:
        img = _batch_images(batch, "", normalizer, device)
        fut = _batch_images(batch, "future_", normalizer, device)
        u = batch["primitive"].to(device)
        a_bar = batch["a_bar"].to(device)
        perm = torch.randperm(u.shape[0], generator=generator).to(device)
        with autocast(device, amp):
            z_on, z_on_next = encoder(img), encoder(fut)
            z_tg, z_tg_next = target(img), target(fut)
            heads = predictor.forward_heads(z_on, u, a_bar)
            heads_tg = predictor.forward_heads(z_tg, u, a_bar)
            heads_shuffled = predictor.forward_heads(z_tg, u[perm], a_bar[perm])
        heads, z_tg_next = heads.float(), z_tg_next.float()
        out["head_err"].append((heads - z_tg_next.unsqueeze(0)).pow(2).sum(-1).mean(0).cpu())
        out["z_hat"].append(heads.mean(0).cpu())
        out["z_hat_tg"].append(heads_tg.float().mean(0).cpu())
        out["z_hat_tg_shuffled"].append(heads_shuffled.float().mean(0).cpu())
        for key, value in (("z_on", z_on), ("z_on_next", z_on_next), ("z_tg", z_tg), ("z_tg_next", z_tg_next),
                           ("u", u), ("a_bar", a_bar)):
            out[key].append(value.float().cpu() if value.is_floating_point() else value.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def features(z: Tensor, u: Optional[Tensor], a_bar: Optional[Tensor], num_primitives: int, scale: Tensor) -> Tensor:
    """[z, onehot(u), a_bar / scale, 1] in float64 ([z, 1] without u and a_bar)."""
    parts = [z.double()]
    if u is not None:
        parts += [F.one_hot(u.long(), num_primitives).double(), a_bar.double() / scale.double()]
    parts.append(torch.ones(len(z), 1, dtype=torch.float64))
    return torch.cat(parts, 1)


def ridge(x: Tensor, y: Tensor, rel: float = 1e-4) -> Tensor:
    """Ridge regression weights W with x W ~ y; the penalty is ``rel`` times the mean diagonal of x^T x."""
    xtx = x.T @ x
    lam = rel * float(xtx.diagonal().mean())
    return torch.linalg.solve(xtx + lam * torch.eye(xtx.shape[0], dtype=x.dtype), x.T @ y.double())


def _sq_frames(a: Tensor, b: Tensor) -> np.ndarray:
    return (a.double() - b.double()).pow(2).sum(-1).numpy()


def frame_errors(r: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
    """Per-frame squared distances (summed over the d latent dimensions) behind the table rows."""
    return {"copy_target": _sq_frames(r["z_tg"], r["z_tg_next"]), "copy_online": _sq_frames(r["z_on"], r["z_tg_next"]),
            "gap": _sq_frames(r["z_on"], r["z_tg"]), "change_online": _sq_frames(r["z_on"], r["z_on_next"]),
            "pred_heads": r["head_err"].double().numpy(), "pred_mean": _sq_frames(r["z_hat"], r["z_tg_next"]),
            "pred_from_target": _sq_frames(r["z_hat_tg"], r["z_tg_next"]),
            "pred_from_target_shuffled": _sq_frames(r["z_hat_tg_shuffled"], r["z_tg_next"])}


def metrics(r: Dict[str, Tensor]) -> Dict[str, float]:
    m = {k: float(v.mean()) for k, v in frame_errors(r).items()}
    d_pred, d_true = (r["z_hat_tg"] - r["z_tg"]).double(), (r["z_tg_next"] - r["z_tg"]).double()
    m["cos_change"] = float(F.cosine_similarity(d_pred, d_true, dim=-1, eps=1e-12).mean())
    return m


def episode_interval(num: np.ndarray, den: np.ndarray, episodes: np.ndarray, draws: int, seed: int
                     ) -> Tuple[float, float]:
    """95% interval of mean(num) / mean(den), resampling whole episodes (the frames of one flight are correlated)."""
    _, inverse = np.unique(np.asarray(episodes), return_inverse=True)
    a = np.bincount(inverse, weights=np.asarray(num, dtype=np.float64))
    b = np.bincount(inverse, weights=np.asarray(den, dtype=np.float64))
    counts = np.random.default_rng(seed).multinomial(a.size, np.full(a.size, 1.0 / a.size), size=draws)
    ratios = (counts @ a) / np.maximum(counts @ b, 1e-300)
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _episode_ids(dataset, indices: np.ndarray) -> np.ndarray:
    """Episode of every sampled frame (each frame its own cluster if the dataset does not say)."""
    owner = getattr(dataset, "_owner", None)
    return np.asarray(owner)[np.asarray(indices)] if owner is not None else np.arange(len(indices))


def diagnose(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.fp32
    state = torch.load(args.state, map_location="cpu", weights_only=True)
    saved = dict(state.get("args", {}))
    split = argparse.Namespace(data=args.data or saved.get("data"), val=saved.get("val"),
                               val_fraction=float(saved.get("val_fraction", 0.05)),
                               split_seed=int(saved.get("split_seed", 0)),
                               include_failed=bool(saved.get("include_failed", False)))
    if not split.data:
        raise ValueError("no demonstration files: pass --data")
    cfg = config_from_dict(state["config"]) if "config" in state else VPAConfig()     # older states: Table 1 heads
    encoder, predictor = build_stage1_modules(cfg)
    target = cfg.build_momentum_encoder(encoder)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    target.load_state_dict(state["target"])
    normalizer = cfg.build_rgbd_normalizer()
    modules = {"encoder": encoder, "target": target, "predictor": predictor, "normalizer": normalizer}
    for module in modules.values():
        module.to(device).eval()
    train_ds, val_ds = _datasets(split, cfg)
    if val_ds is None:
        raise ValueError("the state has no held-out episodes (val_fraction 0 and no --val)")
    rng = np.random.default_rng(args.seed)
    pick = lambda ds, n: np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))  # noqa: E731
    t0 = time.time()
    picked = {"train": (train_ds, pick(train_ds, args.train_frames)), "val": (val_ds, pick(val_ds, args.val_frames))}
    results = {name: collect(modules, ds, idx, args, device, amp) for name, (ds, idx) in picked.items()}
    episodes = {name: _episode_ids(ds, idx) for name, (ds, idx) in picked.items()}
    errors = {name: frame_errors(r) for name, r in results.items()}
    table = {name: metrics(r) for name, r in results.items()}
    scale = predictor.setpoint_scale.cpu()
    fit = results["train"]
    def design(r: Dict[str, Tensor], key: str, with_action: bool) -> Tensor:
        return features(r[key], r["u"] if with_action else None, r["a_bar"], cfg.num_primitives, scale)

    for name, key, with_action in (("linear_online", "z_on", True), ("linear_target", "z_tg", True),
                                   ("linear_target_noact", "z_tg", False)):
        w = ridge(design(fit, key, with_action), fit["z_tg_next"])
        for split_name, r in results.items():
            errors[split_name][name] = _sq_frames(design(r, key, with_action) @ w, r["z_tg_next"])
            table[split_name][name] = float(errors[split_name][name].mean())
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(args.state)))
    heads = "residual heads" if cfg.predictor_residual else "Table 1 heads"
    print(f"stage (i) at step {int(state.get('step', -1))}, {heads} (state saved {stamp}); {len(results['val']['u'])} held-out "
          f"frames from {np.unique(episodes['val']).size} episodes and {len(results['train']['u'])} training frames on "
          f"{device}{' with bf16' if amp else ''}; {time.time() - t0:.0f} s")
    print(f"{'':44s}{'held-out':>12s}{'training':>12s}")
    for key, label in ROWS:
        print(f"  {label:42s}{table['val'][key]:12.3f}{table['train'][key]:12.3f}")
    v, ev = table["val"], errors["val"]
    ratios = (("pred_mean", "copy_online", "predictor / no change, online input", "as trained (reads E(I_t))"),
              ("pred_from_target", "copy_target", "predictor fed E_bar / no change, target", "skill without the lag"),
              ("pred_from_target_shuffled", "pred_from_target", "... u, a_bar shuffled / not shuffled",
               "above 1: P_omega uses the action"),
              ("linear_target", "copy_target", "linear map / no change, target space", ""),
              ("gap", "copy_target", "online/target gap / no change, target space", "EMA lag"))
    table["val_ratios"] = {}
    print("held-out ratios (below 1: better than the reference)"
          + (f", 95% interval over {np.unique(episodes['val']).size} episodes:" if args.bootstrap > 0 else ":"))
    for num, den, label, note in ratios:
        value = v[num] / max(v[den], 1e-12)
        lo_hi = episode_interval(ev[num], ev[den], episodes["val"], args.bootstrap, args.seed) if args.bootstrap > 0 \
            else (float("nan"), float("nan"))
        table["val_ratios"][f"{num}/{den}"] = (value, *lo_hi)
        interval = f"  [{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]" if args.bootstrap > 0 else ""
        print(f"  {label:44s}{value:7.3f}{interval}   {note}".rstrip())
    return table


def _self_test() -> None:
    import tempfile

    from train_stage1 import make_demo_data, parse_args as stage1_args, train_stage1

    with tempfile.TemporaryDirectory() as tmp:
        (train_file,) = make_demo_data(tmp, {"train.hdf5": 3})
        out = os.path.join(tmp, "stage1.pt")
        train_stage1(stage1_args([train_file, "--out", out, "--batch_size", "8", "--num_workers", "0", "--device", "cpu",
                                  "--steps", "2", "--val_fraction", "0.34", "--log_every", "1", "--warmup_steps", "1"]))
        small = ["--val_frames", "24", "--train_frames", "48", "--batch_size", "8", "--num_workers", "0", "--device", "cpu"]
        table = diagnose(parse_args([out + ".state.pt", *small]))
        for name in ("val", "train"):
            m = table[name]
            assert all(np.isfinite(list(m.values())))
            assert min(v for k, v in m.items() if k != "cos_change") >= 0.0 and -1.0 <= m["cos_change"] <= 1.0
            assert m["pred_heads"] >= m["pred_mean"] - 1e-6                     # Jensen: heads vs their mean
        # identical encoders: no gap, both "no change" baselines and both predictor inputs coincide
        state = torch.load(out + ".state.pt", map_location="cpu", weights_only=True)
        state["target"] = {f"encoder.{k}": v for k, v in state["encoder"].items()}
        torch.save(state, out + ".same.pt")
        m = diagnose(parse_args([out + ".same.pt", *small]))["val"]
        assert m["gap"] < 1e-9 and abs(m["copy_online"] - m["copy_target"]) < 1e-6
        assert abs(m["pred_from_target"] - m["pred_mean"]) < 1e-6
        # residual heads: the configuration comes from the state
        train_stage1(stage1_args([train_file, "--out", out + ".res", "--batch_size", "8", "--num_workers", "0",
                                  "--device", "cpu", "--steps", "2", "--val_fraction", "0.34", "--log_every", "1",
                                  "--warmup_steps", "1", "--residual_predictor"]))
        m = diagnose(parse_args([out + ".res.state.pt", *small]))["val"]
        assert all(np.isfinite(list(m.values()))) and m["pred_heads"] >= m["pred_mean"] - 1e-6
        # episode intervals: one value per episode -> the interval spans the episode ratios; constant ratio -> exact
        lo, hi = episode_interval(np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4), np.array([0, 0, 1, 1]), 500, 0)
        assert 1.5 <= lo <= hi <= 3.5
        lo, hi = episode_interval(2.0 * np.arange(1.0, 9.0), np.arange(1.0, 9.0), np.arange(8) // 2, 200, 0)
        assert lo == hi == 2.0
        assert set(table) >= {"val", "train", "val_ratios"} and len(table["val_ratios"]) == 5
        # ridge regression recovers an exact linear map
        g = torch.Generator().manual_seed(0)
        x = torch.randn(200, 5, generator=g, dtype=torch.float64)
        w = torch.randn(5, 3, generator=g, dtype=torch.float64)
        assert torch.allclose(ridge(x, x @ w, rel=1e-12), w, atol=1e-6)
    print("diagnose_stage1.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        diagnose(parsed)

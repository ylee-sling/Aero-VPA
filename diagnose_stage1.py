# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
diagnose_stage1.py -- Where does the stage-(i) prediction error come from? (preprint_261026.pdf, Eq. 8)

The training log compares the predictor with the "no change" baseline ||E_bar(I_t) - E_bar(I_t+1)||^2, which
lives entirely in the space of the EMA target encoder E_bar. The predictor, however, reads the online latent
z_t = E(I_t) and has to land in the target space, so its error also contains the lag between E and E_bar.
This tool separates the two. It reads a training state (``<out>.state.pt``, written every --save_minutes
while train_stage1.py runs, so it works during training) and reports on held-out and training frames:

    no change, target space    ||E_bar(I_t) - E_bar(I_t+1)||^2      the baseline of the training log
    no change, online input    ||E(I_t)     - E_bar(I_t+1)||^2      what copying z_t achieves
    online/target gap          ||E(I_t)     - E_bar(I_t)  ||^2      lag of the EMA target encoder
    change in online space     ||E(I_t)     - E(I_t+1)    ||^2
    predictor, Eq. 8 term      (1/N_e) sum_i ||P^(i)(z_t, u_t, a_bar_t) - E_bar(I_t+1)||^2   (the logged "inv")
    predictor, ensemble mean   ||z_hat_t+1 - E_bar(I_t+1)||^2
    predictor fed E_bar(I_t)   the same with the target latent of I_t as input
    linear map from E(I_t)     ridge regression [z_t, onehot(u_t), a_bar_t] -> E_bar(I_t+1), fitted on training frames
    linear map from E_bar(I_t) the same from the target latent
    cos(predicted, true change) mean cosine between z_hat - E_bar(I_t) and E_bar(I_t+1) - E_bar(I_t)

All distances are squared Euclidean norms summed over the d latent dimensions and averaged over frames, the
units of Eq. 8. The episode split (data, --val/--val_fraction, --split_seed) is the one stored in the state.

    python diagnose_stage1.py checkpoints/stage1.pt.state.pt
    python diagnose_stage1.py --self-test
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from checkpoint import build_stage1_modules
from config import VPAConfig
from train_stage1 import _batch_images, _datasets, autocast, pick_device

ROWS = (
    ("copy_target", "no change, target space (log baseline)"),
    ("copy_online", "no change, online input"),
    ("gap", "online/target gap"),
    ("change_online", "change in online space"),
    ("pred_heads", "predictor, Eq. 8 term (log 'inv')"),
    ("pred_mean", "predictor, ensemble mean"),
    ("pred_from_target", "predictor fed E_bar(I_t)"),
    ("linear_online", "linear map from E(I_t)"),
    ("linear_target", "linear map from E_bar(I_t)"),
    ("cos_change", "cos(predicted, true change)"),
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
    p.add_argument("--self-test", dest="self_test", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def collect(modules: Dict[str, torch.nn.Module], dataset, indices: Sequence[int], args: argparse.Namespace,
            device: torch.device, amp: bool) -> Dict[str, Tensor]:
    """Latents of I_t and I_t+1 under both encoders, and the predictor outputs, for ``indices``."""
    encoder, target, predictor, normalizer = (modules[k] for k in ("encoder", "target", "predictor", "normalizer"))
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    out: Dict[str, List[Tensor]] = {k: [] for k in ("z_on", "z_on_next", "z_tg", "z_tg_next", "z_hat", "z_hat_tg",
                                                    "head_err", "u", "a_bar")}
    for batch in loader:
        img = _batch_images(batch, "", normalizer, device)
        fut = _batch_images(batch, "future_", normalizer, device)
        u = batch["primitive"].to(device)
        a_bar = batch["a_bar"].to(device)
        with autocast(device, amp):
            z_on, z_on_next = encoder(img), encoder(fut)
            z_tg, z_tg_next = target(img), target(fut)
            heads = predictor.forward_heads(z_on, u, a_bar)
            heads_tg = predictor.forward_heads(z_tg, u, a_bar)
        heads, heads_tg, z_tg_next = heads.float(), heads_tg.float(), z_tg_next.float()
        out["head_err"].append((heads - z_tg_next.unsqueeze(0)).pow(2).sum(-1).mean(0).cpu())
        out["z_hat"].append(heads.mean(0).cpu())
        out["z_hat_tg"].append(heads_tg.mean(0).cpu())
        for key, value in (("z_on", z_on), ("z_on_next", z_on_next), ("z_tg", z_tg), ("z_tg_next", z_tg_next),
                           ("u", u), ("a_bar", a_bar)):
            out[key].append(value.float().cpu() if value.is_floating_point() else value.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def _sq(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).pow(2).sum(-1).mean())


def features(z: Tensor, u: Tensor, a_bar: Tensor, num_primitives: int, scale: Tensor) -> Tensor:
    """[z, onehot(u), a_bar / scale, 1] in float64."""
    onehot = F.one_hot(u.long(), num_primitives).double()
    return torch.cat([z.double(), onehot, a_bar.double() / scale.double(), torch.ones(len(z), 1, dtype=torch.float64)], 1)


def ridge(x: Tensor, y: Tensor, rel: float = 1e-4) -> Tensor:
    """Ridge regression weights W with x W ~ y; the penalty is ``rel`` times the mean diagonal of x^T x."""
    xtx = x.T @ x
    lam = rel * float(xtx.diagonal().mean())
    return torch.linalg.solve(xtx + lam * torch.eye(xtx.shape[0], dtype=x.dtype), x.T @ y.double())


def metrics(r: Dict[str, Tensor]) -> Dict[str, float]:
    m = {"copy_target": _sq(r["z_tg"], r["z_tg_next"]), "copy_online": _sq(r["z_on"], r["z_tg_next"]),
         "gap": _sq(r["z_on"], r["z_tg"]), "change_online": _sq(r["z_on"], r["z_on_next"]),
         "pred_heads": float(r["head_err"].double().mean()), "pred_mean": _sq(r["z_hat"], r["z_tg_next"]),
         "pred_from_target": _sq(r["z_hat_tg"], r["z_tg_next"])}
    d_pred, d_true = (r["z_hat"] - r["z_tg"]).double(), (r["z_tg_next"] - r["z_tg"]).double()
    m["cos_change"] = float(F.cosine_similarity(d_pred, d_true, dim=-1, eps=1e-12).mean())
    return m


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
    cfg = VPAConfig()
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
    results = {"train": collect(modules, train_ds, pick(train_ds, args.train_frames), args, device, amp),
               "val": collect(modules, val_ds, pick(val_ds, args.val_frames), args, device, amp)}
    table = {name: metrics(r) for name, r in results.items()}
    scale = predictor.setpoint_scale.cpu()
    for space, key in (("online", "z_on"), ("target", "z_tg")):
        fit = results["train"]
        w = ridge(features(fit[key], fit["u"], fit["a_bar"], cfg.num_primitives, scale), fit["z_tg_next"])
        for name, r in results.items():
            pred = features(r[key], r["u"], r["a_bar"], cfg.num_primitives, scale) @ w
            table[name][f"linear_{space}"] = _sq(pred, r["z_tg_next"])
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(args.state)))
    print(f"stage (i) at step {int(state.get('step', -1))} (state saved {stamp}); {len(results['val']['u'])} held-out "
          f"and {len(results['train']['u'])} training frames on {device}{' with bf16' if amp else ''}; "
          f"{time.time() - t0:.0f} s")
    print(f"{'':44s}{'held-out':>12s}{'training':>12s}")
    for key, label in ROWS:
        print(f"  {label:42s}{table['val'][key]:12.3f}{table['train'][key]:12.3f}")
    v = table["val"]
    print("held-out ratios:")
    print(f"  predictor (mean) / no change, target space  {v['pred_mean'] / max(v['copy_target'], 1e-12):7.3f}   "
          "(the log ratio uses the Eq. 8 term)")
    print(f"  predictor (mean) / no change, online input  {v['pred_mean'] / max(v['copy_online'], 1e-12):7.3f}")
    print(f"  online/target gap / no change, target space {v['gap'] / max(v['copy_target'], 1e-12):7.3f}")
    print(f"  linear map from E(I_t) / predictor (mean)   {v['linear_online'] / max(v['pred_mean'], 1e-12):7.3f}")
    return table


def _self_test() -> None:
    import tempfile

    from train_stage1 import make_demo_data, parse_args as stage1_args, train_stage1

    with tempfile.TemporaryDirectory() as tmp:
        (train_file,) = make_demo_data(tmp, {"train.hdf5": 3})
        out = os.path.join(tmp, "stage1.pt")
        train_stage1(stage1_args([train_file, "--out", out, "--batch_size", "8", "--num_workers", "0", "--device", "cpu",
                                  "--steps", "2", "--val_fraction", "0.34", "--log_every", "1", "--warmup_steps", "1"]))
        args = parse_args([out + ".state.pt", "--val_frames", "24", "--train_frames", "48", "--batch_size", "8",
                           "--num_workers", "0", "--device", "cpu"])
        table = diagnose(args)
        for name in ("val", "train"):
            m = table[name]
            assert all(np.isfinite(list(m.values())))
            assert min(v for k, v in m.items() if k != "cos_change") >= 0.0 and -1.0 <= m["cos_change"] <= 1.0
            assert m["pred_heads"] >= m["pred_mean"] - 1e-6                     # Jensen: heads vs their mean
        # identical encoders: no gap, and the two "no change" baselines coincide
        state = torch.load(out + ".state.pt", map_location="cpu", weights_only=True)
        state["target"] = {f"encoder.{k}": v for k, v in state["encoder"].items()}
        torch.save(state, out + ".same.pt")
        table = diagnose(parse_args([out + ".same.pt", "--val_frames", "24", "--train_frames", "48", "--batch_size", "8",
                                     "--num_workers", "0", "--device", "cpu"]))
        assert table["val"]["gap"] < 1e-9 and abs(table["val"]["copy_online"] - table["val"]["copy_target"]) < 1e-6
        assert abs(table["val"]["pred_from_target"] - table["val"]["pred_mean"]) < 1e-6
    print("diagnose_stage1.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        diagnose(parsed)

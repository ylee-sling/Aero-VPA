# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
train_stage1.py -- Stage (i): representation learning of E_psi and P_omega (preprint_261026.pdf, Secs. 3.2, 3.8).

Trains the RGB-D encoder E_psi and the 5-head predictor P_omega on expert flights with the
VICReg-regularised JEPA objective (Eq. 8)

    L_JEPA = E_t[ (1/N_e) sum_i || P_omega^(i)(z_t, Embed(u*_t), a_bar_t) - sg(E_psi_bar(I_{t+1})) ||^2 ]
             + lambda_v v(Z) + lambda_c c(Z),        z_t = E_psi(I_t),  I_{t+1} = frame at tau_t + T_nu (0.1 s),

with the EMA target encoder psi_bar <- mu psi_bar + (1 - mu) psi after every step, tracks the realised
latent scale gamma_bar (Eq. 11), and at the end freezes both networks with (gamma*, mu_Z) into a "stage1"
checkpoint (checkpoint.py). The primitive token is the expert label u*_t (the selector is trained in stage ii).

    python train_stage1.py "data/*.hdf5" --out checkpoints/stage1.pt --epochs 20
    python train_stage1.py "data/*.hdf5" --out checkpoints/stage1.pt --epochs 20 --resume   # after a stop
    python train_stage1.py --self-test

Outputs next to ``--out``: the checkpoint, ``<out>.state.pt`` (optimiser, EMA target and tracker, for
--resume; written every --save_minutes) and ``<out>.log.jsonl`` (one JSON record per log line and validation).

Validation (held-out episodes: --val files, or a hash split of the training files with --val_fraction)
reports the three loss terms and collapse diagnostics: per-dimension latent std, the effective rank of Z,
the ensemble disagreement sigma (Eq. 22), and the prediction error against "no change": the ratio to
||E_psi(I_t) - E_psi_bar(I_{t+1})||^2 (copying the predictor's own input z_t; below 1: the predictor forecasts
more than persistence), persistence in the target space ||E_psi_bar(I_t) - E_psi_bar(I_{t+1})||^2, and the EMA
gap ||E_psi(I_t) - E_psi_bar(I_t)||^2 that separates the two. diagnose_stage1.py breaks the error down further.

Choices the paper leaves open, as defaults here: AdamW (betas 0.9/0.95, weight decay 0.05 except biases,
norms, embeddings and position/[CLS] tokens), learning rate 3e-4 for batch 256 with linear warm-up and a
cosine decay to 1 %, gradient clipping at 1.0, bf16 autocast on CUDA (losses in float32), constant EMA
momentum mu = config.ema_momentum. This file also holds the training utilities shared with stage (ii).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from checkpoint import atomic_torch_save, build_stage1_modules, config_from_dict, config_to_dict, environment_info, \
    load_checkpoint, make_checkpoint, save_checkpoint
from config import VPAConfig
from dataset import AeroVPADataset, episode_split, expand_paths

__all__ = ["cosine_lr", "param_groups", "RunLog", "pick_device", "set_seed", "autocast", "data_summary",
           "make_demo_data", "train_stage1", "parse_args"]


# ---------------------------------------------------------------------------------------------
# Shared training utilities (also used by train_stage2.py)
# ---------------------------------------------------------------------------------------------
def cosine_lr(step: int, total: int, base: float, warmup: int, final_ratio: float = 0.01) -> float:
    """Linear warm-up to ``base`` over ``warmup`` steps, then cosine decay to ``final_ratio * base`` at ``total``."""
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


_NO_DECAY = ("bias", "norm", "pos_embed", "cls_token", "embedding", "null_embedding")


def param_groups(named: Iterable[tuple], weight_decay: float) -> List[Dict[str, Any]]:
    """AdamW groups: no weight decay for biases, norms, embeddings and position/[CLS] tokens."""
    decay, no_decay = [], []
    for name, p in named:
        if not p.requires_grad:
            continue
        (no_decay if any(key in name for key in _NO_DECAY) or p.dim() < 2 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class RunLog:
    """Console lines plus one JSON record per event in ``<out>.log.jsonl``."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __call__(self, kind: str, text: str, **record: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.time(), "kind": kind, **record}, default=float) + "\n")


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def autocast(device: torch.device, enabled: bool) -> Any:
    """bf16 autocast on CUDA when enabled, else a no-op context."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def data_summary(paths: Sequence[str]) -> List[Dict[str, Any]]:
    return [{"file": os.path.basename(p), "bytes": os.path.getsize(p)} for p in paths]


# ---------------------------------------------------------------------------------------------
# Stage (i)
# ---------------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aero-VPA stage (i): JEPA training of E_psi and P_omega (Eq. 8).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("data", nargs="*", help="demonstration files or glob patterns (quote the pattern)")
    p.add_argument("--out", default="checkpoints/stage1.pt", help="stage-1 checkpoint to write")
    p.add_argument("--val", nargs="*", default=None, help="validation files (e.g. validation worlds); "
                   "default: a hash split of the training files with --val_fraction")
    p.add_argument("--val_fraction", type=float, default=0.05, help="held-out episode fraction when --val is not given")
    p.add_argument("--split_seed", type=int, default=0, help="seed of the episode split (use the same in stage ii)")
    p.add_argument("--epochs", type=float, default=20.0, help="passes over the training frames")
    p.add_argument("--steps", type=int, default=0, help="optimiser steps (overrides --epochs when > 0)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=min(12, max(0, (os.cpu_count() or 2) - 2)))
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true", help="disable bf16 autocast on CUDA")
    p.add_argument("--compile", action="store_true", help="torch.compile the encoder (faster on recent GPUs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--save_minutes", type=float, default=20.0, help="training-state save interval")
    p.add_argument("--resume", action="store_true", help="continue from <out>.state.pt")
    p.add_argument("--include_failed", action="store_true", help="also train on unsuccessful episodes")
    p.add_argument("--residual_predictor", action="store_true",
                   help="residual heads: z_hat^(i) = z_t + f^(i)(...), same parameters (config predictor_residual)")
    p.add_argument("--residual_init", type=float, default=1.0,
                   help="factor on the default init of the residual output layer (0: start at persistence)")
    p.add_argument("--self-test", dest="self_test", action="store_true", help="run the self-test and exit")
    args = p.parse_args(argv)
    if not args.self_test and not args.data:
        p.error("give demonstration files (or --self-test)")
    return args


def _batch_images(batch: Dict[str, Any], prefix: str, normalizer: nn.Module, device: torch.device) -> Tensor:
    rgb = batch[f"{prefix}rgb"].to(device, non_blocking=True)
    depth = batch[f"{prefix}depth"].to(device, non_blocking=True)
    return normalizer(rgb, depth).image


def _ensemble_sigma(heads: Tensor, gamma: float) -> Tensor:
    """Eq. 22 with an explicit scale (the predictor's own buffer is set only when stage (i) ends)."""
    mean = heads.mean(dim=0, keepdim=True)
    return (heads - mean).pow(2).sum(dim=-1).mean(dim=0).sqrt() / (math.sqrt(heads.shape[-1]) * gamma)


def _effective_rank(z: Tensor) -> float:
    """exp(entropy of the normalised singular values of the centred batch): 1 (collapsed) ... d."""
    s = torch.linalg.svdvals((z - z.mean(dim=0, keepdim=True)).double())
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


@torch.no_grad()
def evaluate_stage1(encoder: nn.Module, target: nn.Module, predictor: nn.Module, loss_fn: nn.Module,
                    normalizer: nn.Module, loader: DataLoader, gamma: float, device: torch.device,
                    amp: bool) -> Dict[str, float]:
    """Loss terms and collapse diagnostics on a fixed held-out subset."""
    was_training = encoder.training, predictor.training
    encoder.eval()
    predictor.eval()
    sums: Dict[str, float] = {"inv": 0.0, "var": 0.0, "cov": 0.0, "copy": 0.0, "copy_online": 0.0, "gap": 0.0}
    n_batches, zs, sigmas = 0, [], []
    for batch in loader:
        img = _batch_images(batch, "", normalizer, device)
        fut = _batch_images(batch, "future_", normalizer, device)
        u = batch["primitive"].to(device)
        a_bar = batch["a_bar"].to(device)
        with autocast(device, amp):
            z = encoder(img)
            z_next = target(fut)
            z_now = target(img)
            heads = predictor.forward_heads(z, u, a_bar)
        z, z_next, z_now, heads = z.float(), z_next.float(), z_now.float(), heads.float()
        if z.shape[0] >= 2:
            sums["var"] += float(loss_fn.variance(z))
            sums["cov"] += float(loss_fn.covariance(z))
        sums["inv"] += float(loss_fn.invariance(heads, z_next))
        sums["copy"] += float((z_now - z_next).pow(2).sum(dim=-1).mean())          # persistence, target space
        sums["copy_online"] += float((z - z_next).pow(2).sum(dim=-1).mean())       # persistence of the input z_t
        sums["gap"] += float((z - z_now).pow(2).sum(dim=-1).mean())                # EMA lag of the target
        zs.append(z.cpu())
        sigmas.append(_ensemble_sigma(heads, gamma).cpu())
        n_batches += 1
    encoder.train(was_training[0])
    predictor.train(was_training[1])
    if n_batches == 0:
        return {}
    out = {k: v / n_batches for k, v in sums.items()}
    out["total"] = out["inv"] + loss_fn.lambda_v * out["var"] + loss_fn.lambda_c * out["cov"]
    out["inv_over_copy"] = out["inv"] / max(out["copy"], 1e-12)
    out["inv_over_copy_online"] = out["inv"] / max(out["copy_online"], 1e-12)
    z_all = torch.cat(zs)[:4096]
    std = z_all.std(dim=0) if z_all.shape[0] >= 2 else torch.zeros(z_all.shape[1])
    sigma = torch.cat(sigmas)
    out.update(latent_std_mean=float(std.mean()), latent_std_min=float(std.min()),
               dims_std_below_0p1=int((std < 0.1).sum()), effective_rank=_effective_rank(z_all) if z_all.shape[0] >= 2 else 1.0,
               sigma_mean=float(sigma.mean()), sigma_p90=float(torch.quantile(sigma, 0.9)), frames=int(z_all.shape[0]))
    return out


def _datasets(args: argparse.Namespace, cfg: VPAConfig) -> tuple:
    only_ok = not args.include_failed
    if args.val:
        train_ds = AeroVPADataset(args.data, 1, cfg, only_successful=only_ok, normalize=False)
        val_ds = AeroVPADataset(args.val, 1, cfg, only_successful=only_ok, normalize=False)
    else:
        train_f, val_f = episode_split(args.val_fraction, args.split_seed)
        train_ds = AeroVPADataset(args.data, 1, cfg, only_successful=only_ok, episode_filter=train_f, normalize=False)
        try:
            val_ds = AeroVPADataset(args.data, 1, cfg, only_successful=only_ok, episode_filter=val_f, normalize=False) \
                if args.val_fraction > 0 else None
        except ValueError:                                      # no held-out episode (tiny datasets)
            val_ds = None
    return train_ds, val_ds


def train_stage1(args: argparse.Namespace) -> str:
    """Run stage (i); returns the path of the stage-1 checkpoint."""
    device = pick_device(args.device)
    amp = (not args.no_amp) and device.type == "cuda"
    set_seed(args.seed)
    cfg = VPAConfig(predictor_residual=bool(args.residual_predictor), predictor_residual_init=float(args.residual_init))
    log = RunLog(args.out + ".log.jsonl")
    train_ds, val_ds = _datasets(args, cfg)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None)
    if len(loader) == 0:
        raise ValueError(f"{len(train_ds)} training samples is fewer than one batch of {args.batch_size}")
    val_loader = None
    if val_ds is not None:
        rng = np.random.default_rng(1234)
        subset = rng.choice(len(val_ds), size=min(len(val_ds), args.val_batches * args.batch_size), replace=False)
        val_loader = DataLoader(Subset(val_ds, np.sort(subset).tolist()), batch_size=args.batch_size, shuffle=False,
                                num_workers=min(args.num_workers, 4), pin_memory=device.type == "cuda")
    total = args.steps if args.steps > 0 else max(1, math.ceil(args.epochs * len(loader)))

    encoder, predictor = build_stage1_modules(cfg)
    target = cfg.build_momentum_encoder(encoder)
    tracker = cfg.build_latent_scale_tracker()
    loss_fn = cfg.build_vicreg_loss()
    normalizer = cfg.build_rgbd_normalizer()
    for module in (encoder, predictor, target, tracker, normalizer):
        module.to(device)
    named = list(encoder.named_parameters(prefix="encoder")) + list(predictor.named_parameters(prefix="predictor"))
    optimizer = torch.optim.AdamW(param_groups(named, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    step = 0
    state_path = args.out + ".state.pt"
    if args.resume:
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"--resume: no training state {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=True)
        saved_cfg = config_from_dict(state["config"]) if "config" in state else VPAConfig()
        mode = lambda c: (c.predictor_residual, c.predictor_residual_init if c.predictor_residual else None)  # noqa: E731
        if mode(saved_cfg) != mode(cfg):
            raise ValueError(f"--resume: {state_path} was trained with predictor_residual="
                             f"{saved_cfg.predictor_residual} (init {saved_cfg.predictor_residual_init}); "
                             "pass the same --residual_predictor/--residual_init")
        encoder.load_state_dict(state["encoder"])
        predictor.load_state_dict(state["predictor"])
        target.load_state_dict(state["target"])
        tracker.load_state_dict(state["tracker"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        log("resume", f"resumed from {state_path} at step {step}", step=step)
    forward_encoder = torch.compile(encoder) if args.compile else encoder

    def save_state() -> None:
        atomic_torch_save({"step": step, "encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                           "target": target.state_dict(), "tracker": tracker.state_dict(),
                           "optimizer": optimizer.state_dict(), "args": vars(args), "config": config_to_dict(cfg)},
                          state_path)

    log("start", f"stage (i): {len(train_ds)} training samples from {len(train_ds.episodes)} episodes in "
        f"{len(train_ds.paths)} files; {len(val_ds) if val_ds else 0} validation samples; {total} steps of "
        f"{args.batch_size} ({total * args.batch_size / len(train_ds):.1f} epochs) on {device}"
        f"{' with bf16' if amp else ''}"
        f"{f'; residual predictor (init x{cfg.predictor_residual_init:g})' if cfg.predictor_residual else ''}",
        samples=len(train_ds), steps=total, device=str(device), predictor_residual=cfg.predictor_residual)
    encoder.train()
    predictor.train()
    last_save, t_log, wait, seen = time.time(), time.time(), 0.0, 0
    terms: Dict[str, Tensor] = {}
    loss = torch.zeros(())
    gamma = float("nan")
    while step < total:
        t_fetch = time.time()
        for batch in loader:
            wait += time.time() - t_fetch
            img = _batch_images(batch, "", normalizer, device)
            fut = _batch_images(batch, "future_", normalizer, device)
            u = batch["primitive"].to(device, non_blocking=True)
            a_bar = batch["a_bar"].to(device, non_blocking=True)
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(step, total, args.lr, args.warmup_steps)
            with autocast(device, amp):
                z = forward_encoder(img)
                heads = predictor.forward_heads(z, u, a_bar)
                with torch.no_grad():
                    z_next = target(fut)
            loss, terms = loss_fn(z.float(), heads.float(), z_next.float())
            if not torch.isfinite(loss):
                save_state()
                raise FloatingPointError(f"non-finite loss at step {step}: {terms}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), args.grad_clip)
            optimizer.step()
            target.update(encoder)
            gamma = tracker.update(z.detach().float())
            step += 1
            seen += img.shape[0]
            if step % args.log_every == 0 or step == total:
                dt = time.time() - t_log
                loss_value = loss.detach().item()
                log("train", f"step {step}/{total}  loss {loss_value:.3f} (inv {float(terms['invariance']):.3f} "
                    f"var {float(terms['variance']):.4f} cov {float(terms['covariance']):.4f})  gamma_bar {gamma:.4f}  "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}  {seen / dt:.0f} samples/s  data-wait {100 * wait / dt:.0f}%",
                    step=step, loss=loss_value, **{k: float(v) for k, v in terms.items()}, gamma_bar=gamma,
                    lr=optimizer.param_groups[0]["lr"], samples_per_s=seen / dt, data_wait=wait / dt)
                t_log, wait, seen = time.time(), 0.0, 0
            if val_loader is not None and (step % args.val_every == 0 or step == total):
                metrics = evaluate_stage1(encoder, target, predictor, loss_fn, normalizer, val_loader, gamma, device, amp)
                log("val", f"val @ {step}: total {metrics['total']:.3f} inv {metrics['inv']:.3f} "
                    f"(no change: from z_t {metrics['copy_online']:.3f}, ratio {metrics['inv_over_copy_online']:.3f}; "
                    f"in target space {metrics['copy']:.3f}; EMA gap {metrics['gap']:.3f})  "
                    f"latent std mean {metrics['latent_std_mean']:.3f} min {metrics['latent_std_min']:.3f}  "
                    f"eff. rank {metrics['effective_rank']:.1f}/{cfg.latent_dim}  sigma mean {metrics['sigma_mean']:.4f}",
                    step=step, **metrics)
            if time.time() - last_save > 60.0 * args.save_minutes:
                save_state()
                last_save = time.time()
            if step >= total:
                break
            t_fetch = time.time()
    save_state()

    # ---- freeze: (gamma*, mu_Z) from the Eq. 11 tracker; P_omega gets gamma* for Eq. 22
    gamma_star, mu_z = tracker.freeze()
    predictor.set_latent_scale(gamma_star)
    final_val = evaluate_stage1(encoder, target, predictor, loss_fn, normalizer, val_loader, gamma_star, device, amp) \
        if val_loader is not None else {}
    provenance = {"stage": "i", "steps": step, "batch_size": args.batch_size, "lr": args.lr,
                  "epochs": step * args.batch_size / len(train_ds), "train_samples": len(train_ds),
                  "train_episodes": len(train_ds.episodes), "train_files": data_summary(train_ds.paths),
                  "val_samples": len(val_ds) if val_ds else 0, "val_metrics": final_val,
                  "split": {"val": args.val, "val_fraction": args.val_fraction, "split_seed": args.split_seed},
                  "final_train_loss": loss.detach().item(), **environment_info()}
    ckpt = make_checkpoint("stage1", cfg, {"vision_encoder": encoder, "predictor": predictor},
                           {"gamma_bar_star": gamma_star, "latent_mean": mu_z}, provenance)
    save_checkpoint(args.out, ckpt)
    log("done", f"stage (i) done: gamma* = {gamma_star:.4f}; wrote {args.out} (id {ckpt['checkpoint_id'][:8]})",
        gamma_bar_star=gamma_star, checkpoint=args.out, checkpoint_id=ckpt["checkpoint_id"], val=final_val)
    return args.out


# ---------------------------------------------------------------------------------------------
# Self-test helpers (also used by the other training scripts)
# ---------------------------------------------------------------------------------------------
def make_demo_data(directory: str, files: Dict[str, int], frames: int = 40, seed: int = 0) -> List[str]:
    """Tiny dry-run demonstration files ``{name: episodes}`` with ``frames`` frames per episode, all marked
    successful (a dry run cut by ``max_frames`` would otherwise be an unsuccessful episode)."""
    import h5py

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim"))
    import expert as E

    world = E.synthetic_forest(seed=seed, half_size=25.0)
    paths = []
    for k, (name, episodes) in enumerate(files.items()):
        path = os.path.join(directory, name)
        E.dry_run(path, world, E.DryRunConfig(num_episodes=episodes, seed=seed + k, max_frames=frames,
                                              spacing=(8.0, 10.0), num_waypoints=(3, 3), speed_limits=(5.0,)))
        with h5py.File(path, "r+") as f:
            for group in f.values():
                group.attrs["success"] = True
        paths.append(path)
    return paths


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        train_file, val_file = make_demo_data(tmp, {"train.hdf5": 2, "val.hdf5": 1})
        out = os.path.join(tmp, "ckpt", "stage1.pt")
        base = [train_file, "--val", val_file, "--out", out, "--batch_size", "8", "--num_workers", "0",
                "--device", "cpu", "--log_every", "1", "--val_every", "2", "--val_batches", "1", "--warmup_steps", "2"]
        train_stage1(parse_args(base + ["--steps", "3"]))
        ckpt = load_checkpoint(out, kind="stage1")
        gamma = float(ckpt["calibration"]["gamma_bar_star"])
        assert math.isfinite(gamma) and gamma > 0.0
        assert ckpt["calibration"]["latent_mean"].shape == (256,)
        assert float(ckpt["state_dict"]["predictor"]["latent_scale"]) == gamma
        assert ckpt["provenance"]["steps"] == 3 and ckpt["provenance"]["val_metrics"]["frames"] > 0
        records = [json.loads(line) for line in open(out + ".log.jsonl", encoding="utf-8")]
        assert [r["step"] for r in records if r["kind"] == "train"] == [1, 2, 3]
        assert any(r["kind"] == "val" for r in records)
        first_id = ckpt["checkpoint_id"]
        # resume continues from step 3 to step 5 with the saved optimiser, target and tracker
        train_stage1(parse_args(base + ["--steps", "5", "--resume"]))
        ckpt2 = load_checkpoint(out, kind="stage1")
        assert ckpt2["provenance"]["steps"] == 5 and ckpt2["checkpoint_id"] != first_id
        records = [json.loads(line) for line in open(out + ".log.jsonl", encoding="utf-8")]
        assert [r["step"] for r in records if r["kind"] == "train"][-2:] == [4, 5]
        # learning-rate schedule and parameter groups
        assert math.isclose(cosine_lr(0, 100, 1.0, 10), 0.1) and math.isclose(cosine_lr(9, 100, 1.0, 10), 1.0)
        assert math.isclose(cosine_lr(100, 100, 1.0, 10), 0.01) and cosine_lr(55, 100, 1.0, 10) < 1.0
        groups = param_groups([("blocks.0.fc.weight", nn.Parameter(torch.zeros(2, 2))),
                               ("blocks.0.fc.bias", nn.Parameter(torch.zeros(2))),
                               ("pos_embed", nn.Parameter(torch.zeros(1, 3, 2)))], 0.05)
        assert len(groups[0]["params"]) == 1 and len(groups[1]["params"]) == 2
        assert expand_paths(os.path.join(tmp, "*.hdf5")) == sorted([train_file, val_file])
        # residual heads: kept in the checkpoint and state config, rebuilt downstream, guarded on --resume
        from checkpoint import load_stage1_modules

        out_r = os.path.join(tmp, "ckpt", "stage1_res.pt")
        base_r = [arg if arg != out else out_r for arg in base]
        train_stage1(parse_args(base_r + ["--steps", "2", "--residual_predictor", "--residual_init", "0.5"]))
        ckpt_r = load_checkpoint(out_r, kind="stage1")
        assert ckpt_r["config"]["predictor_residual"] is True and ckpt_r["config"]["predictor_residual_init"] == 0.5
        assert load_stage1_modules(ckpt_r)[2].residual and not load_stage1_modules(ckpt2)[2].residual
        assert torch.load(out_r + ".state.pt", weights_only=True)["config"]["predictor_residual"] is True
        try:
            train_stage1(parse_args(base_r + ["--steps", "3", "--resume"]))             # flag left out
        except ValueError:
            pass
        else:
            raise AssertionError("--resume accepted a different predictor parameterisation")
        train_stage1(parse_args(base_r + ["--steps", "3", "--resume", "--residual_predictor", "--residual_init", "0.5"]))
        assert load_checkpoint(out_r, kind="stage1")["provenance"]["steps"] == 3
    print("train_stage1.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        train_stage1(parsed)

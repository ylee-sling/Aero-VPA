# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
train_stage2.py -- Stage (ii): policy learning of the primitive selector and the CFM solver
(preprint_261026.pdf, Secs. 3.3, 3.4, 3.8).

With E_psi and P_omega frozen and their outputs cached by cache_latents.py, trains pi_phi^h, the solver
(Embed and v_theta) and the learned null instruction embedding on

    L_pol(phi, theta) = L_CFM(theta) + lambda_h L_CE(phi)                                                (Eq. 30)
    L_CFM = E || v_theta(A^(rho), rho | e_t) - (A_t - xi) ||_F^2,   A^(rho) = rho A_t + (1 - rho) xi     (Eqs. 16, 17)
    L_CE  = -E log pi_phi^h(u*_t | z_t, g_t, S_t, c_text)                                              (Eq. 13)
    e_t   = Concat(Embed(u*_t), (z_t - mu_Z)/gamma*, (z_hat_{t+1} - mu_Z)/gamma*, (S_t - mu_S)/sigma_S, g_t, c_text)  (Eq. 15)

on expert chunks A_t (heading frame of their capture time) divided by the half-widths of the action box A_0.
--policy_state_inputs withholds groups of S_t from the selector and from e_t (they enter at their training mean;
P_omega still receives a_bar): e.g. "velocity attitude rates" leaves out the last command a_bar, which is within
a few cm/s of the expert's next command and lets a policy repeat it instead of using the image (input_sensitivity.py).
mu_S and sigma_S are computed here from the training frames and fixed before training; the selector reads
its inputs through the same standardisation. c_text is the learned null embedding for episodes without an
instruction and the frozen CLIP embedding otherwise (computed once per distinct instruction).

    python train_stage2.py checkpoints/stage1.pt "data/*.hdf5" --latents latents --out checkpoints/policy.pt
    python train_stage2.py ... --resume
    python train_stage2.py --self-test

All frames are held in memory (about 2.4 KB each; on the GPU when it fits), so a step is a random index draw
and needs no data loader. Validation (held-out episodes) reports L_CFM, L_CE, selector accuracy and recall
per primitive, and the error of chunks generated with K Euler steps (Eqs. 18, 19) against the expert, in
physical units. The written "policy" checkpoint still has sigma_lo = sigma_crit = None: run
calibrate_sigma.py next.

Choices the paper leaves open, as defaults here: AdamW (lr 3e-4, weight decay 0.01 except biases, norms and
embeddings), batch 1024, 1000 warm-up steps and cosine decay, gradient clipping at 1.0 per network, an exponential moving
average of the trained weights (decay 0.999, used for validation and saved; --ema 0 disables it), unweighted
cross-entropy (--class_weights inverse_sqrt to reweight rare primitives).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cache_latents import cache_is_current, cache_path, complete_episodes
from checkpoint import atomic_torch_save, build_policy_modules, config_from_dict, environment_info, load_checkpoint, \
    make_checkpoint, save_checkpoint
from config import PRIMITIVES, STATE_GROUPS
from dataset import episode_split, expand_paths
from train_stage1 import RunLog, cosine_lr, data_summary, param_groups, pick_device, set_seed

__all__ = ["LatentStore", "InstructionTable", "state_statistics", "evaluate_stage2", "train_stage2", "parse_args"]


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
class LatentStore:
    """Every frame of the selected episodes in memory: cached z_t and z_hat_{t+1}, S_t, g_t, A_t, u*_t and the
    instruction of the episode.

    Args:
        sources: demonstration files.
        latent_dir: directory of the caches written by cache_latents.py.
        checkpoint_id: id of the stage-1 checkpoint the caches must come from.
        episode_filter: optional ``f(path, name, attrs) -> bool``.
        only_successful: skip episodes whose ``success`` attribute is False.
    """

    FIELDS = ("z", "z_hat", "state", "goal", "actions", "primitive", "instruction", "episode")

    def __init__(self, sources: Sequence[str], latent_dir: str, checkpoint_id: str, episode_filter: Any = None,
                 only_successful: bool = True) -> None:
        chosen = []                                                          # (source, episode, frames, instruction)
        for source in sources:
            cache = cache_path(latent_dir, source)
            if not cache_is_current(cache, source, checkpoint_id):
                raise ValueError(f"{cache} is missing or was not built from stage-1 checkpoint {checkpoint_id[:8]} "
                                 f"and the current {os.path.basename(source)}: run cache_latents.py first")
            with h5py.File(source, "r") as f:
                for name in complete_episodes(f):
                    attrs = dict(f[name].attrs)
                    if only_successful and not bool(attrs.get("success", True)):
                        continue
                    if episode_filter is not None and not episode_filter(source, name, attrs):
                        continue
                    instruction = attrs.get("instruction", "")
                    instruction = instruction.decode() if isinstance(instruction, bytes) else str(instruction)
                    chosen.append((source, name, int(f[name]["timestamps"].shape[0]), instruction.strip()))
        if not chosen:
            raise ValueError("no usable episodes")
        n = sum(c[2] for c in chosen)
        self.instructions: List[str] = sorted({c[3] for c in chosen if c[3]})
        ids = {text: k + 1 for k, text in enumerate(self.instructions)}       # 0 = no instruction (null embedding)
        self.episode_names = [f"{os.path.basename(c[0])}:{c[1]}" for c in chosen]
        self.sources = sorted({c[0] for c in chosen})
        first = chosen[0]
        with h5py.File(cache_path(latent_dir, first[0]), "r") as cf:
            d = cf[first[1]]["z"].shape[1]
        arrays = {
            "z": np.empty((n, d), np.float32), "z_hat": np.empty((n, d), np.float32),
            "state": np.empty((n, 13), np.float32), "goal": np.empty((n, 4), np.float32),
            "actions": np.empty((n, 16, 4), np.float32), "primitive": np.empty(n, np.int64),
            "instruction": np.empty(n, np.int64), "episode": np.empty(n, np.int64),
        }
        row = 0
        for k, (source, name, frames, instruction) in enumerate(chosen):
            with h5py.File(source, "r") as f, h5py.File(cache_path(latent_dir, source), "r") as cf:
                ep, cep = f[name], cf[name]
                if cep["z"].shape[0] != frames:
                    raise ValueError(f"cache of {source}:{name} has {cep['z'].shape[0]} frames, the episode {frames}")
                sl = slice(row, row + frames)
                arrays["z"][sl] = cep["z"][...]
                arrays["z_hat"][sl] = cep["z_hat"][...]
                arrays["state"][sl] = ep["observations/state"][...]
                arrays["goal"][sl] = ep["observations/goal"][...]
                arrays["actions"][sl] = ep["actions"][...]
                arrays["primitive"][sl] = ep["primitives"][...]
                arrays["instruction"][sl] = ids.get(instruction, 0)
                arrays["episode"][sl] = k
            row += frames
        self.tensors: Dict[str, Tensor] = {key: torch.from_numpy(value) for key, value in arrays.items()}
        for key in ("z", "z_hat", "state", "goal", "actions"):
            if not bool(torch.isfinite(self.tensors[key]).all()):
                raise ValueError(f"non-finite values in {key}")

    def __len__(self) -> int:
        return int(self.tensors["z"].shape[0])

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def to(self, device: torch.device) -> "LatentStore":
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}
        return self

    @property
    def device(self) -> torch.device:
        return self.tensors["z"].device

    def batch(self, index: Tensor, device: torch.device) -> Dict[str, Tensor]:
        index = index.to(self.device)
        return {k: v[index].to(device, non_blocking=True) for k, v in self.tensors.items()}


def state_statistics(state: Tensor, eps: float = 1e-6) -> tuple:
    """(mu_S, sigma_S) per component; components that never vary get sigma = 1 (shift only)."""
    s = state.double()
    mean, std = s.mean(dim=0), s.std(dim=0, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean.float().cpu(), std.float().cpu()


class InstructionTable(nn.Module):
    """c_text per instruction id: 0 -> the learned null embedding, k >= 1 -> a frozen CLIP embedding (row k - 1).

    The same values as ``perception.TextEncoderWrapper`` (CLIP is frozen and deterministic), without running the
    text tower in every step.
    """

    def __init__(self, text_dim: int, frozen: Optional[Tensor] = None) -> None:
        super().__init__()
        self.null_embedding = nn.Parameter(torch.zeros(text_dim))
        self.register_buffer("frozen", torch.zeros(0, text_dim) if frozen is None else frozen.detach().float().clone())

    def forward(self, ids: Tensor) -> Tensor:
        table = torch.cat([self.null_embedding.unsqueeze(0), self.frozen.to(self.null_embedding.dtype)], dim=0)
        return table[ids]


def clip_embeddings(model_name: str, instructions: Sequence[str]) -> Tensor:
    """Frozen CLIP embeddings of the distinct instructions (needs ``transformers`` and the model weights)."""
    from perception import TextEncoderWrapper

    encoder = TextEncoderWrapper(model_name)
    with torch.no_grad():
        return encoder(list(instructions)).float().cpu()


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aero-VPA stage (ii): selector and CFM solver on cached latents (Eq. 30).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("stage1", nargs="?", help="stage-1 checkpoint (train_stage1.py)")
    p.add_argument("data", nargs="*", help="demonstration files or glob patterns")
    p.add_argument("--latents", default="latents", help="cache directory of cache_latents.py")
    p.add_argument("--out", default="checkpoints/policy.pt")
    p.add_argument("--val", nargs="*", default=None, help="validation files (default: hash split, as in stage i)")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--epochs", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=0, help="optimiser steps (overrides --epochs when > 0)")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay of the trained weights (0 = off)")
    p.add_argument("--class_weights", choices=["none", "inverse_sqrt"], default="none")
    p.add_argument("--policy_state_inputs", nargs="+", choices=list(STATE_GROUPS), default=None,
                   help="groups of S_t the selector and the CFM conditioning see (default: all four, Eq. 2)")
    p.add_argument("--device", default="auto")
    p.add_argument("--data_on_gpu", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument("--val_frames", type=int, default=20000)
    p.add_argument("--save_minutes", type=float, default=10.0)
    p.add_argument("--resume", action="store_true", help="continue from <out>.state.pt")
    p.add_argument("--include_failed", action="store_true")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if not args.self_test and (not args.stage1 or not args.data):
        p.error("give the stage-1 checkpoint and demonstration files (or --self-test)")
    return args


def _ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters(), strict=True):
            pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for be, b in zip(ema.buffers(), model.buffers(), strict=True):
            be.copy_(b)


def _losses(policy: nn.ModuleDict, batch: Dict[str, Tensor], lambda_h: float, class_weight: Optional[Tensor],
            generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
    selector, solver, text = policy["selector"], policy["solver"], policy["text"]
    c_text = text(batch["instruction"])
    u = batch["primitive"]
    e_t = solver.build_conditioning(u, batch["z"], batch["z_hat"], batch["state"], batch["goal"], c_text)
    cfm = solver.flow_matching_loss(solver.normalize_actions(batch["actions"]), e_t, generator=generator)
    logits = selector(batch["z"], batch["goal"], batch["state"], c_text)
    ce = F.cross_entropy(logits, u, weight=class_weight)
    return {"total": cfm + lambda_h * ce, "cfm": cfm, "ce": ce, "logits": logits, "e_t": e_t}


@torch.no_grad()
def evaluate_stage2(policy: nn.ModuleDict, store: "LatentStore", index: Tensor, lambda_h: float, batch_size: int,
                    device: torch.device, seed: int = 0) -> Dict[str, Any]:
    """Validation losses, selector accuracy and generated-chunk errors (physical units) on ``index``."""
    policy.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    n_cls = policy["selector"].num_primitives
    sums = {"cfm": 0.0, "ce": 0.0}
    correct, count = 0, 0
    confusion = torch.zeros(n_cls, n_cls, dtype=torch.long)
    sq = torch.zeros(4, dtype=torch.float64)
    sq0 = torch.zeros(4, dtype=torch.float64)
    for start in range(0, len(index), batch_size):
        batch = store.batch(index[start:start + batch_size], device)
        out = _losses(policy, batch, lambda_h, None, generator=gen)
        b = batch["z"].shape[0]
        sums["cfm"] += float(out["cfm"]) * b
        sums["ce"] += float(out["ce"]) * b
        pred = out["logits"].argmax(dim=-1)
        correct += int((pred == batch["primitive"]).sum())
        count += b
        confusion += torch.bincount((batch["primitive"] * n_cls + pred).cpu(), minlength=n_cls * n_cls).view(n_cls, n_cls)
        chunk = policy["solver"].generate_chunk(out["e_t"], generator=gen)            # K Euler steps, physical
        err = (chunk - batch["actions"]).double()
        sq += err.pow(2).sum(dim=(0, 1)).cpu()
        sq0 += err[:, 0].pow(2).sum(dim=0).cpu()
    policy.train()
    knots = count * policy["solver"].horizon
    rmse = (sq / max(knots, 1)).sqrt().tolist()
    rmse0 = (sq0 / max(count, 1)).sqrt().tolist()
    support = confusion.sum(dim=1)
    recall = {PRIMITIVES[k]: (float(confusion[k, k]) / int(support[k]) if int(support[k]) else None) for k in range(n_cls)}
    out = {"cfm": sums["cfm"] / max(count, 1), "ce": sums["ce"] / max(count, 1), "accuracy": correct / max(count, 1),
           "recall": recall, "support": {PRIMITIVES[k]: int(support[k]) for k in range(n_cls)}, "frames": count,
           "rmse": dict(zip(("vx", "vy", "vz", "yaw_rate"), rmse, strict=True)),
           "rmse_knot0": dict(zip(("vx", "vy", "vz", "yaw_rate"), rmse0, strict=True))}
    out["total"] = out["cfm"] + lambda_h * out["ce"]
    return out


def _stores(args: argparse.Namespace, ckpt_id: str) -> tuple:
    only_ok = not args.include_failed
    data = expand_paths(args.data)
    if args.val:
        return (LatentStore(data, args.latents, ckpt_id, only_successful=only_ok),
                LatentStore(expand_paths(args.val), args.latents, ckpt_id, only_successful=only_ok))
    train_f, val_f = episode_split(args.val_fraction, args.split_seed)
    train = LatentStore(data, args.latents, ckpt_id, episode_filter=train_f, only_successful=only_ok)
    val = None
    if args.val_fraction > 0:
        try:
            val = LatentStore(data, args.latents, ckpt_id, episode_filter=val_f, only_successful=only_ok)
        except ValueError:
            val = None
    return train, val


def train_stage2(args: argparse.Namespace) -> str:
    """Run stage (ii); returns the path of the "policy" checkpoint."""
    device = pick_device(args.device)
    set_seed(args.seed)
    log = RunLog(args.out + ".log.jsonl")
    stage1 = load_checkpoint(args.stage1)
    overrides: Dict[str, Any] = {"sigma_lo": None, "sigma_crit": None}
    if args.policy_state_inputs:
        overrides["policy_state_inputs"] = tuple(args.policy_state_inputs)
    cfg = config_from_dict(stage1["config"], **overrides)
    gamma = float(stage1["calibration"]["gamma_bar_star"])
    mu_z = stage1["calibration"]["latent_mean"].float()
    train, val = _stores(args, stage1["checkpoint_id"])
    mu_s, sd_s = state_statistics(train.tensors["state"])
    instructions = sorted(set(train.instructions) | set(val.instructions if val else []))
    frozen = clip_embeddings(cfg.clip_model_name, instructions) if instructions else None
    # instruction ids of each store refer to its own list: remap to the union
    for store in (train, val):
        if store is not None and store.instructions:
            remap = torch.tensor([0] + [instructions.index(t) + 1 for t in store.instructions])
            store.tensors["instruction"] = remap[store.tensors["instruction"]]
    on_gpu = device.type == "cuda" and (args.data_on_gpu == "yes" or (
        args.data_on_gpu == "auto" and (train.nbytes + (val.nbytes if val else 0)) < 0.5 * torch.cuda.mem_get_info()[0]))
    if on_gpu:
        train.to(device)
        if val is not None:
            val.to(device)

    selector, solver = build_policy_modules(cfg)
    solver.set_standardization_stats(mu_z, gamma, mu_s, sd_s)
    selector.set_input_standardization(mu_z, gamma, mu_s, sd_s)
    policy = nn.ModuleDict({"selector": selector, "solver": solver, "text": InstructionTable(cfg.text_dim, frozen)})
    policy.to(device).train()
    ema = copy.deepcopy(policy).requires_grad_(False) if args.ema > 0 else None
    optimizer = torch.optim.AdamW(param_groups(policy.named_parameters(), args.weight_decay), lr=args.lr,
                                  betas=(0.9, 0.95))
    counts = torch.bincount(train.tensors["primitive"].cpu(), minlength=cfg.num_primitives).double()
    class_weight = None
    if args.class_weights == "inverse_sqrt":
        w = 1.0 / counts.clamp_min(1.0).sqrt()
        class_weight = (w * cfg.num_primitives / w.sum()).float().to(device)
    total = args.steps if args.steps > 0 else max(1, math.ceil(args.epochs * len(train) / args.batch_size))
    step = 0
    state_path = args.out + ".state.pt"
    if args.resume:
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"--resume: no training state {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=True)
        saved_inputs = (state.get("args") or {}).get("policy_state_inputs")
        if list(saved_inputs or []) != list(args.policy_state_inputs or []):
            raise SystemExit(f"--resume: {state_path} was trained with --policy_state_inputs {saved_inputs}, "
                             f"not {args.policy_state_inputs}")
        policy.load_state_dict(state["policy"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        log("resume", f"resumed from {state_path} at step {step}", step=step)

    def save_state() -> None:
        atomic_torch_save({"step": step, "policy": policy.state_dict(), "ema": ema.state_dict() if ema else None,
                           "optimizer": optimizer.state_dict(), "args": vars(args)}, state_path)

    val_index = None
    if val is not None:
        rng = np.random.default_rng(1234)
        val_index = torch.from_numpy(np.sort(rng.choice(len(val), size=min(len(val), args.val_frames), replace=False)))
    log("start", f"stage (ii): {len(train)} training frames from {len(train.episode_names)} episodes, "
        f"{len(val) if val else 0} validation frames; {total} steps of {args.batch_size} "
        f"({total * args.batch_size / len(train):.1f} epochs) on {device}, data on "
        f"{'GPU' if on_gpu else 'CPU'} ({(train.nbytes + (val.nbytes if val else 0)) / 1e9:.2f} GB); "
        f"{len(instructions)} distinct instructions; primitive counts {counts.long().tolist()}; S_t groups seen by "
        f"the selector and e_t: {', '.join(cfg.policy_state_inputs) or 'none'}",
        frames=len(train), steps=total, instructions=instructions, primitive_counts=counts.long().tolist(),
        policy_state_inputs=list(cfg.policy_state_inputs))
    gen = torch.Generator(device=train.device).manual_seed(args.seed)
    last_save, t_log = time.time(), time.time()
    out: Dict[str, Tensor] = {}
    while step < total:
        idx = torch.randint(len(train), (args.batch_size,), generator=gen, device=train.device)
        batch = train.batch(idx, device)
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, total, args.lr, args.warmup_steps)
        out = _losses(policy, batch, cfg.lambda_h, class_weight)
        if not torch.isfinite(out["total"]):
            save_state()
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        out["total"].backward()
        if args.grad_clip > 0:                       # per network: the CFM gradients must not throttle the selector's
            for module in policy.values():
                nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip)
        optimizer.step()
        if ema is not None:
            _ema_update(ema, policy, args.ema)
        step += 1
        if step % args.log_every == 0 or step == total:
            dt = time.time() - t_log
            acc = float((out["logits"].argmax(-1) == batch["primitive"]).float().mean())
            log("train", f"step {step}/{total}  L_pol {out['total'].item():.4f} (L_CFM {out['cfm'].item():.4f}  "
                f"L_CE {out['ce'].item():.4f}, acc {acc:.3f})  lr {optimizer.param_groups[0]['lr']:.2e}  "
                f"{args.log_every * args.batch_size / max(dt, 1e-9):.0f} frames/s",
                step=step, total=out["total"].item(), cfm=out["cfm"].item(), ce=out["ce"].item(), accuracy=acc,
                lr=optimizer.param_groups[0]["lr"])
            t_log = time.time()
        if val_index is not None and (step % args.val_every == 0 or step == total):
            metrics = evaluate_stage2(ema if ema is not None else policy, val, val_index, cfg.lambda_h,
                                      args.batch_size, device)
            r = metrics["rmse"]
            log("val", f"val @ {step}: L_CFM {metrics['cfm']:.4f}  L_CE {metrics['ce']:.4f}  accuracy "
                f"{metrics['accuracy']:.3f}  chunk RMSE vx {r['vx']:.3f} vy {r['vy']:.3f} vz {r['vz']:.3f} m/s, "
                f"yaw rate {r['yaw_rate']:.3f} rad/s", step=step, **metrics)
        if time.time() - last_save > 60.0 * args.save_minutes:
            save_state()
            last_save = time.time()
    save_state()

    final = ema if ema is not None else policy
    final_val = evaluate_stage2(final, val, val_index, cfg.lambda_h, args.batch_size, device) if val_index is not None else {}
    states = {"vision_encoder": stage1["state_dict"]["vision_encoder"], "predictor": stage1["state_dict"]["predictor"],
              "selector": final["selector"], "solver": final["solver"],
              "text_null_embedding": final["text"].null_embedding.detach()}
    calibration = {"gamma_bar_star": gamma, "latent_mean": mu_z, "proprio_mean": mu_s, "proprio_std": sd_s}
    provenance = {"stage": "ii", "stage1_checkpoint_id": stage1["checkpoint_id"], "steps": step,
                  "batch_size": args.batch_size, "lr": args.lr, "ema": args.ema, "class_weights": args.class_weights,
                  "epochs": step * args.batch_size / len(train), "train_frames": len(train),
                  "train_episodes": len(train.episode_names), "train_files": data_summary(train.sources),
                  "val_frames": len(val) if val else 0, "val_metrics": final_val, "instructions": instructions,
                  "primitive_counts": counts.long().tolist(), "stage1_provenance": stage1["provenance"],
                  "split": {"val": args.val, "val_fraction": args.val_fraction, "split_seed": args.split_seed},
                  **environment_info()}
    ckpt = make_checkpoint("policy", cfg, states, calibration, provenance)
    save_checkpoint(args.out, ckpt)
    log("done", f"stage (ii) done: wrote {args.out} (id {ckpt['checkpoint_id'][:8]}); next: calibrate_sigma.py",
        checkpoint=args.out, checkpoint_id=ckpt["checkpoint_id"], val=final_val)
    return args.out


# ---------------------------------------------------------------------------------------------
# Self-test (run: python train_stage2.py --self-test)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import json
    import tempfile

    from cache_latents import cache_latents, parse_args as cache_args
    from pipeline import Observation, VPAInferencePipeline
    from train_stage1 import make_demo_data, parse_args as stage1_args, train_stage1

    # instruction table: id 0 is the trainable null embedding, ids >= 1 are frozen rows
    table = InstructionTable(4, torch.arange(8.0).view(2, 4))
    out = table(torch.tensor([0, 1, 2, 0]))
    assert torch.equal(out[1], torch.arange(4.0)) and torch.equal(out[0], torch.zeros(4))
    out.sum().backward()
    assert torch.equal(table.null_embedding.grad, torch.full((4,), 2.0)) and not table.frozen.requires_grad

    with tempfile.TemporaryDirectory() as tmp:
        train_file, val_file = make_demo_data(tmp, {"train.hdf5": 2, "val.hdf5": 1}, frames=30)
        stage1_path = os.path.join(tmp, "stage1.pt")
        train_stage1(stage1_args([train_file, "--out", stage1_path, "--batch_size", "8", "--num_workers", "0",
                                  "--device", "cpu", "--val_fraction", "0", "--steps", "2", "--log_every", "100"]))
        latents = os.path.join(tmp, "latents")
        cache_latents(cache_args([stage1_path, train_file, val_file, "--out_dir", latents, "--num_workers", "0",
                                  "--device", "cpu"]))
        out_path = os.path.join(tmp, "policy.pt")
        base = [stage1_path, train_file, "--latents", latents, "--val", val_file, "--out", out_path,
                "--batch_size", "16", "--device", "cpu", "--log_every", "1", "--val_every", "2", "--warmup_steps", "1",
                "--ema", "0.5"]
        train_stage2(parse_args(base + ["--steps", "3"]))
        ckpt = load_checkpoint(out_path, kind="policy")
        cfg = config_from_dict(ckpt["config"])
        assert cfg.sigma_lo is None and cfg.sigma_crit is None and ckpt["provenance"]["instructions"] == []
        cal = ckpt["calibration"]
        store = LatentStore([train_file], latents, load_checkpoint(stage1_path)["checkpoint_id"])
        assert len(store) == 60 and store.tensors["z"].shape == (60, 256)
        mu_s, sd_s = state_statistics(store.tensors["state"])
        assert torch.allclose(cal["proprio_mean"], mu_s) and torch.allclose(cal["proprio_std"], sd_s)
        assert bool((cal["proprio_std"] > 0).all())
        sel = ckpt["state_dict"]["selector"]
        assert torch.allclose(sel["proprio_mean"], mu_s) and torch.allclose(sel["latent_mean"], cal["latent_mean"])
        assert float(sel["latent_scale"]) == float(cal["gamma_bar_star"])
        assert torch.allclose(ckpt["state_dict"]["solver"]["proprio_std"], sd_s)
        val_metrics = ckpt["provenance"]["val_metrics"]
        assert val_metrics["frames"] == 30 and 0.0 <= val_metrics["accuracy"] <= 1.0
        assert all(math.isfinite(v) for v in val_metrics["rmse"].values())
        # stage-1 networks are carried over unchanged
        s1 = load_checkpoint(stage1_path)
        assert all(torch.equal(s1["state_dict"]["vision_encoder"][k], v)
                   for k, v in ckpt["state_dict"]["vision_encoder"].items())
        # the closed loop runs from the checkpoint (thresholds passed until calibrate_sigma.py sets them)
        pipe = VPAInferencePipeline.from_checkpoint(out_path, latency_bound=0.05, tracking_lag=0.03, sigma_lo=1e3,
                                                    sigma_crit=2e3)
        pipe.reset(None, torch.tensor([[[10.0, 0.0, -2.0]]]))
        s_t = pipe.proprioception(torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 1.0]]), torch.zeros(1, 3), torch.zeros(1))
        act = pipe.act(0.0, observation=Observation(torch.randn(1, 4, 128, 160), s_t, torch.zeros(1, 3), torch.zeros(1),
                                                    torch.tensor([30.0]), 0.0))
        assert act.step_output is not None and act.step_output.chunk.shape == (1, 16, 4)
        assert torch.equal(pipe.text_encoder.null_embedding, ckpt["state_dict"]["text_null_embedding"])
        # resume
        first_id = ckpt["checkpoint_id"]
        train_stage2(parse_args(base + ["--steps", "5", "--resume"]))
        ckpt2 = load_checkpoint(out_path, kind="policy")
        assert ckpt2["provenance"]["steps"] == 5 and ckpt2["checkpoint_id"] != first_id
        records = [json.loads(line) for line in open(out_path + ".log.jsonl", encoding="utf-8")]
        assert [r["step"] for r in records if r["kind"] == "train"] == [1, 2, 3, 4, 5]
        # a cache from another stage-1 checkpoint is refused
        try:
            LatentStore([train_file], latents, "0" * 32)
            raise AssertionError("expected a stale-cache error")
        except ValueError:
            pass
        # --policy_state_inputs: the last command a_bar withheld from the selector and e_t, kept in the checkpoint
        masked_path = os.path.join(tmp, "policy_no_command.pt")
        masked = [stage1_path, train_file, "--latents", latents, "--out", masked_path, "--batch_size", "16",
                  "--device", "cpu", "--log_every", "100", "--val_fraction", "0", "--warmup_steps", "1", "--steps", "2",
                  "--policy_state_inputs", "velocity", "attitude", "rates"]
        train_stage2(parse_args(masked))
        mcfg = config_from_dict(load_checkpoint(masked_path, kind="policy")["config"])
        assert mcfg.policy_state_inputs == ("velocity", "attitude", "rates") and cfg.policy_state_inputs == tuple(STATE_GROUPS)
        mpipe = VPAInferencePipeline.from_checkpoint(masked_path, latency_bound=0.05, tracking_lag=0.03, sigma_lo=1e3,
                                                     sigma_crit=2e3)
        z = torch.randn(4, 256)
        g, c = torch.randn(4, 4), mpipe.text_encoder(None).expand(4, -1)
        s_a, s_b = torch.randn(4, 13), torch.randn(4, 13)
        s_b[:, :9] = s_a[:, :9]                                                   # differ in a_bar only
        with torch.no_grad():
            assert torch.equal(mpipe.selector(z, g, s_a, c), mpipe.selector(z, g, s_b, c))
            u = torch.zeros(4, dtype=torch.long)
            assert torch.equal(mpipe.solver.build_conditioning(u, z, z, s_a, g, c),
                               mpipe.solver.build_conditioning(u, z, z, s_b, g, c))
        try:
            train_stage2(parse_args(masked[:-4] + ["--steps", "3", "--resume"]))   # without --policy_state_inputs
            raise AssertionError("resumed with other --policy_state_inputs")
        except SystemExit as exc:
            assert "--policy_state_inputs" in str(exc), exc
    print("train_stage2.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        train_stage2(parsed)

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
checkpoint.py -- One-file checkpoints of Aero-VPA (preprint_261026.pdf, Secs. 3.5 and 3.8).

A checkpoint is ``torch.save`` of one dict of tensors and plain Python values, so it loads with
``torch.load(..., weights_only=True)``:

    format           "aero-vpa/checkpoint"
    format_version   1
    kind             "stage1"  E_psi and P_omega after stage (i), with gamma* and mu_Z
                     "policy"  every network after stage (ii), with mu_S and sigma_S as well
    checkpoint_id    uuid4 hex, new on every save; the latent cache records the stage-1 id it was built from
    config           the ``VPAConfig`` fields; sigma_lo and sigma_crit stay None until calibrate_sigma.py
    state_dict       "vision_encoder", "predictor"                                         (both kinds)
                     "selector", "solver", "text_null_embedding"                           ("policy")
    calibration      "gamma_bar_star", "latent_mean"                  frozen at the end of stage (i) (Eq. 11)
                     "proprio_mean", "proprio_std"                    dataset statistics of S_t ("policy")
    provenance       free-form record: data, steps, metrics, versions, the sigma calibration report

The EMA target encoder and optimiser state are not part of a checkpoint: they live in the training-state
file (``<checkpoint>.state.pt``) that the training scripts write for resuming. The frozen CLIP text tower
is not stored either; only the learned null instruction embedding is.

``VPAInferencePipeline.from_checkpoint`` (pipeline.py) rebuilds the closed loop from a "policy"
checkpoint. ``python checkpoint.py`` runs the self-test.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import subprocess
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:  # package import
    from .config import VPAConfig
except ImportError:  # flat import (files side by side)
    from config import VPAConfig

__all__ = [
    "CHECKPOINT_FORMAT", "CHECKPOINT_VERSION", "KINDS", "config_to_dict", "config_from_dict", "make_checkpoint",
    "save_checkpoint", "load_checkpoint", "build_stage1_modules", "build_policy_modules", "load_stage1_modules",
    "null_text_encoder", "environment_info", "atomic_torch_save",
]

CHECKPOINT_FORMAT = "aero-vpa/checkpoint"
CHECKPOINT_VERSION = 1
KINDS: Tuple[str, ...] = ("stage1", "policy")
_REQUIRED_STATES = {
    "stage1": ("vision_encoder", "predictor"),
    "policy": ("vision_encoder", "predictor", "selector", "solver", "text_null_embedding"),
}
_REQUIRED_CALIBRATION = {
    "stage1": ("gamma_bar_star", "latent_mean"),
    "policy": ("gamma_bar_star", "latent_mean", "proprio_mean", "proprio_std"),
}


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
def config_to_dict(cfg: VPAConfig) -> Dict[str, Any]:
    """All ``VPAConfig`` fields as a plain dict (tuples stay tuples)."""
    return dataclasses.asdict(cfg)


def config_from_dict(values: Mapping[str, Any], **overrides: Any) -> VPAConfig:
    """``VPAConfig`` from a stored dict plus overrides; unknown keys are an error (a newer format)."""
    known = {f.name for f in dataclasses.fields(VPAConfig)}
    merged = {**dict(values), **overrides}
    unknown = sorted(set(merged) - known)
    if unknown:
        raise ValueError(f"unknown VPAConfig fields {unknown}: checkpoint from a newer version?")
    tuple_fields = {f.name for f in dataclasses.fields(VPAConfig) if isinstance(getattr(VPAConfig(), f.name), tuple)}
    for name in tuple_fields & set(merged):
        merged[name] = tuple(merged[name])
    return VPAConfig(**merged)


# ---------------------------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------------------------
def _cpu(value: Any) -> Any:
    """Plain, loadable values: detached CPU tensors (a checkpoint never pins a GPU), built-in str/int/float/bool
    instead of subclasses such as ``torch.__version__`` or NumPy scalars, which ``weights_only`` loading rejects."""
    if isinstance(value, Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {str(k): _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if hasattr(value, "item") and callable(value.item):        # NumPy scalars
        return _cpu(value.item())
    if hasattr(value, "tolist") and callable(value.tolist):    # NumPy arrays
        return _cpu(value.tolist())
    return str(value)


def make_checkpoint(kind: str, cfg: VPAConfig, states: Mapping[str, Any], calibration: Mapping[str, Any],
                    provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Assemble and validate a checkpoint dict. ``states`` maps names to modules, state dicts or tensors."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    out_states: Dict[str, Any] = {}
    for name, value in states.items():
        out_states[name] = value.state_dict() if isinstance(value, nn.Module) else value
    ckpt = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "kind": kind,
        "checkpoint_id": uuid.uuid4().hex,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config_to_dict(cfg),
        "state_dict": _cpu(out_states),
        "calibration": _cpu({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in calibration.items()}),
        "provenance": _cpu(dict(provenance or {})),
    }
    _validate(ckpt, kind)
    return ckpt


def _validate(ckpt: Mapping[str, Any], kind: Optional[str]) -> None:
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"not an Aero-VPA checkpoint (format {ckpt.get('format')!r})")
    if ckpt.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint format version {ckpt.get('format_version')} is not supported "
                         f"(this code reads version {CHECKPOINT_VERSION})")
    if ckpt.get("kind") not in KINDS:
        raise ValueError(f"unknown checkpoint kind {ckpt.get('kind')!r}")
    if kind is not None and ckpt["kind"] != kind:
        raise ValueError(f"expected a {kind!r} checkpoint, got {ckpt['kind']!r}")
    missing = [k for k in _REQUIRED_STATES[ckpt["kind"]] if k not in ckpt["state_dict"]]
    missing += [f"calibration.{k}" for k in _REQUIRED_CALIBRATION[ckpt["kind"]] if k not in ckpt["calibration"]]
    if missing:
        raise ValueError(f"checkpoint is missing {missing}")
    config_from_dict(ckpt["config"])                     # raises on unknown or invalid fields


def atomic_torch_save(obj: Any, path: str) -> None:
    """``torch.save`` to ``path`` via a temporary file and ``os.replace`` (never a half-written file)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_checkpoint(path: str, ckpt: Mapping[str, Any]) -> None:
    """Validate and write atomically (values reduced to plain, ``weights_only``-loadable types)."""
    _validate(ckpt, None)
    atomic_torch_save(_cpu(dict(ckpt)), path)


def load_checkpoint(path: str, kind: Optional[str] = None, map_location: Any = "cpu") -> Dict[str, Any]:
    """Read and validate a checkpoint (``weights_only=True``: no code is executed while loading)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path} does not contain a checkpoint dict")
    _validate(ckpt, kind)
    return ckpt


# ---------------------------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------------------------
def build_stage1_modules(cfg: VPAConfig) -> Tuple[nn.Module, nn.Module]:
    """(E_psi, P_omega) with the Table 1 dimensions of ``cfg``."""
    return cfg.build_vision_encoder(), cfg.build_predictor()


def build_policy_modules(cfg: VPAConfig) -> Tuple[nn.Module, nn.Module]:
    """(pi_phi^h, CFM solver) with the Table 1 dimensions of ``cfg``."""
    try:
        from .selector import PrimitiveSelector
        from .solver import CFMSolver
    except ImportError:
        from selector import PrimitiveSelector
        from solver import CFMSolver
    selector = PrimitiveSelector(
        latent_dim=cfg.latent_dim, goal_dim=cfg.goal_dim, proprio_dim=cfg.proprio_dim, text_dim=cfg.text_dim,
        num_primitives=cfg.num_primitives, hidden_dims=cfg.selector_hidden,
    )
    return selector, CFMSolver.from_config(cfg)


def load_stage1_modules(ckpt: Mapping[str, Any], device: Any = "cpu"
                        ) -> Tuple[VPAConfig, nn.Module, nn.Module, float, Tensor]:
    """Frozen (cfg, E_psi, P_omega, gamma*, mu_Z) from a "stage1" or "policy" checkpoint, in eval mode."""
    cfg = config_from_dict(ckpt["config"])
    encoder, predictor = build_stage1_modules(cfg)
    encoder.load_state_dict(ckpt["state_dict"]["vision_encoder"])
    predictor.load_state_dict(ckpt["state_dict"]["predictor"])
    gamma = float(ckpt["calibration"]["gamma_bar_star"])
    mu_z = ckpt["calibration"]["latent_mean"].to(torch.float32)
    predictor.set_latent_scale(gamma)
    for module in (encoder, predictor):
        module.to(device).eval().requires_grad_(False)
    return cfg, encoder, predictor, gamma, mu_z.to(device)


class _NoTextModel(nn.Module):
    """Stand-in text tower for missions without instructions: it refuses to encode any text."""

    def __init__(self, projection_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=int(projection_dim))

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Any:
        raise RuntimeError("this pipeline has no text encoder; use instructions None or '' "
                           "(or load the checkpoint with text_encoder='clip')")


def _no_tokenizer(texts: Sequence[str], **_: Any) -> Dict[str, Tensor]:
    raise RuntimeError("this pipeline has no text encoder; use instructions None or '' "
                       "(or load the checkpoint with text_encoder='clip')")


def null_text_encoder(text_dim: int = 512) -> nn.Module:
    """``TextEncoderWrapper`` that only provides the learned null embedding (no CLIP download)."""
    try:
        from .perception import TextEncoderWrapper
    except ImportError:
        from perception import TextEncoderWrapper
    return TextEncoderWrapper(text_model=_NoTextModel(text_dim), tokenizer=_no_tokenizer)


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------
def environment_info() -> Dict[str, Any]:
    """Versions, device and repository revision, recorded in every checkpoint."""
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        revision = subprocess.run(["git", "-C", root, "describe", "--always", "--dirty", "--tags"],
                                  capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    info = {"git_revision": revision, "python": platform.python_version(), "torch": str(torch.__version__),
            "host": platform.node(), "cuda_device": None}
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


# ---------------------------------------------------------------------------------------------
# Self-test (run: python checkpoint.py)
# ---------------------------------------------------------------------------------------------
def _expect(exc: type, fn: Any) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _self_test() -> None:
    import tempfile

    try:
        from .pipeline import VPAInferencePipeline
    except ImportError:
        from pipeline import VPAInferencePipeline

    torch.manual_seed(0)
    # ---- config round trip, overrides, unknown keys
    cfg = VPAConfig(sigma_lo=0.2, sigma_crit=0.7, noise_refresh_period=10)
    back = config_from_dict(config_to_dict(cfg))
    assert back == cfg and isinstance(back.image_size, tuple) and isinstance(back.selector_hidden, tuple)
    as_lists = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config_to_dict(cfg).items()}
    assert config_from_dict(as_lists) == cfg                                  # tuples restored
    assert config_from_dict(config_to_dict(cfg), sigma_crit=0.9).sigma_crit == 0.9
    _expect(ValueError, lambda: config_from_dict({**config_to_dict(cfg), "unknown_field": 1}))

    # ---- a full policy checkpoint from a pipeline's modules, saved and reloaded
    pipe_a = VPAInferencePipeline.from_config(cfg, null_text_encoder(cfg.text_dim), latency_bound=0.05, tracking_lag=0.03)
    with torch.no_grad():
        pipe_a.text_encoder.null_embedding.normal_()
    mu_z, gamma = torch.randn(cfg.latent_dim) * 0.1, 0.8
    mu_s, sd_s = torch.randn(cfg.proprio_dim) * 0.1, torch.rand(cfg.proprio_dim) + 0.5
    pipe_a.selector.set_input_standardization(mu_z, gamma, mu_s, sd_s)
    pipe_a.calibrate(gamma, mu_z, mu_s, sd_s)
    states = {"vision_encoder": pipe_a.vision_encoder, "predictor": pipe_a.predictor, "selector": pipe_a.selector,
              "solver": pipe_a.solver, "text_null_embedding": pipe_a.text_encoder.null_embedding.detach()}
    calibration = {"gamma_bar_star": gamma, "latent_mean": mu_z, "proprio_mean": mu_s, "proprio_std": sd_s}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "policy.pt")
        ckpt = make_checkpoint("policy", cfg, states, calibration, {"note": "self-test", **environment_info()})
        save_checkpoint(path, ckpt)
        assert not [f for f in os.listdir(tmp) if ".tmp-" in f]
        loaded = load_checkpoint(path, kind="policy")
        assert loaded["checkpoint_id"] == ckpt["checkpoint_id"] and len(loaded["checkpoint_id"]) == 32
        _expect(ValueError, lambda: load_checkpoint(path, kind="stage1"))
        cfg_l, enc_l, pred_l, gamma_l, mu_l = load_stage1_modules(loaded)
        assert cfg_l == cfg and gamma_l == gamma and torch.equal(mu_l, mu_z) and not enc_l.training
        assert all(not p.requires_grad for p in enc_l.parameters())

        pipe_b = VPAInferencePipeline.from_checkpoint(path, latency_bound=0.05, tracking_lag=0.03, text_encoder="null")
        for name in ("vision_encoder", "predictor", "selector", "solver"):
            sa, sb = getattr(pipe_a, name).state_dict(), getattr(pipe_b, name).state_dict()
            assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa), name
        assert torch.equal(pipe_a.text_encoder.null_embedding, pipe_b.text_encoder.null_embedding)

        # identical closed-loop decisions from the original and the reloaded pipeline
        waypoints = torch.tensor([[[10.0, 0.0, -2.0], [20.0, 5.0, -3.0]]])
        image = torch.randn(1, *cfg.frame_shape)
        outs = []
        for pipe in (pipe_a, pipe_b):
            pipe.reset(None, waypoints)
            s_t = pipe.proprioception(torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 1.0]]), torch.zeros(1, 3), torch.zeros(1))
            gen = torch.Generator().manual_seed(3)
            outs.append(pipe.step(image, s_t, torch.zeros(1, 3), torch.zeros(1), torch.tensor([30.0]), 0.0, generator=gen))
        assert torch.equal(outs[0].selected_primitive, outs[1].selected_primitive)
        assert torch.allclose(outs[0].sigma, outs[1].sigma) and torch.allclose(outs[0].z_hat, outs[1].z_hat)
        if outs[0].chunk is not None:
            assert torch.allclose(outs[0].chunk, outs[1].chunk)
        assert outs[0].num_sequential_evaluations == outs[1].num_sequential_evaluations

        # the stub text encoder refuses text; uncalibrated thresholds are refused with a clear error
        _expect(RuntimeError, lambda: pipe_b.reset("fly to the gate", waypoints))
        raw = load_checkpoint(path)
        raw["config"]["sigma_lo"] = raw["config"]["sigma_crit"] = None
        save_checkpoint(path, raw)
        _expect(ValueError, lambda: VPAInferencePipeline.from_checkpoint(path, latency_bound=0.05, tracking_lag=0.03,
                                                                         text_encoder="null"))
        pipe_c = VPAInferencePipeline.from_checkpoint(path, latency_bound=0.05, tracking_lag=0.03, text_encoder="null",
                                                      sigma_lo=0.1, sigma_crit=0.5)
        assert pipe_c.safety.sigma_crit == 0.5

        # format checks
        bad = dict(raw, format_version=99)
        _expect(ValueError, lambda: save_checkpoint(path, bad))
        stage1 = make_checkpoint("stage1", cfg, {"vision_encoder": pipe_a.vision_encoder, "predictor": pipe_a.predictor},
                                 {"gamma_bar_star": gamma, "latent_mean": mu_z})
        _expect(ValueError, lambda: make_checkpoint("stage1", cfg, {"vision_encoder": pipe_a.vision_encoder},
                                                    {"gamma_bar_star": gamma, "latent_mean": mu_z}))
        save_checkpoint(path, stage1)
        _expect(ValueError, lambda: VPAInferencePipeline.from_checkpoint(path, latency_bound=0.05, tracking_lag=0.03))
    print("checkpoint.py self-test passed")


if __name__ == "__main__":
    _self_test()

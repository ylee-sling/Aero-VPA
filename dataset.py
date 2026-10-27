# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
dataset.py -- PyTorch datasets over Aero-VPA expert demonstrations (preprint_261026.pdf, Secs. 3.2, 3.4, 3.8).

Reads the HDF5 files written by ``sim/collect_expert_dataset.py`` (schema in ``sim/expert.py``; check a file
with ``python inspect_hdf5.py``) and yields one decision step per item:

    stage 1 -- representation learning, L_JEPA (Eq. 8):
        image        [4, 128, 160] float32   I_t, normalised by perception.RGBDNormalizer
        future_image [4, 128, 160] float32   I_{t+1}: the frame captured closest to tau_t + T_nu (0.1 s)
        state        [13]          float32   S_t (Eq. 2)
        primitive    []            int64     u_t: the expert label u*_t (the selector is not trained yet)
        a_bar        [4]           float32   a_bar_t = S_t[9:13], the executing setpoint (Eq. 7)
        lookahead_s  []            float64   tau_{t+1} - tau_t (0.1 s at 50 Hz)

    stage 2 -- policy learning, L_pol = L_CFM + lambda_h L_CE (Eq. 30):
        image        [4, 128, 160] float32   I_t (encoded by the frozen E_psi into z_t, and P_omega into z_hat)
        state        [13]          float32   S_t (raw; CFMSolver.build_conditioning standardises it)
        goal         [4]           float32   g_t (Eq. 3)
        instruction  str                     x ("" = none); c_text = text_conditioning(text_encoder, batch)
        actions      [16, 4]       float32   A_t in physical units, heading frame H_t (Eq. 4)
        actions_norm [16, 4]       float32   A_t / (v_h, v_h, v_z, yaw_max) = CFMSolver.normalize_actions(A_t)
        primitive    []            int64     u*_t, the target of L_CE (Eq. 13)

c_text is not precomputed: without an instruction it is the learned null embedding of
``perception.TextEncoderWrapper``, a parameter trained in stage (ii), so it must come from the model.
Every item also carries ``timestamp``, ``episode`` (``"<file>:<episode>"``) and ``frame`` for debugging.

    from dataset import AeroVPADataset, make_dataloader
    loader = make_dataloader("data/*.hdf5", stage=1, batch_size=256, num_workers=8)

HDF5 handles are opened lazily in each DataLoader worker, so multi-process loading is safe.
``python dataset.py`` runs the self-test: dry-run episodes -> inspect_hdf5 -> both stages -> one training
step of each stage through the Table 1 modules.
"""

from __future__ import annotations

import glob
import hashlib
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:  # package import
    from .config import VPAConfig
    from .perception import RGBDNormalizer, TextEncoderWrapper
except ImportError:  # flat import (files side by side)
    from config import VPAConfig
    from perception import RGBDNormalizer, TextEncoderWrapper

__all__ = ["SCHEMA_NAME", "AeroVPADataset", "make_dataloader", "text_conditioning", "episode_split", "expand_paths"]

SCHEMA_NAME = "aero-vpa/expert-demonstrations"
SUPPORTED_VERSIONS = ("1.0.0",)
EpisodeFilter = Callable[[str, str, Dict[str, Any]], bool]


def expand_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    """Files from paths and glob patterns, sorted and de-duplicated."""
    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    out: List[str] = []
    for item in items:
        matches = sorted(glob.glob(str(item))) if any(ch in str(item) for ch in "*?[") else [str(item)]
        out.extend(m for m in matches if m not in out)
    if not out:
        raise FileNotFoundError(f"no demonstration files match {paths!r}")
    for path in out:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return out


def episode_split(val_fraction: float = 0.1, seed: int = 0) -> Tuple[EpisodeFilter, EpisodeFilter]:
    """(train, validation) episode filters by a hash of ``file:episode``: frames of one flight never cross."""
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must lie in [0, 1]")

    def in_val(path: str, name: str, _attrs: Dict[str, Any]) -> bool:
        digest = hashlib.sha1(f"{seed}:{os.path.basename(path)}:{name}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2.0 ** 64 < val_fraction

    return (lambda p, n, a: not in_val(p, n, a)), in_val


def _attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


class AeroVPADataset(Dataset):
    """Decision steps of the demonstration files for training stage 1 or 2 (see the module docstring).

    Args:
        paths: file, glob pattern, or a sequence of them.
        stage: 1 (JEPA/VICReg) or 2 (CFM solver and primitive selector).
        config: ``VPAConfig`` (dimensions are checked against every file; depth range of the normaliser).
        only_successful: skip episodes whose ``success`` attribute is False (aborted expert flights).
        episode_filter: optional ``f(path, episode_name, episode_attrs) -> bool`` (e.g. ``episode_split``).
        normalize: True: ``image`` is the normalised network input. False: ``rgb`` [3, H, W] uint8 and
            ``depth`` [H, W] float32 (m, 0 = invalid) instead, for normalisation on the GPU.
    """

    def __init__(self, paths: Union[str, Sequence[str]], stage: int, config: Optional[VPAConfig] = None,
                 only_successful: bool = True, episode_filter: Optional[EpisodeFilter] = None,
                 normalize: bool = True) -> None:
        import h5py

        if stage not in (1, 2):
            raise ValueError("stage must be 1 (JEPA) or 2 (policy)")
        self.stage = int(stage)
        self.cfg = config or VPAConfig()
        self.paths = expand_paths(paths)
        self.normalize = bool(normalize)
        self.normalizer = RGBDNormalizer(depth_range=self.cfg.depth_range)
        self.action_scale = torch.tensor(self.cfg.action_scale, dtype=torch.float32)
        self.episodes: List[Tuple[int, str]] = []            # (file index, episode name)
        self.instructions: List[str] = []
        self.depth_scale: List[float] = []
        frames, futures, owners = [], [], []
        for file_id, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                self._check_file(path, f)
                self.depth_scale.append(float(f.attrs.get("depth_scale_m", 1e-3)))
                for name in sorted(k for k in f.keys() if k.startswith("episode_")):
                    ep = f[name]
                    attrs = {k: _attr(v) for k, v in ep.attrs.items()}
                    if not bool(attrs.get("complete", True)) or "future_targets" not in ep:
                        continue                                   # left unfinished by an interrupted writer
                    if only_successful and not bool(attrs.get("success", True)):
                        continue
                    if episode_filter is not None and not episode_filter(path, name, attrs):
                        continue
                    future = ep["future_targets"][...].astype(np.int64)
                    idx = np.nonzero(future >= 0)[0] if self.stage == 1 else np.arange(future.shape[0])
                    if idx.size == 0:
                        continue
                    owners.append(np.full(idx.size, len(self.episodes), dtype=np.int32))
                    frames.append(idx.astype(np.int32))
                    futures.append(future[idx].astype(np.int32))
                    self.episodes.append((file_id, name))
                    self.instructions.append(str(attrs.get("instruction", "")))
        if not self.episodes:
            raise ValueError(f"no usable episodes in {self.paths} for stage {self.stage}")
        self._owner = np.concatenate(owners)
        self._frame = np.concatenate(frames)
        self._future = np.concatenate(futures)
        self._handles: Dict[int, Any] = {}
        self._pid: Optional[int] = None

    def _check_file(self, path: str, f: Any) -> None:
        cfg = self.cfg
        schema, version = _attr(f.attrs.get("schema")), _attr(f.attrs.get("schema_version"))
        if schema != SCHEMA_NAME or version not in SUPPORTED_VERSIONS:
            raise ValueError(f"{path}: schema {schema!r} {version!r}, expected {SCHEMA_NAME!r} {SUPPORTED_VERSIONS}")
        want = {"image_height": cfg.image_size[0], "image_width": cfg.image_size[1], "horizon": cfg.horizon,
                "action_dim": cfg.action_dim, "proprio_dim": cfg.proprio_dim, "goal_dim": cfg.goal_dim}
        got = {k: _attr(f.attrs.get(k)) for k in want}
        if got != want:
            raise ValueError(f"{path}: dimensions {got} do not match the config {want}")
        if abs(float(f.attrs.get("predictor_horizon_s", -1.0)) - cfg.predictor_horizon_s) > 1e-9:
            raise ValueError(f"{path}: lookahead targets were computed for T_nu = {f.attrs.get('predictor_horizon_s')} s, "
                             f"config has {cfg.predictor_horizon_s} s")

    # ---------------- access ----------------
    def __len__(self) -> int:
        return int(self._frame.shape[0])

    def _group(self, episode: int) -> Any:
        import h5py

        if self._pid != os.getpid():                      # new DataLoader worker: never reuse inherited handles
            self._handles, self._pid = {}, os.getpid()
        file_id, name = self.episodes[episode]
        if file_id not in self._handles:
            self._handles[file_id] = h5py.File(self.paths[file_id], "r")
        return self._handles[file_id][name]

    def _image(self, ep: Any, t: int, depth_scale: float) -> Dict[str, Tensor]:
        raw = ep["observations/images"][t]                                      # [H, W, 4] uint16
        rgb = torch.from_numpy(raw[..., :3].astype(np.uint8)).permute(2, 0, 1)  # [3, H, W]
        depth = torch.from_numpy(raw[..., 3].astype(np.float32) * np.float32(depth_scale))   # [H, W], 0 = invalid
        if not self.normalize:
            return {"rgb": rgb, "depth": depth}
        return {"image": self.normalizer(rgb.unsqueeze(0), depth.unsqueeze(0)).image[0]}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        episode = int(self._owner[index])
        t = int(self._frame[index])
        ep = self._group(episode)
        file_id, name = self.episodes[episode]
        scale = self.depth_scale[file_id]
        state = torch.from_numpy(np.asarray(ep["observations/state"][t], dtype=np.float32))
        item: Dict[str, Any] = {
            **self._image(ep, t, scale),
            "state": state,
            "primitive": torch.tensor(int(ep["primitives"][t]), dtype=torch.long),
            "timestamp": float(ep["timestamps"][t]),
            "episode": f"{os.path.basename(self.paths[file_id])}:{name}",
            "frame": t,
        }
        if self.stage == 1:
            j = int(self._future[index])
            future = self._image(ep, j, scale)
            item.update({f"future_{k}": v for k, v in future.items()})
            item["a_bar"] = state[-self.cfg.action_dim:].clone()
            item["lookahead_s"] = float(ep["timestamps"][j]) - item["timestamp"]
        else:
            actions = torch.from_numpy(np.asarray(ep["actions"][t], dtype=np.float32))
            item["goal"] = torch.from_numpy(np.asarray(ep["observations/goal"][t], dtype=np.float32))
            item["actions"] = actions
            item["actions_norm"] = actions / self.action_scale
            item["instruction"] = self.instructions[episode]
        return item

    # ---------------- statistics ----------------
    def state_statistics(self, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
        """(mu_S, sigma_S) over the indexed frames, for ``CFMSolver.set_standardization_stats`` (Sec. 3.4).

        Components that never vary in the data (sigma below ``eps``) get sigma = 1 so standardisation is a
        shift only, never a division by ~0.
        """
        total, count = np.zeros(self.cfg.proprio_dim), 0
        sq = np.zeros(self.cfg.proprio_dim)
        for episode in range(len(self.episodes)):
            rows = self._frame[self._owner == episode]
            states = np.asarray(self._group(episode)["observations/state"][...], dtype=np.float64)[rows]
            total += states.sum(axis=0)
            sq += (states ** 2).sum(axis=0)
            count += states.shape[0]
        mean = total / count
        std = np.sqrt(np.maximum(sq / count - mean ** 2, 0.0))
        std = np.where(std < eps, 1.0, std)
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def label_counts(self) -> Tensor:
        """Frames per primitive over the indexed frames ``[N_u]`` (for class weights)."""
        counts = np.zeros(self.cfg.num_primitives, dtype=np.int64)
        for episode in range(len(self.episodes)):
            rows = self._frame[self._owner == episode]
            labels = self._group(episode)["primitives"][...][rows]
            counts += np.bincount(labels, minlength=self.cfg.num_primitives)
        return torch.from_numpy(counts)

    def __repr__(self) -> str:
        return (f"AeroVPADataset(stage={self.stage}, files={len(self.paths)}, episodes={len(self.episodes)}, "
                f"samples={len(self)})")


def make_dataloader(paths: Union[str, Sequence[str]], stage: int, batch_size: int, shuffle: bool = True,
                    num_workers: int = 4, config: Optional[VPAConfig] = None, only_successful: bool = True,
                    episode_filter: Optional[EpisodeFilter] = None, normalize: bool = True,
                    pin_memory: Optional[bool] = None, drop_last: Optional[bool] = None) -> DataLoader:
    """``DataLoader`` over ``AeroVPADataset``. Stage 1 drops the last partial batch by default, because the
    variance and covariance terms of VICReg are batch statistics (Eqs. 9, 10)."""
    ds = AeroVPADataset(paths, stage, config, only_successful, episode_filter, normalize)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        drop_last=(stage == 1) if drop_last is None else drop_last,
        persistent_workers=num_workers > 0,
    )


def text_conditioning(text_encoder: TextEncoderWrapper, instructions: Sequence[str],
                      device: Optional[torch.device] = None) -> Tensor:
    """c_text ``[B, d_c]`` for a batch's ``instruction`` list: CLIP for text, the learned null embedding for ""."""
    c_text = text_encoder(list(instructions))
    if c_text.shape[0] == 1 and len(instructions) > 1:
        c_text = c_text.expand(len(instructions), -1)
    return c_text if device is None else c_text.to(device)


# ---------------------------------------------------------------------------------------------
# Self-test (run: python dataset.py)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import sys
    import tempfile

    import h5py

    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root, "sim"))
    import expert as E  # noqa: E402  (sim/expert.py: the writer of the files)

    import inspect_hdf5
    from perception import _toy_tokenizer, _ToyTextModel
    from selector import PrimitiveSelector
    from solver import CFMSolver

    torch.manual_seed(0)
    cfg = VPAConfig()
    with tempfile.TemporaryDirectory() as tmp:
        # ---- two short dry-run episodes in the collector's schema
        path = os.path.join(tmp, "pilot_sample.hdf5")
        world = E.synthetic_forest(seed=0, half_size=25.0)
        E.dry_run(path, world, E.DryRunConfig(num_episodes=2, seed=1, max_frames=40, spacing=(8.0, 10.0),
                                              num_waypoints=(3, 3), speed_limits=(5.0,)))
        report = inspect_hdf5.validate_file(path, cfg)
        assert not report["errors"], report["errors"]
        with h5py.File(path, "r") as f:
            n = {k: f[k]["timestamps"].shape[0] for k in f if k.startswith("episode_")}
            n_future = {k: int((f[k]["future_targets"][...] >= 0).sum()) for k in n}
            raw = f["episode_0000/observations/images"][7]
            state7 = f["episode_0000/observations/state"][7]
            target7 = int(f["episode_0000/future_targets"][7])
            raw_future = f["episode_0000/observations/images"][target7]
            actions7 = f["episode_0000/actions"][7]
        assert n == {"episode_0000": 40, "episode_0001": 40} and target7 == 12

        # ---- episode filters: max_frames episodes are unsuccessful
        try:
            AeroVPADataset(path, 1)
            raise AssertionError("expected no successful episodes")
        except ValueError:
            pass
        with h5py.File(path, "r+") as f:
            f["episode_0001"].attrs["success"] = True
        assert len(AeroVPADataset(path, 2)) == n["episode_0001"]
        train, val = episode_split(0.5, seed=3)
        names = sorted(n)
        assert all(train(path, k, {}) != val(path, k, {}) for k in names)              # disjoint and complete
        assert [val(path, k, {}) for k in names] == [val(os.path.join("x", os.path.basename(path)), k, {}) for k in names]

        # ---- stage 1 items
        ds1 = AeroVPADataset(path, 1, cfg, only_successful=False)
        assert len(ds1) == sum(n_future.values())
        item = ds1[7]
        assert item["episode"].endswith("episode_0000") and item["frame"] == 7
        assert item["image"].shape == (4, 128, 160) and item["image"].dtype == torch.float32
        assert item["future_image"].shape == (4, 128, 160) and abs(item["lookahead_s"] - 0.1) < 1e-6
        assert torch.equal(item["a_bar"], item["state"][9:13]) and item["primitive"].dtype == torch.long
        norm = RGBDNormalizer(depth_range=cfg.depth_range)
        ref = norm(torch.from_numpy(raw[..., :3].astype(np.uint8)).permute(2, 0, 1)[None],
                   torch.from_numpy(raw[..., 3].astype(np.float32) * 1e-3)[None]).image[0]
        ref_future = norm(torch.from_numpy(raw_future[..., :3].astype(np.uint8)).permute(2, 0, 1)[None],
                          torch.from_numpy(raw_future[..., 3].astype(np.float32) * 1e-3)[None]).image[0]
        assert torch.equal(item["image"], ref) and torch.equal(item["future_image"], ref_future)
        assert torch.equal(item["state"], torch.from_numpy(state7))
        raw_item = AeroVPADataset(path, 1, cfg, only_successful=False, normalize=False)[7]
        assert raw_item["rgb"].dtype == torch.uint8 and raw_item["depth"].shape == (128, 160)

        # ---- stage 2 items
        ds2 = AeroVPADataset(path, 2, cfg, only_successful=False)
        assert len(ds2) == sum(n.values())
        item2 = ds2[7]
        assert item2["actions"].shape == (16, 4) and item2["goal"].shape == (4,) and item2["instruction"] == ""
        assert torch.equal(item2["actions"], torch.from_numpy(actions7))
        mu_s, sd_s = ds2.state_statistics()
        assert mu_s.shape == (13,) and bool((sd_s > 0).all()) and int(ds2.label_counts().sum()) == len(ds2)

        # ---- DataLoaders (multi-process: HDF5 handles opened per worker)
        loader1 = make_dataloader(path, 1, batch_size=6, num_workers=2, only_successful=False, pin_memory=False)
        batch1 = next(iter(loader1))
        assert batch1["image"].shape == (6, 4, 128, 160) and batch1["future_image"].shape == (6, 4, 128, 160)
        assert batch1["state"].shape == (6, 13) and batch1["a_bar"].shape == (6, 4) and batch1["primitive"].shape == (6,)
        loader2 = make_dataloader(path, 2, batch_size=6, num_workers=0, only_successful=False, pin_memory=False)
        batch2 = next(iter(loader2))
        assert batch2["actions"].shape == (6, 16, 4) and batch2["goal"].shape == (6, 4)
        assert isinstance(batch2["instruction"], list) and len(batch2["instruction"]) == 6

        # ---- one stage-1 step (Eq. 8) through the Table 1 encoder and predictor
        enc = cfg.build_vision_encoder()
        target_enc = cfg.build_momentum_encoder(enc)
        pred = cfg.build_predictor()
        vicreg = cfg.build_vicreg_loss()
        tracker = cfg.build_latent_scale_tracker()
        opt = torch.optim.AdamW([*enc.parameters(), *pred.parameters()], lr=1e-4)
        z = enc(batch1["image"])
        with torch.no_grad():
            z_target = target_enc(batch1["future_image"])
        heads = pred.forward_heads(z, batch1["primitive"], batch1["a_bar"])
        loss1, terms = vicreg(z, heads, z_target)
        opt.zero_grad()
        loss1.backward()
        assert torch.isfinite(loss1) and all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None)
        assert pred.heads[0].weight.grad is not None and enc.patch_embed.proj.weight.grad is not None
        opt.step()
        target_enc.update(enc)
        tracker.update(z.detach())
        gamma_star, mu_z = tracker.freeze()
        assert set(terms) == {"invariance", "variance", "covariance"}

        # ---- one stage-2 step (Eq. 30): selector + CFM solver on frozen latents, c_text from the text wrapper
        pred.set_latent_scale(gamma_star)
        enc.requires_grad_(False)
        pred.requires_grad_(False)
        selector = PrimitiveSelector(latent_dim=cfg.latent_dim, goal_dim=cfg.goal_dim, proprio_dim=cfg.proprio_dim,
                                     text_dim=cfg.text_dim, num_primitives=cfg.num_primitives, hidden_dims=cfg.selector_hidden)
        solver = CFMSolver.from_config(cfg)
        solver.set_standardization_stats(mu_z, gamma_star, mu_s, sd_s)
        selector.set_input_standardization(mu_z, gamma_star, mu_s, sd_s)
        text = TextEncoderWrapper(text_model=_ToyTextModel(projection_dim=cfg.text_dim), tokenizer=_toy_tokenizer)
        c_text = text_conditioning(text, batch2["instruction"])
        assert c_text.shape == (6, cfg.text_dim)
        with torch.no_grad():
            z2 = enc(batch2["image"])
            z_hat = pred(z2, batch2["primitive"], batch2["state"][:, -cfg.action_dim:]).z_hat
        logits = selector(z2, batch2["goal"], batch2["state"], c_text)
        loss_ce = selector.loss(logits, batch2["primitive"])
        e_t = solver.build_conditioning(batch2["primitive"], z2, z_hat, batch2["state"], batch2["goal"], c_text)
        assert e_t.shape == (6, cfg.cond_dim)
        assert torch.allclose(batch2["actions_norm"], solver.normalize_actions(batch2["actions"]))
        assert float(batch2["actions_norm"].abs().max()) <= 1.0 + 1e-6
        loss2 = solver.flow_matching_loss(batch2["actions_norm"], e_t) + cfg.lambda_h * loss_ce
        loss2.backward()
        assert torch.isfinite(loss2) and solver.vector_field.action_out.weight.grad is not None
        assert selector.mlp[0].weight.grad is not None and text.null_embedding.grad is not None
        assert not z2.requires_grad and not z_hat.requires_grad          # stage (ii) trains on frozen latents
    print("dataset.py self-test passed")


if __name__ == "__main__":
    _self_test()

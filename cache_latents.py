# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
cache_latents.py -- Run the frozen stage-(i) networks once over every frame (preprint_261026.pdf, Sec. 3.8:
"with latents cached, the selector and solver minimize L_pol").

    python cache_latents.py checkpoints/stage1.pt "data/*.hdf5" --out_dir latents

For every demonstration file ``data/X.hdf5`` this writes ``latents/data__X.latents.hdf5`` (``<directory>__<name>``:
the DAgger rounds record the worlds of the demonstrations under the same names, dagger_r1/X.hdf5; caches named
``X.latents.hdf5``, as before 2026-10-31, are still used while they are current for their source):

    /                         attrs: schema, schema_version, stage1_checkpoint_id, gamma_bar_star, source_file,
                                     source_bytes, source_mtime_ns, primitive ("expert")
    /episode_0000/z           (T, 256) float32   z_t = E_psi(I_t)                                    (Eq. 6)
    /episode_0000/z_hat       (T, 256) float32   z_hat_{t+1} = mean_i P_omega^(i)(z_t, Embed(u*_t), a_bar_t)  (Eq. 7)
    /episode_0000/sigma       (T,)     float32   sigma_{t+1}, the normalised ensemble disagreement     (Eq. 22)

for every complete episode (successful or not; train_stage2.py chooses). The primitive is the expert label
u*_t (teacher forcing, as in stage ii). A cache is rebuilt when the stage-1 checkpoint or the source file
changes, and skipped otherwise, so the command can be rerun after more data arrives. Files are written
under a temporary name and renamed when complete.

Frames are read in contiguous chunks by DataLoader workers (decompression in parallel) and encoded on the
GPU in float32/TF32 (``--amp`` for bf16).
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from checkpoint import load_checkpoint, load_stage1_modules
from dataset import expand_paths
from train_stage1 import autocast, pick_device

__all__ = ["CACHE_SCHEMA", "CACHE_VERSION", "cache_path", "legacy_cache_path", "find_cache", "cache_is_current",
           "FrameChunks", "cache_latents", "parse_args"]

CACHE_SCHEMA = "aero-vpa/latent-cache"
CACHE_VERSION = "1.0.0"


def cache_path(out_dir: str, source: str) -> str:
    """Where the cache of ``source`` is written: ``<directory>__<name>.latents.hdf5``, so that data/X.hdf5 and
    dagger_r1/X.hdf5 (the same world) get separate caches."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(source)))
    return os.path.join(out_dir, f"{folder}__{os.path.splitext(os.path.basename(source))[0]}.latents.hdf5")


def legacy_cache_path(out_dir: str, source: str) -> str:
    """The name of caches written before 2026-10-31 (``<name>.latents.hdf5``), shared by sources of the same name."""
    return os.path.join(out_dir, os.path.splitext(os.path.basename(source))[0] + ".latents.hdf5")


def find_cache(out_dir: str, source: str, checkpoint_id: str) -> Optional[str]:
    """The current cache of ``source`` in ``out_dir`` (``cache_path``, else ``legacy_cache_path``), or None."""
    for path in (cache_path(out_dir, source), legacy_cache_path(out_dir, source)):
        if cache_is_current(path, source, checkpoint_id):
            return path
    return None


def _source_signature(source: str) -> Tuple[int, int]:
    st = os.stat(source)
    return int(st.st_size), int(st.st_mtime_ns)


def cache_is_current(path: str, source: str, checkpoint_id: str) -> bool:
    """True if ``path`` is a complete cache of ``source`` built from the stage-1 checkpoint ``checkpoint_id``."""
    if not os.path.isfile(path):
        return False
    size, mtime = _source_signature(source)
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            return (a.get("schema") == CACHE_SCHEMA and a.get("stage1_checkpoint_id") == checkpoint_id
                    and int(a.get("source_bytes", -1)) == size and int(a.get("source_mtime_ns", -1)) == mtime)
    except OSError:
        return False


def complete_episodes(f: Any) -> List[str]:
    """Episode groups that the writer finished (an interrupted writer leaves complete = False)."""
    names = sorted(k for k in f.keys() if k.startswith("episode_"))
    return [n for n in names if bool(f[n].attrs.get("complete", True)) and "future_targets" in f[n]]


class FrameChunks(Dataset):
    """Contiguous frame chunks ``(file, episode, start, stop)``; one item is already a batch."""

    def __init__(self, items: Sequence[Tuple[str, str, int, int]], action_dim: int = 4) -> None:
        self.items = list(items)
        self.action_dim = action_dim
        self._files: Dict[str, Any] = {}
        self._pid: Optional[int] = None

    def __len__(self) -> int:
        return len(self.items)

    def _file(self, path: str) -> Any:
        if self._pid != os.getpid():
            self._files, self._pid = {}, os.getpid()
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, name, a, b = self.items[index]
        f = self._file(path)
        ep = f[name]
        raw = ep["observations/images"][a:b]                                          # [n, H, W, 4] uint16
        scale = np.float32(f.attrs.get("depth_scale_m", 1e-3))
        state = np.asarray(ep["observations/state"][a:b], dtype=np.float32)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(raw[..., :3].astype(np.uint8))).permute(0, 3, 1, 2),
            "depth": torch.from_numpy(raw[..., 3].astype(np.float32) * scale),
            "state": torch.from_numpy(state),
            "goal": torch.from_numpy(np.asarray(ep["observations/goal"][a:b], dtype=np.float32)),
            "a_bar": torch.from_numpy(np.ascontiguousarray(state[:, -self.action_dim:])),
            "primitive": torch.from_numpy(np.asarray(ep["primitives"][a:b], dtype=np.int64)),
            "path": path, "episode": name, "start": a, "stop": b,
        }


def chunk_items(paths: Sequence[str], chunk: int, episode_filter: Any = None) -> List[Tuple[str, str, int, int]]:
    items = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for name in complete_episodes(f):
                if episode_filter is not None and not episode_filter(path, name, dict(f[name].attrs)):
                    continue
                n = f[name]["timestamps"].shape[0]
                items += [(path, name, a, min(a + chunk, n)) for a in range(0, n, chunk)]
    return items


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache z_t, z_hat_{t+1} and sigma_{t+1} of the frozen stage-(i) networks.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="stage-1 checkpoint (train_stage1.py)")
    p.add_argument("data", nargs="*", help="demonstration files or glob patterns")
    p.add_argument("--out_dir", default="latents")
    p.add_argument("--chunk", type=int, default=256, help="frames per GPU batch")
    p.add_argument("--num_workers", type=int, default=min(8, max(0, (os.cpu_count() or 2) - 2)))
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (default: float32/TF32)")
    p.add_argument("--force", action="store_true", help="rebuild caches that are current")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if not args.self_test and (not args.checkpoint or not args.data):
        p.error("give the stage-1 checkpoint and demonstration files (or --self-test)")
    return args


@torch.no_grad()
def cache_latents(args: argparse.Namespace) -> List[str]:
    """Build the caches; returns the cache paths (current ones included)."""
    device = pick_device(args.device)
    ckpt = load_checkpoint(args.checkpoint)
    cfg, encoder, predictor, gamma, _ = load_stage1_modules(ckpt, device)
    normalizer = cfg.build_rgbd_normalizer().to(device)
    ckpt_id = ckpt["checkpoint_id"]
    sources = expand_paths(args.data)
    names = [cache_path(args.out_dir, s) for s in sources]
    if len(set(names)) < len(names):
        raise ValueError("sources in different directories of the same name would share caches")
    os.makedirs(args.out_dir, exist_ok=True)
    todo = [s for s in sources if args.force or find_cache(args.out_dir, s, ckpt_id) is None]
    print(f"[cache] {len(sources)} files, {len(sources) - len(todo)} current, {len(todo)} to build "
          f"(stage-1 checkpoint {ckpt_id[:8]}, gamma* = {gamma:.4f}, device {device})", flush=True)
    items = chunk_items(todo, args.chunk)
    loader = DataLoader(FrameChunks(items, cfg.action_dim), batch_size=None, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        prefetch_factor=2 if args.num_workers > 0 else None)
    out: Optional[h5py.File] = None
    out_source = None
    frames, files_done = 0, 0

    def finish() -> None:
        nonlocal out, files_done
        if out is not None:
            tmp_name = out.filename
            out.close()
            os.replace(tmp_name, cache_path(args.out_dir, out_source))
            files_done += 1
            out = None

    for chunk in loader:
        source = chunk["path"]
        if source != out_source:
            finish()
            out_source = source
            final = cache_path(args.out_dir, source)
            out = h5py.File(final + ".tmp", "w")
            size, mtime = _source_signature(source)
            out.attrs.update({"schema": CACHE_SCHEMA, "schema_version": CACHE_VERSION, "stage1_checkpoint_id": ckpt_id,
                              "gamma_bar_star": gamma, "source_file": os.path.basename(source), "source_bytes": size,
                              "source_mtime_ns": mtime, "primitive": "expert"})
            with h5py.File(source, "r") as src:
                for name in complete_episodes(src):
                    n = src[name]["timestamps"].shape[0]
                    g = out.create_group(name)
                    g.create_dataset("z", shape=(n, cfg.latent_dim), dtype=np.float32)
                    g.create_dataset("z_hat", shape=(n, cfg.latent_dim), dtype=np.float32)
                    g.create_dataset("sigma", shape=(n,), dtype=np.float32)
        image = normalizer(chunk["rgb"].to(device, non_blocking=True), chunk["depth"].to(device, non_blocking=True)).image
        with autocast(device, args.amp):
            z = encoder(image)
            pred = predictor(z, chunk["primitive"].to(device), chunk["a_bar"].to(device))
        a, b = int(chunk["start"]), int(chunk["stop"])
        group = out[chunk["episode"]]
        group["z"][a:b] = z.float().cpu().numpy()
        group["z_hat"][a:b] = pred.z_hat.float().cpu().numpy()
        group["sigma"][a:b] = pred.sigma.float().cpu().numpy()
        frames += b - a
        if frames % (50 * args.chunk) < (b - a):
            print(f"[cache] {frames} frames encoded, {files_done}/{len(todo)} files", flush=True)
    finish()
    print(f"[cache] done: {frames} frames in {files_done} files -> {args.out_dir}", flush=True)
    return [find_cache(args.out_dir, s, ckpt_id) or cache_path(args.out_dir, s) for s in sources]


# ---------------------------------------------------------------------------------------------
# Self-test (run: python cache_latents.py --self-test)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile
    import time

    from train_stage1 import make_demo_data, parse_args as stage1_args, train_stage1

    with tempfile.TemporaryDirectory() as tmp:
        files = make_demo_data(tmp, {"a.hdf5": 1, "b.hdf5": 2}, frames=30)
        ckpt_path = os.path.join(tmp, "stage1.pt")
        common = ["--out", ckpt_path, "--batch_size", "8", "--num_workers", "0", "--device", "cpu", "--val_fraction", "0",
                  "--log_every", "100", "--warmup_steps", "1"]
        train_stage1(stage1_args(files + common + ["--steps", "2"]))
        out_dir = os.path.join(tmp, "latents")
        args = parse_args([ckpt_path, *files, "--out_dir", out_dir, "--chunk", "7", "--num_workers", "2", "--device", "cpu"])
        caches = cache_latents(args)
        folder = os.path.basename(tmp)
        assert [os.path.basename(c) for c in caches] == [f"{folder}__a.latents.hdf5", f"{folder}__b.latents.hdf5"]
        assert not [f for f in os.listdir(out_dir) if f.endswith(".tmp")]

        # cached values equal a direct evaluation of the frozen networks
        ckpt = load_checkpoint(ckpt_path)
        cfg, encoder, predictor, gamma, _ = load_stage1_modules(ckpt)
        normalizer = cfg.build_rgbd_normalizer()
        with h5py.File(files[1], "r") as src, h5py.File(caches[1], "r") as cache:
            assert cache.attrs["stage1_checkpoint_id"] == ckpt["checkpoint_id"]
            assert sorted(cache.keys()) == complete_episodes(src) == ["episode_0000", "episode_0001"]
            ep, cep = src["episode_0001"], cache["episode_0001"]
            n = ep["timestamps"].shape[0]
            assert cep["z"].shape == (n, 256) and cep["sigma"].shape == (n,)
            idx = [0, 6, 7, n - 1]                                          # across chunk boundaries
            raw = ep["observations/images"][idx]
            rgb = torch.from_numpy(raw[..., :3].astype(np.uint8)).permute(0, 3, 1, 2)
            depth = torch.from_numpy(raw[..., 3].astype(np.float32) * 1e-3)
            state = torch.from_numpy(ep["observations/state"][idx])
            with torch.no_grad():
                z = encoder(normalizer(rgb, depth).image)
                pred = predictor(z, torch.from_numpy(ep["primitives"][idx]), state[:, -4:])
            assert np.allclose(cep["z"][idx], z.numpy(), atol=1e-5)
            assert np.allclose(cep["z_hat"][idx], pred.z_hat.numpy(), atol=1e-5)
            assert np.allclose(cep["sigma"][idx], pred.sigma.numpy(), atol=1e-6)
            assert np.all(np.isfinite(cep["z"][...]))

        # a rerun skips current caches; a changed source or a new checkpoint rebuilds
        mtimes = [os.stat(c).st_mtime_ns for c in caches]
        time.sleep(0.01)
        cache_latents(args)
        assert [os.stat(c).st_mtime_ns for c in caches] == mtimes
        with h5py.File(files[0], "r+") as f:
            f.attrs["touched"] = True
        cache_latents(args)
        assert os.stat(caches[0]).st_mtime_ns != mtimes[0] and os.stat(caches[1]).st_mtime_ns == mtimes[1]
        train_stage1(stage1_args(files + common + ["--steps", "3", "--resume"]))
        assert not cache_is_current(caches[1], files[1], load_checkpoint(ckpt_path)["checkpoint_id"])
        cache_latents(args)
        assert cache_is_current(caches[1], files[1], load_checkpoint(ckpt_path)["checkpoint_id"])
        assert math.isfinite(gamma)

        # a DAgger round records a world of the demonstrations under the same name (dagger_r1/a.hdf5): separate
        # caches, and neither overwrites the other; a current cache under the old shared name is still used
        cid = load_checkpoint(ckpt_path)["checkpoint_id"]
        r1 = os.path.join(tmp, "dagger_r1")
        os.makedirs(r1)
        r1_files = make_demo_data(r1, {"a.hdf5": 1}, frames=20)
        mtimes = [os.stat(c).st_mtime_ns for c in caches]
        both = cache_latents(parse_args([ckpt_path, *files, *r1_files, "--out_dir", out_dir, "--chunk", "7",
                                         "--num_workers", "0", "--device", "cpu"]))
        assert os.path.basename(both[2]) == "dagger_r1__a.latents.hdf5" and [os.stat(c).st_mtime_ns for c in caches] == mtimes
        assert all(find_cache(out_dir, s, cid) == c for s, c in zip(files + r1_files, both, strict=True))
        os.replace(both[2], legacy_cache_path(out_dir, r1_files[0]))       # as round 1 wrote it (2026-10-31)
        assert find_cache(out_dir, r1_files[0], cid) == legacy_cache_path(out_dir, r1_files[0])
        assert find_cache(out_dir, files[0], cid) == caches[0]               # the legacy name is not the demo's
        os.remove(caches[0])
        assert find_cache(out_dir, files[0], cid) is None                   # ... so a lost demo cache is rebuilt
        cache_latents(args)
        assert find_cache(out_dir, files[0], cid) == caches[0]
    print("cache_latents.py self-test passed")


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        _self_test()
    else:
        cache_latents(parsed)

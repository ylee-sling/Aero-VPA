# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
inspect_hdf5.py -- Print and validate an Aero-VPA demonstration file against the paper's dimensions.

    python inspect_hdf5.py pilot_sample.hdf5                # tree, shapes, dtypes, statistics, checks
    python inspect_hdf5.py data/*.hdf5 --quiet              # checks only (exit status 1 on any violation)
    python inspect_hdf5.py pilot_sample.hdf5 --json report.json

Checks every episode against ``config.VPAConfig`` (Secs. 3.1-3.4, Table 1) and the schema written by
``sim/expert.py``:

    observations/images  (T, 128, 160, 4) uint16   R, G, B in [0, 255]; D in mm (0 = invalid)
    observations/state   (T, 13)          float32  S_t: unit n^B, finite
    observations/goal    (T, 4)           float32  g_t: |direction| <= 1, distance in [0, 1]
    actions              (T, 16, 4)       float32  A_t inside the action box A_0
    primitives           (T,)             int64    u*_t in {0, ..., 5}
    timestamps           (T,)             float64  strictly increasing
    future_targets       (T,)             int64    -1 or a later frame about T_nu = 0.1 s ahead

Needs only ``h5py`` and ``numpy`` (``config.py`` imports nothing else).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from config import PRIMITIVES, VPAConfig

SCHEMA_NAME = "aero-vpa/expert-demonstrations"
SUPPORTED_VERSIONS = ("1.0.0",)


def expected_layout(cfg: VPAConfig) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Per-episode datasets: name -> (shape after T, dtype)."""
    h, w = cfg.image_size
    return {
        "observations/images": ((h, w, cfg.in_channels), "uint16"),
        "observations/state": ((cfg.proprio_dim,), "float32"),
        "observations/goal": ((cfg.goal_dim,), "float32"),
        "actions": ((cfg.horizon, cfg.action_dim), "float32"),
        "primitives": ((), "int64"),
        "timestamps": ((), "float64"),
        "future_targets": ((), "int64"),
    }


def _describe(obj: Any, indent: int = 0, lines: List[str] = None) -> List[str]:
    import h5py

    lines = [] if lines is None else lines
    pad = "  " * indent
    for key in obj:
        item = obj[key]
        if isinstance(item, h5py.Group):
            n_attrs = len(item.attrs)
            lines.append(f"{pad}{key}/" + (f"  ({n_attrs} attrs)" if n_attrs else ""))
            _describe(item, indent + 1, lines)
        else:
            extra = ([f"chunks {item.chunks}"] if item.chunks else []) + ([item.compression] if item.compression else [])
            lines.append(f"{pad}{key:<28s} {str(item.shape):<22s} {str(item.dtype):<8s} {', '.join(extra)}".rstrip())
    return lines


def _attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_file(path: str, cfg: VPAConfig = None) -> Dict[str, Any]:
    """Validate one file. Returns a report with ``errors`` (list of str), ``warnings`` and statistics."""
    import h5py

    cfg = cfg or VPAConfig()
    errors: List[str] = []
    warnings: List[str] = []
    report: Dict[str, Any] = {"path": path, "episodes": {}, "errors": errors, "warnings": warnings}
    layout = expected_layout(cfg)
    scale = np.asarray(cfg.action_scale, dtype=np.float64)
    with h5py.File(path, "r") as f:
        attrs = {k: _attr(v) for k, v in f.attrs.items()}
        report["attrs"] = attrs
        if attrs.get("schema") != SCHEMA_NAME:
            errors.append(f"root: schema {attrs.get('schema')!r}, expected {SCHEMA_NAME!r}")
        if attrs.get("schema_version") not in SUPPORTED_VERSIONS:
            errors.append(f"root: schema_version {attrs.get('schema_version')!r} not in {SUPPORTED_VERSIONS}")
        expect_attrs = {
            "image_height": cfg.image_size[0], "image_width": cfg.image_size[1], "horizon": cfg.horizon,
            "action_dim": cfg.action_dim, "proprio_dim": cfg.proprio_dim, "goal_dim": cfg.goal_dim,
        }
        for key, want in expect_attrs.items():
            if attrs.get(key) != want:
                errors.append(f"root: attribute {key} = {attrs.get(key)!r}, config expects {want}")
        for key, want in (("knot_dt", cfg.knot_dt), ("predictor_horizon_s", cfg.predictor_horizon_s)):
            if attrs.get(key) is None or abs(float(attrs[key]) - want) > 1e-9:
                errors.append(f"root: attribute {key} = {attrs.get(key)!r}, config expects {want}")
        if "primitives" in attrs and tuple(json.loads(attrs["primitives"])) != tuple(PRIMITIVES):
            errors.append(f"root: primitives {attrs['primitives']} differ from config.PRIMITIVES {PRIMITIVES}")
        if "action_scale" in attrs and not np.allclose(attrs["action_scale"], scale):
            errors.append(f"root: action_scale {attrs['action_scale']} differs from config {cfg.action_scale}")
        depth_scale = float(attrs.get("depth_scale_m", 1e-3))

        episodes = sorted(k for k in f.keys() if k.startswith("episode_"))
        if not episodes:
            errors.append("no episode_* groups")
        totals = {"frames": 0, "successful": 0, "labels": np.zeros(len(PRIMITIVES), dtype=np.int64)}
        for name in episodes:
            ep = f[name]
            where = f"{name}"
            if not bool(_attr(ep.attrs.get("complete", True))):
                warnings.append(f"{where}: incomplete (the writer was interrupted); skipped by dataset.py and "
                                "removed when the file is reopened with --append")
                continue
            missing = [k for k in layout if k not in ep]
            if missing:
                errors.append(f"{where}: missing datasets {missing}")
                continue
            n = ep["timestamps"].shape[0]
            for key, (shape, dtype) in layout.items():
                ds = ep[key]
                if ds.shape != (n, *shape):
                    errors.append(f"{where}/{key}: shape {ds.shape}, expected {(n, *shape)}")
                if ds.dtype != np.dtype(dtype):
                    errors.append(f"{where}/{key}: dtype {ds.dtype}, expected {dtype}")
            if n == 0:
                warnings.append(f"{where}: empty episode")
                continue
            ts = ep["timestamps"][...]
            state = ep["observations/state"][...].astype(np.float64)
            goal = ep["observations/goal"][...].astype(np.float64)
            actions = ep["actions"][...].astype(np.float64)
            labels = ep["primitives"][...]
            targets = ep["future_targets"][...]
            dts = np.diff(ts)
            if np.any(dts <= 0):
                errors.append(f"{where}/timestamps: not strictly increasing")
            for key, arr in (("state", state), ("goal", goal), ("actions", actions)):
                if not np.all(np.isfinite(arr)):
                    errors.append(f"{where}/{key}: non-finite values")
            n_b = np.linalg.norm(state[:, 3:6], axis=1)
            if np.any(np.abs(n_b - 1.0) > 1e-3):
                errors.append(f"{where}/observations/state: n^B (entries 3:6) not unit, max |norm-1| "
                              f"{np.abs(n_b - 1).max():.2e}")
            if np.any(np.linalg.norm(goal[:, :3], axis=1) > 1.0 + 1e-5) or np.any((goal[:, 3] < -1e-6) | (goal[:, 3] > 1 + 1e-6)):
                errors.append(f"{where}/observations/goal: outside the range of Eq. 3")
            excess = np.abs(actions) - scale[None, None, :]
            if np.any(excess > 1e-4):
                errors.append(f"{where}/actions: outside the box A_0 by up to {excess.max():.3f}")
            if labels.min() < 0 or labels.max() >= len(PRIMITIVES):
                errors.append(f"{where}/primitives: labels outside [0, {len(PRIMITIVES) - 1}]")
            valid = targets >= 0
            idx = np.nonzero(valid)[0]
            if np.any(targets[valid] >= n) or np.any(targets[valid] <= idx) or np.any(targets < -1):
                errors.append(f"{where}/future_targets: indices out of range or not in the future")
            elif idx.size:
                deviation = np.abs(ts[targets[valid]] - ts[idx] - cfg.predictor_horizon_s).max()
                if deviation > 0.75 * np.median(dts):
                    errors.append(f"{where}/future_targets: lookahead deviates from T_nu by up to {deviation:.3f} s")
            if valid.sum() == 0:
                warnings.append(f"{where}: no frame has a lookahead target (stage 1 cannot use it)")
            ep_attrs = {k: _attr(v) for k, v in ep.attrs.items()}
            if "success" not in ep_attrs:
                warnings.append(f"{where}: no 'success' attribute")
            img = ep["observations/images"][n // 2]
            depth = img[..., 3].astype(np.float64) * depth_scale
            label_counts = np.bincount(labels, minlength=len(PRIMITIVES))
            report["episodes"][name] = {
                "frames": int(n), "duration_s": float(ts[-1] - ts[0]) if n > 1 else 0.0,
                "median_dt_s": float(np.median(dts)) if dts.size else 0.0,
                "max_gap_s": float(dts.max()) if dts.size else 0.0,
                "stage1_frames": int(valid.sum()), "labels": label_counts.tolist(),
                "speed_max": float(np.linalg.norm(actions[:, 0, :2], axis=1).max()),
                "rgb_max": int(img[..., :3].max()),
                "depth_valid_fraction": float((img[..., 3] > 0).mean()),
                "depth_range_m": [float(depth[depth > 0].min()), float(depth.max())] if (depth > 0).any() else None,
                "attrs": ep_attrs,
            }
            if int(img[..., :3].max()) > 255:
                errors.append(f"{where}/observations/images: RGB channel above 255")
            totals["frames"] += int(n)
            totals["successful"] += int(bool(ep_attrs.get("success", False)))
            totals["labels"] += label_counts
        report["totals"] = {"episodes": len(episodes), "frames": totals["frames"], "successful": totals["successful"],
                            "labels": dict(zip(PRIMITIVES, totals["labels"].tolist(), strict=True))}
        report["tree"] = ([f"{episodes[0]}/"] + _describe(f[episodes[0]], 1)) if episodes else []
    return report


def print_report(report: Dict[str, Any], quiet: bool = False) -> None:
    print(f"== {report['path']}")
    if not quiet:
        print("-- root attributes")
        for key in sorted(report.get("attrs", {})):
            value = report["attrs"][key]
            text = json.dumps(value) if not isinstance(value, str) else value
            print(f"   {key:<22s} {text[:110]}{'...' if len(text) > 110 else ''}")
        print("-- hierarchy of the first episode (shape, dtype, chunks, compression)")
        for line in report.get("tree", []):
            print("   " + line)
        if len(report["episodes"]) > 1:
            print(f"   ... and {len(report['episodes']) - 1} more episodes (all checked against the same layout)")
        print("-- episodes")
        print(f"   {'episode':<14s}{'frames':>7s}{'dur s':>8s}{'dt ms':>7s}{'gap ms':>8s}{'stage1':>8s}"
              f"{'v_h':>6s}{'v_max':>7s}{'depth ok':>9s}  success  termination")
        for name, ep in report["episodes"].items():
            a = ep["attrs"]
            print(f"   {name:<14s}{ep['frames']:>7d}{ep['duration_s']:>8.1f}{1e3 * ep['median_dt_s']:>7.1f}"
                  f"{1e3 * ep['max_gap_s']:>8.1f}{ep['stage1_frames']:>8d}{float(a.get('speed_limit', 'nan')):>6.1f}"
                  f"{ep['speed_max']:>7.2f}{ep['depth_valid_fraction']:>9.2f}  {str(a.get('success', '?')):<7s}  "
                  f"{a.get('termination', '')}")
        totals = report.get("totals", {})
        if totals:
            print(f"-- totals: {totals['episodes']} episodes ({totals['successful']} successful), {totals['frames']} frames")
            n = max(1, totals["frames"])
            print("   labels: " + ", ".join(f"{k} {v} ({100 * v / n:.1f}%)" for k, v in totals["labels"].items()))
    for w in report["warnings"]:
        print(f"   WARNING {w}")
    for e in report["errors"]:
        print(f"   ERROR   {e}")
    print("   schema OK" if not report["errors"] else f"   {len(report['errors'])} schema violations")


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate Aero-VPA demonstration HDF5 files.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--quiet", action="store_true", help="only warnings, errors and the verdict")
    parser.add_argument("--json", help="write the reports (without the tree) to this JSON file")
    args = parser.parse_args(argv)
    reports, failed = [], 0
    for path in args.paths:
        try:
            report = validate_file(path)
        except OSError as exc:
            report = {"path": path, "episodes": {}, "errors": [f"cannot open: {exc}"], "warnings": []}
        print_report(report, quiet=args.quiet)
        failed += bool(report["errors"])
        reports.append({k: v for k, v in report.items() if k != "tree"})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=1, default=str)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

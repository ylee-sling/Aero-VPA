# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/crash_study.py -- Where traced missions crashed: the obstacle hit relative to the camera, and how the vehicle
moved in the last second (fly_policy.py --trace writes the traces).

For every mission that ended in a collision or on the ground it prints the state at the last decision (speed,
vertical speed, altitude, direction of motion relative to the camera axis), the nearest obstacle in the
privileged map of the world and its bearing relative to the camera, which is "in view" within half the
horizontal field of view (73 deg / 2), "beside" up to 100 deg, "behind" beyond; the blind share, executor mode,
fallback and primitive of the last second; and per policy tag a summary: obstacles hit by position, the share of
collisions while moving outside the view, speeds, modes and primitives.

    python3 sim/crash_study.py 'logs/trace_r1/*.jsonl' 'logs/trace_r1_shield2/*.jsonl'
    python3 sim/crash_study.py 'bench/traces/trace_r1/*.jsonl.gz' --world_root bench/worlds/test   # the dataset
    python3 sim/crash_study.py --self-test

The mission record (first trace line) names the world's privileged map (world_json). When that path does not
exist here (records written on another machine), the file of the same name is looked up in --world_root and in
sim/worlds/test of this repository.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import expert as E  # noqa: E402

HALF_FOV = math.degrees(1.274) / 2.0          # x500_vpa camera: HFOV 1.274 rad
BESIDE_DEG = 100.0
SHORT = ("CRU", "LAT", "VER", "YAW", "APP", "BRK")
MODES = {0: "TRACK", 1: "BRAKE", 2: "HOLD"}
CRASH = ("collision", "ground")


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def where(rel: Optional[float]) -> str:
    if rel is None:
        return "unknown"
    a = abs(rel)
    return "in view" if a <= HALF_FOV else ("beside" if a <= BESIDE_DEG else "behind")


def nearest(world: Any, p: Sequence[float]) -> Tuple[float, float]:
    """(distance to the surface, NED bearing in degrees) of the obstacle nearest to p; trunks horizontally."""
    c, r = world._c, world._r
    horizontal = np.hypot(c[:, 0] - p[0], c[:, 1] - p[1]) - r
    spatial = np.linalg.norm(c - np.asarray(p, dtype=np.float64)[None, :], axis=1) - r
    d = np.where(world._kind == "cylinder", horizontal, spatial)
    i = int(np.argmin(d))
    return float(d[i]), math.degrees(math.atan2(c[i, 1] - p[1], c[i, 0] - p[0]))


def resolve_world(path: str, roots: Sequence[str]) -> Optional[str]:
    if path and os.path.exists(path):
        return path
    name = os.path.basename(path or "")
    for root in roots:
        candidate = os.path.join(root, name)
        if name and os.path.exists(candidate):
            return candidate
    return None


def study(path: str, roots: Sequence[str]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """(tag, ending, crash details or None) of one trace file (.jsonl, or .jsonl.gz as in the benchmark dataset)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    record, rows = lines[0], [x for x in lines[1:] if x.get("kind") == "decision"]
    end = str(record.get("termination"))
    tag = str(record.get("tag", "?"))
    if end not in CRASH or not rows:
        return tag, end, None
    last = rows[-1]
    window = [x for x in rows if x["t"] >= last["t"] - 1.0]
    p, v, yaw = last["p"], last["v"], float(last["yaw"])
    hspeed = math.hypot(v[0], v[1])
    blind = [float(bool(x.get("blind"))) for x in window if "blind" in x]
    u = last.get("u")
    out: Dict[str, Any] = {
        "world": str(record.get("world", "?")).replace("aero_forest_", ""), "k": record.get("episode"), "end": end,
        "t": float(last["t"]), "speed": math.sqrt(sum(x * x for x in v)), "hspeed": hspeed, "vz": float(v[2]),
        "alt": -float(p[2]),
        "motion_rel": wrap(math.degrees(math.atan2(v[1], v[0])) - yaw) if hspeed > 0.2 else None,
        "blind": statistics.mean(blind) if blind else None,
        "views": Counter(str(x.get("view", "-")) for x in window),
        "mode": MODES.get(last.get("mode"), str(last.get("mode"))),
        "fallback": statistics.mean(float(bool(x.get("fallback"))) for x in window),
        "u": SHORT[u] if isinstance(u, int) and 0 <= u < len(SHORT) else "-",
        "d": last.get("d"), "v_safe": last.get("v_safe"), "clear0": window[0].get("clearance"),
        "trunk_rel": None, "trunk_d": None,
    }
    world_json = resolve_world(str(record.get("world_json") or ""), roots)
    if world_json:
        world = E.PrivilegedMap.from_world_json(world_json)
        if len(world):
            dist, bearing = nearest(world, p)
            out["trunk_d"], out["trunk_rel"] = dist, wrap(bearing - yaw)
    return tag, end, out


def fmt(x: Any, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def report(files: Sequence[str], roots: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Print the study; returns per tag {"ends": Counter, "crashes": [...]} for tests."""
    ends: Dict[str, Counter] = defaultdict(Counter)
    crashes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in files:
        try:
            tag, end, c = study(f, roots)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            print("skipped %s: %s" % (f, exc))
            continue
        ends[tag][end] += 1
        if c is not None:
            crashes[tag].append(c)
    for tag in sorted(ends):
        cs = crashes[tag]
        print("== %s: %d traced missions, endings %s" % (tag, sum(ends[tag].values()), dict(ends[tag])))
        print("   world            k  end        t   speed  vz    alt  motion  trunk(rel, at)  blind  mode   fb   u"
              "    d    v_safe clear1s")
        for c in cs:
            print("   %-15s %2s  %-9s %5.1f %5.2f %5.2f %5.2f  %6s  %6s %-7s %5s  %-5s %4s  %-3s %4s %5s %5s" % (
                c["world"][:15], c["k"], c["end"], c["t"], c["speed"], c["vz"], c["alt"], fmt(c["motion_rel"], ".0f"),
                fmt(c["trunk_rel"], ".0f"), where(c["trunk_rel"]), fmt(c["blind"], ".1f"), c["mode"],
                fmt(c["fallback"], ".1f"), c["u"], fmt(c["d"], ".1f"), fmt(c["v_safe"], ".1f"), fmt(c["clear0"], ".2f")))
        if not cs:
            continue
        coll = [c for c in cs if c["end"] == "collision"]
        print("   summary of %d crashes (%d collisions, %d ground):" % (len(cs), len(coll), len(cs) - len(coll)))
        print("     obstacle hit: %s" % dict(Counter(where(c["trunk_rel"]) for c in coll)))
        moving = [c for c in coll if c["motion_rel"] is not None]
        print("     moving outside the view (|motion| > %.0f deg): %d of %d" % (
            HALF_FOV, sum(abs(c["motion_rel"]) > HALF_FOV for c in moving), len(moving)))
        print("     speed at the crash median %.2f m/s (horizontal %.2f), max %.2f" % (
            statistics.median(c["speed"] for c in cs), statistics.median(c["hspeed"] for c in cs),
            max(c["speed"] for c in cs)))
        bl = [c["blind"] for c in cs if c["blind"] is not None]
        print("     blind share of the last second: median %s; crashes blind at least half of it: %d of %d" % (
            fmt(statistics.median(bl) if bl else None, ".2f"), sum(b >= 0.5 for b in bl), len(bl)))
        print("     mode at the crash %s; primitive %s; fallback in the last second %d of %d" % (
            dict(Counter(c["mode"] for c in cs)), dict(Counter(c["u"] for c in cs)),
            sum(c["fallback"] > 0 for c in cs), len(cs)))
        views: Counter = Counter()
        for c in cs:
            views.update(c["views"])
        print("     view of the last second's decisions: %s" % dict(views.most_common()))
    return {tag: {"ends": ends[tag], "crashes": crashes[tag]} for tag in ends}


# ---------------------------------------------------------------------------------------------
# Self-test (python3 sim/crash_study.py --self-test)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import contextlib
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        worlds = os.path.join(tmp, "worlds")
        os.makedirs(worlds)

        def world(name: str, north: float, east: float) -> str:
            # ENU sidecar of make_world.py: NED (north, east) is ENU (east, north)
            path = os.path.join(worlds, name + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"obstacles": [{"type": "cylinder", "center": [east, north, 5.0], "radius": 0.2,
                                          "length": 10.0}]}, f)
            return path

        def trace(name: str, end: str, world_json: str, v: Sequence[float], yaw: float) -> str:
            path = os.path.join(tmp, name + ".jsonl")
            rows = [{"kind": "mission", "tag": "pol", "world": "aero_forest_dense_s9000", "episode": 0,
                     "termination": end, "world_json": world_json}]
            for i in range(5):
                rows.append({"kind": "decision", "t": 10.0 + 0.25 * i, "p": [0.0, 0.0, -2.0], "v": list(v),
                             "yaw": yaw, "blind": i >= 3, "view": "obstacle", "mode": 0, "fallback": False, "u": 1,
                             "d": 1.0, "v_safe": 1.0, "clearance": 0.5})
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            return path

        files = [
            trace("a", "collision", world("w_ahead", 0.6, 0.0), (0.5, 0.0, 0.0), 0.0),          # in view
            # recorded on another machine: only the file name resolves, through --world_root
            trace("b", "collision", "/elsewhere/" + os.path.basename(world("w_east", 0.0, 0.6)), (0.0, 0.5, 0.0), 0.0),
            trace("c", "collision", world("w_behind", -0.6, 0.0), (-0.5, 0.0, 0.0), 0.0),       # behind, backing
            trace("d", "ground", world("w_far", 5.0, 5.0), (0.0, 0.0, 2.5), 0.0),               # descending
            trace("e", "success", world("w_ok", 3.0, 3.0), (1.0, 0.0, 0.0), 0.0),
        ]
        with open(files[2], "rb") as f_in, gzip.open(files[2] + ".gz", "wb") as f_out:   # as packaged for release
            f_out.write(f_in.read())
        os.remove(files[2])
        files[2] += ".gz"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = report(files, [worlds])
        text = buf.getvalue()
        res = out["pol"]
        assert res["ends"] == Counter({"collision": 3, "ground": 1, "success": 1}), res["ends"]
        assert len(res["crashes"]) == 4
        places = sorted(where(c["trunk_rel"]) for c in res["crashes"] if c["end"] == "collision")
        assert places == ["behind", "beside", "in view"], places
        assert "obstacle hit: {" in text and "moving outside the view (|motion| > 36 deg): 2 of 3" in text, text
        assert "crashes blind at least half of it: 0 of 4" in text, text
        ground = [c for c in res["crashes"] if c["end"] == "ground"][0]
        assert abs(ground["vz"] - 2.5) < 1e-9 and ground["motion_rel"] is None
        # wrap and where at the boundaries
        assert wrap(190.0) == -170.0 and where(36.0) == "in view" and where(-99.0) == "beside" and where(101) == "behind"
    print("crash_study self-test passed: 5 traces, 4 crashes classified (in view, beside, behind, ground)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("traces", nargs="*", help="trace files or glob patterns (fly_policy.py --trace)")
    p.add_argument("--world_root", action="append", default=[],
                   help="directory with the worlds' .json maps, when the recorded paths do not exist here")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    patterns = args.traces or ["logs/trace_*/*.jsonl"]
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    if not files:
        print("no trace files match %s" % " ".join(patterns))
        return 1
    roots = list(args.world_root) + [os.path.join(os.path.dirname(HERE), "sim", "worlds", "test")]
    report(files, roots)
    return 0


if __name__ == "__main__":
    sys.exit(main())

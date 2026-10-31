# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/summarize_flights.py -- Success rates and flight statistics of sim/fly_policy.py records, per policy.

    python3 sim/summarize_flights.py flights.jsonl [more.jsonl ...] [--by density] [--compare TAG_A TAG_B]
    python3 sim/summarize_flights.py flights.jsonl --missions          # one line per mission

Per policy (record tag): missions, success rate with a 95% Wilson interval, crashes (collision or ground),
timeouts and other endings, waypoints reached, fallback onsets per mission and the share of missions with one,
time braking or holding, the time ratio to the expert reference and the path length relative to the mission
(successful missions), mean speed, the median smallest clearance, blind decisions, and the 95th percentiles of
decision compute time and decision age (capture to commit). ``--by density`` repeats the success rate per world
density (sparse / medium / dense, from the world name). With two policies (or ``--compare``), missions flown by
both (same world and mission index, hence the same waypoints) are paired: both succeed, only one, neither, and
the exact McNemar test on the discordant pairs. A mission recorded twice for a policy keeps its last record.
``--missions`` lists every mission: ending, duration, path, speed, waypoints, blind share, fallback onsets, the
most frequent primitives, s_t, sigma_bar, the smallest clearance and where the vehicle ended (from home).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise fly_policy.py records.")
    p.add_argument("results", nargs="+", help="JSON lines files")
    p.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"), default=None)
    p.add_argument("--by", choices=["density", "world"], default=None, help="success rate per group")
    p.add_argument("--missions", action="store_true", help="also list every mission")
    return p.parse_args(argv)


def load(paths: Sequence[str]) -> Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]:
    """{tag: {(world, episode): record}} keeping the last record of a repeated mission."""
    out: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                text = raw.strip()
                if not text:
                    continue
                try:
                    r = json.loads(text)
                except ValueError:
                    continue
                out.setdefault(r["tag"], {})[(r["world"], int(r["episode"]))] = r
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of the discordant pairs at 1/2."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def density(world: str) -> str:
    m = re.search(r"(sparse|medium|dense)", world)
    return m.group(1) if m else "other"


def _median(values: List[float]) -> float:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(values)) if values else math.nan


def stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    ok = [r for r in records if r["success"]]
    ends: Dict[str, int] = {}
    for r in records:
        ends[r["termination"]] = ends.get(r["termination"], 0) + 1
    lo, hi = wilson(len(ok), n)
    return {
        "missions": n, "success": len(ok) / n if n else math.nan, "success_ci": (lo, hi),
        "crash": sum(ends.get(t, 0) for t in ("collision", "ground")) / n if n else math.nan,
        "collision": ends.get("collision", 0) / n if n else math.nan, "ground": ends.get("ground", 0) / n if n else math.nan,
        "duration": _median([r.get("duration_s") for r in records]),
        "path": _median([r.get("path_length_m") for r in records]),
        "timeout": ends.get("timeout", 0) / n if n else math.nan, "endings": ends,
        "waypoints_reached": sum(r["waypoints_reached"] for r in records) / max(1, sum(r["waypoints"] for r in records)),
        "fallback_onsets": float(np.mean([r["fallback_onsets"] for r in records])) if n else math.nan,
        "missions_with_fallback": float(np.mean([r["fallback_onsets"] > 0 for r in records])) if n else math.nan,
        "not_tracking_s": float(np.mean([r["not_tracking_s"] for r in records])) if n else math.nan,
        "time_ratio": _median([r.get("time_ratio") for r in ok]),
        "path_ratio": _median([r["path_length_m"] / r["mission_length_m"] for r in ok if r["mission_length_m"] > 0]),
        "mean_speed": float(np.mean([r["mean_speed"] for r in records])) if n else math.nan,
        "min_clearance": _median([r.get("min_clearance_m") for r in records]),
        "blind": sum(r["blind_decisions"] for r in records) / max(1, sum(r["decisions"] for r in records)),
        "latency_p95_ms": _median([r["decision_latency_ms"]["p95"] for r in records]),
        "age_p95_ms": _median([r["decision_age_ms"]["p95"] for r in records]),
        "setpoint_gap": _median([r.get("max_setpoint_gap_s") for r in records]),
        "setpoint_gap_max": max((r["max_setpoint_gap_s"] for r in records if r.get("max_setpoint_gap_s") is not None),
                                default=math.nan),
        **_dagger_stats([r for r in records if r.get("dagger")]),
    }


def _dagger_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stage-(iii) flights (fly_policy.py --dagger_dir): who flew, why the expert took over, what was recorded."""
    if not records:
        return {}
    n = len(records)
    reasons: Dict[str, float] = {}
    for r in records:
        for k, v in (r.get("dagger_takeovers") or {}).items():
            reasons[k] = reasons.get(k, 0.0) + v / n
    shares = [r.get("dagger_policy_share") for r in records]
    crash_modes: Dict[str, int] = {}
    for r in records:
        if r.get("termination") in ("collision", "ground"):
            mode = str(r.get("dagger_final_mode") or "?")
            crash_modes[mode] = crash_modes.get(mode, 0) + 1
    return {"dagger_missions": n, "dagger_policy_share": _median([x for x in shares if x is not None]),
            "dagger_takeovers": reasons, "dagger_replans": float(np.mean([r.get("dagger_replans", 0) for r in records])),
            "dagger_slowdowns": float(np.mean([r.get("dagger_slowdowns", 0) for r in records])),
            "dagger_expert_brakes": float(np.mean([r.get("dagger_expert_brakes", 0) for r in records])),
            "dagger_soft_blocked": float(np.mean([r.get("dagger_soft_blocked", 0) for r in records])),
            "dagger_crash_modes": crash_modes,
            "dagger_frames_policy": float(np.mean([r.get("dagger_frames_policy", 0) for r in records])),
            "dagger_frames_expert": float(np.mean([r.get("dagger_frames_expert", 0) for r in records]))}


PRIMITIVE_SHORT = ("CRU", "LAT", "VER", "YAW", "APP", "BRK")


def mission_lines(data: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]) -> List[str]:
    """One line per mission (policy, world, mission index order)."""
    def f(x: Any, fmt: str) -> str:
        return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)

    lines = [f"{'policy':22s} {'world':24s} {'k':>3s} {'ending':14s} {'dur s':>6s} {'path m':>6s} {'m/s':>5s} {'wp':>5s} "
             f"{'blind':>6s} {'fb':>3s} {'primitives':22s} {'s_t':>5s} {'sig_bar':>7s} {'sig_max':>7s} {'clear':>6s} "
             f"{'end from home (N, E, alt)':>27s}"]
    for tag in sorted(data):
        for (world, k), r in sorted(data[tag].items()):
            counts = np.asarray(r.get("primitive_counts") or [], dtype=np.float64)
            prims = ""
            if counts.sum() > 0:
                order = np.argsort(-counts)[:3]
                prims = " ".join(f"{PRIMITIVE_SHORT[i]}{100 * counts[i] / counts.sum():.0f}" for i in order if counts[i] > 0)
            end = r.get("final_position_ned")
            home = r.get("home") or [0.0, 0.0, 0.0]
            where = "-" if not end else f"{end[0] - home[0]:7.1f} {end[1] - home[1]:7.1f} {-end[2]:5.1f}"
            lines.append(
                f"{tag[:22]:22s} {world.replace('aero_forest_', '')[:24]:24s} {k:3d} {r['termination'][:14]:14s} "
                f"{f(r.get('duration_s'), '6.1f'):>6s} {f(r.get('path_length_m'), '6.1f'):>6s} {f(r.get('mean_speed'), '5.2f'):>5s} "
                f"{r['waypoints_reached']:>2d}/{r['waypoints']:<2d} "
                f"{100.0 * r['blind_decisions'] / max(1, r['decisions']):5.0f}% {r['fallback_onsets']:3d} {prims:22s} "
                f"{f(r.get('scale_mean'), '5.2f'):>5s} {f(r.get('sigma_bar_mean'), '7.3f'):>7s} "
                f"{f(r.get('sigma_bar_max'), '7.3f'):>7s} {f(r.get('min_clearance_m'), '6.2f'):>6s} {where:>27s}")
    return lines


def report(data: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]], by: Optional[str] = None,
           compare: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    tags = sorted(data)
    table = {tag: stats(list(data[tag].values())) for tag in tags}
    rows = [
        ("missions", lambda s: f"{s['missions']}"),
        ("success", lambda s: f"{100 * s['success']:.1f}% [{100 * s['success_ci'][0]:.0f}, {100 * s['success_ci'][1]:.0f}]"),
        ("crash (collision / ground)", lambda s: f"{100 * s['crash']:.1f}% ({100 * s['collision']:.0f} / "
                                                 f"{100 * s['ground']:.0f})"),
        ("timeout", lambda s: f"{100 * s['timeout']:.1f}%"),
        ("other endings", lambda s: ", ".join(f"{k} {v}" for k, v in sorted(s["endings"].items())
                                              if k not in ("success", "collision", "ground", "timeout")) or "-"),
        ("waypoints reached", lambda s: f"{100 * s['waypoints_reached']:.1f}%"),
        ("duration / path median (s, m)", lambda s: f"{s['duration']:.1f} / {s['path']:.1f}"),
        ("fallback onsets / mission", lambda s: f"{s['fallback_onsets']:.2f}"),
        ("missions with a fallback", lambda s: f"{100 * s['missions_with_fallback']:.1f}%"),
        ("braking or holding (s / mission)", lambda s: f"{s['not_tracking_s']:.1f}"),
        ("time / expert reference (median)", lambda s: f"{s['time_ratio']:.2f}"),
        ("path / mission length (median)", lambda s: f"{s['path_ratio']:.2f}"),
        ("mean speed (m/s)", lambda s: f"{s['mean_speed']:.2f}"),
        ("smallest clearance (median, m)", lambda s: f"{s['min_clearance']:.2f}"),
        ("blind decisions", lambda s: f"{100 * s['blind']:.1f}%"),
        ("decision compute p95 (ms)", lambda s: f"{s['latency_p95_ms']:.1f}"),
        ("decision age p95 (ms)", lambda s: f"{s['age_p95_ms']:.1f}"),
        ("longest setpoint gap (s, med/max)", lambda s: "-" if not math.isfinite(s["setpoint_gap_max"])
         else f"{s['setpoint_gap']:.2f} / {s['setpoint_gap_max']:.2f}"),
    ]
    if any("dagger_missions" in table[t] for t in tags):
        def takeovers(st: Dict[str, Any]) -> str:
            reasons = st.get("dagger_takeovers") or {}
            top = sorted(((v, k) for k, v in reasons.items() if v > 0), reverse=True)[:2]
            return f"{sum(reasons.values()):.1f}" + (f" ({', '.join(f'{k} {v:.1f}' for v, k in top)})" if top else "")

        rows += [
            ("DAgger: policy share of the time", lambda st: "-" if st.get("dagger_policy_share") is None
             else f"{100 * st['dagger_policy_share']:.0f}%"),
            ("DAgger: takeovers / mission", lambda st: takeovers(st) if "dagger_missions" in st else "-"),
            ("DAgger: frames / mission (pol/exp)", lambda st: f"{st['dagger_frames_policy']:.0f} / "
             f"{st['dagger_frames_expert']:.0f}" if "dagger_missions" in st else "-"),
            ("DAgger: re-plans / mission", lambda st: f"{st['dagger_replans']:.2f}" if "dagger_missions" in st else "-"),
            ("  expert slow-downs / stops / blocked", lambda st: f"{st['dagger_slowdowns']:.2f} / "
             f"{st['dagger_expert_brakes']:.2f} / {st['dagger_soft_blocked']:.2f}" if "dagger_missions" in st else "-"),
            ("  crashes by who flew", lambda st: (", ".join(f"{k} {v}" for k, v in sorted(st["dagger_crash_modes"].items()))
                                                  or "none") if "dagger_missions" in st else "-"),
        ]
    width = max(24, *(len(t) + 2 for t in tags))
    print(f"{'':34s}" + "".join(f"{t:>{width}s}" for t in tags))
    for label, fmt in rows:
        print(f"{label:34s}" + "".join(f"{fmt(table[t]):>{width}s}" for t in tags))
    if by:
        key = density if by == "density" else (lambda w: w)
        groups = sorted({key(w) for t in tags for (w, _) in data[t]})
        print(f"\nsuccess by {by}")
        for g in groups:
            cells = []
            for t in tags:
                recs = [r for (w, _), r in data[t].items() if key(w) == g]
                k = sum(r["success"] for r in recs)
                cells.append(f"{k}/{len(recs)} ({100 * k / len(recs):.0f}%)" if recs else "-")
            print(f"  {g:32s}" + "".join(f"{c:>{width}s}" for c in cells))
    pair = list(compare) if compare else (tags if len(tags) == 2 else None)
    out: Dict[str, Any] = {"stats": table}
    if pair:
        a, b = pair
        if a not in data or b not in data:
            raise SystemExit(f"--compare: unknown tag(s); have {tags}")
        shared = sorted(set(data[a]) & set(data[b]))
        both = sum(data[a][m]["success"] and data[b][m]["success"] for m in shared)
        only_a = sum(data[a][m]["success"] and not data[b][m]["success"] for m in shared)
        only_b = sum(data[b][m]["success"] and not data[a][m]["success"] for m in shared)
        neither = len(shared) - both - only_a - only_b
        p = mcnemar_exact(only_a, only_b)
        print(f"\npaired missions ({len(shared)} flown by both): both succeed {both}, only {a} {only_a}, "
              f"only {b} {only_b}, neither {neither}; exact McNemar p = {p:.3g}")
        out["paired"] = {"missions": len(shared), "both": both, "only_a": only_a, "only_b": only_b,
                         "neither": neither, "p_value": p}
    return out


def _self_test() -> None:
    import os
    import tempfile

    assert mcnemar_exact(0, 0) == 1.0 and math.isclose(mcnemar_exact(0, 6), 2 / 64)
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and math.isclose(lo + hi, 1.0, rel_tol=1e-9)

    def rec(tag: str, world: str, k: int, ok: bool, end: str) -> Dict[str, Any]:
        return {"tag": tag, "world": world, "episode": k, "success": ok, "termination": end, "waypoints": 3,
                "waypoints_reached": 3 if ok else 1, "fallback_onsets": int(not ok), "not_tracking_s": 0.5,
                "time_ratio": 1.2 if ok else None, "path_length_m": 33.0, "mission_length_m": 30.0, "mean_speed": 3.0,
                "min_clearance_m": 1.0, "blind_decisions": 1, "decisions": 100,
                "decision_latency_ms": {"p95": 9.0}, "decision_age_ms": {"p95": 20.0}}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for k in range(6):
                f.write(json.dumps(rec("A", "aero_forest_dense_s1000", k, True, "success")) + "\n")
                f.write(json.dumps(rec("B", "aero_forest_dense_s1000", k, k < 2, "success" if k < 2 else "collision")) + "\n")
            f.write(json.dumps(rec("B", "aero_forest_dense_s1000", 5, True, "success")) + "\n")   # re-flown: last wins
        with open(path, "a", encoding="utf-8") as f:                      # control-loop stalls (fly_policy.py)
            f.write(json.dumps({**rec("C", "aero_forest_dense_s1000", 0, True, "success"), "max_setpoint_gap_s": 0.02}) + "\n")
            f.write(json.dumps({**rec("C", "aero_forest_dense_s1000", 1, False, "collision"), "max_setpoint_gap_s": 0.74}) + "\n")
        out = report(load([path]), by="density", compare=("A", "B"))
        assert math.isclose(out["stats"]["C"]["setpoint_gap"], 0.38) and out["stats"]["C"]["setpoint_gap_max"] == 0.74
        assert not math.isfinite(out["stats"]["A"]["setpoint_gap_max"])                    # older records: no field
        lines = mission_lines(load([path]))
        assert len(lines) == 1 + 14 and "dense_s1000" in lines[1] and "CRU" not in lines[1]
        assert out["stats"]["A"]["missions"] == 6 and out["stats"]["A"]["success"] == 1.0
        assert out["stats"]["B"]["missions"] == 6 and math.isclose(out["stats"]["B"]["success"], 0.5)
        assert out["paired"] == {"missions": 6, "both": 3, "only_a": 3, "only_b": 0, "neither": 0,
                                 "p_value": mcnemar_exact(3, 0)}
    print("summarize_flights.py self-test passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        _self_test()
    else:
        parsed = parse_args()
        loaded = load(parsed.results)
        if parsed.missions:
            print("\n".join(mission_lines(loaded)) + "\n")
        report(loaded, parsed.by, parsed.compare)

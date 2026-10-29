# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/show_trace.py -- What the policy did in one mission, decision by decision (sim/fly_policy.py --trace).

    python3 sim/show_trace.py logs/trace/policy_cal_ep.pt_aero_forest_sparse_s2000_e000.jsonl
    python3 sim/show_trace.py logs/trace/*.jsonl --brief                  # one block per mission
    python3 sim/show_trace.py FILE --every 0.5 --last 3 --svg logs/trace/e000.svg

For each trace: the outcome; how the decisions split over executor modes, primitives (selected and after the
uncertainty gate), the blind rule (still / in view / behind / outside left-right / above-below) and the speed cap;
where the active waypoint lay relative to the heading and where the policy sent the vehicle (horizontal angle
between the first raw knot and the waypoint), whether it turned towards it; commanded and flown speeds and s_t;
the closest approach to every waypoint; a timeline every --every seconds and every decision of the last --last
seconds. --svg draws the path from above (trunks from the world's JSON sidecar, waypoints, blind decisions)
and the speeds, clearance and sigma_bar over time.

Angles are degrees: bearings in the heading frame (positive to the right), azimuth / elevation of u_t in the
camera (right / up positive). Primitives: CRU(ISE) LAT(ERAL_AVOID) VER(TICAL_AVOID) YAW(_SCAN) APP(ROACH)
BRK (BRAKE_HOVER). Modes: TRACK, BRAKE, HOLD.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SHORT = ("CRU", "LAT", "VER", "YAW", "APP", "BRK")
MODES = {-1: "-", 0: "TRACK", 1: "BRAKE", 2: "HOLD"}
BLIND = ("behind", "horizontal", "vertical", "corner", "invalid")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("traces", nargs="*", help="trace files (fly_policy.py --trace DIR)")
    p.add_argument("--brief", action="store_true", help="summary only, no timeline")
    p.add_argument("--every", type=float, default=1.0, help="timeline step (s)")
    p.add_argument("--last", type=float, default=2.0, help="print every decision of the last seconds")
    p.add_argument("--svg", default=None, help="draw the mission (one trace)")
    p.add_argument("--world_json", default=None, help="world sidecar for the trunks (default: from the record)")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if not args.self_test and not args.traces:
        p.error("give trace files (or --self-test)")
    if args.svg and len(args.traces) != 1:
        p.error("--svg draws one trace")
    return args


def load_trace(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    record: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except ValueError:
                continue
            if item.get("kind") == "mission":
                record = item
            elif item.get("kind") == "decision":
                rows.append(item)
    if not record:
        raise SystemExit(f"{path}: no mission line (not a fly_policy.py --trace file?)")
    return record, rows


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def bearing(row: Dict[str, Any]) -> float:
    """Horizontal bearing of the active waypoint in the heading frame (deg, right positive)."""
    g = row["goal"]
    return math.degrees(math.atan2(g[1], g[0]))


def command_bearing(row: Dict[str, Any], moving: float = 0.3) -> Optional[float]:
    """Horizontal direction of the first raw knot in the heading frame (deg), None if slower than ``moving``."""
    raw = row.get("raw0")
    if raw is None or math.hypot(raw[0], raw[1]) < moving:
        return None
    return math.degrees(math.atan2(raw[1], raw[0]))


def _median(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(values)) if values else None


def _f(x: Optional[float], digits: int = 1, unit: str = "") -> str:
    return "-" if x is None else f"{x:.{digits}f}{unit}"


def _shares(counter: Counter, total: int, names: Optional[Dict[Any, str]] = None) -> str:
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{(names or {}).get(k, k)} {100.0 * v / max(1, total):.0f}%" for k, v in items)


def analyse(record: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The numbers of the summary (also used by the self-test)."""
    n = len(rows)
    v_blind = float(record.get("blind_speed", 1.0))
    radius = float(record.get("waypoint_radius", 1.0))
    with_view = [r for r in rows if "view" in r]
    capped = [r for r in with_view if r.get("blind") and r.get("raw_peak", 0.0) > v_blind + 1e-6]
    bearings = [bearing(r) for r in rows]
    errors, toward, turning = [], 0, 0
    for r, b in zip(rows, bearings, strict=True):
        c = command_bearing(r)
        if c is not None:
            errors.append(abs(_wrap(c - b)))
        raw = r.get("raw0")
        if raw is not None and abs(b) > 20.0:
            turning += 1
            toward += int(raw[3] * b > 0.0)                      # yaw rate (right positive) towards the bearing
    wps: Dict[int, Dict[str, float]] = {}
    for r in rows:
        w = wps.setdefault(int(r["m"]), {"from": r["t"], "closest": math.inf, "at": r["t"]})
        if r["wp_dist"] < w["closest"]:
            w["closest"], w["at"] = r["wp_dist"], r["t"]
        w["to"] = r["t"]
    speeds = [float(np.linalg.norm(r["v"])) for r in rows]
    raw_speeds = [float(np.linalg.norm(r["raw0"][:3])) for r in rows if r.get("raw0") is not None]
    cmd_speeds = [float(np.linalg.norm(r["cmd0"][:3])) for r in rows if r.get("cmd0") is not None]
    return {
        "decisions": n, "v_blind": v_blind, "waypoint_radius": radius,
        "modes": Counter(MODES.get(int(r.get("mode", -1)), "?") for r in rows),
        "selected": Counter(SHORT[int(r["u_sel"])] for r in rows), "final": Counter(SHORT[int(r["u"])] for r in rows),
        "fallback": sum(bool(r["fallback"]) for r in rows),
        "views": Counter(r["view"] for r in with_view), "with_view": len(with_view), "capped": len(capped),
        "bearing_abs_median": _median([abs(b) for b in bearings]),
        "bearing_over_45": sum(abs(b) > 45.0 for b in bearings) / max(1, n),
        "command_error_median": _median(errors), "command_error_over_90": sum(e > 90.0 for e in errors) / max(1, len(errors)),
        "turns_toward": toward / turning if turning else None, "turning_decisions": turning,
        "speed_median": _median(speeds), "raw_speed_median": _median(raw_speeds), "cmd_speed_median": _median(cmd_speeds),
        "scale_median": _median([r["scale"] for r in rows if "scale" in r]),
        "sigma_bar_median": _median([r["sigma_bar"] for r in rows]),
        "clearance_min": min((r["clearance"] for r in rows if r.get("clearance") is not None), default=None),
        "waypoints": wps,
    }


def _row_line(r: Dict[str, Any]) -> str:
    p, v = r["p"], r["v"]
    raw, cmd = r.get("raw0"), r.get("cmd0")
    view = r.get("view", "-")
    blind = "*" if r.get("blind") else " "
    return (f"{r['t']:6.2f} {p[0]:7.1f} {p[1]:7.1f} {-p[2]:5.1f} {np.linalg.norm(v):5.2f} {r['yaw']:6.0f} {r['m']:2d} "
            f"{r['wp_dist']:6.1f} {bearing(r):5.0f}  {SHORT[r['u_sel']]}/{SHORT[r['u']]} {'F' if r['fallback'] else ' '} "
            f"{view[:6]:6s}{blind} {_f(r.get('az'), 0):>4s} {_f(r.get('el'), 0):>4s} {_f(r.get('d'), 1):>5s} "
            f"{_f(r.get('v_safe'), 2):>5s} {_f(r.get('scale'), 2):>5s} "
            f"{_f(None if raw is None else float(np.linalg.norm(raw[:3])), 2):>5s} "
            f"{_f(None if raw is None else command_bearing(r, 0.0), 0):>5s} {_f(None if raw is None else raw[3], 2):>6s} "
            f"{_f(None if cmd is None else float(np.linalg.norm(cmd[:3])), 2):>5s} {_f(r.get('clearance'), 2):>5s} "
            f"{MODES.get(int(r.get('mode', -1)), '?'):5s} {r['sigma_bar']:.3f}")


HEADER = ("     t       x       y   alt   spd    yaw  m  wpdst   brg  u_sel/u   view     az   el     d vsafe     s "
          " |raw|  rbrg  yawr  |cmd|   clr mode  sigma_bar")


def report(path: str, record: Dict[str, Any], rows: List[Dict[str, Any]], brief: bool, every: float, last: float) -> Dict[str, Any]:
    a = analyse(record, rows)
    n = max(1, a["decisions"])
    print(f"== {os.path.basename(path)}")
    print(f"   {record.get('tag')} on {record.get('world')}, mission {record.get('episode')}: {record.get('termination')} "
          f"after {_f(record.get('duration_s'))} s; {record.get('waypoints_reached')}/{record.get('waypoints')} waypoints; "
          f"{a['decisions']} decisions; smallest clearance {_f(record.get('min_clearance_m'), 2)} m; mean speed "
          f"{_f(record.get('mean_speed'), 2)} m/s")
    if not rows:
        print("   (no decisions)")
        return a
    print(f"   modes at decision time: {_shares(a['modes'], n)}; fallback {100.0 * a['fallback'] / n:.0f}% of decisions")
    print(f"   selected u_t: {_shares(a['selected'], n)}")
    if a["final"] != a["selected"]:
        print(f"   after the gate: {_shares(a['final'], n)}")
    wv = max(1, a["with_view"])
    views = a["views"]
    blind = sum(views.get(k, 0) for k in BLIND)
    reasons = ", ".join("%s %.0f%%" % (k, 100.0 * views.get(k, 0) / wv) for k in BLIND if views.get(k))
    print(f"   chunks {a['with_view']}: blind {100.0 * blind / wv:.0f}% ({reasons or 'none'}), capped at v_blind "
          f"{100.0 * a['capped'] / wv:.0f}%; in view: obstacle {100.0 * views.get('obstacle', 0) / wv:.0f}%, "
          f"clear {100.0 * views.get('clear', 0) / wv:.0f}%; still {100.0 * views.get('still', 0) / wv:.0f}%")
    print(f"   active waypoint off the heading: median {_f(a['bearing_abs_median'], 0)} deg, > 45 deg in "
          f"{100.0 * a['bearing_over_45']:.0f}% of decisions")
    toward = "-" if a["turns_toward"] is None else "%.0f%%" % (100.0 * a["turns_toward"])
    print(f"   first raw knot vs the waypoint (horizontal): median {_f(a['command_error_median'], 0)} deg, > 90 deg in "
          f"{100.0 * a['command_error_over_90']:.0f}%; turns towards it in {toward} of {a['turning_decisions']} "
          "decisions with the waypoint > 20 deg off")
    print(f"   speed median: raw first knot {_f(a['raw_speed_median'], 2)}, committed {_f(a['cmd_speed_median'], 2)}, flown "
          f"{_f(a['speed_median'], 2)} m/s; s_t median {_f(a['scale_median'], 2)}; sigma_bar median "
          f"{_f(a['sigma_bar_median'], 3)} (lo {_f(record.get('sigma_lo'), 3)}, crit {_f(record.get('sigma_crit'), 3)})")
    for m, w in sorted(a["waypoints"].items()):
        print(f"   waypoint {m}: active {w['from']:.1f}-{w['to']:.1f} s, closest {w['closest']:.2f} m at {w['at']:.1f} s"
              f"{'' if w['closest'] >= a['waypoint_radius'] else ' (reached)'}")
    if brief:
        return a
    print("   timeline (* blind; F fallback; brg: waypoint bearing; rbrg / yawr: raw first knot bearing and yaw rate; "
          "|cmd|: after s_t and the rate limit)")
    print("   " + HEADER)
    t_next = -math.inf
    t_end = rows[-1]["t"]
    for r in rows:
        if r["t"] >= t_end - last:
            break
        if r["t"] >= t_next:
            print("   " + _row_line(r))
            t_next = r["t"] + every
    print(f"   last {last:.1f} s, every decision:")
    for r in rows:
        if r["t"] >= t_end - last:
            print("   " + _row_line(r))
    return a


# ---------------------------------------------------------------------------------------------
# SVG (no plotting library needed)
# ---------------------------------------------------------------------------------------------
def svg(record: Dict[str, Any], rows: List[Dict[str, Any]], world_json: Optional[str], path: str) -> None:
    pos = np.array([r["p"] for r in rows]) if rows else np.zeros((0, 3))
    wps = np.asarray(record.get("waypoints_ned", []), dtype=np.float64).reshape(-1, 3)
    home = np.asarray(record.get("home", [0.0, 0.0, 0.0]), dtype=np.float64)
    pts = np.vstack([pos[:, :2], wps[:, :2], home[None, :2]]) if len(pos) else np.vstack([wps[:, :2], home[None, :2]])
    lo, hi = pts.min(axis=0) - 4.0, pts.max(axis=0) + 4.0
    size, pad, head = 640.0, 20.0, 40.0
    span = float(max(hi - lo))
    k = (size - 2 * pad) / span

    def xy(n: float, e: float) -> Tuple[float, float]:                       # north up, east right
        return pad + (e - lo[1]) * k, head + size - pad - (n - lo[0]) * k

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size:.0f}" height="{head + size + 280:.0f}" '
           f'font-family="sans-serif" font-size="11">', '<rect width="100%" height="100%" fill="white"/>',
           f'<defs><clipPath id="map"><rect x="0" y="{head}" width="{size}" height="{size}"/></clipPath></defs>',
           f'<rect x="0.5" y="{head + 0.5}" width="{size - 1}" height="{size - 1}" fill="none" stroke="#ccc"/>',
           '<g clip-path="url(#map)">']
    trunks = []
    wj = world_json or record.get("world_json") or ""
    if wj and os.path.isfile(wj):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import expert as E

        world = E.load_world(wj)
        for o in world.obstacles:
            c = o.center
            if not (lo[0] - 3 <= c[0] <= hi[0] + 3 and lo[1] - 3 <= c[1] <= hi[1] + 3):
                continue
            if o.kind == "cylinder":
                trunks.append((c[0], c[1], o.radius, "#8b5a2b", ""))
            elif o.kind == "sphere" and -(c[2]) - o.radius < 5.0:           # canopies that reach below 5 m
                trunks.append((c[0], c[1], o.radius, "#3a7d3a", ' stroke-dasharray="3,3" fill="none"'))
    for n_, e_, r_, col, extra in trunks:
        x, y = xy(n_, e_)
        fill = "" if "fill" in extra else f' fill="{col}"'
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(1.0, r_ * k):.1f}" stroke="{col}"{fill}{extra}/>')
    for i in range(1, len(pos)):
        x0, y0 = xy(pos[i - 1, 0], pos[i - 1, 1])
        x1, y1 = xy(pos[i, 0], pos[i, 1])
        r = rows[i]
        col = "#d62728" if r.get("blind") else ("#ff7f0e" if r.get("fallback") or int(r.get("mode", 0)) != 0 else "#1f77b4")
        out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{col}" stroke-width="2"/>')
    hx, hy = xy(home[0], home[1])
    out.append(f'<rect x="{hx - 4:.1f}" y="{hy - 4:.1f}" width="8" height="8" fill="black"/>')
    for i, w in enumerate(wps):
        x, y = xy(w[0], w[1])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(3.0, k * 1.0):.1f}" fill="none" stroke="black"/>'
                   f'<text x="{x + 6:.1f}" y="{y - 6:.1f}">{i + 1}</text>')
    if len(pos):
        x, y = xy(pos[-1, 0], pos[-1, 1])
        out.append(f'<text x="{x + 6:.1f}" y="{y + 12:.1f}" fill="#d62728">{record.get("termination", "")}</text>')
    out.append("</g>")
    out.append(f'<text x="{pad}" y="16">{record.get("tag")} {record.get("world")} mission {record.get("episode")}: '
               f'{record.get("termination")} after {_f(record.get("duration_s"))} s, '
               f'{record.get("waypoints_reached")}/{record.get("waypoints")} waypoints</text>')
    out.append(f'<text x="{pad}" y="31" fill="#555">north up, {span:.0f} m across | path: blue track, orange brake, '
               'red blind | brown trunks, dashed canopies, black waypoints</text>')
    # time series: flown speed, raw and committed first-knot speed, clearance, sigma_bar / sigma_crit
    if rows:
        top, h, w = head + size + 30.0, 220.0, size - 2 * pad
        t = np.array([r["t"] for r in rows])
        t_max = max(float(t[-1]), 1e-3)
        series = [
            ("flown speed", "#1f77b4", [float(np.linalg.norm(r["v"])) for r in rows], 5.0),
            ("raw |v0|", "#9467bd", [float(np.linalg.norm(r["raw0"][:3])) if r.get("raw0") else math.nan for r in rows], 5.0),
            ("committed |v0|", "#2ca02c", [float(np.linalg.norm(r["cmd0"][:3])) if r.get("cmd0") else math.nan for r in rows], 5.0),
            ("clearance", "#8b5a2b", [r["clearance"] if r.get("clearance") is not None else math.nan for r in rows], 5.0),
            ("sigma_bar/crit", "#d62728", [r["sigma_bar"] / max(1e-9, float(record.get("sigma_crit") or 1.0)) * 5.0
                                           for r in rows], 5.0),
        ]
        out.append(f'<rect x="{pad}" y="{top}" width="{w}" height="{h}" fill="none" stroke="#999"/>')
        for r in rows:
            if r.get("blind"):
                x = pad + w * r["t"] / t_max
                out.append(f'<line x1="{x:.1f}" y1="{top + h:.1f}" x2="{x:.1f}" y2="{top + h - 6:.1f}" stroke="#d62728"/>')
        for j, (label, col, vals, vmax) in enumerate(series):
            ptsl = [f"{pad + w * ti / t_max:.1f},{top + h - h * min(max(v, 0.0), vmax) / vmax:.1f}"
                    for ti, v in zip(t, vals, strict=True) if math.isfinite(v)]
            if ptsl:
                out.append(f'<polyline points="{" ".join(ptsl)}" fill="none" stroke="{col}" stroke-width="1.2"/>')
            out.append(f'<text x="{pad + 4 + 120 * j}" y="{top + h + 16}" fill="{col}">{label}</text>')
        out.append(f'<text x="{pad}" y="{top - 4}">0-5 m/s, 0-5 m, sigma_bar/sigma_crit x5 (red ticks: blind); '
                   f'0-{t_max:.0f} s</text>')
    out.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    for path in args.traces:
        record, rows = load_trace(path)
        report(path, record, rows, args.brief, args.every, args.last)
        print()
        if args.svg:
            svg(record, rows, args.world_json, args.svg)
            print(f"wrote {args.svg}")
    return 0


def _self_test() -> None:
    import tempfile

    rec = {"kind": "mission", "tag": "p.pt", "world": "w", "episode": 0, "termination": "collision", "duration_s": 3.0,
           "waypoints": 2, "waypoints_reached": 0, "waypoints_ned": [[10.0, 0.0, -2.5], [20.0, 0.0, -2.5]],
           "home": [0.0, 0.0, -2.5], "blind_speed": 1.0, "waypoint_radius": 1.0, "sigma_lo": 0.1, "sigma_crit": 0.5,
           "min_clearance_m": 0.3, "mean_speed": 1.0, "world_json": ""}
    rows = []
    for i in range(30):
        t = 0.1 * i
        side = i >= 15                                       # second half: sideways command, waypoint 60 deg right
        rows.append({"kind": "decision", "t": t, "p": [t, 0.0, -2.5], "v": [1.0, 0.0, 0.0], "yaw": 0.0, "m": 1,
                     "wp_dist": 10.0 - t, "goal": [0.5, 0.866, 0.0, 0.5] if side else [1.0, 0.0, 0.0, 0.5],
                     "u_sel": 1 if side else 0, "u": 1 if side else 0, "fallback": False, "sigma_bar": 0.05,
                     "mode": 0, "clearance": 2.0 - 0.05 * i, "raw0": [0.0, 2.0, 0.0, -0.3] if side else [2.0, 0.0, 0.0, 0.0],
                     "raw_peak": 2.0, "cmd0": [0.0, 1.0, 0.0, -0.3] if side else [2.0, 0.0, 0.0, 0.0], "v_safe": 1.0,
                     "scale": 0.5 if side else 1.0, "d": 19.1, "blind": side, "view": "behind" if side else "clear",
                     "az": 90.0 if side else 0.0, "el": 0.0})
    a = analyse(rec, rows)
    assert a["decisions"] == 30 and a["views"]["behind"] == 15 and a["capped"] == 15
    assert a["selected"]["LAT"] == 15 and a["modes"]["TRACK"] == 30
    assert abs(a["command_error_median"] - 15.0) < 0.01           # half 0 deg, half 30 deg off -> median 15
    assert a["turns_toward"] == 0.0 and a["turning_decisions"] == 15   # waypoint right, yaw rate left
    assert a["waypoints"][1]["closest"] == rows[-1]["wp_dist"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in [rec, *rows]:
                f.write(json.dumps(item) + "\n")
        out_svg = os.path.join(tmp, "t.svg")
        assert main([path, "--svg", out_svg, "--every", "0.5", "--last", "0.5"]) == 0
        text = open(out_svg, encoding="utf-8").read()
        assert text.startswith("<svg") and text.rstrip().endswith("</svg>") and "polyline" in text
        assert main([path, "--brief"]) == 0
    print("show_trace.py self-test passed")


if __name__ == "__main__":
    sys.exit(main())

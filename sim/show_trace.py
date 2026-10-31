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

--response fits the vehicle's horizontal velocity response to the setpoints that were dispatched (all traces
pooled, then by who flew): v' = k a_ff(t - d) + (v_sp(t - d) - v) / tau, at most 7 m/s^2, simulated over 1 s windows
from the flown velocity. It reports the delay d, the time constant tau and the share k of the acceleration
feed-forward that the vehicle follows, and how well the closed-loop prediction of sim/dagger.py (d 0.2 s, tau
0.5 s, k 1) and the same without feed-forward match. Traces before 2026-10-31 do not record the feed-forward
(setpoint_acc); k is then not identifiable and is reported as "-".

--prediction compares, at every check of sim/dagger.py while the expert flew, the smallest clearance it predicted
for the way it took with the smallest clearance then flown over the same horizon (reaction 0.3 s plus the
minimum-jerk braking time at the flown speed, at least 0.5 s), kept only when the expert flew all of it: how
optimistic the predictions are, and the conflict threshold that would cover them (traces since 9c8b3e7).

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
MODES = {-2: "EXPERT", -1: "-", 0: "TRACK", 1: "BRAKE", 2: "HOLD"}
BLIND = ("behind", "horizontal", "vertical", "corner", "invalid")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("traces", nargs="*", help="trace files (fly_policy.py --trace DIR)")
    p.add_argument("--brief", action="store_true", help="summary only, no timeline")
    p.add_argument("--every", type=float, default=1.0, help="timeline step (s)")
    p.add_argument("--last", type=float, default=2.0, help="print every decision of the last seconds")
    p.add_argument("--window", type=float, default=None, help="timeline of the last seconds only (default: all)")
    p.add_argument("--svg", default=None, help="draw the mission (one trace)")
    p.add_argument("--world_json", default=None, help="world sidecar for the trunks (default: from the record)")
    p.add_argument("--response", action="store_true", help="fit the velocity response to the dispatched setpoints")
    p.add_argument("--prediction", action="store_true", help="the expert's predicted against flown clearance")
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
    disagreement, dir_err = [], []
    for r in rows:                                   # DAgger: the policy's raw first knot against the expert's label
        if r.get("controller") == "policy" and r.get("raw0") is not None and r.get("label0") is not None:
            raw, lab = np.asarray(r["raw0"][:3]), np.asarray(r["label0"][:3])
            disagreement.append(float(np.linalg.norm(raw - lab)))
            if np.linalg.norm(raw) > 0.3 and np.linalg.norm(lab) > 0.3:
                cos = float(raw @ lab / (np.linalg.norm(raw) * np.linalg.norm(lab)))
                dir_err.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
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
        "dagger_modes": Counter(r["dagger_mode"] for r in rows if "dagger_mode" in r),
        "labelled": Counter(r["controller"] for r in rows if "controller" in r),
        "label_error_median": _median(disagreement), "label_direction_median": _median(dir_err),
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
            f"{MODES.get(int(r.get('mode', -1)), '?'):6s} {r['sigma_bar']:.3f}"
            + (f"  {r['dagger_mode'][:6]:6s} {_f(r.get('e_path'), 2):>5s} "
               f"{_f(None if r.get('label0') is None else float(np.linalg.norm(r['label0'][:3])), 2):>5s} "
               f"{_f(r.get('dagger_cap'), 1):>4s} {_choice(r)}"
               if "dagger_mode" in r else ""))


HEADER = ("     t       x       y   alt   spd    yaw  m  wpdst   brg  u_sel/u   view     az   el     d vsafe     s "
          " |raw|  rbrg  yawr  |cmd|   clr mode   sigma_bar")
DAGGER_HEADER = "  dagger  epath |lab|  cap  way  pred  gap"


def _choice(r: Dict[str, Any]) -> str:
    """The expert's last way (fly / slow / stop), its predicted smallest clearance and the longest setpoint gap
    since the previous decision (ms), when the trace has them (fly_policy.py since 2026-10-31)."""
    c = r.get("dagger_choice") or {}
    way = c.get("way")
    return (f"{(way or '-')[:4]:>4s} {_f(c.get(way) if way else None, 2):>5s} "
            f"{_f(r.get('tick_gap_ms'), 0):>4s}")


def report(path: str, record: Dict[str, Any], rows: List[Dict[str, Any]], brief: bool, every: float, last: float,
           window: Optional[float] = None) -> Dict[str, Any]:
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
    if a["dagger_modes"]:
        tk = record.get("dagger_takeovers") or {}
        print(f"   DAgger: {_shares(a['dagger_modes'], n)} of decisions; labelled {dict(a['labelled'])}; takeovers "
              f"{', '.join(f'{k} {v}' for k, v in tk.items() if v) or 'none'}; policy vs expert label (policy frames): "
              f"|v0 - v0*| median {_f(a['label_error_median'], 2)} m/s, direction {_f(a['label_direction_median'], 0)} deg")
    for m, w in sorted(a["waypoints"].items()):
        print(f"   waypoint {m}: active {w['from']:.1f}-{w['to']:.1f} s, closest {w['closest']:.2f} m at {w['at']:.1f} s"
              f"{'' if w['closest'] >= a['waypoint_radius'] else ' (reached)'}")
    if brief:
        return a
    print("   timeline (* blind; F fallback; brg: waypoint bearing; rbrg / yawr: raw first knot bearing and yaw rate; "
          "|cmd|: after s_t and the rate limit" + ("; dagger: who flies, epath: distance to the expert path, |lab|: "
                                                  "speed of the expert's label, cap: the expert's speed governor, way "
                                                  "/ pred: its last choice and that way's predicted smallest "
                                                  "clearance, gap: longest setpoint gap (ms)"
                                                  if a["dagger_modes"] else "") + ")")
    events = []
    for r in rows:
        ev = r.get("dagger_event")
        if ev and (not events or events[-1] != ev):
            events.append(ev)
    if events:
        print("   expert takeovers and stops (t, reason, way): "
              + "; ".join(f"{e[0]:.1f} {e[1]} {e[2]}" for e in events[-8:]))
    print("   " + HEADER + (DAGGER_HEADER if a["dagger_modes"] else ""))
    t_end = rows[-1]["t"]
    t_next = -math.inf if window is None else t_end - window
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


# ---------------------------------------------------------------------------------------------
# --response: the vehicle's velocity response to the dispatched setpoints
# ---------------------------------------------------------------------------------------------
RESPONSE_MODEL = (0.24, 0.5, 1.0)         # delay_s, response_s and feed-forward share of sim/dagger.py's prediction


def response_samples(rows: List[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    """Flown horizontal velocity at the frame stamps and the last dispatched setpoint (horizontal velocity and
    acceleration feed-forward) at about the time it went out (stamp + age - compute), with who dispatched it."""
    keep = [r for r in rows if r.get("setpoint") is not None and int(r.get("mode", -1)) != -1]
    if len(keep) < 10:
        return None
    t = np.array([float(r["t"]) for r in keep])
    out = np.array([max(0.0, float(r.get("age_ms") or 0.0) - float(r.get("compute_ms") or 0.0)) / 1e3 for r in keep])
    return {"t": t, "v": np.array([r["v"][:2] for r in keep], dtype=np.float64),
            "t_sp": t + out, "sp": np.array([r["setpoint"][:2] for r in keep], dtype=np.float64),
            "ff": np.array([(r.get("setpoint_acc") or [0.0, 0.0])[:2] for r in keep], dtype=np.float64),
            "has_ff": np.array([r.get("setpoint_acc") is not None for r in keep]),
            "expert": np.array([int(r.get("mode", -1)) == -2 for r in keep])}


def response_windows(samples: Dict[str, np.ndarray], window: float = 1.0, step: float = 0.25, lead: float = 0.6,
                     max_gap: float = 0.2) -> List[Tuple[int, int]]:
    """(first, last) row indices of the fitting windows: ``window`` s of rows without a gap above ``max_gap``,
    preceded by ``lead`` s of setpoints (the largest delay tried), starting at least ``step`` s apart."""
    t = samples["t"]
    breaks = np.concatenate([[0], np.nonzero(np.diff(t) > max_gap)[0] + 1, [len(t)]])
    out: List[Tuple[int, int]] = []
    for a, b in zip(breaks[:-1], breaks[1:], strict=True):
        next_start = t[a] + lead
        for i in range(a, b):
            if t[i] < next_start:
                continue
            if t[i] + window > t[b - 1] + 1e-9:
                break
            j = a + int(np.searchsorted(t[a:b], t[i] + window, side="right")) - 1
            if j > i:
                out.append((i, j))
            next_start = t[i] + step
    return out


def fit_response(groups: Sequence[Tuple[Dict[str, np.ndarray], List[Tuple[int, int]]]], dt: float = 0.01,
                 window: float = 1.0, a_max: float = 7.0, delays: Optional[np.ndarray] = None,
                 taus: Optional[np.ndarray] = None, shares: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                 extra: Sequence[Tuple[float, float, float]] = (RESPONSE_MODEL, RESPONSE_MODEL[:2] + (0.0,))
                 ) -> Dict[str, Any]:
    """Grid search of (delay, tau, share) minimising the squared error of the simulated horizontal velocity at the
    rows of every window (simulation from the flown velocity at the window start). Returns the best point, its
    RMSE (m/s) and the RMSE of each ``extra`` point."""
    delays = np.round(np.arange(0.0, 0.52, 0.02), 3) if delays is None else delays
    taus = np.round(np.arange(0.2, 1.25, 0.05), 3) if taus is None else taus
    steps = int(round(window / dt))
    rel = dt * np.arange(1, steps + 1)                                           # simulation times after t0
    starts, v0, eval_j, eval_v, mask, owners = [], [], [], [], [], []
    for gi, (smp, wins) in enumerate(groups):
        for i, j in wins:
            t0 = smp["t"][i]
            idx = np.arange(i + 1, j + 1)
            starts.append(t0)
            v0.append(smp["v"][i])
            eval_j.append(np.clip(np.round((smp["t"][idx] - t0) / dt).astype(int) - 1, 0, steps - 1))
            eval_v.append(smp["v"][idx])
            owners.append(gi)
    n = len(starts)
    if n == 0:
        return {"windows": 0}
    width = max(len(e) for e in eval_j)
    ej = np.zeros((n, width), dtype=int)
    ev = np.zeros((n, width, 2))
    mask = np.zeros((n, width))
    for w in range(n):
        k = len(eval_j[w])
        ej[w, :k], ev[w, :k], mask[w, :k] = eval_j[w], eval_v[w], 1.0
    v0_arr = np.array(v0)
    owners_arr = np.array(owners)
    rows = np.arange(n)[:, None]

    def inputs(d: float) -> Tuple[np.ndarray, np.ndarray]:
        u, f = np.zeros((n, steps, 2)), np.zeros((n, steps, 2))
        for gi, (smp, _) in enumerate(groups):
            sel = np.nonzero(owners_arr == gi)[0]
            if sel.size == 0:
                continue
            q = (np.array(starts)[sel, None] + rel[None] - d).ravel()
            for c in range(2):
                u[sel, :, c] = np.interp(q, smp["t_sp"], smp["sp"][:, c]).reshape(sel.size, steps)
                f[sel, :, c] = np.interp(q, smp["t_sp"], smp["ff"][:, c]).reshape(sel.size, steps)
        return u, f

    def sse(u: np.ndarray, f: np.ndarray, tau: float, share: float) -> float:
        v = v0_arr.copy()
        hist = np.empty((n, steps, 2))
        for s in range(steps):
            acc = share * f[:, s] + (u[:, s] - v) / tau
            norm = np.linalg.norm(acc, axis=1, keepdims=True)
            acc *= np.minimum(1.0, a_max / np.maximum(norm, 1e-9))
            v = v + dt * acc
            hist[:, s] = v
        err = hist[rows, ej] - ev
        return float(np.sum(mask[..., None] * err * err))

    count = float(mask.sum())
    best = (math.inf, 0.0, 0.0, 0.0)
    for d in delays:
        u, f = inputs(float(d))
        for tau in taus:
            for share in shares:
                e = sse(u, f, float(tau), float(share))
                if e < best[0]:
                    best = (e, float(d), float(tau), float(share))
    out = {"windows": n, "points": int(count), "delay": best[1], "tau": best[2], "share": best[3],
           "rmse": math.sqrt(best[0] / count)}
    for d, tau, share in extra:
        u, f = inputs(d)
        out[(d, tau, share)] = math.sqrt(sse(u, f, tau, share) / count)
    return out


def response_report(paths: Sequence[str], delays: Optional[np.ndarray] = None,
                    taus: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
    """Print the --response fit for the pooled traces and by who flew; returns the fits."""
    traces = []
    for path in paths:
        _, rows = load_trace(path)
        smp = response_samples(rows)
        if smp is not None:
            traces.append(smp)
    has_ff = bool(traces) and all(bool(s["has_ff"].all()) for s in traces)
    shares = (0.0, 0.25, 0.5, 0.75, 1.0) if has_ff else (0.0,)
    fits: Dict[str, Dict[str, Any]] = {}
    for name, who in (("all", None), ("expert", True), ("policy", False)):
        groups = []
        for smp in traces:
            wins = [(i, j) for i, j in response_windows(smp)
                    if who is None or bool(np.all(smp["expert"][i:j + 1] == who))]
            groups.append((smp, wins))
        fits[name] = fit_response(groups, shares=shares, delays=delays, taus=taus)
    m = RESPONSE_MODEL
    print(f"velocity response (horizontal), {len(traces)} traces, 1 s windows from the flown velocity"
          + ("" if has_ff else "; no feed-forward recorded (traces before 2026-10-31): share not identifiable"))
    print(f"   {'flown by':8s} {'windows':>7s} {'delay s':>8s} {'tau s':>6s} {'ff share':>8s} {'rmse m/s':>9s}   "
          f"prediction {m[0]:.2f}/{m[1]:.1f}/{m[2]:.0f}   without ff {m[0]:.2f}/{m[1]:.1f}/0")
    for name, fit in fits.items():
        if not fit.get("windows"):
            print(f"   {name:8s} {0:7d}")
            continue
        share = "%.2f" % fit["share"] if has_ff else "-"
        print(f"   {name:8s} {fit['windows']:7d} {fit['delay']:8.2f} {fit['tau']:6.2f} {share:>8s} "
              f"{fit['rmse']:9.3f}   {fit[m]:16.3f}   {fit[(m[0], m[1], 0.0)]:14.3f}")
    return fits


# ---------------------------------------------------------------------------------------------
# --prediction: what the expert's closed-loop prediction expected against what was flown
# ---------------------------------------------------------------------------------------------
COLLISION_CLEARANCE = 0.35                # fly_policy.py --collision_distance


def prediction_checks(record: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every check (``dagger_choice``) made while the expert flew, with the smallest clearance predicted for the way it
    took (and for flying on) and the smallest clearance flown after it over the prediction horizon, when the expert
    flew all of that horizon (a mission ending in a crash counts up to the crash)."""
    if not rows:
        return []
    t = np.array([float(r["t"]) for r in rows])
    clr = np.array([math.nan if r.get("clearance") is None else float(r["clearance"]) for r in rows])
    expert = np.array([r.get("dagger_mode") == "EXPERT" for r in rows])
    crashed = record.get("termination") in ("collision", "ground")
    out, seen = [], set()
    for i, r in enumerate(rows):
        c, ct = r.get("dagger_choice"), r.get("dagger_choice_t")
        if not expert[i] or not c or ct is None or ct in seen or not (t[i] - 0.06 <= ct <= t[i] + 0.06):
            continue
        seen.add(ct)
        way = c.get("way")
        pred = c.get(way) if way else None
        if pred is None:
            continue
        speed = float(np.linalg.norm(r["v"]))
        horizon = max(0.5, 0.3 + 1.5 * speed / 5.0)
        after = (t > ct) & (t <= ct + horizon)
        if not after.any() or (t[-1] < ct + horizon and not crashed):
            continue
        idx = np.nonzero(after)[0]
        if not expert[idx].all() or np.all(np.isnan(clr[idx])):
            continue
        out.append({"t": float(ct), "speed": speed, "way": way, "pred": float(pred), "fly": c.get("fly"),
                    "threshold": float(c.get("threshold", math.nan)), "flown": float(np.nanmin(clr[idx]))})
    return out


def takeover_clearances(rows: List[Dict[str, Any]], after: float = 2.0) -> List[Dict[str, Any]]:
    """Every handover from the policy to the expert: the clearance and speed at the first expert decision, the reason
    (``dagger_event``) and the smallest clearance over the next ``after`` seconds."""
    out = []
    for i in range(1, len(rows)):
        if rows[i].get("dagger_mode") != "EXPERT" or rows[i - 1].get("dagger_mode") != "POLICY":
            continue
        t0 = float(rows[i]["t"])
        nxt = [float(r["clearance"]) for r in rows[i:] if r["t"] <= t0 + after and r.get("clearance") is not None]
        ev = rows[i].get("dagger_event") or [None, "?", "?"]
        if rows[i].get("clearance") is not None and nxt:
            out.append({"t": t0, "clearance": float(rows[i]["clearance"]), "speed": float(np.linalg.norm(rows[i]["v"])),
                        "reason": ev[1], "way": ev[2], "next_min": min(nxt)})
    return out


def prediction_report(paths: Sequence[str]) -> Dict[str, Any]:
    """Print how optimistic the expert's predictions were (predicted minus flown smallest clearance), and how close
    to obstacles the policy handed over."""
    checks: List[Dict[str, Any]] = []
    handovers: List[Dict[str, Any]] = []
    used = 0
    for path in paths:
        record, rows = load_trace(path)
        found = prediction_checks(record, rows)
        used += bool(found)
        checks.extend(found)
        handovers.extend(takeover_clearances(rows))
    if handovers:
        print(f"handovers from the policy to the expert: {len(handovers)}; clearance at the handover and smallest in the "
              "next 2 s (m):")
        for reason in sorted({h["reason"] for h in handovers}):
            hs = [h for h in handovers if h["reason"] == reason]
            at, nxt = np.array([h["clearance"] for h in hs]), np.array([h["next_min"] for h in hs])
            print(f"   {str(reason):16s} {len(hs):5d}   at handover median {np.median(at):.2f}, p10 {np.percentile(at, 10):.2f}, "
                  f"min {at.min():.2f}; next 2 s median {np.median(nxt):.2f}, p10 {np.percentile(nxt, 10):.2f}, min {nxt.min():.2f}; "
                  f"below 0.5 m: {int((nxt < 0.5).sum())}")
    print(f"expert's predicted against flown smallest clearance: {len(checks)} checks in {used} of {len(paths)} traces "
          "(the expert flying the whole horizon)")
    if not checks:
        print("   none: traces before 9c8b3e7 do not record the expert's checks")
        return {"checks": 0}
    opt = np.array([c["pred"] - c["flown"] for c in checks])

    def line(label: str, sel: np.ndarray) -> None:
        if sel.any():
            q = np.percentile(opt[sel], [50, 90, 95, 99])
            print(f"   {label:16s} {int(sel.sum()):6d}   {q[0]:+6.2f} {q[1]:+6.2f} {q[2]:+6.2f} {q[3]:+6.2f} {opt[sel].max():+6.2f}")

    print(f"   {'checks':16s} {'n':>6s}   {'median':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s} {'max':>6s}  (predicted - flown, m)")
    line("all", np.ones(len(checks), dtype=bool))
    for way in ("fly", "slow", "stop"):
        line(f"way {way}", np.array([c["way"] == way for c in checks]))
    speed = np.array([c["speed"] for c in checks])
    for label, lo, hi in (("below 1 m/s", 0.0, 1.0), ("1-3 m/s", 1.0, 3.0), ("above 3 m/s", 3.0, math.inf)):
        line(label, (speed >= lo) & (speed < hi))
    pred, flown, thr = (np.array([c[k] for c in checks]) for k in ("pred", "flown", "threshold"))
    kept = pred >= thr
    miss = kept & (flown < thr)
    print(f"   predicted to keep the threshold: {int(kept.sum())}; flew below it: {int(miss.sum())}; below the collision "
          f"line {COLLISION_CLEARANCE} m: {int((flown < COLLISION_CLEARANCE).sum())} (of which predicted kept: "
          f"{int((kept & (flown < COLLISION_CLEARANCE)).sum())})")
    p95 = float(np.percentile(opt, 95))
    print(f"   a threshold covering 95% of these errors: {COLLISION_CLEARANCE} + {p95:.2f} = "
          f"{COLLISION_CLEARANCE + p95:.2f} m (now 0.45 m in flight, 0.40 m when already inside, 0.50 m at takeovers, "
          "each + 0.06 m per m/s of predicted speed)")
    return {"checks": len(checks), "p95": p95, "misses": int(miss.sum())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.response:
        response_report(args.traces)
        return 0
    if args.prediction:
        prediction_report(args.traces)
        return 0
    for path in args.traces:
        record, rows = load_trace(path)
        report(path, record, rows, args.brief, args.every, args.last, args.window)
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
    rows[-1].update(dagger_mode="EXPERT", e_path=0.4, label0=[1.0, 0.0, 0.0, 0.0], dagger_cap=1.5, tick_gap_ms=12.0,
                    dagger_choice={"way": "slow", "threshold": 0.45, "fly": 0.3, "slow": 0.52},
                    dagger_event=[2.8, "obstacle", "slow"])
    assert _choice(rows[-1]).split() == ["slow", "0.52", "12"] and _choice(rows[0]).split() == ["-", "-", "-"]
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
        assert main([path, "--every", "0.1", "--last", "0", "--window", "0.5"]) == 0
    # --response: a vehicle with a 0.24 s delay, a 0.6 s time constant and a quarter of the feed-forward
    rng = np.random.default_rng(3)
    d_true, tau_true, k_true, fine = 0.24, 0.6, 0.25, 0.001

    def sp_ff(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sp = np.stack([2.5 * np.sin(0.9 * t) + np.sin(2.3 * t + 1.0), 2.0 * np.cos(0.7 * t) + 0.8 * np.sin(1.9 * t)], -1)
        ff = np.stack([2.25 * np.cos(0.9 * t) + 2.3 * np.cos(2.3 * t + 1.0), -1.4 * np.sin(0.7 * t) + 1.52 * np.cos(1.9 * t)], -1)
        return sp, ff

    grid = np.arange(0.0, 24.0, fine)
    sp_g, ff_g = sp_ff(np.maximum(grid - d_true, 0.0))
    vel = np.zeros((len(grid), 2))
    for i in range(1, len(grid)):
        acc = k_true * ff_g[i - 1] + (sp_g[i - 1] - vel[i - 1]) / tau_true
        vel[i] = vel[i - 1] + fine * acc * min(1.0, 7.0 / max(float(np.linalg.norm(acc)), 1e-9))
    stamps = np.cumsum(rng.uniform(0.025, 0.04, 800))
    stamps = stamps[stamps < 23.5]
    sp_r, ff_r = sp_ff(stamps + 0.015)                                    # dispatched 15 ms after the stamp
    flown = np.stack([np.interp(stamps, grid, vel[:, 0]), np.interp(stamps, grid, vel[:, 1])], -1)
    trace = [{"t": float(t), "v": [float(v[0]), float(v[1]), 0.0], "setpoint": [float(a[0]), float(a[1]), 0.0, 0.0],
              "setpoint_acc": [float(c[0]), float(c[1]), 0.0], "age_ms": 35.0, "compute_ms": 20.0,
              "mode": -2 if t < 12.0 else 0}
             for t, v, a, c in zip(stamps, flown, sp_r, ff_r, strict=True)]
    smp = response_samples(trace)
    assert smp is not None and bool(smp["has_ff"].all()) and abs(float(smp["t_sp"][0] - smp["t"][0]) - 0.015) < 1e-9
    wins = response_windows(smp)
    assert wins and all(smp["t"][j] - smp["t"][i] <= 1.0 + 1e-9 and smp["t"][i] >= 0.6 for i, j in wins)
    fit = fit_response([(smp, wins)], delays=np.array([0.16, 0.2, 0.24, 0.28, 0.32]),
                       taus=np.array([0.4, 0.5, 0.6, 0.7, 0.8]), shares=(0.0, 0.25, 0.5, 1.0))
    assert (fit["delay"], fit["tau"], fit["share"]) == (0.24, 0.6, 0.25), fit
    assert fit["rmse"] < 0.02 and fit[RESPONSE_MODEL] > 5.0 * fit["rmse"], fit           # the true model fits best
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in [{"kind": "mission", "tag": "p"}, *({"kind": "decision", **r} for r in trace)]:
                f.write(json.dumps(item) + "\n")
        fits = response_report([path], delays=np.array([0.2, 0.24]), taus=np.array([0.5, 0.6]))
        assert fits["expert"]["windows"] > 0 and fits["policy"]["windows"] > 0
        assert fits["all"]["windows"] >= fits["expert"]["windows"] + fits["policy"]["windows"]
    # --prediction: checks every 50 ms predicting 0.6 m while the flown clearance falls to 0.5 m (0.1 m optimistic)
    exp_rows = [{"t": round(0.03 * i, 2), "v": [2.0, 0.0, 0.0], "clearance": max(0.5, 1.5 - 0.03 * i),
                 "dagger_mode": "EXPERT" if 10 <= i < 80 else "POLICY", "dagger_event": [0.3, "obstacle", "fly"],
                 "dagger_choice": {"way": "fly", "threshold": 0.45, "fly": 0.6}, "dagger_choice_t": round(0.05 * (i * 3 // 5), 2)}
                for i in range(100)]
    checks = prediction_checks({"termination": "success"}, exp_rows)
    assert checks and all(c["way"] == "fly" and c["t"] + 0.9 <= 2.4 + 1e-9 for c in checks)       # expert all along
    late = [c for c in checks if c["t"] >= 0.15]
    assert late and all(abs(c["flown"] - 0.5) < 1e-9 for c in late)
    hand = takeover_clearances(exp_rows)
    assert len(hand) == 1 and hand[0]["reason"] == "obstacle" and abs(hand[0]["clearance"] - 1.2) < 1e-9
    assert abs(hand[0]["next_min"] - 0.5) < 1e-9
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in [{"kind": "mission", "termination": "success"}, *({"kind": "decision", **r} for r in exp_rows)]:
                f.write(json.dumps(item) + "\n")
        out = prediction_report([path])
        assert out["checks"] == len(checks) and out["p95"] <= 0.1 + 1e-9 and out["misses"] == 0
        assert main([path, "--prediction"]) == 0
    print("show_trace.py self-test passed")


if __name__ == "__main__":
    sys.exit(main())

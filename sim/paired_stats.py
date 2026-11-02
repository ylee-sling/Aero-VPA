# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/paired_stats.py -- Paired statistics of the shielded benchmark from the flight records (fly_policy.py).

For every checkpoint of the benchmark (the "benchmark" list of scripts/release/release_manifest.json: a name,
the records of its unshielded and of its shielded evaluation) on the same test missions:

    exact ending counts of both evaluations;
    successes and crashes paired mission by mission (both / only one / neither, exact McNemar p), on all missions
    and on the missions from --holdout on (mission 0 of every test world was flown by the shield pilots);
    every shielded checkpoint paired with the first one of the list (r1 + shield);
    shielded endings by world density, and what the shielded missions did by ending (waypoints reached,
    duration, speed, clearance, the shield's interventions).

    python3 sim/paired_stats.py                          # records in the current directory (~/Aero-VPA)
    python3 sim/paired_stats.py --root records           # the records/ folder of the benchmark dataset
    python3 sim/paired_stats.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from summarize_flights import density, mcnemar_exact, wilson  # noqa: E402

MANIFEST = os.path.join(os.path.dirname(HERE), "scripts", "release", "release_manifest.json")
CRASH = ("collision", "ground")
Key = Tuple[str, int]
Records = Dict[Key, Dict[str, Any]]


def load_records(pattern: str, root: str = ".") -> Records:
    """{(world, episode): record} of the files matching ``pattern`` under ``root``; the last record of a mission
    counts, placeholder records ("cancelled") are skipped."""
    out: Records = {}
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict) or r.get("termination") in (None, "cancelled"):
                    continue
                out[(str(r["world"]), int(r["episode"]))] = r
    return out


def benchmark_list(manifest: str = MANIFEST) -> List[Dict[str, str]]:
    with open(manifest, encoding="utf-8") as f:
        return list(json.load(f)["benchmark"])


def ok(r: Dict[str, Any]) -> bool:
    return bool(r["success"])


def crashed(r: Dict[str, Any]) -> bool:
    return r["termination"] in CRASH


def paired(a: Records, b: Records, keys: Sequence[Key], test: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    xa = [test(a[k]) for k in keys]
    xb = [test(b[k]) for k in keys]
    both = sum(x and y for x, y in zip(xa, xb, strict=True))
    only_a = sum(x and not y for x, y in zip(xa, xb, strict=True))
    only_b = sum(y and not x for x, y in zip(xa, xb, strict=True))
    return {"n": len(keys), "a": sum(xa), "b": sum(xb), "both": both, "only_a": only_a, "only_b": only_b,
            "neither": len(keys) - both - only_a - only_b, "p": mcnemar_exact(only_a, only_b)}


def endings(data: Records) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in data.values():
        out[r["termination"]] = out.get(r["termination"], 0) + 1
    return dict(sorted(out.items()))


def _med(xs: Sequence[Optional[float]], spec: str = ".2f") -> str:
    vals = [x for x in xs if x is not None]
    return format(statistics.median(vals), spec) if vals else "-"


def analyse(bench: Sequence[Dict[str, str]], root: str = ".", holdout: int = 1) -> Dict[str, Any]:
    """Print the report; returns the numbers (for tests and for the model card)."""
    names = [b["name"] for b in bench]
    U = {b["name"]: load_records(b["unshielded"], root) for b in bench}
    S = {b["name"]: load_records(b["shielded"], root) for b in bench}
    out: Dict[str, Any] = {"endings": {}, "all": {}, "holdout": {}, "vs_first": {}, "density": {}}

    print("== endings (exact counts)")
    for n in names:
        for label, data in (("unshielded", U[n]), ("shielded", S[n])):
            ends = endings(data)
            out["endings"][(n, label)] = ends
            print("  %-9s %-10s n=%3d  %s" % (n, label, len(data), ", ".join("%s %d" % kv for kv in ends.items())))

    for key, title, sel in (
            ("all", "all missions", lambda k: True),
            ("holdout", "missions %d and later only (mission 0 of every test world was the shield pilots' set)"
             % holdout, lambda k: k[1] >= holdout)):
        print("\n== shielded against unshielded, same missions: %s" % title)
        for n in names:
            keys = sorted(k for k in set(U[n]) & set(S[n]) if sel(k))
            s, c = paired(U[n], S[n], keys, ok), paired(U[n], S[n], keys, crashed)
            lo, hi = wilson(s["b"], s["n"])
            out[key][n] = {"success": s, "crash": c, "ci": (lo, hi)}
            print("  %-9s n=%3d success %3d -> %3d [%.1f, %.1f]%% (both %d, only unshielded %d, only shielded %d,"
                  " neither %d; p = %.3g); crashes %3d -> %3d (only unshielded %d, only shielded %d; p = %.3g)"
                  % (n, s["n"], s["a"], s["b"], 100 * lo, 100 * hi, s["both"], s["only_a"], s["only_b"], s["neither"],
                     s["p"], c["a"], c["b"], c["only_a"], c["only_b"], c["p"]))

    first = names[0]
    print("\n== shielded checkpoints against %s + shield (same missions)" % first)
    for n in names[1:]:
        keys = sorted(set(S[first]) & set(S[n]))
        s = paired(S[first], S[n], keys, ok)
        out["vs_first"][n] = s
        print("  %s %3d vs %-9s %3d: both %d, only %s %d, only %s %d, neither %d; exact McNemar p = %.3g"
              % (first, s["a"], n, s["b"], s["both"], first, s["only_a"], n, s["only_b"], s["neither"], s["p"]))

    print("\n== shielded endings by density")
    for n in names:
        row, cells = [], {}
        for d in ("dense", "medium", "sparse"):
            rs = [r for k, r in S[n].items() if density(k[0]) == d]
            cells[d] = {"n": len(rs), "success": sum(ok(r) for r in rs), "crash": sum(crashed(r) for r in rs),
                        "timeout": sum(r["termination"] == "timeout" for r in rs)}
            row.append("%s: success %d, crash %d, timeout %d of %d" % (
                d, cells[d]["success"], cells[d]["crash"], cells[d]["timeout"], cells[d]["n"]))
        out["density"][n] = cells
        print("  %-9s %s" % (n, " | ".join(row)))

    print("\n== what the shielded missions were doing, by ending (%s)" % ", ".join(names[:2]))
    for n in names[:2]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in S[n].values():
            groups.setdefault(r["termination"], []).append(r)
        for end in sorted(groups):
            rs = groups[end]
            sec: Dict[str, float] = {}
            for r in rs:
                for k, v in ((r.get("shield") or {}).get("seconds") or {}).items():
                    sec[k] = sec.get(k, 0.0) + float(v) / len(rs)
            cnt = {k: sum(float((r.get("shield") or {}).get(k) or 0) for r in rs) / len(rs)
                   for k in ("backoffs", "nudges", "stalls", "holds")}
            wp = [r["waypoints_reached"] / max(1, r["waypoints"]) for r in rs]
            print("  %-9s %-14s n=%3d  waypoints reached %s, duration %s s, mean speed %s m/s, smallest clearance %s m;"
                  " per mission %s; seconds %s"
                  % (n, end, len(rs), _med(wp), _med([r.get("duration_s") for r in rs], ".1f"),
                     _med([r.get("mean_speed") for r in rs]), _med([r.get("min_clearance_m") for r in rs]),
                     " ".join("%s %.2f" % kv for kv in cnt.items()), " ".join("%s %.1f" % kv for kv in sorted(sec.items()))))
    return out


# ---------------------------------------------------------------------------------------------
# Self-test (python3 sim/paired_stats.py --self-test)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import contextlib
    import io
    import tempfile

    worlds = ["aero_forest_%s_s20%02d" % (d, s) for d in ("dense", "medium", "sparse") for s in range(2)]

    def write(path: str, outcome: Callable[[int, int], str], shield: bool) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for w_i, w in enumerate(worlds):
                for k in range(5):
                    end = outcome(w_i, k)
                    r = {"tag": os.path.basename(path), "world": w, "episode": k, "termination": end,
                         "success": end == "success", "waypoints_reached": 4 if end == "success" else 1,
                         "waypoints": 4, "duration_s": 50.0, "mean_speed": 1.2, "min_clearance_m": 0.5}
                    if shield:
                        r["shield"] = {"backoffs": 1, "nudges": 2, "stalls": 0, "holds": 1,
                                       "seconds": {"barrier": 4.0, "blind": 2.0}}
                    f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"tag": "x", "world": worlds[0], "episode": 0, "termination": "cancelled",
                                "success": False}) + "\n")              # a placeholder never counts

    with tempfile.TemporaryDirectory() as tmp:
        # unshielded: success on mission 0 only; shielded: success on missions 0-2, timeout on 3, collision on 4
        write(os.path.join(tmp, "flights_a_i0.jsonl"), lambda w, k: "success" if k == 0 else "collision", False)
        write(os.path.join(tmp, "flights_shield_a_i0.jsonl"),
              lambda w, k: "success" if k <= 2 else ("timeout" if k == 3 else "collision"), True)
        write(os.path.join(tmp, "flights_b_i0.jsonl"), lambda w, k: "collision", False)
        write(os.path.join(tmp, "flights_shield_b_i0.jsonl"), lambda w, k: "success" if k == 1 else "ground", True)
        bench = [{"name": "a", "unshielded": "flights_a_i*.jsonl", "shielded": "flights_shield_a_i*.jsonl"},
                 {"name": "b", "unshielded": "flights_b_i*.jsonl", "shielded": "flights_shield_b_i*.jsonl"}]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = analyse(bench, root=tmp, holdout=1)
        n_missions = len(worlds) * 5
        a = out["all"]["a"]
        assert a["success"]["n"] == n_missions and a["success"]["a"] == 6 and a["success"]["b"] == 18
        assert (a["success"]["both"], a["success"]["only_a"], a["success"]["only_b"]) == (6, 0, 12)
        assert abs(a["success"]["p"] - mcnemar_exact(0, 12)) < 1e-15
        assert a["crash"]["a"] == 24 and a["crash"]["b"] == 6 and a["crash"]["only_a"] == 18
        h = out["holdout"]["a"]
        assert h["success"]["n"] == 24 and h["success"]["a"] == 0 and h["success"]["b"] == 12
        assert out["endings"][("a", "shielded")] == {"collision": 6, "success": 18, "timeout": 6}
        assert out["vs_first"]["b"]["only_a"] == 12 and out["vs_first"]["b"]["both"] == 6
        assert out["density"]["b"]["dense"] == {"n": 10, "success": 2, "crash": 8, "timeout": 0}
        text = buf.getvalue()
        assert "missions 1 and later only" in text and "per mission backoffs 1.00" in text, text
    print("paired_stats self-test passed: endings, paired successes and crashes, hold-out, density")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=".", help="directory holding the flight records [.]")
    p.add_argument("--manifest", default=MANIFEST, help="release manifest with the benchmark list")
    p.add_argument("--holdout", type=int, default=1, help="first mission index of the hold-out comparison [1]")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    analyse(benchmark_list(args.manifest), root=args.root, holdout=args.holdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

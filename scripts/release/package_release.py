# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
scripts/release/package_release.py -- Stage the Hugging Face release of Aero-VPA: the weights and videos (model
repository) and the closed-loop evaluation records, traces, worlds and reports (dataset repository), each with a
card whose numbers are computed from the records, a MANIFEST.json and SHA256SUMS.

Run in the repository root of the machine that flew the evaluations (~/Aero-VPA on the VM):

    python3 scripts/release/package_release.py --dry_run                     # inventory: what goes where, sizes
    python3 scripts/release/package_release.py --namespace ORG               # stage release/models, release/datasets
    python3 scripts/release/upload_hf.py --namespace ORG                     # then upload (private first)
    python3 scripts/release/package_release.py --self-test

What is published is listed in scripts/release/release_manifest.json. Every text file and checkpoint is copied
with the machine's absolute paths replaced (the repository root by a relative path, the home directory by ~);
checkpoints keep their tensors and checkpoint_id bit for bit, which is verified after saving. Nothing here
uploads anything, and the sources are only read.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import glob
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from string import Template
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

import paired_stats as PS  # noqa: E402
from summarize_flights import density as world_density  # noqa: E402
from summarize_flights import mcnemar_exact, wilson  # noqa: E402

MANIFEST = os.path.join(ROOT, "scripts", "release", "release_manifest.json")
GITHUB = "https://github.com/ylee-sling/Aero-VPA"
DOI = "10.5281/zenodo.23633375"
COMMERCIAL = "For commercial integration, proprietary licensing, or defense adaptation, contact ylee@sling.ai.kr."
LICENSES = {"gpl-3.0": "GNU GPL v3.0 only", "cc-by-4.0": "CC BY 4.0", "cc-by-nc-4.0": "CC BY-NC 4.0",
            "apache-2.0": "Apache 2.0", "mit": "MIT"}
ABSOLUTE = re.compile(r"/home/[^/\s\"']+/|/root/|/Users/[^/\s\"']+/")
TEXT_SUFFIXES = (".txt", ".json", ".jsonl", ".sdf", ".md", ".csv", ".svg")


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%.1f TB" % n


def du(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(dirpath, name))
    return total


class Scrubber:
    """Replaces the machine's absolute paths in strings: the repository root by a relative path, the home
    directory by ~. Counts replacements and remembers what is left that still looks absolute."""

    def __init__(self, repo_root: str, home: Optional[str] = None) -> None:
        root = os.path.realpath(repo_root).rstrip("/") + "/"
        home_dir = os.path.realpath(home or os.path.expanduser("~")).rstrip("/") + "/"
        pairs = {root: "", os.path.abspath(repo_root).rstrip("/") + "/": "", home_dir: "~/",
                 os.path.abspath(home or os.path.expanduser("~")).rstrip("/") + "/": "~/"}
        self.pairs = sorted(pairs.items(), key=lambda kv: -len(kv[0]))     # longest prefix first
        self.count = 0
        self.leftovers: Dict[str, int] = {}

    def text(self, s: str) -> str:
        for old, new in self.pairs:
            if old in s:
                self.count += s.count(old)
                s = s.replace(old, new)
        for m in ABSOLUTE.findall(s):
            self.leftovers[m] = self.leftovers.get(m, 0) + 1
        return s

    def obj(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.obj(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.obj(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.obj(v) for v in value)
        return value


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("format") != "aero-vpa/release-manifest":
        raise SystemExit("%s is not a release manifest" % path)
    return m


def expand(patterns: Sequence[str], root: str) -> List[str]:
    out = set()
    for pat in patterns:
        for p in glob.glob(os.path.join(root, pat)):
            if os.path.isfile(p):
                out.add(os.path.relpath(p, root))
    return sorted(out)


# ---------------------------------------------------------------------------------------------
# Inventory: what goes where
# ---------------------------------------------------------------------------------------------
def inventory(manifest: Dict[str, Any], root: str) -> Dict[str, Any]:
    inv: Dict[str, Any] = {"checkpoints": [], "missing": [], "records": [], "dagger_records": [], "traces": {},
                           "reports": [], "worlds": [], "media": [], "not_released": []}
    for c in manifest["checkpoints"]:
        path = os.path.join(root, c["file"])
        if os.path.isfile(path):
            inv["checkpoints"].append({**c, "bytes": os.path.getsize(path)})
        else:
            inv["missing"].append(("checkpoint", c["file"], bool(c.get("optional"))))
    for rel in expand(manifest["records"], root):
        if os.path.basename(rel).startswith("dagger_r"):
            inv["dagger_records"].append(rel)
        else:
            inv["records"].append(rel)
    for d in manifest["traces"]:
        files = expand([os.path.join(d, "*.jsonl")], root)
        if files:
            inv["traces"][os.path.basename(d.rstrip("/"))] = files
        else:
            inv["missing"].append(("traces", d, True))
    for pat in manifest["reports"]:
        found = expand([pat], root)
        inv["reports"] += found
        if not found and not any(ch in pat for ch in "*?["):
            inv["missing"].append(("report", pat, True))
    for d in manifest["worlds"]:
        files = expand([os.path.join(d, "*.sdf"), os.path.join(d, "*.json")], root)
        if files:
            inv["worlds"] += files
        else:
            inv["missing"].append(("worlds", d, d.endswith("dev")))
    inv["media"] = expand(manifest["media"], root)
    for name, why in manifest.get("not_released", {}).items():
        size = sum(du(os.path.join(root, p)) for p in expand([name], root)) if "*" in name else (
            du(os.path.join(root, name)) if os.path.exists(os.path.join(root, name)) else None)
        inv["not_released"].append((name, size, why))
    return inv


def print_inventory(inv: Dict[str, Any], root: str, model_id: str, dataset_id: str) -> None:
    def size(rels: Sequence[str]) -> int:
        return sum(os.path.getsize(os.path.join(root, r)) for r in rels)

    print("== model repository %s" % model_id)
    for c in inv["checkpoints"]:
        print("  checkpoints/%-32s %9s  %-7s %s" % (os.path.basename(c["file"]), human(c["bytes"]), c["group"], c["name"]))
    print("  media/: %d videos, %s" % (len(inv["media"]), human(size(inv["media"]))))
    for m in inv["media"]:
        print("    %s (%s)" % (m, human(os.path.getsize(os.path.join(root, m)))))
    print("\n== dataset repository %s" % dataset_id)
    print("  records/: %d evaluation files (%s) + %d DAgger round files (%s)" % (
        len(inv["records"]), human(size(inv["records"])), len(inv["dagger_records"]), human(size(inv["dagger_records"]))))
    runs: Dict[str, List[str]] = {}
    for r in inv["records"]:
        runs.setdefault(run_name(r), []).append(r)
    for name, files in sorted(runs.items()):
        n = 0
        for rel in files:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
                n += sum(1 for line in f if line.strip() and '"cancelled"' not in line)
        print("    %-34s %d file(s), %4d records" % (name, len(files), n))
    for name, files in sorted(inv["traces"].items()):
        print("  traces/%s/: %d traced missions (%s before compression)" % (name, len(files), human(size(files))))
    print("  worlds/: %d files (%s)" % (len(inv["worlds"]), human(size(inv["worlds"]))))
    print("  reports/: %s" % ", ".join(os.path.basename(r) for r in inv["reports"]))
    print("\n== not released (stays on this machine)")
    for name, nbytes, why in inv["not_released"]:
        print("  %-26s %10s  %s" % (name, "-" if nbytes is None else human(nbytes), why))
    if inv["missing"]:
        print("\n== not found")
        for kind, rel, optional in inv["missing"]:
            print("  %-10s %s%s" % (kind, rel, " (optional)" if optional else "  <-- required"))


# ---------------------------------------------------------------------------------------------
# Numbers for the cards, from the records
# ---------------------------------------------------------------------------------------------
def benchmark_numbers(bench: Sequence[Dict[str, str]], root: str, holdout: int = 1) -> List[Dict[str, Any]]:
    rows = []
    for b in bench:
        u, s = PS.load_records(b["unshielded"], root), PS.load_records(b["shielded"], root)
        keys = sorted(set(u) & set(s))
        row: Dict[str, Any] = {"name": b["name"], "n_u": len(u), "n_s": len(s), "n_pair": len(keys)}
        if not keys:
            rows.append(row)
            continue
        succ = PS.paired(u, s, keys, PS.ok)
        crash = PS.paired(u, s, keys, PS.crashed)
        hk = [k for k in keys if k[1] >= holdout]
        hsucc = PS.paired(u, s, hk, PS.ok)
        row.update({
            "succ_u": succ["a"], "succ_s": succ["b"], "ci_s": wilson(succ["b"], len(keys)),
            "ci_u": wilson(succ["a"], len(keys)), "p_success": succ["p"],
            "only_u": succ["only_a"], "only_s": succ["only_b"], "both": succ["both"],
            "crash_u": crash["a"], "crash_s": crash["b"], "p_crash": crash["p"],
            "timeout_s": sum(s[k]["termination"] == "timeout" for k in keys),
            "timeout_u": sum(u[k]["termination"] == "timeout" for k in keys),
            "h_n": len(hk), "h_succ_u": hsucc["a"], "h_succ_s": hsucc["b"], "h_ci_s": wilson(hsucc["b"], len(hk)),
            "h_p": hsucc["p"],
            "density": {d: {"n": sum(world_density(k[0]) == d for k in keys),
                            "u": sum(PS.ok(u[k]) for k in keys if world_density(k[0]) == d),
                            "s": sum(PS.ok(s[k]) for k in keys if world_density(k[0]) == d)}
                        for d in ("dense", "medium", "sparse")},
        })
        rows.append(row)
    return rows


def _pct(k: int, n: int) -> str:
    return "%.1f%%" % (100.0 * k / n) if n else "-"


def _p(p: float) -> str:
    if p >= 1e-3:
        return "%.3g" % p
    mant, exp = ("%.2e" % p).split("e")
    return "%s × 10<sup>%d</sup>" % (mant, int(exp))


def results_tables(rows: Sequence[Dict[str, Any]]) -> str:
    done = [r for r in rows if r.get("n_pair")]
    if not done:
        return "_No paired evaluation records were found when this card was generated._\n"
    lines = ["| checkpoint | success, no shield | success, shield [95% CI] | crashes, no shield → shield "
             "| timeouts, shield | paired McNemar p (success) |", "|---|---|---|---|---|---|"]
    for r in done:
        n = r["n_pair"]
        lines.append("| %s | %d/%d (%s) | **%d/%d (%s)** [%.1f, %.1f] | %s → %s | %s | %s |" % (
            r["name"], r["succ_u"], n, _pct(r["succ_u"], n), r["succ_s"], n, _pct(r["succ_s"], n),
            100 * r["ci_s"][0], 100 * r["ci_s"][1], _pct(r["crash_u"], n), _pct(r["crash_s"], n),
            _pct(r["timeout_s"], n), _p(r["p_success"])))
    lines += ["", "**Success by world density** (shield / no shield, missions per density in brackets):", "",
              "| checkpoint | dense | medium | sparse |", "|---|---|---|---|"]
    for r in done:
        cells = []
        for d in ("dense", "medium", "sparse"):
            c = r["density"][d]
            cells.append("%d / %d (%d)" % (c["s"], c["u"], c["n"]))
        lines.append("| %s | %s |" % (r["name"], " | ".join(cells)))
    lines += ["", "**Missions the shield was not revised on** (mission 1 and later of every world; the shield "
              "pilots flew mission 0):", "", "| checkpoint | missions | success, no shield | success, shield [95% CI] "
              "| paired p |", "|---|---|---|---|---|"]
    for r in done:
        lines.append("| %s | %d | %s | %s [%.1f, %.1f] | %s |" % (
            r["name"], r["h_n"], _pct(r["h_succ_u"], r["h_n"]), _pct(r["h_succ_s"], r["h_n"]),
            100 * r["h_ci_s"][0], 100 * r["h_ci_s"][1], _p(r["h_p"])))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------------------------
MODEL_CARD = Template("""---
license: ${license_id}
library_name: pytorch
pipeline_tag: robotics
tags:
- robotics
- quadrotor
- drone
- navigation
- imitation-learning
- dagger
- flow-matching
- jepa
- world-model
- safety-shield
- px4
- gazebo
datasets:
- ${dataset_id}
---

# Aero-VPA: vision-based quadrotor navigation policies

Aero-VPA is an autonomous navigation framework incorporating a JEPA-based latent world model. The JEPA predictor
serves as the latent dynamics world model predicting forward state transitions, while a flow-matching policy acts
as the downstream controller. At deployment a deterministic onboard safety shield (15 s depth memory, barrier
braking, breadcrumb backoff; no map) filters every setpoint.

These are the checkpoints of the closed-loop evaluation in PX4 SITL + Gazebo Harmonic. Code, training and
evaluation scripts: [${github}](${github}). Evaluation records, traces and worlds:
[${dataset_id}](https://huggingface.co/datasets/${dataset_id}). Method: [doi:${doi}](https://doi.org/${doi}).

## Results in simulation (${n_missions} test missions per checkpoint)

Generated from the evaluation records in the dataset by `scripts/release/package_release.py` (${generated}).
36 test worlds (seeds 2000-2011 × sparse, medium, dense) × 5 missions; every checkpoint, with and without the
shield, flew the same missions, so the comparisons are paired. Crash = collision (privileged distance below
0.35 m) or ground contact.

${results}
${videos}
## Checkpoints

${checkpoints}

`r1` is the recommended policy. Every policy file holds all networks of the paper's Table 1 (18.13M trainable
parameters: ViT encoder, JEPA predictor, primitive selector, CFM solver), the frozen statistics and the calibrated
uncertainty thresholds,
and loads with `torch.load(..., weights_only=True)`; the frozen CLIP text tower is not included (null
instruction embedding only). The `stage1*` files are the frozen stage (i) networks, needed to retrain stage (ii).

## Use

```bash
git clone ${github} && cd Aero-VPA && pip install -r requirements.txt
hf download ${model_id} --include "checkpoints/*" --local-dir .
python3 sim/make_world.py --seed 2000 --count 12 --density sparse medium dense --out sim/worlds/test
sim/run_benchmark.sh --shield r1_shield checkpoints/policy_r1_cal_ep.pt     # PX4 SITL + Gazebo, see the README
```

```python
from pipeline import VPAInferencePipeline
pipe = VPAInferencePipeline.from_checkpoint("checkpoints/policy_r1_cal_ep.pt", latency_bound=0.06, tracking_lag=0.15)
```

## Limitations and intended use

- Research artefacts evaluated **in simulation only** (PX4 SITL, Gazebo Harmonic, static procedurally generated
  forests, no wind, no moving obstacles). They have not flown a real aircraft and are not a safety system.
- Mean flight speed is about 1.2 m/s; about half of the missions succeed with the shield.
- Mission outcomes have a large chance component (simulator timing, CFM noise); one training seed per recipe.
- The shield was revised after its first pilot, which flew mission 0 of every test world; the table above also
  reports the missions after it.

## License

Weights: ${license_name}. Code: GPL-3.0-only. ${commercial}

## Citation

```bibtex
@misc{lee2026aerovpa,
  author    = {Lee, Yeonseok},
  title     = {Aero-VPA: Few-Step Flow-Matching Control over Non-Reconstructive JEPA Latents for Agile Flight on Edge Hardware},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {${doi}},
  url       = {${github}},
  note      = {Preprint}
}
```
""")

DATASET_CARD = Template("""---
license: ${license_id}
pretty_name: Aero-VPA closed-loop benchmark
task_categories:
- robotics
tags:
- quadrotor
- drone
- navigation
- benchmark
- simulation
- px4
- gazebo
- safety-shield
size_categories:
- ${size_category}
configs:
- config_name: missions
  default: true
  data_files:
  - split: test
    path: missions/missions.${missions_format}
---

# Aero-VPA closed-loop benchmark (PX4 SITL, procedurally generated forests)

Every closed-loop mission flown to evaluate the [Aero-VPA](${github}) policies
([weights](https://huggingface.co/${model_id})): one record per mission, decision-by-decision traces of the
traced runs, the worlds with their privileged obstacle maps, and the reports. Generated by
`scripts/release/package_release.py` (${generated}).

## Contents

| path | contents |
|---|---|
| `missions/missions.${missions_format}` | one row per mission of every run (${n_rows} rows): run, tag, world, density, episode, ending, success, duration, path, speeds, clearance, waypoints, fallbacks, blind decisions, shield interventions |
| `records/` | the raw records written by `sim/fly_policy.py` (JSON lines), one file per run and simulator |
| `records/dagger/` | the DAgger rounds' supervised flights (expert-gated, not autonomous performance) |
| `traces/<set>/` | decision-by-decision traces (`fly_policy.py --trace`), gzipped JSON lines; the first line is the mission record |
| `worlds/` | the worlds flown (Gazebo SDF) and their privileged obstacle maps (JSON) |
| `reports/` | text reports of the analysis scripts |

## Protocol

36 test worlds (seeds 2000-2011 × sparse, medium, dense; Poisson-disc trunk spacing 7, 5 and 3.5 m), 5 missions
each: the benchmark runs fly 180 missions; pilot and traced runs fly mission 0 of every world, and earlier
baselines subsets of the worlds (`run` and `tag` identify each). Mission k of a world is seeded by
(world seed, k), so every run flies the same missions. A mission takes off to 2.5 m and visits 3-6 waypoints 10-40 m apart; it times out at 1.5 × the expert
reference (3 m/s) + 15 s. Endings: `success`, `collision` (privileged distance below 0.35 m), `ground` (below
0.3 m), `timeout`, `camera_timeout` (no decision for 0.5 s), `out_of_bounds`. Shielded runs fly with
`fly_policy.py --shield` (onboard depth memory, barrier, blind-motion cap, altitude floor, breadcrumb backoff;
no map).

## Results

${results}
## Reproduce the numbers

```bash
git clone ${github} && cd Aero-VPA
hf download ${dataset_id} --repo-type dataset --local-dir bench
python3 sim/paired_stats.py --root bench/records
python3 sim/summarize_flights.py bench/records/flights_shield_r1_i*.jsonl --by density
python3 sim/crash_study.py 'bench/traces/trace_r1/*.jsonl.gz' --world_root bench/worlds/test
```

```python
import pandas as pd
df = pd.read_${missions_format}("hf://datasets/${dataset_id}/missions/missions.${missions_format}")
print(df.groupby("run")["success"].mean())
```

## Limitations

Simulation only; static forests; no wind; one training seed per policy; mission outcomes have a large chance
component. Collisions are detected with the privileged map. The shield was revised after its first pilot, which
flew mission 0 of every test world.

## License

${license_name}. The code that produced and reads these files is GPL-3.0-only: ${github}. ${commercial}
""")


def checkpoint_table(entries: Sequence[Dict[str, Any]]) -> str:
    lines = ["| file | name | kind | values stored | calibrated | SHA-256 | description |",
             "|---|---|---|---|---|---|---|"]
    for e in entries:
        lines.append("| `checkpoints/%s` | %s%s | %s | %.2fM | %s | `%s…` | %s |" % (
            os.path.basename(e["file"]), e["name"], " (recommended)" if e.get("recommended") else "", e["kind"],
            e["parameters"] / 1e6, "yes" if e["calibrated"] else "-", e["sha256"][:12], e["description"]))
    return "\n".join(lines) + "\n"


def video_block(media: Sequence[str], model_id: str, posters: Sequence[str] = ()) -> str:
    """Video players for the model card (files in media/, served from the repository's main branch)."""
    if not media:
        return ""
    base = "https://huggingface.co/%s/resolve/main/media/" % model_id
    lines = ["## Videos", ""]
    for rel in media:
        name = os.path.basename(rel)
        poster = os.path.splitext(name)[0] + ".jpg"
        attr = ' poster="%s%s"' % (base, poster) if poster in posters else ""
        lines += ["**%s**" % os.path.splitext(name)[0].replace("_", " "), "",
                  '<video controls muted width="640"%s src="%s%s"></video>' % (attr, base, name), ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------------------------
def copy_text(src: str, dst: str, scrub: Scrubber, compress: bool = False) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, encoding="utf-8", errors="replace") as f:
        text = scrub.text(f.read())
    if compress:
        with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write(text)
    else:
        with open(dst, "w", encoding="utf-8") as f:
            f.write(text)


def stage_checkpoint(src: str, dst: str, scrub: Scrubber) -> Dict[str, Any]:
    """Copy a checkpoint with its strings scrubbed; verify tensors, checkpoint_id and format after saving."""
    import torch

    from checkpoint import load_checkpoint, save_checkpoint

    original = load_checkpoint(src)
    clean = scrub.obj({k: v for k, v in original.items() if k != "state_dict"})
    clean["state_dict"] = original["state_dict"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_checkpoint(dst, clean)
    again = load_checkpoint(dst)

    def tensors(sd: Any, prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if isinstance(sd, dict):
            for k, v in sd.items():
                out.update(tensors(v, prefix + str(k) + "."))
        elif isinstance(sd, torch.Tensor):
            out[prefix.rstrip(".")] = sd
        return out

    a, b = tensors(original["state_dict"]), tensors(again["state_dict"])
    if a.keys() != b.keys() or not all(torch.equal(a[k], b[k]) for k in a):
        raise RuntimeError("%s: tensors changed while staging" % src)
    if again["checkpoint_id"] != original["checkpoint_id"] or again["kind"] != original["kind"]:
        raise RuntimeError("%s: checkpoint_id or kind changed while staging" % src)
    for key in ("calibration",):
        ca, cb = tensors(original.get(key, {})), tensors(again.get(key, {}))
        if ca.keys() != cb.keys() or not all(torch.equal(ca[k], cb[k]) for k in ca):
            raise RuntimeError("%s: %s changed while staging" % (src, key))
    cfg = again.get("config", {})
    return {"kind": again["kind"], "checkpoint_id": again["checkpoint_id"],
            "parameters": int(sum(t.numel() for t in b.values())),
            "calibrated": cfg.get("sigma_crit") is not None, "source_sha256": sha256(src)}


def mission_row(r: Dict[str, Any], run: str, file: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"run": run, "file": file, "density": world_density(str(r.get("world", "")))}
    for k, v in r.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[k] = v
        elif isinstance(v, dict) and k in ("shield", "decision_latency_ms", "decision_age_ms"):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float, bool)) or v2 is None:
                    row["%s_%s" % (k, k2)] = v2
                elif isinstance(v2, dict) and k2 == "seconds":
                    for k3, v3 in v2.items():
                        row["%s_seconds_%s" % (k, k3)] = v3
    row["crash"] = r.get("termination") in PS.CRASH
    row["shielded"] = bool(r.get("shield"))
    return row


def normalise(rows: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[Any]]]:
    """Column-wise values with one type per column (bool, int, float or str), None where absent."""
    names: List[str] = []
    for r in rows:
        for k in r:
            if k not in names:
                names.append(k)
    cols: Dict[str, List[Any]] = {}
    for k in names:
        vals = [r.get(k) for r in rows]
        kinds = {type(v) for v in vals if v is not None}
        if kinds <= {bool}:
            cols[k] = vals
        elif kinds <= {int, bool} and bool not in kinds:
            cols[k] = vals
        elif kinds <= {int, float}:
            cols[k] = [None if v is None else float(v) for v in vals]
        else:
            cols[k] = [None if v is None else (v if isinstance(v, str) else json.dumps(v)) for v in vals]
    return names, cols


def write_missions(rows: List[Dict[str, Any]], directory: str) -> str:
    """missions.parquet with pyarrow, else missions.csv; returns the format."""
    os.makedirs(directory, exist_ok=True)
    names, cols = normalise(rows)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        with open(os.path.join(directory, "missions.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(names)
            for i in range(len(rows)):
                w.writerow(["" if cols[k][i] is None else cols[k][i] for k in names])
        return "csv"
    pq.write_table(pa.table({k: cols[k] for k in names}), os.path.join(directory, "missions.parquet"),
                   compression="zstd")
    return "parquet"


def run_name(rel: str) -> str:
    return re.sub(r"_i\d+$", "", os.path.splitext(os.path.basename(rel))[0])


def write_manifest(repo_dir: str, sources: Dict[str, Dict[str, Any]], extra: Dict[str, Any]) -> None:
    files = []
    for dirpath, _, names in os.walk(repo_dir):
        for name in sorted(names):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, repo_dir)
            if rel in ("MANIFEST.json", "SHA256SUMS"):
                continue
            files.append({"path": rel, "bytes": os.path.getsize(path), "sha256": sha256(path),
                          **sources.get(rel, {})})
    files.sort(key=lambda f: f["path"])
    with open(os.path.join(repo_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({**extra, "files": files}, f, indent=1)
        f.write("\n")
    with open(os.path.join(repo_dir, "SHA256SUMS"), "w", encoding="utf-8") as f:
        for e in files:
            f.write("%s  %s\n" % (e["sha256"], e["path"]))


def git_revision(root: str) -> str:
    try:
        return subprocess.run(["git", "-C", root, "describe", "--always", "--dirty", "--tags"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def stage(args: argparse.Namespace, manifest: Dict[str, Any], root: str) -> Dict[str, Any]:
    model_id = "%s/%s" % (args.namespace, manifest["model_repo"])
    dataset_id = "%s/%s" % (args.namespace, manifest["dataset_repo"])
    inv = inventory(manifest, root)
    required = [rel for kind, rel, optional in inv["missing"] if not optional]
    if required and not args.allow_missing:
        raise SystemExit("required files are missing (see --dry_run): %s" % ", ".join(required))
    out = os.path.abspath(args.out)
    model_dir = os.path.join(out, "models", manifest["model_repo"])
    data_dir = os.path.join(out, "datasets", manifest["dataset_repo"])
    for d in (model_dir, data_dir):
        if os.path.exists(d):
            if not args.force:
                raise SystemExit("%s exists: pass --force to rebuild it" % d)
            shutil.rmtree(d)
        os.makedirs(d)
    scrub = Scrubber(root, args.home)
    generated = "%s, code %s" % (datetime.date.today().isoformat(), git_revision(root))
    model_src: Dict[str, Dict[str, Any]] = {}
    data_src: Dict[str, Dict[str, Any]] = {}

    # model repository: checkpoints and videos
    entries = []
    for c in inv["checkpoints"]:
        rel = os.path.join("checkpoints", os.path.basename(c["file"]))
        info = stage_checkpoint(os.path.join(root, c["file"]), os.path.join(model_dir, rel), scrub)
        entries.append({**c, **info, "sha256": sha256(os.path.join(model_dir, rel))})
        model_src[rel] = {"source": c["file"], "source_sha256": info["source_sha256"],
                          "checkpoint_id": info["checkpoint_id"], "note": "strings scrubbed; tensors identical"}
        print("  staged %s (%s, %.2fM parameters)" % (rel, info["kind"], info["parameters"] / 1e6))
    for rel in inv["media"]:
        dst = os.path.join("media", os.path.basename(rel))
        os.makedirs(os.path.join(model_dir, "media"), exist_ok=True)
        shutil.copy2(os.path.join(root, rel), os.path.join(model_dir, dst))
        model_src[dst] = {"source": rel}
        if args.posters and shutil.which("ffmpeg"):
            poster = os.path.splitext(dst)[0] + ".jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", os.path.join(root, rel),
                            "-frames:v", "1", "-vf", "scale=640:-2", os.path.join(model_dir, poster)], check=False)

    # dataset repository: records, missions table, traces, worlds, reports
    rows = []
    for rel in inv["records"] + inv["dagger_records"]:
        sub = "records/dagger" if rel in inv["dagger_records"] else "records"
        dst = os.path.join(sub, os.path.basename(rel))
        copy_text(os.path.join(root, rel), os.path.join(data_dir, dst), scrub)
        data_src[dst] = {"source": rel}
        if sub == "records":
            with open(os.path.join(data_dir, dst), encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and r.get("termination") not in (None, "cancelled"):
                        rows.append(mission_row(r, run_name(rel), os.path.basename(rel)))
    fmt = write_missions(rows, os.path.join(data_dir, "missions"))
    if not args.skip_traces:
        for name, files in sorted(inv["traces"].items()):
            for rel in files:
                dst = os.path.join("traces", name, os.path.basename(rel) + ".gz")
                copy_text(os.path.join(root, rel), os.path.join(data_dir, dst), scrub, compress=True)
                data_src[dst] = {"source": rel}
    for rel in inv["worlds"]:
        dst = os.path.join("worlds", os.path.relpath(rel, os.path.join("sim", "worlds")))
        copy_text(os.path.join(root, rel), os.path.join(data_dir, dst), scrub)
        data_src[dst] = {"source": rel, "source_sha256": sha256(os.path.join(root, rel))}
    for rel in inv["reports"]:
        dst = os.path.join("reports", os.path.basename(rel))
        copy_text(os.path.join(root, rel), os.path.join(data_dir, dst), scrub)
        data_src[dst] = {"source": rel}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        PS.analyse(manifest["benchmark"], root=os.path.join(data_dir, "records"), holdout=1)
    os.makedirs(os.path.join(data_dir, "reports"), exist_ok=True)
    with open(os.path.join(data_dir, "reports", "paired_stats.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    # cards, from the staged (scrubbed) records
    numbers = benchmark_numbers(manifest["benchmark"], os.path.join(data_dir, "records"))
    results = results_tables(numbers)
    n_missions = max((r.get("n_pair", 0) for r in numbers), default=0)
    common = {"github": GITHUB, "doi": DOI, "model_id": model_id, "dataset_id": dataset_id, "generated": generated,
              "commercial": COMMERCIAL, "results": results}
    with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(MODEL_CARD.substitute(common, license_id=args.weights_license,
                                      license_name=LICENSES.get(args.weights_license, args.weights_license),
                                      n_missions=n_missions, checkpoints=checkpoint_table(entries),
                                      videos=video_block([os.path.basename(m) for m in inv["media"]], model_id,
                                                         os.listdir(os.path.join(model_dir, "media"))
                                                         if os.path.isdir(os.path.join(model_dir, "media")) else ())))
    size_category = "n<1K" if len(rows) < 1000 else ("1K<n<10K" if len(rows) < 10000 else "10K<n<100K")
    with open(os.path.join(data_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(DATASET_CARD.substitute(common, license_id=args.data_license,
                                        license_name=LICENSES.get(args.data_license, args.data_license),
                                        missions_format=fmt, n_rows=len(rows), size_category=size_category))
    meta = {"format": "aero-vpa/release", "generated": generated, "scrubbed_paths": scrub.count}
    write_manifest(model_dir, model_src, {**meta, "repo_id": model_id, "repo_type": "model",
                                          "license": args.weights_license})
    write_manifest(data_dir, data_src, {**meta, "repo_id": dataset_id, "repo_type": "dataset",
                                        "license": args.data_license, "missions": len(rows)})

    print("\nstaged %s: %s in %s" % (model_id, human(du(model_dir)), os.path.relpath(model_dir)))
    print("staged %s: %s in %s (%d mission rows, %s)" % (dataset_id, human(du(data_dir)), os.path.relpath(data_dir),
                                                       len(rows), fmt))
    print("absolute paths replaced: %d" % scrub.count)
    if scrub.leftovers:
        print("WARNING: absolute paths left in the staged files (check before uploading): %s"
              % ", ".join("%s x%d" % kv for kv in sorted(scrub.leftovers.items())))
    for r in numbers:
        if r.get("n_pair") and r["n_pair"] != 180:
            print("WARNING: %s has %d paired missions (expected 180)" % (r["name"], r["n_pair"]))
    print("\n" + results)
    return {"model_dir": model_dir, "data_dir": data_dir, "numbers": numbers, "rows": len(rows),
            "leftovers": dict(scrub.leftovers)}


# ---------------------------------------------------------------------------------------------
# Self-test (python3 scripts/release/package_release.py --self-test)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile

    import torch

    from checkpoint import config_to_dict, load_checkpoint
    from config import VPAConfig

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "Aero-VPA")
        os.makedirs(os.path.join(root, "checkpoints"))
        abs_root = os.path.realpath(root)

        def fake_checkpoint(path: str, kind: str) -> None:
            states = {"vision_encoder": {"w": torch.randn(3, 4)}, "predictor": {"w": torch.randn(2)}}
            calibration = {"gamma_bar_star": torch.tensor(0.5), "latent_mean": torch.zeros(4)}
            if kind == "policy":
                states.update({"selector": {"w": torch.randn(5)}, "solver": {"w": torch.randn(6)},
                               "text_null_embedding": torch.randn(8)})
                calibration.update({"proprio_mean": torch.zeros(13), "proprio_std": torch.ones(13)})
            cfg = config_to_dict(VPAConfig())
            if kind == "policy":
                cfg["sigma_lo"], cfg["sigma_crit"] = 0.01, 0.02
            torch.save({"format": "aero-vpa/checkpoint", "format_version": 1, "kind": kind,
                        "checkpoint_id": "ab" * 16, "config": cfg, "state_dict": states, "calibration": calibration,
                        "provenance": {"data": abs_root + "/data/*.hdf5", "out": abs_root + "/checkpoints/x.pt",
                                       "note": "plain"}}, path)

        fake_checkpoint(os.path.join(root, "checkpoints", "policy_r1_cal_ep.pt"), "policy")
        fake_checkpoint(os.path.join(root, "checkpoints", "stage1_res.pt"), "stage1")
        worlds = ["aero_forest_%s_s20%02d" % (d, s) for d in ("dense", "medium", "sparse") for s in range(2)]
        os.makedirs(os.path.join(root, "sim", "worlds", "test"))
        for w in worlds:
            with open(os.path.join(root, "sim", "worlds", "test", w + ".json"), "w", encoding="utf-8") as f:
                json.dump({"obstacles": [], "sdf": abs_root + "/sim/worlds/test/" + w + ".sdf"}, f)
            with open(os.path.join(root, "sim", "worlds", "test", w + ".sdf"), "w", encoding="utf-8") as f:
                f.write("<sdf version='1.9'><world name='%s'/></sdf>\n" % w)

        def records(name: str, tag: str, succeed: Any, shield: bool) -> None:
            with open(os.path.join(root, name), "w", encoding="utf-8") as f:
                for w in worlds:
                    for k in range(5):
                        end = "success" if succeed(k) else ("timeout" if shield and k == 4 else "collision")
                        r = {"tag": tag, "world": w, "episode": k, "termination": end, "success": end == "success",
                             "checkpoint": abs_root + "/checkpoints/policy_r1_cal_ep.pt",
                             "world_json": abs_root + "/sim/worlds/test/%s.json" % w, "waypoints": 4,
                             "waypoints_reached": 4 if end == "success" else 1, "duration_s": 40.0 + k,
                             "decision_latency_ms": {"p50": 20.0, "p95": 30}, "primitive_counts": [1, 2, 3, 0, 0, 0]}
                        if shield:
                            r["shield"] = {"backoffs": 1, "nudges": 2, "stalls": 0, "holds": 1,
                                           "seconds": {"barrier": 3.5, "blind": 2}, "config": {"radius": 0.55}}
                        f.write(json.dumps(r) + "\n")

        records("flights_r1_i0.jsonl", "policy_r1_cal_ep.pt", lambda k: k == 0, False)
        records("flights_shield_r1_i0.jsonl", "r1_shield", lambda k: k <= 2, True)
        with open(os.path.join(root, "dagger_r1_i0.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"tag": "d", "world": worlds[0], "episode": 0, "termination": "success",
                                "success": True}) + "\n")
        os.makedirs(os.path.join(root, "logs", "trace_r1"))
        with open(os.path.join(root, "logs", "trace_r1", "r1_trace_x_e000.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "mission", "world_json": abs_root + "/sim/worlds/test/a.json"}) + "\n")
            f.write(json.dumps({"kind": "decision", "t": 0.0}) + "\n")
        with open(os.path.join(root, "logs", "shield_report.txt"), "w", encoding="utf-8") as f:
            f.write("report of %s/flights_shield_r1_i0.jsonl\n" % abs_root)
        os.makedirs(os.path.join(root, "videos"))
        with open(os.path.join(root, "videos", "r1_shield_dense.mp4"), "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
        manifest = {"format": "aero-vpa/release-manifest", "model_repo": "m", "dataset_repo": "d",
                    "checkpoints": [{"name": "r1", "file": "checkpoints/policy_r1_cal_ep.pt", "group": "dagger",
                                     "recommended": True, "description": "test policy"},
                                    {"name": "stage1_res", "file": "checkpoints/stage1_res.pt", "group": "stage1",
                                     "description": "test stage 1"},
                                    {"name": "v1", "file": "checkpoints/policy_cal_ep.pt", "group": "bc",
                                     "optional": True, "description": "absent"}],
                    "benchmark": [{"name": "r1", "unshielded": "flights_r1_i*.jsonl",
                                   "shielded": "flights_shield_r1_i*.jsonl"}],
                    "records": ["flights*.jsonl", "dagger_r[0-9]*_i[0-9]*.jsonl"], "traces": ["logs/trace_r1"],
                    "reports": ["logs/shield_report.txt", "logs/missing_report.txt"], "worlds": ["sim/worlds/test"],
                    "media": ["videos/*.mp4"], "not_released": {"data": "demonstrations"}}

        inv = inventory(manifest, root)
        assert [c["name"] for c in inv["checkpoints"]] == ["r1", "stage1_res"]
        assert ("checkpoint", "checkpoints/policy_cal_ep.pt", True) in inv["missing"]
        assert inv["records"] == ["flights_r1_i0.jsonl", "flights_shield_r1_i0.jsonl"]
        assert inv["dagger_records"] == ["dagger_r1_i0.jsonl"] and len(inv["worlds"]) == 12

        args = argparse.Namespace(namespace="test-org", out=os.path.join(tmp, "release"), force=False,
                                  allow_missing=False, skip_traces=False, posters=False, home=tmp,
                                  weights_license="gpl-3.0", data_license="cc-by-4.0")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = stage(args, manifest, root)
        text = buf.getvalue()
        md, dd = res["model_dir"], res["data_dir"]
        # checkpoints: strings scrubbed, tensors and id unchanged, still loadable
        staged = load_checkpoint(os.path.join(md, "checkpoints", "policy_r1_cal_ep.pt"), kind="policy")
        assert staged["provenance"]["data"] == "data/*.hdf5" and staged["checkpoint_id"] == "ab" * 16
        src = torch.load(os.path.join(root, "checkpoints", "policy_r1_cal_ep.pt"), weights_only=True)
        assert torch.equal(staged["state_dict"]["selector"]["w"], src["state_dict"]["selector"]["w"])
        # records and traces: no absolute path left
        with open(os.path.join(dd, "records", "flights_shield_r1_i0.jsonl"), encoding="utf-8") as f:
            first = json.loads(f.readline())
        assert first["checkpoint"] == "checkpoints/policy_r1_cal_ep.pt", first["checkpoint"]
        assert first["world_json"].startswith("sim/worlds/test/"), first["world_json"]
        with gzip.open(os.path.join(dd, "traces", "trace_r1", "r1_trace_x_e000.jsonl.gz"), "rt",
                       encoding="utf-8") as f:
            assert "sim/worlds/test/a.json" in f.read()
        assert not res["leftovers"], res["leftovers"]
        # missions table and numbers: 30 + 30 rows; unshielded 6 successes, shielded 18
        assert res["rows"] == 60
        r1 = res["numbers"][0]
        assert (r1["succ_u"], r1["succ_s"], r1["only_s"], r1["only_u"]) == (6, 18, 12, 0)
        assert abs(r1["p_success"] - mcnemar_exact(0, 12)) < 1e-15 and r1["h_n"] == 24
        assert r1["density"]["dense"] == {"n": 10, "u": 2, "s": 6}
        assert r1["crash_u"] == 24 and r1["crash_s"] == 6 and r1["timeout_s"] == 6
        card = open(os.path.join(md, "README.md"), encoding="utf-8").read()
        assert "**18/30 (60.0%)**" in card and "license: gpl-3.0" in card and "test-org/d" in card
        assert "resolve/main/media/r1_shield_dense.mp4" in card and "| `checkpoints/stage1_res.pt` |" in card
        assert video_block(["a.mp4"], "o/m", ["a.jpg"]).count('poster="https://huggingface.co/o/m/resolve/main/media/a.jpg"') == 1
        dcard = open(os.path.join(dd, "README.md"), encoding="utf-8").read()
        assert "license: cc-by-4.0" in dcard and "missions/missions." in dcard and "(60 rows)" in dcard
        assert os.path.exists(os.path.join(dd, "reports", "paired_stats.txt"))
        with open(os.path.join(dd, "reports", "shield_report.txt"), encoding="utf-8") as f:
            assert f.read() == "report of flights_shield_r1_i0.jsonl\n"
        man = json.load(open(os.path.join(md, "MANIFEST.json"), encoding="utf-8"))
        paths = {e["path"] for e in man["files"]}
        assert {"checkpoints/policy_r1_cal_ep.pt", "media/r1_shield_dense.mp4", "README.md"} <= paths
        sums = open(os.path.join(dd, "SHA256SUMS"), encoding="utf-8").read().splitlines()
        assert all(len(s.split("  ")[0]) == 64 for s in sums) and any(s.endswith("missions/missions.parquet")
                                                                     or s.endswith("missions/missions.csv")
                                                                     for s in sums)
        assert "staged test-org/m" in text
        # a rebuild needs --force; a missing required file stops it
        try:
            stage(args, manifest, root)
            raise AssertionError("expected SystemExit without --force")
        except SystemExit:
            pass
        bad = dict(manifest, checkpoints=manifest["checkpoints"] + [{"name": "x", "file": "checkpoints/none.pt",
                                                                     "group": "dagger", "description": "-"}])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                stage(argparse.Namespace(**{**vars(args), "force": True}), bad, root)
            raise AssertionError("expected SystemExit for a missing required checkpoint")
        except SystemExit as exc:
            assert "checkpoints/none.pt" in str(exc)
        # the repository's own manifest parses and names the six benchmark checkpoints
        own = load_manifest(MANIFEST)
        assert [b["name"] for b in own["benchmark"]] == ["r1", "kw", "sd", "nomotion", "r2", "noact"]
        assert all(c["file"].startswith("checkpoints/") for c in own["checkpoints"])
    print("package_release self-test passed: checkpoints verified after scrubbing, records, traces, worlds, cards, "
          "manifests")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--namespace", default=None, help="Hugging Face user or organisation of the two repositories")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--root", default=".", help="repository root holding checkpoints/, flights*.jsonl, logs/ [.]")
    p.add_argument("--out", default="release", help="staging directory [release]")
    p.add_argument("--weights_license", default="gpl-3.0", help="Hub licence id of the weights [gpl-3.0]")
    p.add_argument("--data_license", default="cc-by-4.0", help="Hub licence id of the dataset [cc-by-4.0]")
    p.add_argument("--home", default=None, help="home directory to replace by ~ [the current user's]")
    p.add_argument("--skip_traces", action="store_true")
    p.add_argument("--posters", action=argparse.BooleanOptionalAction, default=True,
                   help="a JPEG poster frame of every video (needs ffmpeg) [on]")
    p.add_argument("--allow_missing", action="store_true", help="stage even if required files are missing")
    p.add_argument("--force", action="store_true", help="rebuild an existing staging directory")
    p.add_argument("--dry_run", action="store_true", help="print the inventory and the card numbers; write nothing")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    root = os.path.abspath(args.root)
    manifest = load_manifest(args.manifest)
    ns = args.namespace or "<namespace>"
    if args.dry_run:
        inv = inventory(manifest, root)
        print_inventory(inv, root, "%s/%s" % (ns, manifest["model_repo"]), "%s/%s" % (ns, manifest["dataset_repo"]))
        print("\n== card numbers (from the records as they are now)")
        print(results_tables(benchmark_numbers(manifest["benchmark"], root)))
        return 0
    if not args.namespace:
        raise SystemExit("--namespace is required (the Hugging Face user or organisation)")
    stage(args, manifest, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

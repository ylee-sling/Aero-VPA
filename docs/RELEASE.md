# Release plan: what goes to GitHub and what goes to Hugging Face

The git repository holds the code, the configuration and the documentation. The trained weights, the evaluation
records and the videos are published on the Hugging Face Hub. `scripts/release/package_release.py` stages them
from `scripts/release/release_manifest.json`. The model and dataset cards it writes take every number from the
records. Absolute paths of the evaluation machine are scrubbed. `MANIFEST.json` and `SHA256SUMS` record each
file. `scripts/release/upload_hf.py` uploads the staged repositories, private first, and checks them against
their manifests.

| where | what | size |
|---|---|---|
| GitHub `ylee-sling/Aero-VPA` (public) | code, scripts, world generator, vehicle model, docs, preprint | about 1 MB (33 commits) |
| Hugging Face model repo `<ns>/aero-vpa` | 9 policy checkpoints, 1–2 stage-(i) checkpoints, videos, model card | about 1 GB |
| Hugging Face dataset repo `<ns>/aero-vpa-benchmark` | mission records, missions table, decision traces, worlds, reports, dataset card | a few hundred MB |
| not released (stays on the VM) | demonstrations, validation flights, DAgger frames, latent caches, training states | hundreds of GB; `--dry_run` prints the sizes |

The git history needs no rewrite. On 3 Nov 2026 the full history (33 commits, 1.0 MB packed, largest blob the
0.4 MB preprint) contained no deleted large files, private keys, access tokens or machine paths. Releases are
annotated tags on `main`, never a rewritten history: commit ids are cited in the project notes and the paper.

## 1. GitHub repository

**Committed:**

| group | files |
|---|---|
| model and training | `config.py`, `perception.py`, `predictor.py`, `selector.py`, `solver.py`, `pipeline.py`, `checkpoint.py`, `dataset.py`, `train_stage1.py`, `cache_latents.py`, `train_stage2.py`, `calibrate_sigma.py` |
| diagnostics | `diagnose_stage1.py`, `probe_depth.py`, `clearance.py`, `envelope_check.py`, `input_sensitivity.py`, `inspect_hdf5.py` |
| simulator and data | `sim/run_headless_sitl.sh`, `sim/make_world.py`, `sim/models/x500_vpa/`, `sim/models/aero_vpa_rgbd/`, `sim/expert.py`, `sim/collect_expert_dataset.py`, `sim/collect_worlds.sh` |
| DAgger | `sim/dagger.py`, `sim/dagger_round.sh`, `sim/stress_dagger.py` |
| evaluation | `sim/fly_policy.py`, `sim/shield.py`, `sim/evaluate_worlds.sh`, `sim/run_benchmark.sh`, `sim/summarize_flights.py`, `sim/paired_stats.py`, `sim/crash_study.py`, `sim/show_trace.py` |
| release and CI | `scripts/check_release.py`, `scripts/release.sh`, `scripts/release/package_release.py`, `scripts/release/upload_hf.py`, `scripts/release/release_manifest.json`, `.github/workflows/selftest.yml` |
| documentation and metadata | `README.md`, `docs/README.md`, `docs/RELEASE.md`, `docs/preprint_261026.pdf`, `LICENSE`, `CITATION.cff`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.gitattributes` |

**Reproducing the evaluation needs the following:**
- **Environment** (as verified in the README): Ubuntu 22.04, ROS 2 Humble, PX4 v1.17.0 with px4_msgs `release/1.17`, Gazebo Harmonic 8.15.0, Python 3.10, and `requirements.txt` (torch 2.14.1).
- **Worlds:**
  - Regenerate them with `python3 sim/make_world.py --seed 2000 --count 12 --density sparse medium dense --out sim/worlds/test`.
  - Or take the files the paper flew from the dataset's `worlds/test/`. Their SHA-256 sums are in its `MANIFEST.json`, so a regenerated set can be compared against them.
- **Weights:** `hf download <ns>/aero-vpa --include "checkpoints/*" --local-dir .` puts them where the commands expect them.
- **Runs:** `sim/run_benchmark.sh` (README, "The shielded benchmark"), then `sim/paired_stats.py` and `sim/crash_study.py`.

**Never committed.** These are ignored by `.gitignore`, and `check_release.py` refuses them even when force-added:
- `checkpoints/`, `*.pt`
- `data/`, `data_val/`, `dagger_r*/`, `latents*/`, `*.hdf5`
- `flights*.jsonl`, `*.jsonl`, `*.parquet`
- `logs/`: traces, reports, and the VM's queue and status scripts
- `sim/worlds/`
- `videos/`, `*.mp4`
- `release/`
- archives

The VM-only scripts in `logs/` (`run_*.sh`, `status.py`, `failure_study.py`, `paper_pairs.py`) stay there:
- The benchmark runner, the paired statistics and the crash study are now `sim/run_benchmark.sh`, `sim/paired_stats.py` and `sim/crash_study.py`, with self-tests.
- `status.py` is monitoring for one machine.

**On release day:**
1. Bump `CITATION.cff` to `version: "1.1.0"` and set its date.
2. Add the two Hugging Face links to the README.
3. Tag `v1.1.0`.
4. Create the GitHub Release with `scripts/release.sh --push --gh-release --notes NOTES.md`; the default notes are those of v1.0.0, which shipped no weights.

Zenodo's GitHub integration can mint a software DOI for the tag.

## 2. Hugging Face model repository `<ns>/aero-vpa`

| file | name | role |
|---|---|---|
| `checkpoints/policy_r1_cal_ep.pt` | r1 | **recommended**; DAgger round 1, all inputs |
| `checkpoints/policy_r2_cal_ep.pt` | r2 | DAgger round 2 |
| `checkpoints/policy_r3_noact_cal_ep.pt` | noact | rounds 1–3, last command withheld |
| `checkpoints/policy_r3_nomotion_cal_ep.pt` | nomotion | rounds 1–3, last command and velocity withheld |
| `checkpoints/policy_r3_sd_cal_ep.pt` | sd | rounds 1–3, state dropout |
| `checkpoints/policy_r3_kw_cal_ep.pt` | kw | rounds 1–3, keyframe weighting |
| `checkpoints/policy_cal_ep.pt`, `policy_res_cal_ep.pt`, `policy_res_noact_cal_ep.pt` | v1, v2, noact-bc | behaviour-cloning baselines |
| `checkpoints/stage1_res.pt` (and `stage1.pt`) | stage (i) | frozen encoder + JEPA predictor, to retrain stage (ii) |
| `media/*.mp4`, `media/*.jpg` | videos | the six demo flights, with poster frames |

**What a policy file holds:**
- It is a complete inference checkpoint: all networks, the frozen statistics and the calibrated σ thresholds. It loads with `torch.load(..., weights_only=True)`, so no pickle code runs.
- The training states (`*.state.pt`: EMA target and optimiser) are not released.
- The frozen CLIP text tower is not stored.

**Model card** (`README.md`, generated):
1. YAML front matter: licence, `library_name: pytorch`, `pipeline_tag: robotics`, tags, and the dataset link.
2. The framing.
3. The paired results table: success without and with the shield with Wilson 95% intervals, crashes, timeouts, and the exact McNemar p.
4. Success by density.
5. The missions the shield was not revised on (mission 1 and later).
6. Video players.
7. The checkpoint table with SHA-256.
8. Usage.
9. Limitations: simulation only, about 1.2 m/s, about half the missions succeed.
10. Licence and citation.

## 3. Hugging Face dataset repository `<ns>/aero-vpa-benchmark`

| path | contents | format |
|---|---|---|
| `missions/missions.parquet` | one row per mission of every run; the Hub's viewer and `pd.read_parquet("hf://datasets/...")` read it | Parquet (zstd) |
| `records/flights_*.jsonl` | the raw records of `fly_policy.py`, exactly what `summarize_flights.py`, `paired_stats.py` read | JSON lines |
| `records/dagger/` | the DAgger rounds' supervised flights | JSON lines |
| `traces/<set>/*.jsonl.gz` | decision-by-decision traces (`trace_r1`, `trace_r1_shield`, `trace_r1_shield2`, dev set when flown) | gzipped JSON lines, one file per mission |
| `worlds/test/`, `worlds/dev/` | Gazebo SDF + privileged obstacle maps | as flown |
| `reports/` | `shield_report.txt`, `crash_study*.txt`, pilots, `final_r3_report.txt`, swap tests and open loop, `paper_numbers.txt`, `paired_stats.txt` | text |

**Format choices:**
- Parquet holds the analysis table.
- The raw JSON lines stay alongside, so the repository's scripts reproduce the numbers byte for byte.
- Traces stay one gzipped file per mission. They are lossless and `crash_study.py` reads them directly.
- There are no tarballs. The viewer cannot read inside an archive, users could not download parts of one, and the Hub's Xet storage could not deduplicate it.

**Training data is not part of this release.** The demonstrations (about 1.7 GB per world × 200) can be regenerated with the expert. The three DAgger rounds (about 140 MB per mission file) cannot. If they are released later, a separate, possibly gated repository (`<ns>/aero-vpa-dagger`) keeps the benchmark small.

Free accounts get best-effort public storage on the Hub "beyond the first few gigabytes"
([storage limits](https://huggingface.co/docs/hub/storage-limits)).

## 4. Media
- **Model repository:** the six MP4s go to its `media/`. The card embeds them with `<video src="https://huggingface.co/<ns>/aero-vpa/resolve/main/media/<file>.mp4">`.
- **GitHub README:** links to the model card. MP4s never enter git.
- **GitHub Release:** the videos can also be attached as release assets, which do not enter the history.

## 5. Decisions before publishing
1. **Hugging Face namespace:** a SLING AI organisation or a personal account.
2. **Licence of the weights.** The default `gpl-3.0` keeps one licence for code and weights, together with the commercial-licensing statement. `cc-by-nc-4.0` forbids commercial use outright, but it is not an open-source licence.
3. **Licence of the dataset:** default `cc-by-4.0`.
4. **When to go public.** Upload private, review, then publish with `upload_hf.py --public`. For a double-blind venue, keep the repositories private (or anonymous) until the decision.
5. **Training data:** whether and when to release it.

## 6. Procedure on the VM

```bash
cd ~/Aero-VPA && git pull --ff-only && python3 -m pip install --user -U "huggingface_hub>=0.32" pyarrow
python3 scripts/release/package_release.py --dry_run            # inventory and card numbers, writes nothing
python3 scripts/release/package_release.py --namespace <ns>     # stages release/models/aero-vpa, release/datasets/...
hf auth login                                                   # a write token of <ns>
python3 scripts/release/upload_hf.py --namespace <ns>           # private, uploaded, verified
python3 scripts/release/upload_hf.py --namespace <ns> --tag v1.1.0 --public   # release day
```

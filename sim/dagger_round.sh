#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# dagger_round.sh -- one round of stage (iii), dataset aggregation (sim/dagger.py): fly the current policy over the
# training worlds with the privileged expert as labeller and gate, cache the latents of the new frames, retrain
# stage (ii) on the demonstrations plus every round so far, and calibrate sigma (episode_max).
#
#   sim/dagger_round.sh --round 1 --policy checkpoints/policy_res_noact_cal_ep.pt --stage1 checkpoints/stage1_res.pt
#   sim/dagger_round.sh --round 2 --policy checkpoints/policy_r1_cal_ep.pt --stage1 checkpoints/stage1_res.pt
#
# Round N writes dagger_r<N>/<world>.hdf5 (demonstration schema; one episode per recorded segment), the flight
# records dagger_r<N>_i<k>.jsonl, checkpoints/policy_r<N>.pt and checkpoints/policy_r<N>_cal_ep.pt, and logs in
# logs/dagger_r<N>/. Stage (ii) is trained like --like (default: the flying policy): its steps, batch size, learning
# rate, EMA, class weights, validation split and policy_state_inputs, read from the checkpoint, so rounds differ
# only in their data. Frames of the DAgger rounds are drawn --weight times as often as demonstration frames.
# Every step is resumable: rerunning the same command continues where it stopped (flights are counted in the
# records, caches are reused, training resumes from its state file).
#
# Exit status: 0 done, 1 a step failed, 2 bad arguments.
set -uo pipefail

usage() {
    sed -n '/^# dagger_round.sh/,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --round N           round number (output names)                                          [required]
  --policy CKPT       calibrated policy that flies this round                              [required]
  --stage1 CKPT       stage-1 checkpoint of the policy (latent caches)                     [required]
  --like CKPT         stage-2 checkpoint whose training settings to reuse                  [the policy]
  --latents DIR       latent cache directory of --stage1                                   [found automatically]
  --worlds GLOB       training worlds                                                      ['sim/worlds/*.sdf']
  --missions N        missions per world                                                   [1]
  --instances N       parallel simulators (instances 0..N-1)                               [3]
  --weight W          sampling weight of DAgger frames against demonstration frames        [3]
  --data GLOB         demonstrations                                                       ['data/*.hdf5']
  --val GLOB          validation flights for the calibration                               ['data_val/*.hdf5']
  --skip_flights      only cache, train and calibrate (the flights of this round are done)
  -h, --help
Arguments after "--" are passed to fly_policy.py (e.g. -- --gate_distance 2.0); the flights use a mission timeout
of 3 x the expert's duration + 30 s, since takeovers that re-plan cost several seconds each.
EOF
}

ROUND=""; POLICY=""; STAGE1=""; LIKE=""; LATENTS=""; WORLDS='sim/worlds/*.sdf'; MISSIONS=1; INSTANCES=3; WEIGHT=3
DATA='data/*.hdf5'; VAL='data_val/*.hdf5'; SKIP_FLIGHTS=0; EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --round) ROUND="$2"; shift 2 ;;
        --policy) POLICY="$2"; shift 2 ;;
        --stage1) STAGE1="$2"; shift 2 ;;
        --like) LIKE="$2"; shift 2 ;;
        --latents) LATENTS="$2"; shift 2 ;;
        --worlds) WORLDS="$2"; shift 2 ;;
        --missions) MISSIONS="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --weight) WEIGHT="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --val) VAL="$2"; shift 2 ;;
        --skip_flights) SKIP_FLIGHTS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$ROUND" && -n "$POLICY" && -n "$STAGE1" ]] || { usage >&2; exit 2; }
[[ -f "$POLICY" && -f "$STAGE1" ]] || { echo "missing $POLICY or $STAGE1" >&2; exit 2; }
LIKE="${LIKE:-$POLICY}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
PYTHON="${AERO_VPA_PYTHON:-python3}"
OUT="dagger_r$ROUND"; LOGS="logs/$OUT"; TAG="$OUT"
mkdir -p "$OUT" "$LOGS"
log() { printf '[dagger r%s %(%F %T)T] %s\n' "$ROUND" -1 "$*"; }

# ---- 1. flights: worlds dealt round-robin to the simulators; resumable through the records
if [[ $SKIP_FLIGHTS -eq 0 ]]; then
    mapfile -t ALL < <(ls $WORLDS 2>/dev/null)
    [[ ${#ALL[@]} -gt 0 ]] || { log "no worlds match $WORLDS"; exit 2; }
    log "flying $POLICY over ${#ALL[@]} worlds x $MISSIONS missions on $INSTANCES simulators"
    pids=()
    for ((i = 0; i < INSTANCES; i++)); do
        mine=()
        for ((j = i; j < ${#ALL[@]}; j += INSTANCES)); do mine+=("${ALL[j]}"); done
        [[ ${#mine[@]} -gt 0 ]] || continue
        sim/evaluate_worlds.sh --instance "$i" --episodes "$MISSIONS" --results "${OUT}_i$i.jsonl" --tag "$TAG" \
            --log-dir "$LOGS" "$POLICY" "${mine[@]}" -- --dagger_dir "$OUT" --timeout_factor 3 --timeout_extra_s 30 \
            "${EXTRA[@]}" > "$LOGS/flights_i$i.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || log "a simulator sweep ended with errors (see $LOGS); continuing"; done
fi
ls "$OUT"/*.hdf5 > /dev/null 2>&1 || { log "no DAgger frames in $OUT"; exit 1; }

# ---- 2. latents of the new frames, in the cache directory of the demonstrations
if [[ -z "$LATENTS" ]]; then
    LATENTS=$("$PYTHON" - "$STAGE1" "$DATA" <<'PY'
import glob, os, sys
from cache_latents import cache_is_current, cache_path
from checkpoint import load_checkpoint
from dataset import expand_paths
cid = load_checkpoint(sys.argv[1])["checkpoint_id"]
src = expand_paths(sys.argv[2])[0]
print(next((d for d in sorted(glob.glob("latents*")) if os.path.isdir(d) and cache_is_current(cache_path(d, src), src, cid)), ""))
PY
)
fi
[[ -n "$LATENTS" ]] || { log "no latent cache of $STAGE1 found; pass --latents"; exit 1; }
log "caching latents of $OUT in $LATENTS"
"$PYTHON" cache_latents.py "$STAGE1" "$OUT/*.hdf5" --out_dir "$LATENTS" > "$LOGS/cache.log" 2>&1 || { log "cache_latents failed"; exit 1; }

# ---- 3. stage (ii) on the demonstrations and every round so far, trained like --like
TRAIN_ARGS=$("$PYTHON" - "$LIKE" <<'PY'
import shlex, sys
from checkpoint import config_from_dict, load_checkpoint
ckpt = load_checkpoint(sys.argv[1])
prov, cfg = ckpt["provenance"], config_from_dict(ckpt["config"])
args = ["--steps", str(prov["steps"]), "--batch_size", str(prov["batch_size"]), "--lr", repr(prov["lr"]),
        "--ema", repr(prov["ema"]), "--class_weights", str(prov["class_weights"]),
        "--policy_state_inputs", *cfg.policy_state_inputs]
split = prov.get("split") or {}
if split.get("val"):
    args += ["--val", *split["val"]]
else:
    args += ["--val_fraction", repr(split.get("val_fraction", 0.05)), "--split_seed", str(split.get("split_seed", 0))]
print(" ".join(shlex.quote(a) for a in args))
PY
) || { log "cannot read the training settings of $LIKE"; exit 1; }
NAME="checkpoints/policy_r$ROUND"
RESUME=""; [[ -f "$NAME.pt.state.pt" && ! -f "$NAME.pt" ]] && RESUME="--resume"
if [[ ! -f "$NAME.pt" ]]; then
    log "stage (ii): $DATA + dagger_r*/ (weight $WEIGHT), $TRAIN_ARGS"
    eval "\"$PYTHON\" train_stage2.py \"$STAGE1\" \"$DATA\" 'dagger_r[0-9]*/*.hdf5' --latents \"$LATENTS\" --out $NAME.pt \
        --source_weight 'dagger_r[0-9]*/*.hdf5=$WEIGHT' $TRAIN_ARGS $RESUME" > "$LOGS/train.log" 2>&1 \
        || { log "train_stage2 failed (see $LOGS/train.log)"; exit 1; }
fi

# ---- 4. sigma_lo / sigma_crit on the validation flights
if [[ ! -f "${NAME}_cal_ep.pt" ]]; then
    log "calibration"
    "$PYTHON" calibrate_sigma.py "$NAME.pt" "$VAL" --method episode_max --out "${NAME}_cal_ep.pt" > "$LOGS/calibrate.log" 2>&1 \
        || { log "calibrate_sigma failed (see $LOGS/calibrate.log)"; exit 1; }
fi
"$PYTHON" sim/summarize_flights.py "${OUT}"_i*.jsonl > "$LOGS/flights_summary.txt" 2>&1 || true
log "done: ${NAME}_cal_ep.pt (flights: $LOGS/flights_summary.txt)"

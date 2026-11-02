#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# run_benchmark.sh -- one checkpoint over the test worlds on several simulators at once (the benchmark of the
# paper: 36 test worlds x 5 missions, 3 simulators), resumably.
#
# The worlds are dealt round-robin to the simulator instances (instance i flies worlds i, i+N, i+2N, ...); every
# instance runs sim/evaluate_worlds.sh in the background with its own results file <stem>_i<i>.jsonl, and the
# script waits for all of them, then prints the summary. Rerun to continue: evaluate_worlds.sh skips complete
# worlds and continues at the next mission. Mission k of a world is the same for every checkpoint and option,
# so two runs over the same worlds are paired mission by mission.
#
# Usage:
#   sim/run_benchmark.sh [options] NAME CHECKPOINT [-- extra fly_policy.py arguments]
#   sim/run_benchmark.sh --shield r1_shield checkpoints/policy_r1_cal_ep.pt          # flights_r1_shield_i*.jsonl
#   sim/run_benchmark.sh --stem flights_shield_r1 r1_shield checkpoints/policy_r1_cal_ep.pt -- --shield
#   python3 sim/summarize_flights.py flights_r1_i*.jsonl flights_r1_shield_i*.jsonl --compare policy_r1_cal_ep.pt r1_shield
#
# Run from the repository root. NAME is the tag of the records (fly_policy.py --tag). Test worlds:
#   python3 sim/make_world.py --seed 2000 --count 12 --density sparse medium dense --out sim/worlds/test
# or the worlds/test folder of the benchmark dataset: the files the paper's runs flew (SHA-256 in its MANIFEST.json).
set -uo pipefail

usage() {
    sed -n '/^# run_benchmark.sh/,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --shield            fly with the onboard safety shield (fly_policy.py --shield)
  --trace DIR         write decision traces (fly_policy.py --trace DIR)
  --episodes N        missions per world                                    [5]
  --instances N       simulators side by side (at most 3 on an L4 VM)       [3]
  --first-instance N  number of the first simulator instance                [0]
  --worlds GLOB       worlds to fly                                         [sim/worlds/test/*.sdf]
  --stem STEM         results files STEM_i<instance>.jsonl                  [flights_NAME]
  --log-dir DIR       logs of evaluate_worlds.sh and its runs               [logs/STEM]
  --dry-run           print the commands, run nothing
  -h, --help
EOF
}

SHIELD=0; TRACE=""; EPISODES=5; INSTANCES=3; FIRST=0; WORLDS_GLOB="sim/worlds/test/*.sdf"; STEM=""; LOG_DIR=""
DRY=0; NAME=""; CHECKPOINT=""; EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shield) SHIELD=1; shift ;;
        --trace) TRACE="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --first-instance) FIRST="$2"; shift 2 ;;
        --worlds) WORLDS_GLOB="$2"; shift 2 ;;
        --stem) STEM="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) if [[ -z "$NAME" ]]; then NAME="$1"; elif [[ -z "$CHECKPOINT" ]]; then CHECKPOINT="$1"; else
               echo "unexpected argument: $1" >&2; exit 2; fi; shift ;;
    esac
done
[[ -n "$NAME" && -n "$CHECKPOINT" ]] || { usage >&2; exit 2; }
[[ "$INSTANCES" =~ ^[1-9][0-9]*$ && "$EPISODES" =~ ^[1-9][0-9]*$ && "$FIRST" =~ ^[0-9]+$ ]] \
    || { echo "--instances and --episodes must be positive integers, --first-instance a non-negative one" >&2; exit 2; }
[[ $DRY -eq 1 || -f "$CHECKPOINT" ]] || { echo "no checkpoint $CHECKPOINT" >&2; exit 2; }
[[ -x sim/evaluate_worlds.sh ]] || { echo "run from the repository root (sim/evaluate_worlds.sh not found)" >&2; exit 2; }
STEM="${STEM:-flights_$NAME}"
LOG_DIR="${LOG_DIR:-logs/$STEM}"
# shellcheck disable=SC2206  # the glob is meant to expand
WORLDS=($WORLDS_GLOB)
[[ ${#WORLDS[@]} -gt 0 && -f "${WORLDS[0]}" ]] || { echo "no worlds match $WORLDS_GLOB" >&2; exit 2; }
FLY=()
[[ $SHIELD -eq 1 ]] && FLY+=(--shield)
[[ -n "$TRACE" ]] && FLY+=(--trace "$TRACE")
FLY+=(${EXTRA[@]+"${EXTRA[@]}"})
log() { printf '[run_benchmark %s] %s\n' "$(date '+%F %T')" "$*"; }

log "$NAME: $CHECKPOINT over ${#WORLDS[@]} worlds x $EPISODES missions on $INSTANCES simulators -> ${STEM}_i*.jsonl"
[[ $DRY -eq 1 ]] || mkdir -p "$LOG_DIR"
pids=()
for ((i = 0; i < INSTANCES; i++)); do
    G=()
    for ((j = i; j < ${#WORLDS[@]}; j += INSTANCES)); do G+=("${WORLDS[j]}"); done
    [[ ${#G[@]} -gt 0 ]] || continue
    inst=$((FIRST + i))
    cmd=(sim/evaluate_worlds.sh --instance "$inst" --episodes "$EPISODES" --results "${STEM}_i$i.jsonl" --tag "$NAME"
         --log-dir "$LOG_DIR" "$CHECKPOINT" "${G[@]}")
    [[ ${#FLY[@]} -gt 0 ]] && cmd+=(-- "${FLY[@]}")
    if [[ $DRY -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
        continue
    fi
    "${cmd[@]}" > "$LOG_DIR/evaluate_i$i.out" 2>&1 &
    pids+=("$!")
done
[[ $DRY -eq 1 ]] && exit 0
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[[ $status -eq 0 ]] || log "a sweep ended with errors (see $LOG_DIR/evaluate_i*.out); rerun to continue"
log "$NAME: $(cat "${STEM}"_i*.jsonl 2>/dev/null | grep -c .) records"
python3 sim/summarize_flights.py "${STEM}"_i*.jsonl --by density
exit $status

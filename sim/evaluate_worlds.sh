#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# evaluate_worlds.sh -- fly a policy checkpoint over many worlds in PX4 SITL, resumably (sim/fly_policy.py).
#
# For every world it counts the missions already in the results file for this policy (--tag) and world and,
# while fewer than --episodes, starts run_headless_sitl.sh with fly_policy.py for the rest, continuing at the
# next mission index. A run that ends after a collision (exit 3; the vehicle may be damaged, so the simulator
# is restarted) has recorded its mission and is simply continued. Any other early end costs one attempt, and a
# world still short after --attempts such runs is reported and skipped. Mission k of a world is the same for
# every policy (its seed is the number in the world name), so checkpoints evaluated with the same options fly
# identical missions. Safe to rerun: complete worlds are skipped.
#
# Usage:
#   sim/evaluate_worlds.sh [options] CHECKPOINT WORLD.sdf... [-- extra fly_policy.py arguments]
#   sim/evaluate_worlds.sh --instance 0 checkpoints/policy_cal_ep.pt sim/worlds/val/*.sdf
#   python3 sim/summarize_flights.py flights.jsonl
#
# Exit status: 0 when every world has --episodes missions, 1 otherwise, 130 when interrupted.
set -uo pipefail

usage() {
    sed -n '/^# evaluate_worlds.sh/,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --instance N        simulator instance (own DDS port, ROS domain, gz partition)    [0]
  --episodes N        missions per world                                            [10]
  --attempts N        failed runs per world before giving up on it                  [3]
  --results FILE      JSON lines file of fly_policy.py                              [flights.jsonl]
  --tag NAME          name of the policy in the results                             [checkpoint file name]
  --log-dir DIR       per-run logs                                                  [logs]
  --px4-dir DIR       PX4-Autopilot checkout                                        [~/PX4-Autopilot]
  --ros-ws DIR        colcon workspace with px4_msgs                                [~/px4_ros_ws]
  -h, --help
EOF
}

INSTANCE=0; EPISODES=10; ATTEMPTS=3; RESULTS=flights.jsonl; TAG=""; LOG_DIR=logs
PX4_DIR="$HOME/PX4-Autopilot"; ROS_WS="$HOME/px4_ros_ws"
CHECKPOINT=""; WORLDS=(); EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance) INSTANCE="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --attempts) ATTEMPTS="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --px4-dir) PX4_DIR="$2"; shift 2 ;;
        --ros-ws) ROS_WS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) if [[ -z "$CHECKPOINT" ]]; then CHECKPOINT="$1"; else WORLDS+=("$1"); fi; shift ;;
    esac
done
[[ -n "$CHECKPOINT" && ${#WORLDS[@]} -gt 0 ]] || { usage >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "no checkpoint $CHECKPOINT" >&2; exit 2; }
TAG="${TAG:-$(basename "$CHECKPOINT")}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="${AERO_VPA_LAUNCHER:-$REPO/sim/run_headless_sitl.sh}"
PYTHON="${AERO_VPA_PYTHON:-python3}"
mkdir -p "$LOG_DIR"

log() { printf '[e%s %(%F %T)T] %s\n' "$INSTANCE" -1 "$*"; }

progress() {  # progress WORLD: "<missions recorded> <next mission index>" for this tag
    "$PYTHON" - "$RESULTS" "$TAG" "$1" <<'EOF'
import json
import os
import sys

path, tag, world = sys.argv[1:4]
seen = set()
if os.path.isfile(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("tag") == tag and r.get("world") == world:
                seen.add(int(r["episode"]))
print(len(seen), max(seen) + 1 if seen else 0)
EOF
}

interrupted=0
trap 'interrupted=1' INT TERM
short=()
for world in "${WORLDS[@]}"; do
    [[ $interrupted -eq 1 ]] && break
    name="$(basename "$world" .sdf)"
    if [[ ! -f "$world" ]]; then log "missing world file $world"; short+=("$name (no world)"); continue; fi
    if [[ "$name" =~ _s0*([0-9]+)$ ]]; then seed="${BASH_REMATCH[1]}"; else seed=0; fi
    failed=0; runs=0
    while :; do
        read -r have next < <(progress "$name")
        if (( have >= EPISODES )); then
            (( runs == 0 )) && log "skip $name ($have/$EPISODES)" || log "done $name ($have/$EPISODES)"
            break
        fi
        if (( failed >= ATTEMPTS )); then
            log "giving up on $name after $failed failed runs ($have/$EPISODES)"
            short+=("$name ($have/$EPISODES)")
            break
        fi
        runs=$((runs + 1))
        runlog="$LOG_DIR/e${INSTANCE}_${name}_$(date +%m%d-%H%M%S).log"
        log "start $name: $have/$EPISODES flown, missions $next.., seed $seed (log $runlog)"
        "$LAUNCHER" --px4-dir "$PX4_DIR" --ros-ws "$ROS_WS" --instance "$INSTANCE" --world "$world" -- \
            "$PYTHON" "$REPO/sim/fly_policy.py" "$CHECKPOINT" --episodes $(( EPISODES - have )) \
            --episode_start "$next" --seed "$seed" --results "$RESULTS" --tag "$TAG" "${EXTRA[@]}" \
            > "$runlog" 2>&1 < /dev/null
        code=$?
        reason="$(grep -hE 'ERROR|mission [0-9]+: [a-z_]+ after' "$runlog" | tail -n 1 | sed -E 's/.*\]: //; s/^\[[0-9:]+\] //')"
        log "end   $name: exit $code${reason:+ -- $reason}"
        [[ $interrupted -eq 1 ]] && break
        read -r after _ < <(progress "$name")
        if (( code != 0 && (code != 3 || after <= have) )); then   # exit 3 after recording a mission is progress
            failed=$((failed + 1))
            sleep 5
        fi
    done
done

if [[ $interrupted -eq 1 ]]; then log "interrupted"; exit 130; fi
if [[ ${#short[@]} -gt 0 ]]; then
    log "finished; ${#short[@]} world(s) short: ${short[*]}"
    exit 1
fi
log "finished; all ${#WORLDS[@]} worlds have $EPISODES missions of $TAG in $RESULTS"

#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# collect_worlds.sh -- collect expert demonstrations over many worlds with one simulator stack, resumably.
#
# For every world it counts the successful episodes already in data/<world>.hdf5 and, while fewer than
# --episodes, starts run_headless_sitl.sh with collect_expert_dataset.py --append for the rest. Anything that
# ends a run early (a collision, which needs a fresh simulator; a stack that did not start; a take-off
# timeout) costs one attempt, and the next attempt continues the same file with a new mission seed. A world
# that still falls short after --attempts runs is reported and skipped, so one bad world never stops the
# sweep. Safe to rerun: complete worlds are skipped.
#
# Usage:
#   sim/collect_worlds.sh [options] WORLD.sdf... [-- extra collect_expert_dataset.py arguments]
#   sim/collect_worlds.sh --instance 1 $(seq -f 'sim/worlds/aero_forest_medium_s%04g.sdf' 67 133) -- --compression lzf
#
# The mission seed of a world is the number in its name (..._s0125 -> 125), offset by 1000 per episode
# already recorded and 100 per retry, so a retry never repeats the missions of the run before it.
# Exit status: 0 when every world has --episodes successful episodes, 1 otherwise, 130 when interrupted.
set -uo pipefail

usage() {
    sed -n '/^# collect_worlds.sh/,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --instance N        simulator instance (own DDS port, ROS domain, gz partition)    [0]
  --episodes N        successful episodes per world                                 [10]
  --attempts N        simulator runs per world before giving up on it               [5]
  --out-dir DIR       output directory, one <world>.hdf5 per world                  [data]
  --log-dir DIR       per-run logs                                                  [logs]
  --px4-dir DIR       PX4-Autopilot checkout                                        [~/PX4-Autopilot]
  --ros-ws DIR        colcon workspace with px4_msgs                                [~/px4_ros_ws]
  -h, --help
EOF
}

INSTANCE=0; EPISODES=10; ATTEMPTS=5; OUT_DIR=data; LOG_DIR=logs
PX4_DIR="$HOME/PX4-Autopilot"; ROS_WS="$HOME/px4_ros_ws"
WORLDS=(); EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance) INSTANCE="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --attempts) ATTEMPTS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --px4-dir) PX4_DIR="$2"; shift 2 ;;
        --ros-ws) ROS_WS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) WORLDS+=("$1"); shift ;;
    esac
done
[[ ${#WORLDS[@]} -gt 0 ]] || { usage >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="${AERO_VPA_LAUNCHER:-$REPO/sim/run_headless_sitl.sh}"
PYTHON="${AERO_VPA_PYTHON:-python3}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

log() { printf '[w%s %(%F %T)T] %s\n' "$INSTANCE" -1 "$*"; }

successes() {  # successes FILE: complete, successful episodes (0 if missing); exit 2 if unreadable
    [[ -f "$1" ]] || { echo 0; return 0; }
    "$PYTHON" - "$1" <<'EOF'
import sys
import h5py
try:
    with h5py.File(sys.argv[1], "r") as f:
        print(sum(1 for k in f if k.startswith("episode_")
                  and bool(f[k].attrs.get("complete", True)) and bool(f[k].attrs.get("success", False))))
except OSError:
    sys.exit(2)
EOF
}

interrupted=0
trap 'interrupted=1' INT TERM
short=()
for world in "${WORLDS[@]}"; do
    [[ $interrupted -eq 1 ]] && break
    name="$(basename "$world" .sdf)"
    out="$OUT_DIR/$name.hdf5"
    if [[ ! -f "$world" ]]; then log "missing world file $world"; short+=("$name (no world)"); continue; fi
    if [[ "$name" =~ _s0*([0-9]+)$ ]]; then base_seed="${BASH_REMATCH[1]}"; else base_seed=0; fi
    attempt=0
    while :; do
        have="$(successes "$out")"
        if [[ $? -eq 2 ]]; then
            log "$out is unreadable; moving it to $out.corrupt"
            mv -f "$out" "$out.corrupt"
            have=0
        fi
        if (( have >= EPISODES )); then
            (( attempt == 0 )) && log "skip $name ($have/$EPISODES)" || log "done $name ($have/$EPISODES)"
            break
        fi
        if (( attempt >= ATTEMPTS )); then
            log "giving up on $name after $attempt runs ($have/$EPISODES)"
            short+=("$name ($have/$EPISODES)")
            break
        fi
        attempt=$((attempt + 1))
        seed=$(( base_seed + 1000 * have + 100 * (attempt - 1) ))
        runlog="$LOG_DIR/w${INSTANCE}_${name}_$(date +%m%d-%H%M%S).log"
        log "start $name: $have/$EPISODES recorded, run $attempt/$ATTEMPTS, seed $seed (log $runlog)"
        "$LAUNCHER" --px4-dir "$PX4_DIR" --ros-ws "$ROS_WS" --instance "$INSTANCE" --world "$world" -- \
            "$PYTHON" "$REPO/sim/collect_expert_dataset.py" --append --num_episodes $(( EPISODES - have )) \
            --seed "$seed" --output "$out" "${EXTRA[@]}" > "$runlog" 2>&1 < /dev/null
        code=$?
        reason="$(grep -hE 'ERROR|episode [0-9]+ failed' "$runlog" | tail -n 1 | sed -E 's/.*\]: //; s/^\[[0-9:]+\] //')"
        log "end   $name: exit $code${reason:+ -- $reason}"
        [[ $interrupted -eq 1 ]] && break            # Ctrl-C; a collector stopped on its own is retried
        (( code == 0 )) || sleep 5
    done
done

if [[ $interrupted -eq 1 ]]; then log "interrupted"; exit 130; fi
if [[ ${#short[@]} -gt 0 ]]; then
    log "finished; ${#short[@]} world(s) short: ${short[*]}"
    exit 1
fi
log "finished; all ${#WORLDS[@]} worlds have $EPISODES successful episodes"

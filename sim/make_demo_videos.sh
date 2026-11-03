#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# make_demo_videos.sh -- FPV demonstration videos of the shielded benchmark: for each density (dense, medium,
# sparse), a test mission that both checkpoints succeeded on in the benchmark, re-flown by each with the shield
# and frame recording (fly_policy.py --shield --trace --frames) on one simulator instance, and rendered by
# sim/render_fpv.py to OUT/<name>_<density>.mp4 with a poster OUT/<name>_<density>.jpg and OUT/<name>_<density>.json.
#
# Missions are tried best first (sim/render_fpv.py pick: no backoff of the shield, then the largest clearance,
# then the shortest flight). A re-flight is a new flight and may end differently from the benchmark; a mission is
# used only when every checkpoint succeeds on it again, else the next candidate is tried. Every re-flight stays in
# WORK/flights_video.jsonl and is listed in OUT/videos.json. Resumable: rerun to continue; flown missions are not
# flown again.
#
# Usage (from the repository root, e.g. in tmux):
#   sim/make_demo_videos.sh                       # r1 and kw with the shield, instance 3, videos/
#   sim/make_demo_videos.sh --dry-run             # the candidate missions and what would be flown
# Options:
#   --instance N      simulator instance for the re-flights (the free fourth slot next to 3 sweeps)   [3]
#   --candidates N    missions tried per density                                                     [3]
#   --out DIR         videos and posters (scripts/release/release_manifest.json publishes videos/)  [videos]
#   --work DIR        records, traces, frames and logs of the re-flights                            [logs/fpv]
#   --wait SESSION    first wait until this tmux session has ended (e.g. a running sweep)
#   --keep-frames     keep the raw frames (about 100 kB per decision) after rendering
#   --px4-dir DIR, --ros-ws DIR   as sim/evaluate_worlds.sh                [~/PX4-Autopilot, ~/px4_ros_ws]
#   --dry-run         print the candidates and the plan, fly nothing
# Rendering needs Pillow and an ffmpeg; without them in the system Python, they are installed once with
# pip --target into ~/.release_libs, which only the rendering step puts on its PYTHONPATH.
set -uo pipefail

NAMES=(r1_shield kw_shield)
declare -A CK=([r1_shield]=checkpoints/policy_r1_cal_ep.pt [kw_shield]=checkpoints/policy_r3_kw_cal_ep.pt)
declare -A REC=([r1_shield]="flights_shield_r1_i*.jsonl" [kw_shield]="flights_shield_kw_i*.jsonl")
declare -A TITLE=([r1_shield]="Aero-VPA r1 + onboard safety shield" [kw_shield]="Aero-VPA kw + onboard safety shield")
DENSITIES=(dense medium sparse)
INSTANCE=3; CANDIDATES=3; OUT=videos; WORK=logs/fpv; WAIT=""; KEEP=0; DRY=0
PX4_DIR="$HOME/PX4-Autopilot"; ROS_WS="$HOME/px4_ros_ws"; LIBS="$HOME/.release_libs"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance) INSTANCE="$2"; shift 2 ;;
        --candidates) CANDIDATES="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --work) WORK="$2"; shift 2 ;;
        --wait) WAIT="$2"; shift 2 ;;
        --keep-frames) KEEP=1; shift ;;
        --px4-dir) PX4_DIR="$2"; shift 2 ;;
        --ros-ws) ROS_WS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '/^# make_demo_videos.sh/,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -x sim/run_headless_sitl.sh && -f sim/render_fpv.py ]] || { echo "run from the repository root" >&2; exit 2; }
grep -q -- '"--frames"' sim/fly_policy.py || { echo "sim/fly_policy.py has no --frames: git pull first" >&2; exit 2; }
log() { printf '[fpv %(%F %T)T] %s\n' -1 "$*"; }
mkdir -p "$WORK/logs" "$OUT"
RESULTS="$WORK/flights_video.jsonl"

for name in "${NAMES[@]}"; do
    [[ -f "${CK[$name]}" ]] || { log "missing ${CK[$name]}: stopped"; exit 1; }
    compgen -G "${REC[$name]}" > /dev/null || { log "no benchmark records ${REC[$name]}: stopped"; exit 1; }
done

# status NAME WORLD K: success / failed / none for the re-flights of this checkpoint and mission
status() {
    python3 - "$RESULTS" "$1_video" "$2" "$3" <<'PY'
import json, os, sys
path, tag, world, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
last = None
if os.path.exists(path):
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("tag") == tag and r.get("world") == world and int(r.get("episode", -1)) == k:
            last = r
print("none" if last is None else ("success" if last.get("success") else "failed"))
PY
}

frames_of() { echo "$WORK/frames/$1_video_$2_e$(printf %03d "$3").frames"; }
trace_of() { echo "$WORK/trace/$1_video_$2_e$(printf %03d "$3").jsonl"; }

# 1. candidate missions: every checkpoint succeeded on them in the benchmark
picks=()
for name in "${NAMES[@]}"; do picks+=(--run "$name=${REC[$name]}"); done
python3 sim/render_fpv.py pick "${picks[@]}" --per_density "$CANDIDATES" > "$WORK/candidates.tsv" \
    || { log "choosing missions failed: stopped"; exit 1; }
log "candidates (best first): $WORK/candidates.tsv"
sed 's/^/    /; s/\t/   /g' "$WORK/candidates.tsv"
if [[ $DRY -eq 1 ]]; then
    log "dry run: would re-fly on instance $INSTANCE, best candidate first per density, until every checkpoint succeeds"
    exit 0
fi

# 2. the renderer's tools (Pillow, ffmpeg), isolated from the Python that flies
RENDER=(python3)
if ! python3 -c "import PIL" 2>/dev/null || { ! command -v ffmpeg >/dev/null && ! python3 -c "import imageio_ffmpeg" 2>/dev/null; }; then
    if ! PYTHONPATH="$LIBS" python3 -c "import PIL, imageio_ffmpeg" 2>/dev/null; then
        log "installing Pillow and imageio-ffmpeg into $LIBS (only the rendering step uses them)"
        python3 -m pip install -q --target "$LIBS" -U pillow imageio-ffmpeg || { log "pip failed: stopped"; exit 1; }
    fi
    RENDER=(env "PYTHONPATH=$LIBS" python3)
fi

if [[ -n "$WAIT" ]]; then
    log "waiting for tmux session $WAIT to end"
    while tmux has-session -t "=$WAIT" 2>/dev/null; do sleep 60; done
fi
if pgrep -f "run_headless_sitl.sh.*--instance $INSTANCE( |$)" > /dev/null; then
    log "a simulator already runs on instance $INSTANCE: stopped (choose another with --instance)"; exit 1
fi
log "simulators running now: $(pgrep -fc 'run_headless_sitl.sh' || true); re-flights use instance $INSTANCE"

fly() {  # fly NAME WORLD K SEED
    local name=$1 world=$2 k=$3 seed=$4 runlog
    runlog="$WORK/logs/${name}_${world}_e${k}_$(date +%m%d-%H%M%S).log"
    rm -f "$(frames_of "$name" "$world" "$k")" "$(frames_of "$name" "$world" "$k").json"
    log "fly $name on $world mission $k (seed $seed; log $runlog)"
    sim/run_headless_sitl.sh --px4-dir "$PX4_DIR" --ros-ws "$ROS_WS" --instance "$INSTANCE" \
        --world "sim/worlds/test/$world.sdf" -- \
        python3 sim/fly_policy.py "${CK[$name]}" --shield --episodes 1 --episode_start "$k" --seed "$seed" \
        --results "$RESULTS" --tag "${name}_video" --trace "$WORK/trace" --frames "$WORK/frames" \
        > "$runlog" 2>&1 < /dev/null
    log "  exit $?: $(grep -hoE 'mission [0-9]+: [a-z_]+ after [0-9.]+ s' "$runlog" | tail -n 1)"
}

# 3. per density: the first candidate every checkpoint succeeds on again, rendered
declare -A CHOSEN=()
for dens in "${DENSITIES[@]}"; do
    while IFS=$'\t' read -r d world k seed _; do
        [[ "$d" == "$dens" ]] || continue
        ok=1
        for name in "${NAMES[@]}"; do
            s="$(status "$name" "$world" "$k")"
            if [[ "$s" == none ]]; then
                fly "$name" "$world" "$k" "$seed"
                s="$(status "$name" "$world" "$k")"
            fi
            if [[ "$s" != success ]]; then
                log "  $name on $world mission $k: $s; next candidate"
                ok=0
                [[ $KEEP -eq 1 ]] || rm -f "$(frames_of "$name" "$world" "$k")"
                break
            fi
        done
        if [[ $ok -eq 1 ]]; then CHOSEN[$dens]="$world $k"; break; fi
    done < <(grep -v '^#' "$WORK/candidates.tsv")
    if [[ -z "${CHOSEN[$dens]:-}" ]]; then
        log "$dens: no candidate succeeded for every checkpoint (rerun with --candidates $((CANDIDATES + 2)))"
        continue
    fi
    read -r world k <<< "${CHOSEN[$dens]}"
    for name in "${NAMES[@]}"; do
        video="$OUT/${name}_${dens}.mp4"
        if [[ -f "$video" && -f "${video%.mp4}.json" ]] && grep -q "\"world\": \"$world\"" "${video%.mp4}.json" \
                && grep -q "\"episode\": $k," "${video%.mp4}.json"; then
            log "$video exists for $world mission $k: kept"; continue
        fi
        fr="$(frames_of "$name" "$world" "$k")"
        [[ -f "$fr" ]] || { log "  no frames $fr (flown before --frames?): delete its record line to re-fly"; continue; }
        "${RENDER[@]}" sim/render_fpv.py render --trace "$(trace_of "$name" "$world" "$k")" --frames "$fr" \
            --title "${TITLE[$name]}" --out "$video" --poster "${video%.mp4}.jpg" \
            > "$WORK/logs/render_${name}_${dens}.log" 2>&1 \
            && log "  $(tail -n 1 "$WORK/logs/render_${name}_${dens}.log")" \
            || { log "  rendering $video failed: $WORK/logs/render_${name}_${dens}.log"; continue; }
        [[ $KEEP -eq 1 ]] || rm -f "$fr"
    done
done

# 4. what the videos show, and every re-flight
"${RENDER[@]}" sim/render_fpv.py index --videos "$OUT" --results "$RESULTS" | sed 's/^/    /'
log "videos: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l) in $OUT/ (re-flights $(grep -c . "$RESULTS" 2>/dev/null || echo 0))"
log "all done"

#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# run_headless_sitl.sh -- headless Gazebo Harmonic + PX4 SITL + uXRCE-DDS + ros_gz_bridge for Aero-VPA
# Phase 1 data collection (preprint_261026.pdf, Secs. 4.3 and 5.1).
#
# Starts, in this order, each in its own process group with its own log:
#   1. Xvfb                (only with --render xvfb)
#   2. MicroXRCEAgent      udp4 -p 8888+INSTANCE
#   3. gz sim server       -s -r --headless-rendering (EGL on the NVIDIA GPU), the chosen world
#   4. PX4 SITL            standalone (PX4_GZ_STANDALONE=1), airframe 4002 (gz_x500_depth), model x500_vpa
#   5. ros_gz_bridge       /clock and the RGB-D topics into ROS 2
# then checks that the camera renders and that PX4 topics reach ROS 2, and either waits (Ctrl-C to stop)
# or runs the command given after "--" and stops everything when it exits (its exit code is returned).
#
# Examples
#   sim/run_headless_sitl.sh --px4-dir ~/PX4-Autopilot --ros-ws ~/px4_ros_ws
#   sim/run_headless_sitl.sh --px4-dir ~/PX4-Autopilot --ros-ws ~/px4_ros_ws -- \
#       python3 sim/collect_expert_dataset.py --num_episodes 2 --output pilot_sample.hdf5
#
# Requirements (Ubuntu 22.04): gz-harmonic, PX4-Autopilot v1.17.0 built with `make px4_sitl` (no need to
# start it), Micro-XRCE-DDS-Agent v2.4.x, ROS 2 Humble with ros-humble-ros-gzharmonic, and px4_msgs
# (branch release/1.17) built in the workspace given by --ros-ws.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ------------------------------------------------------------------------------------------- options
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
ROS_WS="${ROS_WS:-}"
WORLD="${WORLD:-}"
MODEL="${MODEL:-x500_vpa}"
RENDER="${RENDER:-egl}"
INSTANCE="${INSTANCE:-0}"
POSE="${POSE:-}"
RUN_DIR="${RUN_DIR:-}"
START_TIMEOUT="${START_TIMEOUT:-120}"
BRIDGE=1
LOCALHOST_ONLY=1
FRESH_PARAMS=1
CMD=()

usage() {
    sed -n '/^# run_headless_sitl.sh/,/^# Requirements/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options (environment variable in brackets):
  --px4-dir DIR        PX4-Autopilot checkout built with 'make px4_sitl'          [PX4_DIR, default ~/PX4-Autopilot]
  --ros-ws DIR         colcon workspace with px4_msgs (its install/setup.bash is sourced)        [ROS_WS]
  --world PATH         world .sdf; default: generate sim/worlds/aero_forest_sparse_s0000.sdf     [WORLD]
  --model NAME         x500_vpa (aligned 160x128 RGB-D, 50 Hz) or x500_depth (stock OakD-Lite)   [MODEL]
  --render egl|xvfb    egl: GPU headless rendering; xvfb: X virtual framebuffer (software GL)  [RENDER]
  --instance N         run N-th parallel stack: DDS port 8888+N, ROS domain +N, own gz partition [INSTANCE]
  --pose x,y,z,R,P,Y   spawn pose in the world (ENU); default from the world's JSON sidecar         [POSE]
  --run-dir DIR        logs and runtime files; default runs/sitl_<time>_i<N>                     [RUN_DIR]
  --timeout S          start-up timeout per stage in seconds (default 120)                 [START_TIMEOUT]
  --no-bridge          do not start ros_gz_bridge
  --no-localhost-only  allow DDS discovery beyond the loopback interface
  --keep-params        keep the PX4 SITL parameter file (default: start from airframe defaults)
  -- CMD ...           run CMD once the stack is ready, then shut down
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --px4-dir) PX4_DIR="$2"; shift 2 ;;
        --ros-ws) ROS_WS="$2"; shift 2 ;;
        --world) WORLD="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --render) RENDER="$2"; shift 2 ;;
        --instance) INSTANCE="$2"; shift 2 ;;
        --pose) POSE="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --timeout) START_TIMEOUT="$2"; shift 2 ;;
        --no-bridge) BRIDGE=0; shift ;;
        --no-localhost-only) LOCALHOST_ONLY=0; shift ;;
        --keep-params) FRESH_PARAMS=0; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; CMD=("$@"); break ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

log()  { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }
die()  { printf '[%(%H:%M:%S)T] ERROR: %s\n' -1 "$*" >&2; exit 1; }

[[ "$INSTANCE" =~ ^[0-9]+$ ]] || die "--instance must be a non-negative integer"
[[ "$RENDER" == egl || "$RENDER" == xvfb ]] || die "--render must be egl or xvfb"
case "$MODEL" in
    x500_vpa|x500_depth) ;;
    *) die "--model must be x500_vpa or x500_depth" ;;
esac

RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/sitl_$(date +%Y%m%d_%H%M%S)_i${INSTANCE}}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

# ------------------------------------------------------------------------------- process management
PGIDS=()
NAMES=()
CLEANED=0

# Background jobs of a non-interactive shell start with SIGINT ignored; restore it so Ctrl-C style shutdown
# reaches every process (env --default-signal: coreutils >= 8.31, Ubuntu 22.04 has 8.32).
SIGNAL_RESET=()
# shellcheck disable=SC2054  # the comma is part of one argument
if env --default-signal=INT true 2>/dev/null; then SIGNAL_RESET=(env --default-signal=INT,QUIT); fi

start_bg() {  # start_bg NAME LOGFILE CMD...: own session (process group = PID) so the whole tree can be stopped
    local name="$1" logfile="$2"
    shift 2
    setsid "${SIGNAL_RESET[@]}" "$@" >"$logfile" 2>&1 </dev/null &
    local pid=$!
    PGIDS+=("$pid")
    NAMES+=("$name")
    log "started $name (pid $pid, log $logfile)"
}

alive() { kill -0 "$1" 2>/dev/null; }

stop_group() {  # stop_group PGID NAME: SIGINT, then SIGTERM after 8 s, then SIGKILL after 4 s
    local pgid="$1" name="$2" i
    kill -0 -- "-$pgid" 2>/dev/null || return 0
    kill -INT -- "-$pgid" 2>/dev/null || true
    for i in $(seq 1 80); do kill -0 -- "-$pgid" 2>/dev/null || { log "stopped $name"; return 0; }; sleep 0.1; done
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for i in $(seq 1 40); do kill -0 -- "-$pgid" 2>/dev/null || { log "terminated $name"; return 0; }; sleep 0.1; done
    kill -KILL -- "-$pgid" 2>/dev/null || true
    log "killed $name"
}

cleanup() {
    local code=$?
    [[ $CLEANED -eq 1 ]] && return
    CLEANED=1
    trap - EXIT INT TERM
    log "shutting down (exit code $code)"
    local i
    for (( i=${#PGIDS[@]}-1; i>=0; i-- )); do stop_group "${PGIDS[i]}" "${NAMES[i]}"; done
    log "logs: $RUN_DIR"
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_alive() {  # check_alive NAME LOGFILE: fail fast with the log tail if a started process died
    local i
    for i in "${!NAMES[@]}"; do
        if [[ "${NAMES[i]}" == "$1" ]] && ! alive "${PGIDS[i]}"; then
            tail -n 40 "$2" >&2 || true
            die "$1 exited during start-up (log above: $2)"
        fi
    done
}

wait_for() {  # wait_for DESCRIPTION TIMEOUT NAME LOGFILE COMMAND...: poll COMMAND every second
    local what="$1" timeout="$2" name="$3" logfile="$4"
    shift 4
    local start=$SECONDS
    while (( SECONDS - start < timeout )); do              # wall time: a probe may itself take seconds
        if "$@" >/dev/null 2>&1; then log "$what: ok ($(( SECONDS - start )) s)"; return 0; fi
        check_alive "$name" "$logfile"
        sleep 1
    done
    tail -n 40 "$logfile" >&2 || true
    die "timed out after ${timeout} s waiting for: $what (log above: $logfile)"
}

# ------------------------------------------------------------------------------------- environment
# shellcheck source=/dev/null
source_quiet() { set +u; source "$1"; set -u; }

if [[ -z "${ROS_DISTRO:-}" && -f /opt/ros/humble/setup.bash ]]; then
    source_quiet /opt/ros/humble/setup.bash
fi
if [[ -n "$ROS_WS" ]]; then
    [[ -f "$ROS_WS/install/setup.bash" ]] || die "no $ROS_WS/install/setup.bash (build px4_msgs with colcon first)"
    source_quiet "$ROS_WS/install/setup.bash"
fi

PX4_DIR="$(cd "$PX4_DIR" 2>/dev/null && pwd)" || die "PX4 directory not found (use --px4-dir)"
PX4_BUILD="$PX4_DIR/build/px4_sitl_default"
[[ -x "$PX4_BUILD/bin/px4" ]] || die "$PX4_BUILD/bin/px4 missing: run 'make px4_sitl' in $PX4_DIR once"
[[ -f "$PX4_BUILD/rootfs/gz_env.sh" ]] || die "$PX4_BUILD/rootfs/gz_env.sh missing: rebuild PX4 with gz support"

command -v gz >/dev/null || die "gz not found: install gz-harmonic"
GZ_VERSION="$(gz sim --versions 2>/dev/null | head -n 1 | tr -d ' ')"
[[ -n "$GZ_VERSION" ]] || die "'gz sim --versions' failed"
[[ "$(printf '%s\n8.0.0\n' "$GZ_VERSION" | sort -V | head -n 1)" == "8.0.0" ]] || die "Gazebo $GZ_VERSION < Harmonic (8)"

AGENT_BIN="$(command -v MicroXRCEAgent || command -v micro-xrce-dds-agent || true)"
[[ -n "$AGENT_BIN" ]] || die "MicroXRCEAgent not found (build Micro-XRCE-DDS-Agent v2.4.x or install the snap)"

if [[ $BRIDGE -eq 1 || ${#CMD[@]} -gt 0 ]]; then
    command -v ros2 >/dev/null || die "ros2 not found: source ROS 2 Humble or pass --no-bridge"
fi
if [[ $BRIDGE -eq 1 ]]; then
    ros2 pkg prefix ros_gz_bridge >/dev/null 2>&1 || die "ros_gz_bridge not found: apt install ros-humble-ros-gzharmonic"
fi

# Isolation: one gz-transport partition, DDS port and ROS domain per instance; gz discovery on loopback.
export GZ_PARTITION="aero_vpa_${INSTANCE}"
export GZ_IP="127.0.0.1"
export ROS_DOMAIN_ID=$(( ${ROS_DOMAIN_ID:-0} + INSTANCE ))
DDS_PORT=$(( 8888 + INSTANCE ))
if [[ $LOCALHOST_ONLY -eq 1 ]]; then
    export ROS_LOCALHOST_ONLY=1
    export PX4_PARAM_UXRCE_DDS_PTCFG=1        # agent participants on localhost only, to match ROS_LOCALHOST_ONLY
fi
if command -v ss >/dev/null && ss -lun 2>/dev/null | awk '{print $4}' | grep -q ":${DDS_PORT}\$"; then
    die "UDP port $DDS_PORT is in use (another agent?); stop it or use --instance"
fi

# Gazebo resources. PX4's gz_env.sh sets PX4_GZ_MODELS, PX4_GZ_WORLDS, GZ_SIM_RESOURCE_PATH,
# GZ_SIM_SYSTEM_PLUGIN_PATH (PX4's gz plugins) and GZ_SIM_SERVER_CONFIG_PATH (the server.config that loads the
# Sensors system with ogre2, IMU, NavSat, ...). In standalone mode PX4 does not source it, so it is done here,
# and the model directory is replaced by a merged one in which the Aero-VPA models override PX4's.
source_quiet "$PX4_BUILD/rootfs/gz_env.sh"
[[ -d "${PX4_GZ_MODELS:-}" && -d "${PX4_GZ_WORLDS:-}" ]] || die "gz_env.sh did not set PX4_GZ_MODELS / PX4_GZ_WORLDS"
MODELS_DIR="$RUN_DIR/models"
mkdir -p "$MODELS_DIR"
for d in "$PX4_GZ_MODELS"/* "$SCRIPT_DIR"/models/*; do
    [[ -d "$d" ]] && ln -sfn "$d" "$MODELS_DIR/$(basename "$d")"
done
export PX4_GZ_MODELS="$MODELS_DIR"
export GZ_SIM_RESOURCE_PATH="$MODELS_DIR:$PX4_GZ_WORLDS:${GZ_SIM_RESOURCE_PATH:-}"

# World: generate the pilot forest unless one was given; the JSON sidecar carries the privileged map.
if [[ -z "$WORLD" ]]; then
    WORLD="$SCRIPT_DIR/worlds/aero_forest_sparse_s0000.sdf"
    [[ -f "$WORLD" ]] || python3 "$SCRIPT_DIR/make_world.py" --seed 0 --density sparse --out "$SCRIPT_DIR/worlds" >/dev/null
fi
[[ -f "$WORLD" ]] || { [[ -f "$PX4_GZ_WORLDS/$WORLD.sdf" ]] && WORLD="$PX4_GZ_WORLDS/$WORLD.sdf"; }
[[ -f "$WORLD" ]] || die "world not found: $WORLD"
WORLD="$(cd "$(dirname "$WORLD")" && pwd)/$(basename "$WORLD")"
WORLD_NAME="$(sed -n 's/.*<world[[:space:]]\+name="\([^"]*\)".*/\1/p' "$WORLD" | head -n 1)"
[[ -n "$WORLD_NAME" ]] || die "no <world name=...> in $WORLD"
WORLD_JSON="${WORLD%.sdf}.json"
if [[ -f "$WORLD_JSON" ]]; then
    export AERO_VPA_WORLD_JSON="$WORLD_JSON"
    if [[ -z "$POSE" ]]; then
        POSE="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["spawn_enu"]; print(f"{s[0]},{s[1]},{s[2]},0,0,0")' "$WORLD_JSON")"
    fi
else
    log "warning: no privileged map $WORLD_JSON; the expert will plan without obstacles"
    unset AERO_VPA_WORLD_JSON || true
fi
export AERO_VPA_WORLD_NAME="$WORLD_NAME" AERO_VPA_RUN_DIR="$RUN_DIR"

# Rendering: EGL on the NVIDIA GPU (no X server), or an X virtual framebuffer.
GZ_RENDER_ARGS=()
if [[ "$RENDER" == egl ]]; then
    unset DISPLAY || true
    GZ_RENDER_ARGS=(--headless-rendering)
    if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    else
        log "warning: NVIDIA EGL vendor file missing (driver without libnvidia-gl?); EGL may fall back to software"
    fi
    command -v nvidia-smi >/dev/null && log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
else
    command -v Xvfb >/dev/null || die "Xvfb not found: apt install xvfb"
    export DISPLAY=":$(( 99 + INSTANCE ))"
    start_bg xvfb "$RUN_DIR/xvfb.log" Xvfb "$DISPLAY" -screen 0 1280x1024x24 -nolisten tcp -noreset
    sleep 1
    check_alive xvfb "$RUN_DIR/xvfb.log"
fi
export HEADLESS=1

cat >"$RUN_DIR/environment.txt" <<EOF
world=$WORLD
world_name=$WORLD_NAME
world_json=${AERO_VPA_WORLD_JSON:-}
model=$MODEL
render=$RENDER
gz_version=$GZ_VERSION
px4_dir=$PX4_DIR
px4_version=$(git -C "$PX4_DIR" describe --tags --always 2>/dev/null || echo unknown)
gz_partition=$GZ_PARTITION
gz_sim_resource_path=$GZ_SIM_RESOURCE_PATH
gz_sim_system_plugin_path=${GZ_SIM_SYSTEM_PLUGIN_PATH:-}
gz_sim_server_config_path=${GZ_SIM_SERVER_CONFIG_PATH:-}
ros_domain_id=$ROS_DOMAIN_ID
dds_port=$DDS_PORT
EOF
log "world $WORLD_NAME, model $MODEL, render $RENDER, gz $GZ_VERSION, PX4 $(sed -n 's/^px4_version=//p' "$RUN_DIR/environment.txt")"

# ----------------------------------------------------------------------------------- 2. DDS agent
start_bg agent "$RUN_DIR/agent.log" "$AGENT_BIN" udp4 -p "$DDS_PORT"
sleep 1
check_alive agent "$RUN_DIR/agent.log"

# ----------------------------------------------------------------------------------- 3. gz server
start_bg gz "$RUN_DIR/gz.log" gz sim -s -r -v 2 "${GZ_RENDER_ARGS[@]}" "$WORLD"
wait_for "gz world '$WORLD_NAME'" "$START_TIMEOUT" gz "$RUN_DIR/gz.log" \
    bash -c "gz service -i --service /world/$WORLD_NAME/scene/info | grep -q 'Service providers'"

# ----------------------------------------------------------------------------------- 4. PX4 SITL
PX4_ARGS=(-d)
PX4_WD="$PX4_BUILD/rootfs"
if [[ "$INSTANCE" -ne 0 ]]; then PX4_ARGS+=(-i "$INSTANCE"); PX4_WD="$PX4_WD/$INSTANCE"; fi
if [[ $FRESH_PARAMS -eq 1 ]]; then rm -f "${PX4_WD:?}/parameters.bson" "${PX4_WD:?}/parameters_backup.bson"; fi

export PX4_SYS_AUTOSTART=4002                 # gz_x500_depth airframe parameters
export PX4_SIM_MODEL="gz_${MODEL}"            # model spawned from $PX4_GZ_MODELS
export PX4_GZ_STANDALONE=1                    # gz is already running: attach, do not launch it
export PX4_GZ_WORLD="$WORLD_NAME"
export PX4_GZ_MODEL_POSE="$POSE"
export PX4_UXRCE_DDS_PORT="$DDS_PORT"
export PX4_UXRCE_DDS_NS=""                    # topics stay /fmu/... for every instance
export PX4_PARAM_UXRCE_DDS_SYNCT=0            # message timestamps stay in PX4 (= Gazebo simulation) time
export PX4_PARAM_COM_RC_IN_MODE=4             # no RC or joystick on a headless VM
export PX4_PARAM_COM_RCL_EXCEPT=4             # no manual-control-loss failsafe in offboard
start_bg px4 "$RUN_DIR/px4.log" "$PX4_BUILD/bin/px4" "${PX4_ARGS[@]}"
wait_for "PX4 start-up script" "$START_TIMEOUT" px4 "$RUN_DIR/px4.log" \
    grep -q "Startup script returned successfully" "$RUN_DIR/px4.log"
# No GCS on a headless VM: NAV_DLL_ACT must be 0, or PX4 refuses to arm ("No connection to the GCS").
# It cannot go through PX4_PARAM_*: 0 is its compiled default, so it is not stored as changed and the
# airframe's "param set-default NAV_DLL_ACT 2" (4001_gz_x500) replaces it. Commander reads it live.
"$PX4_BUILD/bin/px4-param" --instance "$INSTANCE" set NAV_DLL_ACT 0 >"$RUN_DIR/px4-param.log" 2>&1 \
    || die "could not set NAV_DLL_ACT on the running PX4 (see $RUN_DIR/px4-param.log)"
log "PX4 parameter NAV_DLL_ACT set to 0 (no ground station required)"
wait_for "PX4 ready for takeoff" "$START_TIMEOUT" px4 "$RUN_DIR/px4.log" \
    grep -q "Ready for takeoff" "$RUN_DIR/px4.log"
if grep -q "ERROR \[init\]" "$RUN_DIR/px4.log"; then
    grep "ERROR" "$RUN_DIR/px4.log" >&2
    die "PX4 reported start-up errors"
fi

# Camera topics. x500_vpa: one aligned RGB-D sensor (what the collector needs). x500_depth: the stock
# OakD-Lite, whose RGB (1920x1080) and depth (640x480) sensors differ in FOV; bridged for inspection only.
if [[ "$MODEL" == x500_vpa ]]; then
    RGB_TOPIC="/aero_vpa/rgbd/image"
    DEPTH_TOPIC="/aero_vpa/rgbd/depth_image"
    INFO_TOPIC="/aero_vpa/rgbd/camera_info"
else
    RGB_TOPIC="/world/$WORLD_NAME/model/${MODEL}_${INSTANCE}/link/camera_link/sensor/IMX214/image"
    DEPTH_TOPIC="/depth_camera"
    INFO_TOPIC="/depth_camera/camera_info"
    log "warning: x500_depth has unaligned RGB and depth sensors at 30 Hz; collect_expert_dataset.py needs x500_vpa"
fi

# The camera must actually render (this is where a missing GPU/EGL setup shows up).
wait_for "first depth frame on $DEPTH_TOPIC" "$START_TIMEOUT" gz "$RUN_DIR/gz.log" \
    timeout -k 2 10 gz topic -e -n 1 -t "$DEPTH_TOPIC"

# ---------------------------------------------------------------------------------- 5. ROS 2 bridge
if [[ $BRIDGE -eq 1 ]]; then
    # The world's own clock, remapped to /clock. gz adds the root /clock topic only if it sees no other
    # publisher of /clock at start-up, and a server of the same instance stopped seconds earlier can still
    # look like one; the run then never gets /clock ("no data after 60 s on: /clock").
    start_bg bridge "$RUN_DIR/bridge.log" ros2 run ros_gz_bridge parameter_bridge \
        "/world/$WORLD_NAME/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" \
        "$RGB_TOPIC@sensor_msgs/msg/Image[gz.msgs.Image" \
        "$DEPTH_TOPIC@sensor_msgs/msg/Image[gz.msgs.Image" \
        "$INFO_TOPIC@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
        --ros-args -r "/world/$WORLD_NAME/clock:=/clock"
    wait_for "PX4 topics in ROS 2 (domain $ROS_DOMAIN_ID)" "$START_TIMEOUT" agent "$RUN_DIR/agent.log" \
        bash -c "ros2 topic list --no-daemon | grep -qx /fmu/out/vehicle_odometry"
    # Frames, not just topics: a bridge can advertise both image topics and still deliver nothing
    # (the collector then stops with "no data after 60 s on: synchronised ... image + depth_image").
    # Every probe is bounded by SIGKILL (-k): ros2 topic echo can outlive the SIGTERM of a plain timeout,
    # which once stalled a start-up for good at exactly this wait.
    wait_for "RGB and depth frames in ROS 2" "$START_TIMEOUT" bridge "$RUN_DIR/bridge.log" \
        bash -c "timeout -k 2 10 ros2 topic echo --once '$DEPTH_TOPIC' sensor_msgs/msg/Image >/dev/null &&
                 timeout -k 2 10 ros2 topic echo --once '$RGB_TOPIC' sensor_msgs/msg/Image >/dev/null"
    wait_for "simulation clock in ROS 2" "$START_TIMEOUT" bridge "$RUN_DIR/bridge.log" \
        timeout -k 2 10 ros2 topic echo --once /clock rosgraph_msgs/msg/Clock
fi
export AERO_VPA_RGB_TOPIC="$RGB_TOPIC" AERO_VPA_DEPTH_TOPIC="$DEPTH_TOPIC"

log "stack ready: world $WORLD_NAME, ROS_DOMAIN_ID=$ROS_DOMAIN_ID, GZ_PARTITION=$GZ_PARTITION, logs in $RUN_DIR"
[[ -n "${AERO_VPA_WORLD_JSON:-}" ]] && log "privileged map: $AERO_VPA_WORLD_JSON"

if [[ ${#CMD[@]} -gt 0 ]]; then
    log "running: ${CMD[*]}"
    set +e
    "${CMD[@]}"
    status=$?
    set -e
    log "command exited with status $status"
    exit "$status"
fi

log "press Ctrl-C to stop"
while true; do
    for i in "${!NAMES[@]}"; do
        if ! alive "${PGIDS[i]}"; then die "${NAMES[i]} exited unexpectedly (see $RUN_DIR/${NAMES[i]}.log)"; fi
    done
    sleep 2
done

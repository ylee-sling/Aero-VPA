# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/stress_dagger.py -- the DAgger supervisor (sim/dagger.py) against scripted bad policies, without ROS or PX4.

Each mission is sampled and planned as fly_policy.py does (seed, k; speed limit drawn from 3/5/7 m/s), and a
scripted policy flies it under the supervisor at 100 Hz with decisions at ~33 Hz:

    hold    the copycat of the BC policies: keeps the last dispatched setpoint (after a hand-back, the expert's),
            its direction drifting by 25 deg/sqrt(s)
    noisy   the expert's label plus Ornstein-Uhlenbeck noise (1 m/s, 1 s)
    expert  the expert's label itself: every takeover and expert brake is a false alarm

The vehicle is PX4-like: velocity setpoints with acceleration feed-forward (--ff share), a first-order velocity
loop (--tau_xy ~ 1 / MPC_XY_VEL_P_ACC, --tau_z), a dead time (--delay) and tilt-limited acceleration, as fitted on
PX4 SITL (show_trace.py --response on the round-1 replay: delay 0.24 s, tau 0.50 s, feed-forward followed in
full). On top, a horizontal disturbance acceleration (Ornstein-Uhlenbeck, --disturb_tau) stands for what that
model leaves out: its standard deviation is --disturb from 1 m/s on (less below, none in a hover) plus
--disturb_speed per m/s, calibrated so that the expert's predictions err as on the VM (show_trace.py --prediction:
predicted minus flown smallest clearance, 95th / 99th percentile 0.14 / 0.27 m at 1-3 m/s, 0.22 / 0.49 m above);
without it the prediction is exact and thin margins never show. A collision is a privileged distance below
0.35 m (fly_policy.py --collision_distance). Per mission: ending, who flew at a crash, re-plans, expert brakes,
takeovers, the closest approach under the expert; then the totals and how optimistic the expert's predictions were.

    python3 sim/make_world.py --seed 0 --count 4 --density dense medium sparse --out /tmp/worlds
    python3 sim/stress_dagger.py --worlds /tmp/worlds/aero_forest_dense_s000[0-3].json --missions 3 --policy hold
    python3 sim/stress_dagger.py --worlds ... --policy expert                  # false alarms
    python3 sim/stress_dagger.py --worlds ... --debug                          # the last 3 s before each crash
    python3 sim/stress_dagger.py --worlds ... --disturb 0 --disturb_speed 0    # the noise-free vehicle
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dagger as D  # noqa: E402
import expert as E  # noqa: E402

COLLISION_DISTANCE = 0.35
DISTURB = (0.5, 0.03, 0.5)                # disturbance (std from 1 m/s m/s^2, std per m/s 1/s, correlation time s)


class Vehicle:
    """PX4-like response to (velocity, acceleration feed-forward, yaw rate) setpoints."""

    def __init__(self, p: np.ndarray, yaw: float, tau_xy: float, tau_z: float, delay: float, ff: float,
                 a_max: float = 7.0, disturb: float = 0.0, disturb_speed: float = 0.0, disturb_tau: float = 0.5,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.p, self.v = np.asarray(p, float).copy(), np.zeros(3)
        self.yaw, self.yaw_rate = float(yaw), 0.0
        self.tau_xy, self.tau_z, self.delay, self.ff, self.a_max = tau_xy, tau_z, delay, ff, a_max
        self.disturb, self.disturb_speed, self.disturb_tau = disturb, disturb_speed, max(disturb_tau, 1e-3)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.a_d = np.zeros(2)                  # horizontal disturbance acceleration (Ornstein-Uhlenbeck)
        self.queue: Deque[Tuple[float, np.ndarray, np.ndarray, float]] = deque()
        self.sp: Tuple[np.ndarray, np.ndarray, float] = (np.zeros(3), np.zeros(3), 0.0)

    def step(self, t: float, dt: float, vel: np.ndarray, acc: np.ndarray, yaw_rate: float) -> None:
        self.queue.append((t + self.delay, np.asarray(vel, float), np.asarray(acc, float), float(yaw_rate)))
        while self.queue and self.queue[0][0] <= t + 1e-9:
            _, v_sp, a_ff, yr = self.queue.popleft()
            self.sp = (v_sp, a_ff, yr)
        v_sp, a_ff, yr_sp = self.sp
        a = self.ff * a_ff
        a[:2] += (v_sp[:2] - self.v[:2]) / self.tau_xy
        a[2] += (v_sp[2] - self.v[2]) / self.tau_z
        h = float(np.linalg.norm(a[:2]))
        if h > self.a_max:
            a[:2] *= self.a_max / h
        if self.disturb > 0.0 or self.disturb_speed > 0.0:
            speed = float(np.linalg.norm(self.v[:2]))
            sigma = self.disturb * min(1.0, speed) + self.disturb_speed * speed     # none in a hover
            self.a_d += (-self.a_d * dt / self.disturb_tau
                         + sigma * math.sqrt(2.0 * dt / self.disturb_tau) * self.rng.normal(size=2))
            a[:2] += self.a_d
        self.v = self.v + dt * a
        self.p = self.p + dt * self.v
        self.yaw_rate += dt * (yr_sp - self.yaw_rate) / 0.1
        self.yaw = float(E.wrap_angle(self.yaw + dt * self.yaw_rate))


def plan_mission(world: E.PrivilegedMap, planner: E.ExpertPlanner, seed: int, k: int, home: np.ndarray,
                 speeds: List[float]) -> Optional[Tuple[np.ndarray, float, E.ExpertTrajectory]]:
    """Mission k as fly_policy.py samples it (replacement draws when the expert cannot plan it)."""
    speed = float(np.random.default_rng([seed, k, 7]).choice(np.asarray(speeds, dtype=float)))
    for attempt in range(10):
        rng = np.random.default_rng([seed, k] + ([attempt] if attempt else []))
        waypoints = E.sample_mission(world, rng, home, (3, 6), (10.0, 40.0), (1.5, 4.0))
        try:
            return waypoints, speed, planner.plan(home, waypoints, speed, 0.0)
        except E.PlanningError:
            continue
    return None


def fly(world: E.PrivilegedMap, planner: E.ExpertPlanner, args: argparse.Namespace, k: int, gate: D.GateConfig,
        stats: Dict[str, Any]) -> Dict[str, Any]:
    home = np.array([0.0, 0.0, -2.5])
    mission = plan_mission(world, planner, args.seed, k, home, args.speeds)
    if mission is None:
        return {"end": "no_plan"}
    waypoints, speed, traj = mission
    yaw0 = traj.initial_heading() or 0.0
    sup = D.DaggerSupervisor(world, planner, waypoints, speed, E.ExpertTrajectory(traj.spline, yaw0), 0.0,
                             lambda s, w, v: D.plan_now(planner, s, w, v), None, {}, gate)
    veh = Vehicle(home, yaw0, args.tau_xy, args.tau_z, args.delay, args.ff, disturb=args.disturb,
                  disturb_speed=args.disturb_speed, disturb_tau=args.disturb_tau,
                  rng=np.random.default_rng([args.seed, k, 13]))
    checks: List[Tuple[float, float, float, float]] = []      # the expert's checks: (t, predicted, threshold, speed)
    flown: List[Tuple[float, float, bool]] = []               # every tick: (t, clearance, expert flying)
    last_check: Optional[float] = None
    tracker = E.WaypointTracker(waypoints)
    rng = np.random.default_rng([args.seed, k, 11])
    t, dt, deadline = 0.0, 0.01, 3.0 * traj.duration + 30.0
    pol_cmd, pol_yr, last_sp, ou = None, 0.0, np.zeros(3), np.zeros(3)
    flown_by = "POLICY"
    end, end_mode, closest_expert = "timeout", None, math.inf
    history: Deque[Tuple[float, str, float, float, float, float, str]] = deque(maxlen=300)
    events: List[str] = []
    seen_takeovers: Counter = Counter()
    tick = 0
    while t < deadline:
        if tick % 3 == 0:                                            # decisions at ~33 Hz
            tracker.step(veh.p, veh.yaw)
            sup.m = int(tracker.m)
            if tracker.task_complete:
                end = "success"
                break
            if args.policy == "hold":
                if pol_cmd is None or flown_by != "POLICY":
                    pol_cmd = last_sp.copy()
                    if float(np.linalg.norm(pol_cmd[:2])) < 0.5:   # from rest: towards the active waypoint
                        d = waypoints[tracker.m - 1] - veh.p
                        pol_cmd = 2.0 * d / max(float(np.linalg.norm(d)), 1e-6)
                ang = rng.normal(0.0, math.radians(25.0)) * math.sqrt(0.03)
                c, s = math.cos(ang), math.sin(ang)
                pol_cmd = np.array([c * pol_cmd[0] - s * pol_cmd[1], s * pol_cmd[0] + c * pol_cmd[1], pol_cmd[2]])
            else:
                ref = sup.reference
                lab = ref.label(t, ref.nearest(veh.p), veh.p, veh.yaw)[0]
                if args.policy == "noisy":
                    ou += -ou * 0.03 + math.sqrt(2 * 0.03) * rng.normal(size=3) * np.array([1.0, 1.0, 0.3])
                pol_cmd = E.heading_to_world(lab[:3].astype(float), veh.yaw) + ou
                pol_yr = float(lab[3])
        t0 = time.perf_counter()
        cmd = sup.control(t, veh.p, veh.v, veh.yaw)
        stats["tick_s"] += time.perf_counter() - t0
        stats["ticks"] += 1
        flown_by = sup.mode if cmd is not None else "POLICY"
        if args.debug and sup.takeovers != seen_takeovers:
            new = [k for k in sup.takeovers if sup.takeovers[k] != seen_takeovers.get(k, 0)]
            events.append(f"{t:6.2f} {','.join(new)} -> {sup.mode} (speed {float(np.linalg.norm(veh.v)):.2f}, "
                          f"off the path {sup.reference.offset(veh.p):.2f} m, m={tracker.m})")
            seen_takeovers = Counter(sup.takeovers)
        if sup.failed:
            end = sup.failed
            break
        if cmd is None:
            vel = pol_cmd.copy()
            h = float(np.linalg.norm(vel[:2]))
            if h > speed:
                vel[:2] *= speed / h
            yr = pol_yr if args.policy == "expert" else (
                float(np.clip(E.wrap_angle(math.atan2(vel[1], vel[0]) - veh.yaw), -1.0, 1.0)) if h > 0.3 else 0.0)
            cmd = (vel, np.zeros(3), yr)
            sup.dispatched(t, vel, np.zeros(3))
        last_sp = np.asarray(cmd[0], float)
        veh.step(t, dt, *cmd)
        clearance = float(world.distance(veh.p))
        if sup.mode != "POLICY":
            closest_expert = min(closest_expert, clearance)
        choice = sup.last_choice
        if sup.mode == "EXPERT" and choice is not None and choice["t"] is not None and choice["t"] != last_check:
            last_check = choice["t"]
            if choice.get(choice["way"]) is not None:
                checks.append((float(choice["t"]), float(choice[choice["way"]]), float(choice["threshold"]),
                               float(np.linalg.norm(veh.v))))
        flown.append((t, clearance, sup.mode == "EXPERT"))
        if args.debug:
            way = ""
            if sup.mode == "EXPERT" and choice is not None and choice["t"] is not None and t - choice["t"] < 0.06:
                way = f"{choice['way']} " + " ".join(f"{k} {choice[k]:.2f}" for k in ("fly", "slow", "stop")
                                                    if choice.get(k) is not None)
            history.append((t, sup.mode, float(np.linalg.norm(veh.v)), clearance,
                            sup.predicted_clearance(veh.p, veh.v), float(np.linalg.norm(cmd[0])), way))
        if clearance < COLLISION_DISTANCE or veh.p[2] > -0.3:
            end, end_mode = ("collision" if clearance < COLLISION_DISTANCE else "ground"), sup.mode
            if args.debug:
                for i, (ht, mode, spd, clr, pred, c, way) in enumerate(history):
                    if i % 10 == 0 or i >= len(history) - 5:
                        print(f"     t={ht:6.2f} {mode:7s} speed {spd:4.2f} clearance {clr:5.2f} "
                              f"ahead {pred:5.2f} |cmd| {c:4.2f}  {way}")
            break
        t = round(t + dt, 2)
        tick += 1
    if args.debug and events:
        print("     takeovers: " + "; ".join(events[-12:]))
    summary = sup.finish(end)
    return {"end": end, "mode": end_mode, "t": t, "deadline": deadline, "speed": speed, "closest": closest_expert,
            "optimism": optimism(checks, flown, end in ("collision", "ground")), **summary}


def optimism(checks: List[Tuple[float, float, float, float]], flown: List[Tuple[float, float, bool]],
             crashed: bool) -> List[Tuple[float, float, float, float]]:
    """(speed, predicted - flown smallest clearance, predicted, threshold) of every check of the expert whose horizon
    (as show_trace.py --prediction: reaction 0.3 s + minimum-jerk braking at the flown speed, at least 0.5 s) it flew
    in full; a crash ends the record."""
    if not flown:
        return []
    ft = np.array([f[0] for f in flown])
    fc = np.array([f[1] for f in flown])
    fe = np.array([f[2] for f in flown])
    out = []
    for t0, pred, thr, speed in checks:
        horizon = max(0.5, 0.3 + 1.5 * speed / 5.0)
        sel = (ft > t0) & (ft <= t0 + horizon)
        if not sel.any() or (ft[-1] < t0 + horizon and not crashed) or not fe[sel].all():
            continue
        out.append((speed, pred - float(fc[sel].min()), pred, thr))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", nargs="+", required=True, help="world JSON sidecars of sim/make_world.py")
    p.add_argument("--missions", type=int, default=3, help="missions per world")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speeds", type=float, nargs="+", default=[3.0, 5.0, 7.0])
    p.add_argument("--policy", default="hold", choices=("hold", "noisy", "expert"))
    p.add_argument("--tau_xy", type=float, default=0.5, help="horizontal velocity time constant (s)")
    p.add_argument("--tau_z", type=float, default=0.25, help="vertical velocity time constant (s)")
    p.add_argument("--delay", type=float, default=0.24, help="setpoint dead time (s)")
    p.add_argument("--ff", type=float, default=1.0, help="share of the acceleration feed-forward the vehicle uses")
    p.add_argument("--disturb", type=float, default=DISTURB[0], help="disturbance acceleration std from 1 m/s on "
                   "(m/s^2; less below, none in a hover)")
    p.add_argument("--disturb_speed", type=float, default=DISTURB[1], help="... plus this per m/s of speed (1/s)")
    p.add_argument("--disturb_tau", type=float, default=DISTURB[2], help="its correlation time (s)")
    p.add_argument("--gate", default="", help="GateConfig overrides, e.g. expert_clearance=0.5,delay_s=0.2")
    p.add_argument("--debug", action="store_true", help="print the last 3 s before each crash")
    args = p.parse_args(argv)
    gate = D.GateConfig(**{k: float(v) for k, v in (kv.split("=") for kv in args.gate.split(",") if kv)})
    stats: Dict[str, Any] = {"tick_s": 0.0, "ticks": 0}
    ends: Counter = Counter()
    totals: Counter = Counter()
    closest: List[float] = []
    opt: List[Tuple[float, float, float, float]] = []
    n = 0
    for path in args.worlds:
        world = E.PrivilegedMap.from_world_json(path)
        planner = E.ExpertPlanner(world)
        name = os.path.splitext(os.path.basename(path))[0].replace("aero_forest_", "")
        for k in range(args.missions):
            r = fly(world, planner, args, k, gate, stats)
            ends[r["end"] + (f"@{r['mode']}" if r.get("mode") else "")] += 1
            if r["end"] == "no_plan":
                continue
            n += 1
            for key in ("dagger_replans", "dagger_slowdowns", "dagger_expert_brakes", "dagger_soft_blocked",
                        "dagger_backoffs"):
                totals[key] += r[key]
            for key, v in r["dagger_takeovers"].items():
                totals["takeover_" + key] += v
            if math.isfinite(r["closest"]):
                closest.append(r["closest"])
            opt.extend(r["optimism"])
            takeovers = ", ".join(f"{key} {v}" for key, v in r["dagger_takeovers"].items() if v)
            print(f"{name:12s} k={k} {r['speed']:.0f} m/s  {r['end']:9s} {r['mode'] or '':7s} {r['t']:6.1f}/{r['deadline']:5.1f} s"
                  f"  re-plans {r['dagger_replans']}, slow-downs {r['dagger_slowdowns']}, expert stops "
                  f"{r['dagger_expert_brakes']}, blocked "
                  f"{r['dagger_soft_blocked']}; closest under the expert {r['closest']:.2f} m; policy share "
                  f"{r['dagger_policy_share'] or 0.0:.2f}; takeovers {takeovers or 'none'}", flush=True)
    print(f"endings: {dict(ends)}")
    print("per mission: " + ", ".join(f"{key} {v / max(n, 1):.2f}" for key, v in sorted(totals.items())))
    if closest:
        print(f"closest approach under the expert: min {min(closest):.2f} m, p10 {float(np.percentile(closest, 10)):.2f} m")
    if opt:
        sp, err = np.array([o[0] for o in opt]), np.array([o[1] for o in opt])
        kept = np.array([o[2] >= o[3] for o in opt])
        print(f"expert's predicted - flown smallest clearance (m), {len(opt)} checks: p50 / p90 / p95 / p99")
        for label, lo, hi in (("all", 0.0, math.inf), ("below 1 m/s", 0.0, 1.0), ("1-3 m/s", 1.0, 3.0),
                              ("above 3 m/s", 3.0, math.inf)):
            sel = (sp >= lo) & (sp < hi)
            if sel.any():
                q = np.percentile(err[sel], [50, 90, 95, 99])
                print(f"   {label:12s} {int(sel.sum()):6d}   {q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f} / {q[3]:+.2f}")
        print(f"   predicted to keep the threshold: {int(kept.sum())}, of which flown below it: "
              f"{int((kept & (err > np.array([o[2] - o[3] for o in opt]))).sum())}")
    print(f"supervisor tick {1e3 * stats['tick_s'] / max(stats['ticks'], 1):.2f} ms on average (re-plans included)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

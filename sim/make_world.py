# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/make_world.py -- Procedural E1 "dense forest" worlds for Gazebo Harmonic (preprint_261026.pdf, Sec. 5.1).

Writes ``<out>/<name>.sdf`` (the Gazebo world) and ``<out>/<name>.json`` (the privileged occupancy that the
expert plans on). Trees are placed by Poisson-disc sampling at one of three densities; trunk diameters,
heights and canopy radii come from fixed distributions, and the ground and bark colours are randomised.
Everything is a primitive shape (cylinder trunks, sphere canopies), so the JSON sidecar is the exact
collision geometry and nothing is downloaded from Fuel. The world header (physics step 4 ms, real-time
factor 1, sun, spherical coordinates of the PX4 home) follows PX4's ``default.sdf``.

    python sim/make_world.py --seed 0 --density sparse --out sim/worlds
    python sim/make_world.py --seed 0 --count 200 --density sparse medium dense --out sim/worlds

Frames: the SDF and the JSON use the Gazebo ENU world frame (x east, y north, z up). The collector converts
to the PX4 local NED frame with ``expert.PrivilegedMap.from_world_json`` (origin at ``spawn_enu``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

WORLD_SCHEMA = "aero-vpa/world"
WORLD_SCHEMA_VERSION = "1.0.0"
DENSITY_SPACING = {"sparse": 7.0, "medium": 5.0, "dense": 3.5}   # Poisson-disc minimum trunk distance (m)


def poisson_disc(rng: np.random.Generator, half_size: float, radius: float, k: int = 30) -> np.ndarray:
    """Bridson's Poisson-disc sampling in the square [-half_size, half_size]^2. Returns ``[N, 2]``."""
    cell = radius / math.sqrt(2.0)
    n = int(math.ceil(2.0 * half_size / cell))
    grid = -np.ones((n, n), dtype=np.int64)
    points: List[np.ndarray] = []

    def cell_of(p: np.ndarray) -> Tuple[int, int]:
        return (min(n - 1, int((p[0] + half_size) / cell)), min(n - 1, int((p[1] + half_size) / cell)))

    def fits(p: np.ndarray) -> bool:
        if np.any(np.abs(p) > half_size):
            return False
        ci, cj = cell_of(p)
        for i in range(max(0, ci - 2), min(n, ci + 3)):
            for j in range(max(0, cj - 2), min(n, cj + 3)):
                idx = grid[i, j]
                if idx >= 0 and np.linalg.norm(points[idx] - p) < radius:
                    return False
        return True

    first = rng.uniform(-half_size, half_size, 2)
    points.append(first)
    grid[cell_of(first)] = 0
    active = [0]
    while active:
        a = active[int(rng.integers(len(active)))]
        for _ in range(k):
            ang, dist = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(radius, 2.0 * radius)
            cand = points[a] + dist * np.array([math.cos(ang), math.sin(ang)])
            if fits(cand):
                points.append(cand)
                grid[cell_of(cand)] = len(points) - 1
                active.append(len(points) - 1)
                break
        else:
            active.remove(a)
    return np.asarray(points)


def _rgba(c: Sequence[float]) -> str:
    return " ".join(f"{v:.3f}" for v in (*c, 1.0))


def _material(c: Sequence[float]) -> str:
    return (f"<material><ambient>{_rgba(c)}</ambient><diffuse>{_rgba(c)}</diffuse>"
            f"<specular>0.05 0.05 0.05 1</specular></material>")


def _shape(kind: str, name: str, pose: Sequence[float], geometry: str, color: Sequence[float]) -> str:
    p = " ".join(f"{v:.4f}" for v in pose)
    return (f"        <collision name=\"{name}_collision\"><pose>{p} 0 0 0</pose><geometry>{geometry}</geometry></collision>\n"
            f"        <visual name=\"{name}_visual\"><pose>{p} 0 0 0</pose><geometry>{geometry}</geometry>{_material(color)}</visual>\n")


def make_forest(seed: int, density: str, half_size: float = 50.0, clear_radius: float = 4.0
                ) -> Tuple[str, Dict[str, object]]:
    """(SDF text, sidecar dict) of one E1 forest world."""
    if density not in DENSITY_SPACING:
        raise ValueError(f"density must be one of {sorted(DENSITY_SPACING)}")
    rng = np.random.default_rng([seed, list(DENSITY_SPACING).index(density)])
    name = f"aero_forest_{density}_s{seed:04d}"
    ground = np.array([0.30, 0.38, 0.22]) * rng.uniform(0.75, 1.25, 3)
    sky = np.array([0.62, 0.74, 0.88]) * rng.uniform(0.9, 1.1)
    sun_dir = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0])
    sun_dir /= np.linalg.norm(sun_dir)
    models, obstacles = [], []
    for i, (x, y) in enumerate(poisson_disc(rng, half_size, DENSITY_SPACING[density])):
        if math.hypot(x, y) < clear_radius:
            continue
        radius = float(rng.uniform(0.10, 0.30))               # trunk diameter 0.2-0.6 m
        height = float(rng.uniform(6.0, 14.0))
        canopy = float(rng.uniform(1.2, 2.8))
        bark = np.array([0.33, 0.24, 0.16]) * rng.uniform(0.7, 1.3)
        leaf = np.array([0.18, 0.36, 0.14]) * rng.uniform(0.7, 1.3, 3)
        tree = f"tree_{i:04d}"
        body = _shape("cylinder", "trunk", (0.0, 0.0, height / 2),
                      f"<cylinder><radius>{radius:.4f}</radius><length>{height:.4f}</length></cylinder>", bark)
        body += _shape("sphere", "canopy", (0.0, 0.0, height),
                       f"<sphere><radius>{canopy:.4f}</radius></sphere>", leaf)
        models.append(f"    <model name=\"{tree}\">\n      <static>true</static>\n      <pose>{x:.4f} {y:.4f} 0 0 0 0</pose>\n"
                      f"      <link name=\"link\">\n{body}      </link>\n    </model>\n")
        obstacles.append({"type": "cylinder", "name": f"{tree}/trunk", "center": [float(x), float(y), height / 2],
                          "radius": radius, "length": height})
        obstacles.append({"type": "sphere", "name": f"{tree}/canopy", "center": [float(x), float(y), height],
                          "radius": canopy})
    sdf = f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by sim/make_world.py (Aero-VPA): seed {seed}, density {density}. Do not edit; regenerate. -->
<sdf version="1.9">
  <world name="{name}">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>{_rgba(sky)}</background>
      <shadows>true</shadows>
    </scene>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>{4 * half_size:.0f} {4 * half_size:.0f}</size></plane></geometry>{_material(ground)}</visual>
      </link>
    </model>
    <light name="sunUTC" type="directional">
      <pose>0 0 500 0 0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>{sun_dir[0]:.3f} {sun_dir[1]:.3f} {sun_dir[2]:.3f}</direction>
      <diffuse>0.904 0.904 0.904 1</diffuse>
      <specular>0.271 0.271 0.271 1</specular>
      <attenuation><range>2000</range><linear>0</linear><constant>1</constant><quadratic>0</quadratic></attenuation>
    </light>
{''.join(models)}    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
  </world>
</sdf>
"""
    sidecar = {
        "schema": WORLD_SCHEMA, "schema_version": WORLD_SCHEMA_VERSION, "world_name": name,
        "environment_class": "E1_forest", "density": density, "seed": seed,
        "frame": "gazebo ENU (x east, y north, z up), metres",
        "spawn_enu": [0.0, 0.0, 0.0], "bounds_enu": [[-half_size, -half_size], [half_size, half_size]],
        "num_trees": len(models), "obstacles": obstacles,
    }
    return sdf, sidecar


def write_world(out_dir: str, seed: int, density: str, half_size: float = 50.0) -> Tuple[str, str]:
    sdf, sidecar = make_forest(seed, density, half_size)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, sidecar["world_name"])
    sidecar["sdf"] = os.path.basename(base + ".sdf")
    with open(base + ".sdf", "w", encoding="utf-8") as f:
        f.write(sdf)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=1)
    return base + ".sdf", base + ".json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedural E1 forest worlds (SDF + privileged JSON).")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--count", type=int, default=1, help="number of consecutive seeds per density")
    parser.add_argument("--density", nargs="+", default=["sparse"], choices=sorted(DENSITY_SPACING))
    parser.add_argument("--half-size", type=float, default=50.0, help="half width of the square forest (m)")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds"))
    args = parser.parse_args()
    for density in args.density:
        for seed in range(args.seed, args.seed + args.count):
            sdf_path, json_path = write_world(args.out, seed, density, args.half_size)
            print(sdf_path)
            print(json_path)


if __name__ == "__main__":
    main()

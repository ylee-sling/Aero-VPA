# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/render_fpv.py -- FPV videos of traced missions: the onboard RGB-D frames the policy decided on
(fly_policy.py --frames), with the depth image, a top-down map of the trunks around the vehicle, and telemetry
from the trace (fly_policy.py --trace), encoded to MP4 (H.264) with a poster frame.

    # missions every listed run succeeded on in the benchmark, best first, per density (tab-separated)
    python3 sim/render_fpv.py pick --run r1_shield='flights_shield_r1_i*.jsonl' --run kw_shield='flights_shield_kw_i*.jsonl'
    # one video from a re-flown mission
    python3 sim/render_fpv.py render --trace logs/fpv/trace/X.jsonl --frames logs/fpv/frames/X.frames \\
        --title "r1 + onboard shield" --out videos/r1_shield_dense.mp4
    # videos/videos.json: what every video shows, and every re-flight of its density
    python3 sim/render_fpv.py index --videos videos --results logs/fpv/flights_video.jsonl
    python3 sim/render_fpv.py --self-test

Layout (1280 x 720): the 160 x 128 RGB frame scaled 4x with nearest neighbours (as the policy sees it), the depth
frame (turbo colours, 0-10 m), a north-up map of the privileged world 16 px/m around the vehicle with its path,
waypoints and the camera's 73 deg field of view, and the telemetry of the decision shown. Real time by default.
Text needs Pillow (left out without it); encoding needs ffmpeg on the PATH, or the imageio-ffmpeg package.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import expert as E  # noqa: E402
from crash_study import resolve_world  # noqa: E402

PRIMITIVE_NAMES = ("CRUISE", "LATERAL_AVOID", "VERTICAL_AVOID", "YAW_SCAN", "APPROACH", "BRAKE_HOVER")
W, H = 1280, 720
HALF_FOV = 1.274 / 2.0
RGB_AT, RGB_SCALE = (24, 72), 4                  # 640 x 512
DEPTH_AT, DEPTH_SCALE = (688, 72), 3             # 480 x 384
MAP_AT, MAP_SIZE, MAP_PX_M = (688, 472), (568, 232), 16.0
BAR_AT = (1184, 72)
BG = np.array([16, 18, 22], dtype=np.uint8)
PANEL = np.array([28, 32, 38], dtype=np.uint8)
DEPTH_MAX_M = 10.0


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
def frame_dtype(h: int, w: int) -> np.dtype:
    return np.dtype([("t", "<f8"), ("rgb", "u1", (h, w, 3)), ("depth", "<u2", (h, w))])


def read_frames(path: str) -> Tuple[Dict[str, Any], np.ndarray]:
    with open(path + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    dt = frame_dtype(int(meta["height"]), int(meta["width"]))
    if dt.itemsize != int(meta["record_bytes"]):
        raise ValueError("%s: record size %d, layout says %d" % (path, dt.itemsize, meta["record_bytes"]))
    n = os.path.getsize(path) // dt.itemsize
    return meta, np.memmap(path, dtype=dt, mode="r", shape=(n,))


def read_trace(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    return lines[0], [x for x in lines[1:] if x.get("kind") == "decision"]


def world_arrays(record: Dict[str, Any], roots: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """NED centres (north, east) and radii of the cylinders (trunks) of the mission's world; empty without one."""
    path = resolve_world(str(record.get("world_json") or ""), roots)
    if not path:
        return np.zeros((0, 2)), np.zeros(0)
    world = E.PrivilegedMap.from_world_json(path)
    if not len(world):
        return np.zeros((0, 2)), np.zeros(0)
    sel = world._kind == "cylinder"
    return np.asarray(world._c[sel, :2], dtype=np.float64), np.asarray(world._r[sel], dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Drawing (NumPy; text with Pillow when it is installed)
# ---------------------------------------------------------------------------------------------
def turbo(x: np.ndarray) -> np.ndarray:
    """Google's polynomial approximation of the turbo colour map, x in [0, 1] -> uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)


def depth_image(depth_mm: np.ndarray) -> np.ndarray:
    d = depth_mm.astype(np.float32) * 1e-3
    img = turbo(1.0 - d / DEPTH_MAX_M)                       # near: red, far: blue
    img[depth_mm == 0] = 0                                   # no return
    return img


def blit(canvas: np.ndarray, img: np.ndarray, at: Tuple[int, int], scale: int) -> None:
    big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    x, y = at
    canvas[y:y + big.shape[0], x:x + big.shape[1]] = big


def disc(canvas: np.ndarray, cx: float, cy: float, r: float, color: Sequence[int], alpha: float = 1.0,
         clip: Optional[Tuple[int, int, int, int]] = None) -> None:
    x0, y0, x1, y1 = clip or (0, 0, canvas.shape[1], canvas.shape[0])
    xa, xb = max(x0, int(math.floor(cx - r))), min(x1, int(math.ceil(cx + r)) + 1)
    ya, yb = max(y0, int(math.floor(cy - r))), min(y1, int(math.ceil(cy + r)) + 1)
    if xa >= xb or ya >= yb:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    patch = canvas[ya:yb, xa:xb]
    patch[m] = (alpha * np.asarray(color, np.float32) + (1 - alpha) * patch[m]).astype(np.uint8)


def polyline(canvas: np.ndarray, pts: np.ndarray, color: Sequence[int], clip: Tuple[int, int, int, int],
             width: int = 2) -> None:
    if len(pts) < 2:
        return
    seg = np.diff(pts, axis=0)
    n = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) * 2).astype(int))
    samples = np.concatenate([pts[i] + seg[i] * np.linspace(0, 1, n[i], endpoint=False)[:, None]
                              for i in range(len(seg))] + [pts[-1:]])
    xy = np.rint(samples).astype(int)
    x0, y0, x1, y1 = clip
    for dx in range(width):
        for dy in range(width):
            xs, ys = xy[:, 0] + dx, xy[:, 1] + dy
            ok = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            canvas[ys[ok], xs[ok]] = color


class Text:
    """Text on the frame with Pillow, or nothing when Pillow is missing (the video still renders)."""

    def __init__(self) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            self.ok = False
            return
        self.ok, self.Image, self.ImageDraw = True, Image, ImageDraw
        self.fonts: Dict[Tuple[int, bool], Any] = {}
        for size in (14, 16, 20, 26, 34):
            for bold in (False, True):
                self.fonts[(size, bold)] = self._font(ImageFont, size, bold)

    @staticmethod
    def _font(ImageFont: Any, size: int, bold: bool) -> Any:
        names = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold else ("DejaVuSans.ttf",)
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF"):
            for name in names:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
        try:
            return ImageFont.load_default(size=size)
        except TypeError:                                     # Pillow < 10.1
            return ImageFont.load_default()

    def draw(self, canvas: np.ndarray, items: Sequence[Tuple[Tuple[int, int], str, int, bool, Sequence[int]]]) -> np.ndarray:
        if not self.ok or not items:
            return canvas
        img = self.Image.fromarray(canvas)
        d = self.ImageDraw.Draw(img)
        for (x, y), s, size, bold, color in items:
            d.text((x, y), s, font=self.fonts[(size, bold)], fill=tuple(int(c) for c in color))
        return np.asarray(img).copy()


class Renderer:
    def __init__(self, record: Dict[str, Any], rows: List[Dict[str, Any]], trunks: Tuple[np.ndarray, np.ndarray],
                 title: str, subtitle: str) -> None:
        self.record, self.rows, self.title, self.subtitle = record, rows, title, subtitle
        self.centres, self.radii = trunks
        self.waypoints = np.asarray(record.get("waypoints_ned") or [], dtype=np.float64).reshape(-1, 3)
        self.path = np.array([r["p"][:2] for r in rows], dtype=np.float64).reshape(-1, 2)
        self.text = Text()
        self.background = self._background()

    def _background(self) -> np.ndarray:
        c = np.empty((H, W, 3), dtype=np.uint8)
        c[:] = BG
        c[:56] = (10, 11, 14)
        mx, my = MAP_AT
        c[my:my + MAP_SIZE[1], mx:mx + MAP_SIZE[0]] = (34, 40, 36)
        c[600:704, 24:664] = PANEL
        x, y = BAR_AT                                       # colour bar of the depth panel: near at the top
        bar = turbo(np.linspace(1.0, 0.0, 384))[:, None, :].repeat(14, axis=1)
        c[y:y + 384, x:x + 14] = bar
        return c

    def _map(self, c: np.ndarray, i: int) -> None:
        mx, my = MAP_AT
        mw, mh = MAP_SIZE
        clip = (mx, my, mx + mw, my + mh)
        n0, e0 = self.path[i]

        def px(n: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return mx + mw / 2 + (e - e0) * MAP_PX_M, my + mh / 2 - (n - n0) * MAP_PX_M

        # grid every 5 m
        for g in np.arange(math.floor((e0 - 20) / 5) * 5, e0 + 20, 5.0):
            x = int(round(px(np.array(n0), np.array(g))[0]))
            if mx <= x < mx + mw:
                c[my:my + mh, x] = (44, 52, 46)
        for g in np.arange(math.floor((n0 - 10) / 5) * 5, n0 + 10, 5.0):
            y = int(round(px(np.array(g), np.array(e0))[1]))
            if my <= y < my + mh:
                c[y, mx:mx + mw] = (44, 52, 46)
        # field of view (8 m, the shield's memory range), heading north-up
        yaw = math.radians(float(self.rows[i]["yaw"]))
        yy, xx = np.mgrid[my:my + mh, mx:mx + mw]
        de, dn = (xx - (mx + mw / 2)) / MAP_PX_M, ((my + mh / 2) - yy) / MAP_PX_M
        ang = np.arctan2(de, dn) - yaw
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        wedge = (np.hypot(de, dn) <= 8.0) & (np.abs(ang) <= HALF_FOV)
        patch = c[my:my + mh, mx:mx + mw]
        patch[wedge] = (0.75 * patch[wedge] + 0.25 * np.array([80, 200, 230])).astype(np.uint8)
        # trunks
        if len(self.radii):
            near = (np.abs(self.centres[:, 1] - e0) < mw / 2 / MAP_PX_M + 1) & \
                   (np.abs(self.centres[:, 0] - n0) < mh / 2 / MAP_PX_M + 1)
            xs, ys = px(self.centres[near, 0], self.centres[near, 1])
            for x, y, r in zip(xs, ys, self.radii[near], strict=True):
                disc(c, x, y, max(2.0, r * MAP_PX_M), (150, 112, 74), clip=clip)
        # path flown so far, waypoints, vehicle
        xs, ys = px(self.path[:i + 1, 0], self.path[:i + 1, 1])
        polyline(c, np.stack([xs, ys], axis=1), (250, 250, 250), clip)
        m = int(self.rows[i].get("m", 1))
        for j, wp in enumerate(self.waypoints):
            x, y = px(np.array(wp[0]), np.array(wp[1]))
            col = (255, 200, 40) if j + 1 == m else ((120, 120, 120) if j + 1 < m else (200, 170, 60))
            disc(c, float(x), float(y), 6.0 if j + 1 == m else 4.0, col, clip=clip)
        cx, cy = mx + mw / 2, my + mh / 2
        disc(c, cx, cy, 6.0, (80, 200, 230), clip=clip)
        tip = np.array([[cx, cy], [cx + 14 * math.sin(yaw), cy - 14 * math.cos(yaw)]])
        polyline(c, tip, (80, 200, 230), clip, width=3)

    def compose(self, i: int, rgb: np.ndarray, depth_mm: np.ndarray, end: Optional[str] = None) -> np.ndarray:
        row = self.rows[i]
        c = self.background.copy()
        blit(c, rgb, RGB_AT, RGB_SCALE)
        blit(c, depth_image(depth_mm), DEPTH_AT, DEPTH_SCALE)
        self._map(c, i)
        v = np.asarray(row["v"], dtype=np.float64)
        speed, alt = float(np.linalg.norm(v)), -float(row["p"][2])
        u = int(row.get("u", 0))
        shield = str(row.get("shield") or "-")
        mode, *reasons = shield.split("/")
        if mode not in ("fly", "-"):
            label, badge = mode.upper(), (220, 80, 60)              # the shield flies: backoff, nudge, hold
        elif reasons:
            label, badge = "+".join(reasons), (230, 170, 40)        # it filters the policy's command
        else:
            label, badge = "free" if mode == "fly" else "off", (60, 170, 90)
        c[612:640, 432:652] = badge
        m, nwp = int(row.get("m", 1)), len(self.waypoints)
        clear = row.get("clearance")
        t = float(row["t"])
        items = [
            ((24, 10), self.title, 26, True, (240, 240, 240)),
            ((430, 16), self.subtitle, 16, False, (170, 176, 186)),
            ((1110, 12), "t %6.1f s" % t, 26, True, (240, 240, 240)),
            ((30, 78), "onboard RGB 160 x 128 (policy input)", 14, False, (255, 255, 255)),
            ((694, 78), "onboard depth (0-10 m)", 14, False, (255, 255, 255)),
            ((694, 478), "map (privileged, for the viewer only)", 14, False, (230, 230, 230)),
            ((1202, 72), "0 m", 14, False, (200, 200, 200)), ((1202, 440), "10 m", 14, False, (200, 200, 200)),
            ((36, 610), "speed %.2f m/s" % speed, 20, True, (240, 240, 240)),
            ((236, 610), "altitude %.1f m" % alt, 20, False, (220, 220, 220)),
            ((440, 614), "shield: " + label[:18], 16, True, (20, 20, 20)),
            ((36, 646), "primitive %s" % (PRIMITIVE_NAMES[u] if 0 <= u < 6 else u), 16, False, (210, 210, 210)),
            ((300, 646), "waypoint %d / %d  (%.1f m)" % (min(max(m, 1), max(nwp, 1)), nwp,
                                                       float(row.get("wp_dist") or 0.0)), 16, False, (210, 210, 210)),
            ((36, 674), "nearest trunk %s (privileged)" % ("%.2f m" % clear if clear is not None else "-"), 14, False,
             (160, 166, 176)),
            ((300, 674), "fallback (sigma)" if row.get("fallback") else "", 14, True, (230, 120, 90)),
        ]
        if end:
            c[300:420, 240:1040] = (0.35 * c[300:420, 240:1040]).astype(np.uint8)
            items.append(((270, 336), end, 34, True, (120, 230, 140) if end.startswith("SUCCESS") else (240, 120, 90)))
        return self.text.draw(c, items)


# ---------------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------------
def ffmpeg_exe(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit("no ffmpeg: install it, or pip install imageio-ffmpeg") from exc


def encode(frames: Any, out: str, fps: float, exe: str, crf: int = 20) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", "%g" % fps, "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".part.mp4"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for img in frames:
            proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed for %s" % out)
    os.replace(out + ".part.mp4", out)


def save_poster(img: np.ndarray, path: str, exe: str) -> None:
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-i", "-",
           "-frames:v", "1", "-q:v", "3", path]
    subprocess.run(cmd, input=np.ascontiguousarray(img, dtype=np.uint8).tobytes(), check=True)


def render(trace: str, frames_path: str, out: str, title: str, poster: Optional[str] = None, fps: float = 30.0,
           speed: float = 1.0, roots: Sequence[str] = (), exe: Optional[str] = None, hold_s: float = 2.0) -> Dict[str, Any]:
    record, rows = read_trace(trace)
    meta, frames = read_frames(frames_path)
    n = min(len(rows), len(frames))
    if n < 2:
        raise SystemExit("%s: fewer than 2 decisions" % trace)
    if len(rows) != len(frames):
        print("note: %d trace rows, %d frames: using the first %d" % (len(rows), len(frames), n))
    rows = rows[:n]
    t = np.asarray(frames["t"][:n], dtype=np.float64)
    density = (re.search(r"(sparse|medium|dense)", str(record.get("world", ""))) or [None, "?"])[1]
    subtitle = "%s world %s, mission %s  |  PX4 SITL + Gazebo Harmonic" % (
        density, str(record.get("world", "")).replace("aero_forest_", "").split("_")[-1], record.get("episode"))
    rend = Renderer(record, rows, world_arrays(record, list(roots) + [os.path.join(os.path.dirname(HERE), "sim",
                                                                                         "worlds", "test")]),
                    title, subtitle)
    times = np.arange(t[0], t[-1], speed / fps)
    index = np.clip(np.searchsorted(t, times, side="right") - 1, 0, n - 1)
    end = "%s  |  %d/%d waypoints  |  %.1f s" % (str(record.get("termination", "")).upper(),
                                                  int(record.get("waypoints_reached", 0)),
                                                  int(record.get("waypoints", 0)), float(record.get("duration_s") or 0))
    exe = ffmpeg_exe(exe)

    def stream() -> Any:
        for i in index:
            yield rend.compose(int(i), frames["rgb"][i], frames["depth"][i])
        last = rend.compose(n - 1, frames["rgb"][n - 1], frames["depth"][n - 1], end=end)
        for _ in range(int(hold_s * fps)):
            yield last

    encode(stream(), out, fps, exe)
    clear = [r.get("clearance") if r.get("clearance") is not None and float(r["t"]) >= 3.0 else math.inf
             for r in rows]
    k = int(np.argmin(clear)) if np.isfinite(np.min(clear)) else n // 2
    if poster:
        save_poster(rend.compose(k, frames["rgb"][k], frames["depth"][k]), poster, exe)
    info = {"video": os.path.basename(out), "poster": os.path.basename(poster) if poster else None, "title": title,
            "tag": record.get("tag"), "world": record.get("world"), "density": density,
            "episode": record.get("episode"), "seed": record.get("seed"), "termination": record.get("termination"),
            "duration_s": record.get("duration_s"), "mean_speed": record.get("mean_speed"),
            "waypoints": record.get("waypoints"), "waypoints_reached": record.get("waypoints_reached"),
            "min_clearance_m": record.get("min_clearance_m"), "shield": record.get("shield"),
            "checkpoint_id": record.get("checkpoint_id"), "decisions": n, "video_s": round(len(times) / fps, 1),
            "playback_speed": speed}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    return info


# ---------------------------------------------------------------------------------------------
# Choosing the missions, and the index of the videos
# ---------------------------------------------------------------------------------------------
def load(pattern: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("termination") not in (None, "cancelled"):
                    out[(str(r["world"]), int(r["episode"]))] = r
    return out


def pick(runs: Dict[str, str], per_density: int) -> List[Dict[str, Any]]:
    """Missions every run succeeded on, per density, best first: no backoff of the shield, then the largest
    smallest clearance of the worst run, then the shortest mean duration."""
    data = {name: load(pattern) for name, pattern in runs.items()}
    keys = set.intersection(*(set(d) for d in data.values())) if data else set()
    out = []
    for dens in ("dense", "medium", "sparse"):
        cand = []
        for k in keys:
            if dens not in k[0] or not all(bool(d[k]["success"]) for d in data.values()):
                continue
            recs = [d[k] for d in data.values()]
            backoffs = sum(int((r.get("shield") or {}).get("backoffs") or 0) for r in recs)
            clear = min(float(r.get("min_clearance_m") or 0.0) for r in recs)
            dur = sum(float(r.get("duration_s") or 0.0) for r in recs) / len(recs)
            seed = re.search(r"_s0*(\d+)$", k[0])
            cand.append({"density": dens, "world": k[0], "episode": k[1], "seed": int(seed.group(1)) if seed else 0,
                         "backoffs": backoffs, "min_clearance_m": round(clear, 3), "duration_s": round(dur, 1)})
        cand.sort(key=lambda c: (c["backoffs"], -c["min_clearance_m"], c["duration_s"], c["world"], c["episode"]))
        out += cand[:per_density]
    return out


def index(video_dir: str, results: str) -> Dict[str, Any]:
    """videos.json: every rendered video, and every re-flight of the missions tried for its density."""
    videos = []
    for path in sorted(glob.glob(os.path.join(video_dir, "*.json"))):
        if os.path.basename(path) == "videos.json":
            continue
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        if "video" in v:
            videos.append(v)
    attempts = list(load(results).values()) if results else []
    all_lines = []
    if results and os.path.exists(results):
        with open(results, encoding="utf-8") as f:
            for line in f:
                try:
                    all_lines.append(json.loads(line))
                except ValueError:
                    continue
    out = {"about": "FPV demonstration videos: missions both checkpoints succeeded on in the shielded benchmark "
                    "(180 missions each), re-flown in PX4 SITL with frame recording. Every re-flight is listed; "
                    "the benchmark numbers come from the benchmark runs, not from these flights.",
           "videos": videos,
           "reflights": [{k: r.get(k) for k in ("tag", "world", "episode", "termination", "duration_s")}
                         for r in all_lines if isinstance(r, dict)],
           "reflights_total": len(all_lines), "missions_reflown": len(attempts)}
    with open(os.path.join(video_dir, "videos.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


# ---------------------------------------------------------------------------------------------
# Self-test (python3 sim/render_fpv.py --self-test)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile

    assert turbo(np.array([0.0]))[0].tolist() != turbo(np.array([1.0]))[0].tolist()
    with tempfile.TemporaryDirectory() as tmp:
        # a world with two trunks, a straight mission north, 40 decisions with frames
        world_json = os.path.join(tmp, "aero_forest_dense_s9001.json")
        with open(world_json, "w", encoding="utf-8") as f:
            json.dump({"obstacles": [{"type": "cylinder", "center": [1.0, 5.0, 5.0], "radius": 0.25, "length": 10},
                                     {"type": "cylinder", "center": [-2.0, 9.0, 5.0], "radius": 0.3, "length": 10}]}, f)
        record = {"kind": "mission", "tag": "r1_shield_video", "world": "aero_forest_dense_s9001", "episode": 2,
                  "seed": 9001, "termination": "success", "world_json": "/elsewhere/aero_forest_dense_s9001.json",
                  "waypoints_ned": [[6.0, 0.0, -2.5], [12.0, 0.0, -2.5]], "waypoints": 2, "waypoints_reached": 2,
                  "duration_s": 8.0, "mean_speed": 1.5, "min_clearance_m": 0.8}
        n, h, w = 40, 128, 160
        dt = frame_dtype(h, w)
        arr = np.zeros(n, dtype=dt)
        rows = []
        for i in range(n):
            t = 0.2 * i
            arr[i]["t"] = t
            arr[i]["rgb"][..., 1] = 40 + i
            arr[i]["depth"][:, : w // 2] = 1500 + 100 * i
            rows.append({"kind": "decision", "t": t, "p": [1.5 * t, 0.0, -2.5], "v": [1.5, 0.0, 0.0], "yaw": 0.0,
                         "m": 1 if t < 4 else 2, "wp_dist": 6.0 - 1.5 * t % 6, "u": 0, "shield": "fly/barrier" if i % 7 == 0
                         else "fly", "clearance": 2.0 - 0.02 * i if i < 30 else 1.5, "fallback": False})
        frames_path = os.path.join(tmp, "x.frames")
        arr.tofile(frames_path)
        with open(frames_path + ".json", "w", encoding="utf-8") as f:
            json.dump({"format": "aero-vpa/frames", "version": 1, "count": n, "height": h, "width": w,
                       "record_bytes": dt.itemsize}, f)
        trace = os.path.join(tmp, "x.jsonl")
        with open(trace, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in [record] + rows) + "\n")
        meta, frames = read_frames(frames_path)
        assert len(frames) == n and abs(float(frames["t"][3]) - 0.6) < 1e-12 and meta["count"] == n
        rec, rws = read_trace(trace)
        trunks = world_arrays(rec, [tmp])
        assert trunks[0].shape == (2, 2) and np.allclose(trunks[0][0], [5.0, 1.0])     # ENU (1, 5) -> NED (5, 1)
        rend = Renderer(rec, rws, trunks, "r1 + onboard shield", "test")
        img = rend.compose(10, frames["rgb"][10], frames["depth"][10], end="SUCCESS")
        assert img.shape == (H, W, 3) and img.dtype == np.uint8
        x, y = RGB_AT
        assert img[y + 5, x + 5].tolist() == [0, 50, 0]                                 # the RGB frame, scaled
        dx, dy = DEPTH_AT
        assert img[dy + 5, dx + 3 * w - 5].tolist() == [0, 0, 0]                          # no depth return: black
        assert img[dy + 5, dx + 5].tolist() == depth_image(np.array([[2500]], np.uint16))[0, 0].tolist()
        mx, my = MAP_AT
        cx, cy = mx + MAP_SIZE[0] // 2, my + MAP_SIZE[1] // 2
        assert img[cy, cx].tolist() == [80, 200, 230]                                     # the vehicle at the centre
        # a trunk 5 m north, 1 m east of the origin: drawn where the map puts it at decision 0 (vehicle at origin)
        img0 = rend.compose(0, frames["rgb"][0], frames["depth"][0])
        tx, ty = int(cx + 1.0 * MAP_PX_M), int(cy - 5.0 * MAP_PX_M)
        assert img0[ty, tx].tolist() == [150, 112, 74], img0[ty, tx].tolist()
        # choosing missions
        for name, ok_k in (("a", {0, 1, 3}), ("b", {1, 3, 4})):
            with open(os.path.join(tmp, "flights_%s_i0.jsonl" % name), "w", encoding="utf-8") as f:
                for wname in ("aero_forest_dense_s2000", "aero_forest_sparse_s2001"):
                    for k in range(5):
                        f.write(json.dumps({"world": wname, "episode": k, "termination": "success" if k in ok_k
                                            else "collision", "success": k in ok_k, "min_clearance_m": 0.5 + 0.1 * k,
                                            "duration_s": 60.0, "shield": {"backoffs": 1 if k == 3 else 0}}) + "\n")
        best = pick({"a": os.path.join(tmp, "flights_a_i*.jsonl"), "b": os.path.join(tmp, "flights_b_i*.jsonl")}, 2)
        assert [(c["density"], c["episode"]) for c in best] == [("dense", 1), ("dense", 3), ("sparse", 1), ("sparse", 3)]
        assert best[0]["seed"] == 2000 and best[1]["backoffs"] == 2
        # encoding, when an ffmpeg is at hand
        try:
            exe = ffmpeg_exe()
        except SystemExit:
            exe = None
        if exe:
            out = os.path.join(tmp, "videos", "r1_shield_dense.mp4")
            info = render(trace, frames_path, out, "r1 + onboard shield", poster=out[:-4] + ".jpg", fps=10,
                          roots=[tmp], exe=exe, hold_s=0.5)
            assert os.path.getsize(out) > 1000 and os.path.getsize(out[:-4] + ".jpg") > 1000
            assert info["density"] == "dense" and info["decisions"] == n and info["termination"] == "success"
            idx = index(os.path.join(tmp, "videos"), os.path.join(tmp, "flights_a_i0.jsonl"))
            assert len(idx["videos"]) == 1 and idx["reflights_total"] == 10
            encoded = "encoded with %s" % os.path.basename(exe)
        else:
            encoded = "no ffmpeg here: encoding not tested"
    print("render_fpv self-test passed: frames, trace, world, panels, mission choice; %s%s" % (
        encoded, "" if Text().ok else "; no Pillow: text left out"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--self-test", dest="self_test", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    pk = sub.add_parser("pick", help="missions every run succeeded on, best first, per density")
    pk.add_argument("--run", action="append", default=[], metavar="NAME=GLOB", help="records of one run")
    pk.add_argument("--per_density", type=int, default=3)
    rd = sub.add_parser("render", help="one FPV video from a trace and its frames")
    rd.add_argument("--trace", required=True)
    rd.add_argument("--frames", required=True)
    rd.add_argument("--out", required=True, help="the .mp4 to write (a .json with what it shows goes beside it)")
    rd.add_argument("--poster", default=None, help="a .jpg of the closest pass to a trunk")
    rd.add_argument("--title", default="Aero-VPA")
    rd.add_argument("--fps", type=float, default=30.0)
    rd.add_argument("--speed", type=float, default=1.0, help="playback speed (1: real time)")
    rd.add_argument("--world_root", action="append", default=[])
    rd.add_argument("--ffmpeg", default=None)
    ix = sub.add_parser("index", help="videos.json of a video directory")
    ix.add_argument("--videos", required=True)
    ix.add_argument("--results", default=None)
    args = p.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.cmd == "pick":
        runs = dict(r.split("=", 1) for r in args.run)
        print("#density\tworld\tepisode\tseed\tbackoffs\tmin_clearance_m\tduration_s")
        for c in pick(runs, args.per_density):
            print("%s\t%s\t%d\t%d\t%d\t%.3f\t%.1f" % (c["density"], c["world"], c["episode"], c["seed"], c["backoffs"],
                                                     c["min_clearance_m"], c["duration_s"]))
        return 0
    if args.cmd == "render":
        info = render(args.trace, args.frames, args.out, args.title, args.poster, args.fps, args.speed,
                      args.world_root, args.ffmpeg)
        print("wrote %s (%.1f s of video, %d decisions)%s" % (args.out, info["video_s"], info["decisions"],
                                                               ", poster " + args.poster if args.poster else ""))
        return 0
    if args.cmd == "index":
        out = index(args.videos, args.results)
        print("wrote %s: %d videos, %d re-flights" % (os.path.join(args.videos, "videos.json"), len(out["videos"]),
                                                      out["reflights_total"]))
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

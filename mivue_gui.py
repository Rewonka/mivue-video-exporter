#!/usr/bin/env python3
import re
import math
import json
import time
import subprocess
import tempfile
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from tkinter import (
    Tk, Frame, Button, Label, Listbox, Scrollbar, END,
    filedialog, messagebox, StringVar, Toplevel, Radiobutton
)
from tkinter import ttk

import folium

# matplotlib for rendering map video frames
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



PAIR_RE = re.compile(r"^FILE(\d{6})-(\d{6})([FR])\.(MP4|NMEA)$", re.IGNORECASE)
SCRIPT_DIR = Path(__file__).resolve().parent

# Parse ffmpeg progress lines: frame= ... time=00:04:59.96 bitrate=...
TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")

# NMEA time (RMC) example:
# $GPRMC,125121.000,A,4731.7404,N,01901.8975,E,...
RMC_RE = re.compile(r"^\$GPRMC,([^,]+),([AV]),([^,]+),([NS]),([^,]+),([EW]),")


@dataclass
class ClipPair:
    key: str
    front_mp4: Path | None
    rear_mp4: Path | None
    front_nmea: Path | None
    duration_s: float | None = None

    def complete(self) -> bool:
        return bool(self.front_mp4 and self.rear_mp4 and self.front_nmea)


def which(cmd: str) -> str | None:
    from shutil import which as _which
    return _which(cmd)


def hms_to_seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    s = int(round(seconds))
    m = s // 60
    ss = s % 60
    h = m // 60
    mm = m % 60
    if h > 0:
        return f"{h:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def open_in_browser(path: Path) -> None:
    try:
        p = subprocess.run(["xdg-open", str(path)], capture_output=True, text=True)
        if p.returncode == 0:
            return
    except Exception:
        pass
    subprocess.Popen(["firefox", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ffprobe_duration(ffprobe: str, mp4: Path) -> float | None:
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp4)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    out = (p.stdout or "").strip()
    try:
        return float(out)
    except Exception:
        return None


def ffprobe_video_dims(ffprobe: str, mp4: Path) -> tuple[int, int] | None:
    """
    Returns (width, height) for the first video stream.
    """
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(mp4)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    try:
        data = json.loads(p.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None
        w = int(streams[0]["width"])
        h = int(streams[0]["height"])
        return w, h
    except Exception:
        return None


def scan_pairs(normal_dir: Path, ffprobe: str | None) -> list[ClipPair]:
    f_dir = normal_dir / "F"
    r_dir = normal_dir / "R"
    if not f_dir.exists() or not r_dir.exists():
        raise FileNotFoundError("A kiválasztott mappában nem találom az F/ és R/ könyvtárakat (Normal/F és Normal/R).")

    pairs: dict[str, ClipPair] = {}

    def ensure(key: str) -> ClipPair:
        if key not in pairs:
            pairs[key] = ClipPair(key=key, front_mp4=None, rear_mp4=None, front_nmea=None, duration_s=None)
        return pairs[key]

    # Front: MP4 + NMEA
    for p in f_dir.iterdir():
        if not p.is_file():
            continue
        m = PAIR_RE.match(p.name)
        if not m:
            continue
        yymmdd, hhmmss, cam, ext = m.group(1), m.group(2), m.group(3).upper(), m.group(4).upper()
        if cam != "F":
            continue
        key = f"{yymmdd}-{hhmmss}"
        pair = ensure(key)
        if ext.upper() == "MP4":
            pair.front_mp4 = p
        elif ext.upper() == "NMEA":
            pair.front_nmea = p

    # Rear: MP4 only
    for p in r_dir.iterdir():
        if not p.is_file():
            continue
        m = PAIR_RE.match(p.name)
        if not m:
            continue
        yymmdd, hhmmss, cam, ext = m.group(1), m.group(2), m.group(3).upper(), m.group(4).upper()
        if cam != "R" or ext.upper() != "MP4":
            continue
        key = f"{yymmdd}-{hhmmss}"
        pair = ensure(key)
        pair.rear_mp4 = p

    out = [pairs[k] for k in sorted(pairs.keys())]

    if ffprobe:
        for pair in out:
            if pair.front_mp4 and pair.duration_s is None:
                pair.duration_s = ffprobe_duration(ffprobe, pair.front_mp4)

    return out


def build_map_from_nmea(nmea_path: Path, out_html: Path) -> None:
    points: list[tuple[float, float]] = []
    with nmea_path.open("r", errors="ignore") as f:
        for line in f:
            ll = parse_lat_lon_from_gprmc(line)
            if ll:
                points.append(ll)

    if not points:
        raise ValueError("Nem találtam használható $GPRMC koordinátákat ebben a .NMEA fájlban.")

    m = folium.Map(location=points[0], zoom_start=15)
    folium.PolyLine(points, weight=4).add_to(m)
    folium.Marker(points[0], tooltip="Start").add_to(m)
    folium.Marker(points[-1], tooltip="End").add_to(m)

    m.save(str(out_html))


def parse_lat_lon_from_gprmc(line: str) -> tuple[float, float] | None:
    if not line.startswith("$GPRMC"):
        return None
    parts = line.strip().split(",")
    if len(parts) < 7:
        return None
    if parts[2].strip() != "A":
        return None

    lat_raw = parts[3].strip()
    lat_hemi = parts[4].strip().upper()
    lon_raw = parts[5].strip()
    lon_hemi = parts[6].strip().upper()
    if not (lat_raw and lat_hemi and lon_raw and lon_hemi):
        return None

    def ddmm_to_deg(v: str, is_lon: bool) -> float:
        deg_len = 3 if is_lon else 2
        deg = float(v[:deg_len])
        minutes = float(v[deg_len:])
        return deg + minutes / 60.0

    lat = ddmm_to_deg(lat_raw, is_lon=False)
    lon = ddmm_to_deg(lon_raw, is_lon=True)

    if lat_hemi == "S":
        lat = -lat
    if lon_hemi == "W":
        lon = -lon

    return lat, lon


def concat_nmea(front_nmeas: list[Path], out_nmea: Path):
    with out_nmea.open("w", newline="\n") as out:
        for nmea in front_nmeas:
            out.write(f"# ---- {nmea.name} ----\n")
            with nmea.open("r", errors="ignore") as f:
                for line in f:
                    out.write(line.rstrip("\n") + "\n")


def run_and_stream(cmd: list[str], log_cb, on_line=None) -> int:
    log_cb(" ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        log_cb(line)
        if on_line:
            on_line(line)
    return p.wait()


def overlay_segment(
    ffmpeg: str,
    front_mp4: Path,
    rear_mp4: Path,
    out_mp4: Path,
    log_cb,
    duration_s: float | None,
    reduced: bool,
    progress_cb=None,  # progress_cb(seg_ratio 0..1)
):
    """
    reduced=False (native): keep original resolution + lighter compression
    reduced=True: scale to 720p + stronger compression + 64k audio
    rear PiP: scale to main_w/4 (always proportional to final front)
    """
    if reduced:
        # Front -> 720p, Rear -> based on main_w (front width after scaling), then overlay.
        filter_complex = (
            "[0:v]scale=-2:720[front];"
            "[1:v]scale=main_w/4:-1[rear];"
            "[front][rear]overlay=10:10:format=auto[v]"
        )
        crf = "26"
        preset = "slow"
        audio_bitrate = "64k"
    else:
        filter_complex = (
            "[1:v]scale=main_w/4:-1[rear];"
            "[0:v][rear]overlay=10:10:format=auto[v]"
        )
        crf = "20"
        preset = "veryfast"
        audio_bitrate = "128k"

    cmd = [
        ffmpeg, "-y",
        "-i", str(front_mp4),
        "-i", str(rear_mp4),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", crf,
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-shortest",
        str(out_mp4)
    ]

    last_ratio = 0.0

    def on_line(line: str):
        nonlocal last_ratio
        if not duration_s or duration_s <= 0:
            return
        m = TIME_RE.search(line)
        if not m:
            return
        t = hms_to_seconds(m.group(1), m.group(2), m.group(3))
        ratio = max(0.0, min(1.0, t / duration_s))
        if ratio > last_ratio + 0.002:
            last_ratio = ratio
            if progress_cb:
                progress_cb(ratio)

    rc = run_and_stream(cmd, log_cb, on_line=on_line)
    if progress_cb:
        progress_cb(1.0)
    if rc != 0:
        raise RuntimeError(f"ffmpeg overlay hiba (exit={rc}). Nézd a logot.")


def concat_mp4(ffmpeg: str, segments: list[Path], out_mp4: Path, log_cb):
    list_file = out_mp4.with_suffix(".concat.txt")
    with list_file.open("w", newline="\n") as f:
        for seg in segments:
            f.write(f"file {seg.resolve()}\n")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_mp4)
    ]

    rc = run_and_stream(cmd, log_cb)
    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass

    if rc != 0:
        raise RuntimeError(f"ffmpeg concat hiba (exit={rc}). Nézd a logot.")


class FileLogger:
    """
    Writes logs to:
      logs/latest.log (overwritten each run)
      logs/mivue_YYYYMMDD_HHMMSS.log (one per run)
    """
    def __init__(self):
        self.logs_dir = SCRIPT_DIR / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.logs_dir / f"mivue_{ts}.log"
        self.latest_path = self.logs_dir / "latest.log"

        self.latest_path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, line: str):
        with self._lock:
            with self.latest_path.open("a", encoding="utf-8", errors="ignore") as f:
                f.write(line + "\n")
            with self.session_path.open("a", encoding="utf-8", errors="ignore") as f:
                f.write(line + "\n")


class BusyDialog:
    def __init__(self, root: Tk, title="Dolgozom…"):
        self.top = Toplevel(root)
        self.top.title(title)
        self.top.geometry("620x190")
        self.top.transient(root)
        self.top.grab_set()

        self.label_var = StringVar(value="Indítás…")
        Label(self.top, textvariable=self.label_var, wraplength=600, justify="left").pack(anchor="w", padx=12, pady=(12, 6))

        self.progress = ttk.Progressbar(self.top, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12, pady=6)

        self.detail_var = StringVar(value="")
        Label(self.top, textvariable=self.detail_var, wraplength=600, justify="left").pack(anchor="w", padx=12)

    def set_label(self, text: str):
        self.label_var.set(text)

    def set_detail(self, text: str):
        self.detail_var.set(text)

    def set_progress(self, percent: float):
        self.progress["value"] = max(0, min(100, percent))

    def close(self):
        try:
            self.top.grab_release()
        except Exception:
            pass
        self.top.destroy()


# -----------------------------
# Map video (OSM tiles) helpers
# -----------------------------

TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_CACHE_DIR = SCRIPT_DIR / "tiles_cache"
TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "mivue-map-overlay/1.0 (personal use)"


def latlon_to_global_px(lat: float, lon: float, z: int) -> tuple[float, float]:
    """
    WebMercator global pixel coordinates at zoom z.
    """
    siny = math.sin(lat * math.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)
    x = (lon + 180.0) / 360.0
    y = 0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)

    scale = TILE_SIZE * (2 ** z)
    return x * scale, y * scale


def global_px_to_tile_xy(px: float, py: float) -> tuple[int, int]:
    return int(px // TILE_SIZE), int(py // TILE_SIZE)


def fetch_tile(z: int, x: int, y: int, log_cb) -> Path:
    """
    Download tile if not cached.
    """
    # OSM valid range wrap for x, clamp for y
    n = 2 ** z
    x_wrapped = x % n
    if y < 0 or y >= n:
        # outside world - should not happen if bbox is valid
        raise ValueError("Tile Y out of range")

    tile_path = TILE_CACHE_DIR / f"{z}_{x_wrapped}_{y}.png"
    if tile_path.exists():
        return tile_path

    url = OSM_TILE_URL.format(z=z, x=x_wrapped, y=y)
    log_cb(f"[tile] download {url}")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 4):
        try:
            with urlopen(req, timeout=20) as resp:
                data = resp.read()
            tile_path.write_bytes(data)
            return tile_path
        except (HTTPError, URLError) as e:
            log_cb(f"[tile] error attempt {attempt}: {e}")
            time.sleep(1.0 * attempt)
    raise RuntimeError(f"Nem sikerült letölteni tile-t: z={z} x={x} y={y}")


def stitch_tiles(z: int, x_min: int, x_max: int, y_min: int, y_max: int, log_cb):
    """
    Returns a stitched RGB image as numpy array (H,W,3).
    """
    import numpy as np
    from PIL import Image  # pillow is commonly installed; if not, user will see error

    tiles_w = (x_max - x_min + 1)
    tiles_h = (y_max - y_min + 1)
    out = Image.new("RGB", (tiles_w * TILE_SIZE, tiles_h * TILE_SIZE))

    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            p = fetch_tile(z, tx, ty, log_cb)
            img = Image.open(p).convert("RGB")
            ox = (tx - x_min) * TILE_SIZE
            oy = (ty - y_min) * TILE_SIZE
            out.paste(img, (ox, oy))

    return np.array(out)


def parse_nmea_track(nmea_path: Path) -> list[tuple[float, float, float]]:
    """
    Returns list of (t_seconds_from_start, lat, lon), from $GPRMC lines.
    Assumes same day (no midnight crossing).
    """
    samples: list[tuple[float, float, float]] = []

    start_sec: float | None = None

    def ddmm_to_deg(v: str, is_lon: bool) -> float:
        deg_len = 3 if is_lon else 2
        deg = float(v[:deg_len])
        minutes = float(v[deg_len:])
        return deg + minutes / 60.0

    with nmea_path.open("r", errors="ignore") as f:
        for line in f:
            m = RMC_RE.match(line.strip())
            if not m:
                continue
            t_str, status, lat_raw, lat_hemi, lon_raw, lon_hemi = m.groups()
            if status != "A":
                continue

            # time "hhmmss.sss"
            if len(t_str) < 6:
                continue
            hh = int(t_str[0:2])
            mm = int(t_str[2:4])
            ss = float(t_str[4:])
            sec = hh * 3600 + mm * 60 + ss

            lat = ddmm_to_deg(lat_raw, is_lon=False)
            lon = ddmm_to_deg(lon_raw, is_lon=True)
            if lat_hemi.upper() == "S":
                lat = -lat
            if lon_hemi.upper() == "W":
                lon = -lon

            if start_sec is None:
                start_sec = sec
            t_rel = sec - start_sec
            if t_rel < 0:
                # midnight crossing not handled here
                continue
            samples.append((t_rel, lat, lon))

    return samples


def interp_position(track: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    """
    Linear interpolation in time.
    """
    if not track:
        raise ValueError("Üres track")

    if t <= track[0][0]:
        return track[0][1], track[0][2]
    if t >= track[-1][0]:
        return track[-1][1], track[-1][2]

    # find interval (linear scan ok for 5fps; can be optimized)
    for i in range(len(track) - 1):
        t0, lat0, lon0 = track[i]
        t1, lat1, lon1 = track[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return lat0, lon0
            a = (t - t0) / (t1 - t0)
            lat = lat0 + a * (lat1 - lat0)
            lon = lon0 + a * (lon1 - lon0)
            return lat, lon

    return track[-1][1], track[-1][2]


def choose_zoom_for_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float, max_tiles: int = 8) -> int:
    """
    Choose zoom so the bbox doesn't require too many tiles.
    Tries from z=17 down to z=10.
    """
    for z in range(17, 9, -1):
        px1, py1 = latlon_to_global_px(lat_max, lon_min, z)
        px2, py2 = latlon_to_global_px(lat_min, lon_max, z)
        x1, y1 = global_px_to_tile_xy(px1, py1)
        x2, y2 = global_px_to_tile_xy(px2, py2)
        tiles_w = abs(x2 - x1) + 1
        tiles_h = abs(y2 - y1) + 1
        if tiles_w <= max_tiles and tiles_h <= max_tiles:
            return z
    return 10


def ensure_even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def compute_pip_size(ffprobe: str, front_mp4: Path, rear_mp4: Path, reduced: bool) -> tuple[int, int]:
    """
    Returns (pip_w, pip_h) based on the SAME logic as ffmpeg:
      - final front width depends on reduced (720p) or native
      - pip width = main_w/4
      - pip height follows rear aspect
    """
    front_dims = ffprobe_video_dims(ffprobe, front_mp4)
    rear_dims = ffprobe_video_dims(ffprobe, rear_mp4)
    if not front_dims or not rear_dims:
        # fallback reasonable default
        return 480, 270

    fw, fh = front_dims
    rw, rh = rear_dims

    if reduced:
        # front scaled to -2:720 => compute width keeping aspect and even
        main_h = 720
        main_w = int(round((fw / fh) * main_h))
        main_w = ensure_even(main_w)
    else:
        main_w = fw

    pip_w = ensure_even(int(round(main_w / 4)))
    pip_h = ensure_even(int(round(pip_w * (rh / rw))))
    # avoid zero
    pip_w = max(64, pip_w)
    pip_h = max(64, pip_h)
    return pip_w, pip_h

def smooth_track_ema(track: list[tuple[float, float, float]], alpha: float) -> list[tuple[float, float, float]]:
    """
    Exponential Moving Average smoothing for lat/lon, preserves timestamps.
    alpha: 0..1 (smaller = smoother)
    """
    if not track:
        return track
    out = []
    _, lat0, lon0 = track[0]
    s_lat, s_lon = lat0, lon0
    out.append((track[0][0], s_lat, s_lon))
    for t, lat, lon in track[1:]:
        s_lat = (1 - alpha) * s_lat + alpha * lat
        s_lon = (1 - alpha) * s_lon + alpha * lon
        out.append((t, s_lat, s_lon))
    return out

def render_map_video_from_nmea(
    ffmpeg: str,
    ffprobe: str,
    clip: ClipPair,
    out_mp4: Path,
    log_cb,
    progress_cb,
    reduced: bool,
    fps: int = 5,
    follow: bool = True,
):
    """
    Generates an MP4 video of a moving dot on an OpenStreetMap background.
    Output resolution matches rear PiP size (based on reduced/native).
    """
    if not clip.front_nmea or not clip.front_mp4 or not clip.rear_mp4:
        raise ValueError("Hiányzik a szükséges fájl (front_nmea/front_mp4/rear_mp4).")

    dur = clip.duration_s
    if not dur or dur <= 0:
        dur = ffprobe_duration(ffprobe, clip.front_mp4) or 0.0
    if dur <= 0:
        raise ValueError("Nem tudom a videó hosszát (ffprobe).")

    track = parse_nmea_track(clip.front_nmea)
    if len(track) < 2:
        raise ValueError("Kevés GPS pont a .NMEA-ban (legalább 2 kell).")

    # --- GPS track smoothing (EMA) ---
    GPS_EMA_ALPHA = 0.25  # 0.15 = nagyon smooth, 0.35 = kevésbé smooth
    track = smooth_track_ema(track, GPS_EMA_ALPHA)
    log_cb(f"[map] GPS EMA smoothing enabled alpha={GPS_EMA_ALPHA}")

    pip_w, pip_h = compute_pip_size(ffprobe, clip.front_mp4, clip.rear_mp4, reduced=reduced)

    lats = [lat for _, lat, _ in track]
    lons = [lon for _, _, lon in track]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    # choose zoom
    z = choose_zoom_for_bbox(lat_min, lat_max, lon_min, lon_max, max_tiles=8)
    log_cb(f"[map] zoom={z} bbox=({lat_min},{lon_min})..({lat_max},{lon_max})")

    # Build a "big" background bbox with margin so follow-crop has room.
    # Margin in pixels ~ half of window size at chosen zoom.
    # Convert margin px to lat/lon? easiest: compute in global pixels and expand.
    px_min, py_min = latlon_to_global_px(lat_max, lon_min, z)  # top-left-ish
    px_max, py_max = latlon_to_global_px(lat_min, lon_max, z)  # bottom-right-ish

    # global pixel bounds
    gpx_left = min(px_min, px_max)
    gpx_right = max(px_min, px_max)
    gpy_top = min(py_min, py_max)
    gpy_bottom = max(py_min, py_max)

    margin_px_x = pip_w * 1.2
    margin_px_y = pip_h * 1.2

    gpx_left -= margin_px_x
    gpx_right += margin_px_x
    gpy_top -= margin_px_y
    gpy_bottom += margin_px_y

    # tile bounds
    tx1, ty1 = global_px_to_tile_xy(gpx_left, gpy_top)
    tx2, ty2 = global_px_to_tile_xy(gpx_right, gpy_bottom)
    x_min_t, x_max_t = min(tx1, tx2), max(tx1, tx2)
    y_min_t, y_max_t = min(ty1, ty2), max(ty1, ty2)

    tiles_w = (x_max_t - x_min_t + 1)
    tiles_h = (y_max_t - y_min_t + 1)
    log_cb(f"[map] tiles {tiles_w}x{tiles_h} (z={z}) cache_dir={TILE_CACHE_DIR}")

    # stitch tiles (needs pillow)
    try:
        bg = stitch_tiles(z, x_min_t, x_max_t, y_min_t, y_max_t, log_cb)
    except ModuleNotFoundError:
        raise RuntimeError("Hiányzik a Pillow (PIL). Telepítés: pip install pillow")

    bg_h, bg_w, _ = bg.shape

    # Precompute route pixels in background space
    def to_bg_px(lat: float, lon: float) -> tuple[float, float]:
        gpx, gpy = latlon_to_global_px(lat, lon, z)
        # translate into stitched image pixel coords
        return (gpx - x_min_t * TILE_SIZE), (gpy - y_min_t * TILE_SIZE)

    route_px = [to_bg_px(lat, lon) for _, lat, lon in track]

    # Frame render dir
    workdir = Path(tempfile.mkdtemp(prefix="mivue_map_"))
    frame_dir = workdir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(math.ceil(dur * fps))
    log_cb(f"[map] render {total_frames} frames @ {fps}fps, size={pip_w}x{pip_h}, follow={follow}")

    # Pre-draw full background route once onto a matplotlib canvas?
    # We'll draw per-frame by cropping bg and drawing route+dot for that crop.
    # Use matplotlib imshow for background crop, then plot route segment and dot.

    # --- smoothing params (follow “kamera” simítása) ---
    SMOOTH_ALPHA = 0.18  # 0.10 -> nagyon smooth, 0.30 -> gyorsabban követ
    sm_cx = None
    sm_cy = None

    # route relative coords will be computed per-frame after cropping
    for i in range(total_frames):
        t = i / fps
        lat, lon = interp_position(track, t)
        dot_x, dot_y = to_bg_px(lat, lon)

        # EMA smoothing on camera center (NOT on the dot)
        if sm_cx is None:
            sm_cx, sm_cy = dot_x, dot_y
        else:
            sm_cx = (1 - SMOOTH_ALPHA) * sm_cx + SMOOTH_ALPHA * dot_x
            sm_cy = (1 - SMOOTH_ALPHA) * sm_cy + SMOOTH_ALPHA * dot_y

        if follow:
            cx, cy = sm_cx, sm_cy
            left = int(round(cx - pip_w / 2))
            top  = int(round(cy - pip_h / 2))
            left = max(0, min(left, bg_w - pip_w))
            top  = max(0, min(top,  bg_h - pip_h))
        else:
            xs = [p[0] for p in route_px]
            ys = [p[1] for p in route_px]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            left = int(round(cx - pip_w / 2))
            top  = int(round(cy - pip_h / 2))
            left = max(0, min(left, bg_w - pip_w))
            top  = max(0, min(top,  bg_h - pip_h))

        crop = bg[top:top + pip_h, left:left + pip_w, :]

        # PIL image (no matplotlib => no white borders)
        img = Image.fromarray(crop)
        draw = ImageDraw.Draw(img, "RGBA")

        # draw route (only points inside the window to avoid huge lines)
        # optionally: draw only up to current time for "progress line":
        # route_px_now = route_px[:some_index]
        pts = []
        for x, y in route_px:
            rx = x - left
            ry = y - top
            if 0 <= rx < pip_w and 0 <= ry < pip_h:
                pts.append((rx, ry))

        if len(pts) >= 2:
            # blue-ish line with alpha
            draw.line(pts, fill=(30, 90, 200, 220), width=3)

        # draw dot (actual dot position, not smoothed)
        dx = dot_x - left
        dy = dot_y - top
        r = 7
        draw.ellipse((dx - r, dy - r, dx + r, dy + r), fill=(20, 120, 255, 255))

        frame_path = frame_dir / f"frame_{i:06d}.png"
        img.save(frame_path)

        if (i % max(1, total_frames // 100)) == 0 or i == total_frames - 1:
            progress_cb((i + 1) / total_frames)
    # Encode to mp4
    # Use yuv420p for compatibility
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4)
    ]
    rc = run_and_stream(cmd, log_cb)
    if rc != 0:
        raise RuntimeError(f"ffmpeg map encode hiba (exit={rc}).")


# -----------------------------
# App UI
# -----------------------------

class App:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("MiVue 798D + A30 – Simple Viewer/Exporter")

        self.ffmpeg = which("ffmpeg")
        self.ffprobe = which("ffprobe")

        if not self.ffmpeg:
            messagebox.showerror("Hiba", "Nem találom az ffmpeg-et a PATH-ban. Telepítsd: sudo apt install ffmpeg")
            root.destroy()
            return
        if not self.ffprobe:
            messagebox.showerror("Hiba", "Nem találom az ffprobe-ot a PATH-ban. Telepítsd: sudo apt install ffmpeg")
            root.destroy()
            return

        self.file_logger = FileLogger()

        self.base_dir: Path | None = None
        self.pairs: list[ClipPair] = []

        self.ui_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._poll_ui_queue()

        top = Frame(root)
        top.pack(fill="x", padx=10, pady=10)

        self.status = StringVar(value=f"Válaszd ki a Normal mappát. Log: {self.file_logger.latest_path}")
        Label(top, textvariable=self.status, wraplength=980, justify="left").pack(anchor="w")

        btns = Frame(root)
        btns.pack(fill="x", padx=10, pady=5)

        Button(btns, text="Mappa (Normal/)", command=self.pick_dir).pack(side="left")

        # NEW: map video button
        Button(btns, text="Térkép videó (kijelölt)", command=self.make_map_video_for_selection).pack(side="left", padx=10)

        Button(btns, text="Térkép HTML (kijelölt)", command=self.open_map_for_selection).pack(side="left", padx=10)
        Button(btns, text="Egyesítés (több clip → 1 MP4 + 1 NMEA)", command=self.export_selection).pack(side="left", padx=10)

        # Quality selector
        self.quality_mode = StringVar(value="native")
        Label(btns, text="Minőség:").pack(side="left", padx=(18, 4))
        Radiobutton(btns, text="Nativ", variable=self.quality_mode, value="native").pack(side="left")
        Radiobutton(btns, text="Csökkentett (720p + strong)", variable=self.quality_mode, value="reduced").pack(side="left", padx=6)

        mid = Frame(root)
        mid.pack(fill="both", expand=True, padx=10, pady=10)

        self.listbox = Listbox(mid, height=18, width=130, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)

        sb = Scrollbar(mid)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        sb.config(command=self.listbox.yview)

        bottom = Frame(root)
        bottom.pack(fill="x", padx=10, pady=8)
        Label(bottom, text="✅ exportálható. ⚠️ hiányos. (rear PiP: main_w/4). Térkép videó: OSM tiles + 5fps.", wraplength=980, justify="left").pack(anchor="w")

        self.log("=== MiVue GUI started ===")
        self.log(f"Session log: {self.file_logger.session_path}")
        self.log(f"Latest log:  {self.file_logger.latest_path}")
        self.log(f"ffmpeg: {self.ffmpeg}")
        self.log(f"ffprobe:{self.ffprobe}")

    def log(self, msg: str):
        self.file_logger.write(msg)

    def set_status(self, msg: str):
        self.ui_queue.put(("status", msg))

    def _poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status.set(payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_ui_queue)

    def pick_dir(self):
        d = filedialog.askdirectory(title="Válaszd ki a Normal mappát")
        if not d:
            return
        try:
            self.base_dir = Path(d)
            self.set_status("Beolvasás… (fájlok + hossz ffprobe)")
            self.root.update_idletasks()

            self.pairs = scan_pairs(self.base_dir, self.ffprobe)
            self.listbox.delete(0, END)

            total_known = 0.0
            for pair in self.pairs:
                marker = "✅" if pair.complete() else "⚠️"
                dur = fmt_duration(pair.duration_s)
                if pair.duration_s is not None:
                    total_known += pair.duration_s

                self.listbox.insert(
                    END,
                    f"{marker} {pair.key} | dur:{dur} | "
                    f"Fmp4:{'Y' if pair.front_mp4 else 'N'} "
                    f"Rmp4:{'Y' if pair.rear_mp4 else 'N'} "
                    f"Fnmea:{'Y' if pair.front_nmea else 'N'}"
                )

            self.set_status(
                f"Betöltve: {self.base_dir} | Clip-ek: {len(self.pairs)} | Össz-idő(ismert): {fmt_duration(total_known)} | Log: {self.file_logger.latest_path}"
            )
            self.log(f"Selected Normal dir: {self.base_dir}")

        except Exception as e:
            self.log(f"Pick dir error: {e}")
            messagebox.showerror("Hiba", str(e))

    def get_selected_pairs(self) -> list[ClipPair]:
        if not self.pairs:
            return []
        idxs = list(self.listbox.curselection())
        return [self.pairs[i] for i in idxs] if idxs else []

    def open_map_for_selection(self):
        sel = self.get_selected_pairs()
        if not sel:
            messagebox.showinfo("Info", "Jelölj ki legalább 1 elemet.")
            return

        nmeas = [p.front_nmea for p in sel if p.front_nmea]
        if not nmeas:
            messagebox.showerror("Hiba", "A kijelöltek között nincs Front .NMEA.")
            return

        try:
            combined_nmea = SCRIPT_DIR / "combined_selection.nmea"
            out_html = SCRIPT_DIR / "mivue_map_selection.html"

            concat_nmea([n for n in nmeas if n], combined_nmea)
            build_map_from_nmea(combined_nmea, out_html)

            self.log(f"Map HTML: {out_html}")
            open_in_browser(out_html)
        except Exception as e:
            self.log(f"Map HTML error: {e}")
            messagebox.showerror("Hiba", str(e))

    def make_map_video_for_selection(self):
        sel = self.get_selected_pairs()
        if not sel:
            messagebox.showinfo("Info", "Jelölj ki legalább 1 elemet.")
            return

        # Step 1: generate for first selected clip only
        clip = sel[0]
        if not clip.front_nmea or not clip.front_mp4 or not clip.rear_mp4:
            messagebox.showerror("Hiba", "Ehhez kell: front MP4 + rear MP4 + front NMEA (válassz ✅ clipet).")
            return

        reduced = (self.quality_mode.get() == "reduced")

        # follow map by default (as requested)
        follow = True  # change to False to test fixed
        follow_tag = "FOLLOW" if follow else "FIXED"

        pip_w, pip_h = compute_pip_size(self.ffprobe, clip.front_mp4, clip.rear_mp4, reduced=reduced)
        out_mp4 = SCRIPT_DIR / f"MAP_{clip.key}_{follow_tag}_{pip_w}x{pip_h}.mp4"

        busy = BusyDialog(self.root, title="Térkép videó – dolgozom…")
        busy.set_label("Térképes videó készítése (OSM) – részletek a log fájlban")
        busy.set_detail(f"Méret: {pip_w}x{pip_h} | FPS: 5 | Mód: {'Csökkentett' if reduced else 'Nativ'} | Log: {self.file_logger.latest_path}")
        busy.set_progress(0)

        self.log("=== Map video start ===")
        self.log(f"Clip key: {clip.key}")
        self.log(f"Mode: {'reduced' if reduced else 'native'} follow={follow}")
        self.log(f"Output map mp4: {out_mp4}")

        def worker():
            try:
                def progress_cb(ratio: float):
                    self.root.after(0, lambda r=ratio: busy.set_progress(r * 100))

                render_map_video_from_nmea(
                    ffmpeg=self.ffmpeg,
                    ffprobe=self.ffprobe,
                    clip=clip,
                    out_mp4=out_mp4,
                    log_cb=self.log,
                    progress_cb=progress_cb,
                    reduced=reduced,
                    fps=5,
                    follow=follow,
                )

                self.log("=== Map video done ===")
                self.root.after(0, lambda: (busy.close(), messagebox.showinfo("Kész", f"Térképes videó elkészült:\n{out_mp4}\n\nLog: {self.file_logger.latest_path}")))
            except Exception as e:
                err = str(e)
                self.log(f"!!! Map video error: {err}")
                self.root.after(0, lambda err=err: (busy.close(), messagebox.showerror("Hiba", f"{err}\n\nLog: {self.file_logger.latest_path}")))

        threading.Thread(target=worker, daemon=True).start()

    def export_selection(self):
        sel = self.get_selected_pairs()
        if not sel:
            messagebox.showinfo("Info", "Jelölj ki legalább 1 elemet.")
            return

        clips = sorted([p for p in sel if p.complete()], key=lambda x: x.key)
        if not clips:
            messagebox.showerror("Hiba", "A kijelöltek között nincs exportálható (✅) clip.")
            return

        out_dir = filedialog.askdirectory(title="Hova mentsem a végső eredményt?")
        if not out_dir:
            return
        out_dir = Path(out_dir)

        reduced = (self.quality_mode.get() == "reduced")

        start_key = clips[0].key
        end_key = clips[-1].key
        mode_tag = "REDUCED_720P" if reduced else "NATIVE"
        out_mp4 = out_dir / f"COMBINED_{mode_tag}_{start_key}_to_{end_key}.mp4"
        out_nmea = out_dir / f"COMBINED_{mode_tag}_{start_key}_to_{end_key}.nmea"

        busy = BusyDialog(self.root, title="Export – dolgozom…")
        busy.set_label("Export folyamat… (részletek a log fájlban)")
        busy.set_detail(f"Mód: {'Csökkentett 720p' if reduced else 'Nativ'} | Log: {self.file_logger.latest_path}")
        self.root.update_idletasks()

        self.log("=== Export start ===")
        self.log(f"Mode: {'reduced(720p+strong)' if reduced else 'native'}")
        self.log(f"Output MP4: {out_mp4}")
        self.log(f"Output NMEA: {out_nmea}")

        def worker():
            try:
                workdir = Path(tempfile.mkdtemp(prefix="mivue_"))
                segments: list[Path] = []
                total_segments = len(clips)

                def make_progress_cb(done_segments: int):
                    def _cb(seg_ratio: float):
                        overall = (done_segments + seg_ratio) / max(1, total_segments)
                        pct = overall * 90.0
                        self.root.after(0, lambda p=pct: busy.set_progress(p))
                    return _cb

                for i, pair in enumerate(clips, start=1):
                    done = i - 1
                    self.set_status(f"Export: overlay {i}/{total_segments} … | Log: {self.file_logger.latest_path}")
                    self.log(f"--- Overlay {i}/{total_segments}: {pair.key} ---")
                    self.log(f"Front: {pair.front_mp4}")
                    self.log(f"Rear : {pair.rear_mp4}")

                    self.root.after(0, lambda i=i, ts=total_segments: busy.set_detail(f"Szegmens: {i}/{ts} (overlay)"))

                    seg_out = workdir / f"seg_{i:04d}_{pair.key}.mp4"
                    overlay_segment(
                        self.ffmpeg,
                        pair.front_mp4,
                        pair.rear_mp4,
                        seg_out,
                        self.log,
                        duration_s=pair.duration_s,
                        reduced=reduced,
                        progress_cb=make_progress_cb(done),
                    )
                    segments.append(seg_out)

                self.set_status(f"Export: concat… | Log: {self.file_logger.latest_path}")
                self.root.after(0, lambda: (busy.set_progress(92), busy.set_detail("Összefűzés (concat)…")))
                concat_mp4(self.ffmpeg, segments, out_mp4, self.log)
                self.root.after(0, lambda: busy.set_progress(97))

                self.set_status(f"Export: NMEA… | Log: {self.file_logger.latest_path}")
                self.root.after(0, lambda: (busy.set_progress(97), busy.set_detail("NMEA összefűzés…")))
                concat_nmea([p.front_nmea for p in clips if p.front_nmea], out_nmea)
                self.root.after(0, lambda: busy.set_progress(100))

                self.set_status(f"Kész! {out_mp4.name} + {out_nmea.name} | Log: {self.file_logger.latest_path}")
                self.log("=== Export done ===")
                self.root.after(0, lambda: (busy.close(), messagebox.showinfo("Kész", f"Mentve:\n{out_mp4}\n{out_nmea}\n\nLog: {self.file_logger.latest_path}")))

            except Exception as e:
                err = str(e)
                self.log(f"!!! Export error: {err}")
                self.root.after(0, lambda err=err: (busy.close(), messagebox.showerror("Hiba", f"{err}\n\nLog: {self.file_logger.latest_path}")))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

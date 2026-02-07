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
    filedialog, messagebox, StringVar, BooleanVar, Toplevel, Radiobutton
)
from tkinter import ttk

import folium

PAIR_RE = re.compile(r"^FILE(\d{6})-(\d{6})([FR])\.(MP4|NMEA)$", re.IGNORECASE)

# ffmpeg progress time=HH:MM:SS.xx
TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")

# NMEA RMC example:
# $GPRMC,125121.000,A,4731.7404,N,01901.8975,E,...
RMC_RE = re.compile(r"^\$GPRMC,([^,]+),([AV]),([^,]+),([NS]),([^,]+),([EW]),")

SCRIPT_DIR = Path(__file__).resolve().parent


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
    if seconds is None or seconds <= 0:
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


def ensure_even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


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


def scan_pairs(
    normal_dir: Path,
    ffprobe: str | None,
    progress_cb=None,
    detail_cb=None,
) -> list[ClipPair]:
    f_dir = normal_dir / "F"
    r_dir = normal_dir / "R"
    if not f_dir.exists() or not r_dir.exists():
        raise FileNotFoundError("A kiválasztott mappában nem találom az F/ és R/ könyvtárakat (Normal/F és Normal/R).")

    f_files = [p for p in f_dir.iterdir() if p.is_file()]
    r_files = [p for p in r_dir.iterdir() if p.is_file()]

    pairs: dict[str, ClipPair] = {}
    front_mp4_candidates = 0
    for p in f_files:
        m = PAIR_RE.match(p.name)
        if not m:
            continue
        cam = m.group(3).upper()
        ext = m.group(4).upper()
        if cam == "F" and ext == "MP4":
            front_mp4_candidates += 1

    total_steps = len(f_files) + len(r_files) + (front_mp4_candidates if ffprobe else 0)
    done_steps = 0

    def report(detail: str | None = None):
        nonlocal done_steps
        done_steps += 1
        if progress_cb:
            progress_cb(done_steps, max(1, total_steps))
        if detail_cb and detail:
            detail_cb(detail)

    def ensure(key: str) -> ClipPair:
        if key not in pairs:
            pairs[key] = ClipPair(key=key, front_mp4=None, rear_mp4=None, front_nmea=None, duration_s=None)
        return pairs[key]

    # Front: MP4 + NMEA
    for p in f_files:
        m = PAIR_RE.match(p.name)
        if not m:
            report(f"F: {p.name}")
            continue
        yymmdd, hhmmss, cam, ext = m.group(1), m.group(2), m.group(3).upper(), m.group(4).upper()
        if cam != "F":
            report(f"F: {p.name}")
            continue
        key = f"{yymmdd}-{hhmmss}"
        pair = ensure(key)
        if ext == "MP4":
            pair.front_mp4 = p
        elif ext == "NMEA":
            pair.front_nmea = p
        report(f"F: {p.name}")

    # Rear: MP4 only
    for p in r_files:
        m = PAIR_RE.match(p.name)
        if not m:
            report(f"R: {p.name}")
            continue
        yymmdd, hhmmss, cam, ext = m.group(1), m.group(2), m.group(3).upper(), m.group(4).upper()
        if cam != "R" or ext != "MP4":
            report(f"R: {p.name}")
            continue
        key = f"{yymmdd}-{hhmmss}"
        pair = ensure(key)
        pair.rear_mp4 = p
        report(f"R: {p.name}")

    out = [pairs[k] for k in sorted(pairs.keys())]

    if ffprobe:
        for pair in out:
            if pair.front_mp4 and pair.duration_s is None:
                pair.duration_s = ffprobe_duration(ffprobe, pair.front_mp4)
                report(f"ffprobe: {pair.front_mp4.name}")

    return out


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


def concat_nmea(front_nmeas: list[Path], out_nmea: Path):
    with out_nmea.open("w", newline="\n") as out:
        for nmea in front_nmeas:
            out.write(f"# ---- {nmea.name} ----\n")
            with nmea.open("r", errors="ignore") as f:
                for line in f:
                    out.write(line.rstrip("\n") + "\n")


def overlay_segment(
    ffmpeg: str,
    front_mp4: Path,
    rear_mp4: Path,
    out_mp4: Path,
    log_cb,
    duration_s: float | None,
    reduced: bool,
    progress_cb=None,
    map_mp4: Path | None = None,
    map_offset_s: float = 0.0,
    map_duration_s: float | None = None,
    pip_w: int = 480,
    pip_h: int = 270,
):

    """
    reduced=False: keep original res, CRF20 veryfast, audio 128k
    reduced=True:  scale front to 720p, CRF26 slow, audio 64k
    rear PiP: main_w/4 always (proportional to final front)
    map overlay (if map_mp4 is provided): top-right, scaled to same as PiP width (main_w/4)
    """

    # Build inputs:
    # 0: front, 1: rear, 2: map (optional)
    cmd = [ffmpeg, "-y", "-i", str(front_mp4), "-i", str(rear_mp4)]

    if map_mp4 is not None:
        # Cut the relevant time slice from the map video to match this segment
        # Put -ss/-t before -i for faster seeking (good enough for map overlay)
        cmd += ["-ss", f"{map_offset_s:.3f}"]
        if map_duration_s is not None and map_duration_s > 0:
            cmd += ["-t", f"{map_duration_s:.3f}"]
        cmd += ["-i", str(map_mp4)]

    if reduced:
        base_front = "[0:v]scale=-2:720[front];"
    else:
        base_front = "[0:v]null[front];"

    # rear: 1/4 of main video width (after scaling, if reduced)
    rear_scale = f"[1:v]scale={pip_w}:{pip_h}[rear];"

    if map_mp4 is None:
        # Only rear overlay
        if reduced:
            filter_complex = (
                base_front +
                rear_scale +
                "[front][rear]overlay=10:10:format=auto[v]"
            )
        else:
            filter_complex = (
                base_front +
                rear_scale +
                "[front][rear]overlay=10:10:format=auto[v]"
            )
    else:
        # Map scale to same width as rear PiP
        map_scale = f"[2:v]scale={pip_w}:{pip_h}[map];"
        if reduced:
            filter_complex = (
                base_front +
                rear_scale +
                map_scale +
                "[front][rear]overlay=10:10:format=auto[tmp];"
                "[tmp][map]overlay=W-w-10:10:format=auto[v]"
            )
        else:
            filter_complex = (
                base_front +
                rear_scale +
                map_scale +
                "[front][rear]overlay=10:10:format=auto[tmp];"
                "[tmp][map]overlay=W-w-10:10:format=auto[v]"
            )

    if reduced:
        crf = "26"
        preset = "slow"
        audio_bitrate = "64k"
    else:
        crf = "20"
        preset = "veryfast"
        audio_bitrate = "128k"

    cmd += [
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


# -----------------------------
# Logging + UI busy dialog
# -----------------------------

class FileLogger:
    """
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
        self.top.geometry("680x220")
        self.top.transient(root)
        # self.top.grab_set()  # WSL alatt problémás
        self.top.focus_force()


        self.label_var = StringVar(value="Indítás…")
        Label(self.top, textvariable=self.label_var, wraplength=660, justify="left").pack(anchor="w", padx=12, pady=(12, 6))

        self.progress = ttk.Progressbar(self.top, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12, pady=6)

        self.detail_var = StringVar(value="")
        Label(self.top, textvariable=self.detail_var, wraplength=660, justify="left").pack(anchor="w", padx=12)

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
# Map overlay video (OSM tiles)
# -----------------------------

TILE_SIZE = 256
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_CACHE_DIR = SCRIPT_DIR / "tiles_cache"
TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
USER_AGENT = "mivue-map-overlay/1.0 (personal use)"


def latlon_to_global_px(lat: float, lon: float, z: int) -> tuple[float, float]:
    siny = math.sin(lat * math.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)
    x = (lon + 180.0) / 360.0
    y = 0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)
    scale = TILE_SIZE * (2 ** z)
    return x * scale, y * scale


def global_px_to_tile_xy(px: float, py: float) -> tuple[int, int]:
    return int(px // TILE_SIZE), int(py // TILE_SIZE)


def fetch_tile(z: int, x: int, y: int, log_cb) -> Path:
    n = 2 ** z
    x_wrapped = x % n
    if y < 0 or y >= n:
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
    import numpy as np
    from PIL import Image

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
                continue
            samples.append((t_rel, lat, lon))

    return samples


def parse_nmea_tracks_combined(nmea_paths: list[Path], segment_durations: list[float]) -> list[tuple[float, float, float]]:
    if len(nmea_paths) != len(segment_durations):
        raise ValueError("nmea_paths és segment_durations hossza nem egyezik")

    combined: list[tuple[float, float, float]] = []
    offset = 0.0

    for nmea, seg_dur in zip(nmea_paths, segment_durations):
        track = parse_nmea_track(nmea)
        for t, lat, lon in track:
            combined.append((t + offset, lat, lon))
        offset += float(seg_dur)

    combined.sort(key=lambda x: x[0])
    return combined


def smooth_track_ema(track: list[tuple[float, float, float]], alpha: float) -> list[tuple[float, float, float]]:
    if not track:
        return track
    out: list[tuple[float, float, float]] = []
    t0, lat0, lon0 = track[0]
    s_lat, s_lon = lat0, lon0
    out.append((t0, s_lat, s_lon))
    for t, lat, lon in track[1:]:
        s_lat = (1 - alpha) * s_lat + alpha * lat
        s_lon = (1 - alpha) * s_lon + alpha * lon
        out.append((t, s_lat, s_lon))
    return out


def interp_position(track: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    if not track:
        raise ValueError("Üres track")

    if t <= track[0][0]:
        return track[0][1], track[0][2]
    if t >= track[-1][0]:
        return track[-1][1], track[-1][2]

    for i in range(len(track) - 1):
        t0, lat0, lon0 = track[i]
        t1, lat1, lon1 = track[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return lat0, lon0
            a = (t - t0) / (t1 - t0)
            return lat0 + a * (lat1 - lat0), lon0 + a * (lon1 - lon0)

    return track[-1][1], track[-1][2]


def choose_zoom_for_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float, max_tiles: int = 8) -> int:
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


def compute_pip_size(ffprobe: str, front_mp4: Path, rear_mp4: Path, reduced: bool) -> tuple[int, int]:
    front_dims = ffprobe_video_dims(ffprobe, front_mp4)
    rear_dims = ffprobe_video_dims(ffprobe, rear_mp4)
    if not front_dims or not rear_dims:
        return 480, 270

    fw, fh = front_dims
    rw, rh = rear_dims

    if reduced:
        main_h = 720
        main_w = ensure_even(int(round((fw / fh) * main_h)))
    else:
        main_w = fw

    pip_w = ensure_even(int(round(main_w / 4)))
    pip_h = ensure_even(int(round(pip_w * (rh / rw))))
    pip_w = max(64, pip_w)
    pip_h = max(64, pip_h)
    return pip_w, pip_h


def render_map_video_from_track(
    ffmpeg: str,
    track: list[tuple[float, float, float]],
    duration_s: float,
    out_mp4: Path,
    log_cb,
    progress_cb,
    pip_w: int,
    pip_h: int,
    fps: int = 5,
    follow: bool = True,
    gps_ema_alpha: float = 0.25,
    camera_smooth_alpha: float = 0.18,
):
    if duration_s <= 0:
        raise ValueError("duration_s <= 0")
    if len(track) < 2:
        raise ValueError("Kevés GPS pont (track < 2).")

    # GPS smoothing
    if gps_ema_alpha is not None and gps_ema_alpha > 0:
        track = smooth_track_ema(track, gps_ema_alpha)
        log_cb(f"[map] GPS EMA smoothing enabled alpha={gps_ema_alpha}")

    lats = [lat for _, lat, _ in track]
    lons = [lon for _, _, lon in track]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    z = choose_zoom_for_bbox(lat_min, lat_max, lon_min, lon_max, max_tiles=8)
    log_cb(f"[map] zoom={z} bbox=({lat_min},{lon_min})..({lat_max},{lon_max})")

    # Background bounds with margin
    px_min, py_min = latlon_to_global_px(lat_max, lon_min, z)
    px_max, py_max = latlon_to_global_px(lat_min, lon_max, z)

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

    tx1, ty1 = global_px_to_tile_xy(gpx_left, gpy_top)
    tx2, ty2 = global_px_to_tile_xy(gpx_right, gpy_bottom)
    x_min_t, x_max_t = min(tx1, tx2), max(tx1, tx2)
    y_min_t, y_max_t = min(ty1, ty2), max(ty1, ty2)

    tiles_w = (x_max_t - x_min_t + 1)
    tiles_h = (y_max_t - y_min_t + 1)
    log_cb(f"[map] tiles {tiles_w}x{tiles_h} (z={z}) cache_dir={TILE_CACHE_DIR}")

    try:
        bg = stitch_tiles(z, x_min_t, x_max_t, y_min_t, y_max_t, log_cb)
    except ModuleNotFoundError as e:
        raise RuntimeError("Hiányzik a numpy vagy pillow. Telepítés: pip install numpy pillow") from e

    bg_h, bg_w, _ = bg.shape

    def to_bg_px(lat: float, lon: float) -> tuple[float, float]:
        gpx, gpy = latlon_to_global_px(lat, lon, z)
        return (gpx - x_min_t * TILE_SIZE), (gpy - y_min_t * TILE_SIZE)

    route_px = [to_bg_px(lat, lon) for _, lat, lon in track]

    workdir = Path(tempfile.mkdtemp(prefix="mivue_map_"))
    frame_dir = workdir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(math.ceil(duration_s * fps))
    log_cb(f"[map] render {total_frames} frames @ {fps}fps, size={pip_w}x{pip_h}, follow={follow}")

    from PIL import Image, ImageDraw

    sm_cx = None
    sm_cy = None

    if not follow:
        xs = [p[0] for p in route_px]
        ys = [p[1] for p in route_px]
        fixed_cx = (min(xs) + max(xs)) / 2
        fixed_cy = (min(ys) + max(ys)) / 2
    else:
        fixed_cx = fixed_cy = None

    for i in range(total_frames):
        t = i / fps
        lat, lon = interp_position(track, t)
        dot_x, dot_y = to_bg_px(lat, lon)

        if sm_cx is None:
            sm_cx, sm_cy = dot_x, dot_y
        else:
            sm_cx = (1 - camera_smooth_alpha) * sm_cx + camera_smooth_alpha * dot_x
            sm_cy = (1 - camera_smooth_alpha) * sm_cy + camera_smooth_alpha * dot_y

        if follow:
            cx, cy = sm_cx, sm_cy
        else:
            cx, cy = fixed_cx, fixed_cy

        left = int(round(cx - pip_w / 2))
        top = int(round(cy - pip_h / 2))
        left = max(0, min(left, bg_w - pip_w))
        top = max(0, min(top, bg_h - pip_h))

        crop = bg[top:top + pip_h, left:left + pip_w, :]
        img = Image.fromarray(crop)
        draw = ImageDraw.Draw(img, "RGBA")

        pts = []
        for x, y in route_px:
            rx = x - left
            ry = y - top
            if 0 <= rx < pip_w and 0 <= ry < pip_h:
                pts.append((rx, ry))
        if len(pts) >= 2:
            draw.line(pts, fill=(30, 90, 200, 220), width=3)

        dx = dot_x - left
        dy = dot_y - top
        r = 7
        draw.ellipse((dx - r, dy - r, dx + r, dy + r), fill=(20, 120, 255, 255))

        frame_path = frame_dir / f"frame_{i:06d}.png"
        img.save(frame_path)

        if (i % max(1, total_frames // 100)) == 0 or i == total_frames - 1:
            progress_cb((i + 1) / total_frames)

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
# App
# -----------------------------

I18N = {
    "hu": {
        "app_title": "MiVue 798D + A30 – Egyszerű Viewer/Exporter",
        "error_title": "Hiba",
        "info_title": "Info",
        "done_title": "Kész",
        "missing_ffmpeg": "Nem találom az ffmpeg-et a PATH-ban. Telepítsd: sudo apt install ffmpeg",
        "missing_ffprobe": "Nem találom az ffprobe-ot a PATH-ban. Telepítsd: sudo apt install ffmpeg",
        "initial_status": "Válaszd ki a Normal mappát. Log: {log}",
        "btn_pick_dir": "Mappa (Normal/)",
        "btn_map_video": "Térkép videó (kijelölt)",
        "btn_map_html": "Térkép HTML (kijelölt)",
        "btn_export": "Egyesítés (több clip → 1 MP4 + 1 NMEA)",
        "quality_label": "Minőség:",
        "quality_native": "Nativ",
        "quality_reduced": "Csökkentett (720p + strong)",
        "map_overlay_checkbox": "Map overlay exportnál",
        "language_label": "Nyelv / Language:",
        "language_hu": "Magyar",
        "language_en": "English",
        "bottom_hint": "✅ exportálható. ⚠️ hiányos. Rear PiP: main_w/4. Map video: OSM tiles + 5fps. Log: ./logs/latest.log",
        "pick_normal_title": "Válaszd ki a Normal mappát",
        "pick_output_title": "Hova mentsem a végső eredményt?",
        "status_scanning": "Beolvasás… (fájlok + hossz ffprobe)",
        "status_loaded": "Betöltve: {base_dir} | Clip-ek: {count} | Össz-idő(ismert): {duration} | Log: {log}",
        "status_export_overlay": "Export: overlay {i}/{total} … | Log: {log}",
        "status_export_concat": "Export: concat… | Log: {log}",
        "status_export_nmea": "Export: NMEA… | Log: {log}",
        "status_done": "Kész! {mp4} + {nmea} | Log: {log}",
        "msg_select_at_least_one": "Jelölj ki legalább 1 elemet.",
        "msg_no_front_nmea": "A kijelöltek között nincs Front .NMEA.",
        "msg_need_complete_clips_for_map": "Ehhez válassz ✅ clip(ek)et (front MP4 + rear MP4 + front NMEA).",
        "msg_no_exportable_clip": "A kijelöltek között nincs exportálható (✅) clip.",
        "msg_unknown_duration": "Nem tudom a hosszát: {path}",
        "msg_map_video_done": "Térképes videó elkészült:\n{out_mp4}\n\nLog: {log}",
        "msg_export_saved": "Mentve:\n{out_mp4}\n{out_nmea}\n\nLog: {log}",
        "busy_map_title": "Térkép videó – dolgozom…",
        "busy_map_label": "Térképes videó készítése (OSM) – részletek a log fájlban",
        "busy_map_detail": "Szegmensek: {segments} | Méret: {w}x{h} | FPS: 5 | Hossz: {duration}",
        "busy_scan_title": "Beolvasás – dolgozom…",
        "busy_scan_label": "Könyvtár beolvasása (fájlok + ffprobe hossz)",
        "busy_scan_detail": "Előkészítés…",
        "busy_scan_detail_current": "Aktuális: {detail}",
        "busy_export_title": "Export – dolgozom…",
        "busy_export_label": "Export folyamat… (részletek a log fájlban)",
        "busy_export_detail": "Mód: {mode} | Map overlay: {overlay}",
        "busy_export_mode_native": "Nativ",
        "busy_export_mode_reduced": "Csökkentett 720p",
        "busy_generating_map": "Map videó generálás (teljes út)…",
        "busy_segment": "Szegmens: {i}/{total} (rear+map overlay)",
        "busy_concat": "Összefűzés (concat)…",
        "busy_nmea": "NMEA összefűzés…",
        "clip_row": "{marker} {key} | dur:{dur} | Fmp4:{fmp4} Rmp4:{rmp4} Fnmea:{fnmea}",
    },
    "en": {
        "app_title": "MiVue 798D + A30 - Simple Viewer/Exporter",
        "error_title": "Error",
        "info_title": "Info",
        "done_title": "Done",
        "missing_ffmpeg": "ffmpeg was not found in PATH. Install it: sudo apt install ffmpeg",
        "missing_ffprobe": "ffprobe was not found in PATH. Install it: sudo apt install ffmpeg",
        "initial_status": "Select the Normal folder. Log: {log}",
        "btn_pick_dir": "Folder (Normal/)",
        "btn_map_video": "Map video (selected)",
        "btn_map_html": "Map HTML (selected)",
        "btn_export": "Merge (multiple clips -> 1 MP4 + 1 NMEA)",
        "quality_label": "Quality:",
        "quality_native": "Native",
        "quality_reduced": "Reduced (720p + strong)",
        "map_overlay_checkbox": "Map overlay during export",
        "language_label": "Language:",
        "language_hu": "Hungarian",
        "language_en": "English",
        "bottom_hint": "✅ exportable. ⚠️ incomplete. Rear PiP: main_w/4. Map video: OSM tiles + 5fps. Log: ./logs/latest.log",
        "pick_normal_title": "Select the Normal folder",
        "pick_output_title": "Where should I save the final output?",
        "status_scanning": "Scanning... (files + duration via ffprobe)",
        "status_loaded": "Loaded: {base_dir} | Clips: {count} | Total known duration: {duration} | Log: {log}",
        "status_export_overlay": "Export: overlay {i}/{total} ... | Log: {log}",
        "status_export_concat": "Export: concat... | Log: {log}",
        "status_export_nmea": "Export: NMEA... | Log: {log}",
        "status_done": "Done! {mp4} + {nmea} | Log: {log}",
        "msg_select_at_least_one": "Select at least 1 item.",
        "msg_no_front_nmea": "No front .NMEA found in the selection.",
        "msg_need_complete_clips_for_map": "Select ✅ clips (front MP4 + rear MP4 + front NMEA).",
        "msg_no_exportable_clip": "No exportable (✅) clip in the selection.",
        "msg_unknown_duration": "Cannot determine duration: {path}",
        "msg_map_video_done": "Map video created:\n{out_mp4}\n\nLog: {log}",
        "msg_export_saved": "Saved:\n{out_mp4}\n{out_nmea}\n\nLog: {log}",
        "busy_map_title": "Map video - working...",
        "busy_map_label": "Creating map video (OSM) - see details in the log file",
        "busy_map_detail": "Segments: {segments} | Size: {w}x{h} | FPS: 5 | Duration: {duration}",
        "busy_scan_title": "Scanning - working...",
        "busy_scan_label": "Scanning folder (files + ffprobe duration)",
        "busy_scan_detail": "Preparing...",
        "busy_scan_detail_current": "Current: {detail}",
        "busy_export_title": "Export - working...",
        "busy_export_label": "Export in progress... (details in the log file)",
        "busy_export_detail": "Mode: {mode} | Map overlay: {overlay}",
        "busy_export_mode_native": "Native",
        "busy_export_mode_reduced": "Reduced 720p",
        "busy_generating_map": "Generating map video (full route)...",
        "busy_segment": "Segment: {i}/{total} (rear+map overlay)",
        "busy_concat": "Concatenating...",
        "busy_nmea": "Concatenating NMEA...",
        "clip_row": "{marker} {key} | dur:{dur} | Fmp4:{fmp4} Rmp4:{rmp4} Fnmea:{fnmea}",
    },
}

class App:
    def __init__(self, root: Tk):
        self.root = root
        self.lang = StringVar(value="en")
        self.root.title(self.tr("app_title"))

        self.ffmpeg = which("ffmpeg")
        self.ffprobe = which("ffprobe")

        if not self.ffmpeg:
            messagebox.showerror(self.tr("error_title"), self.tr("missing_ffmpeg"))
            root.destroy()
            return
        if not self.ffprobe:
            messagebox.showerror(self.tr("error_title"), self.tr("missing_ffprobe"))
            root.destroy()
            return

        self.file_logger = FileLogger()
        self.base_dir: Path | None = None
        self.pairs: list[ClipPair] = []
        self.total_known_duration = 0.0
        self.ui_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._poll_ui_queue()

        top = Frame(root)
        top.pack(fill="x", padx=10, pady=10)

        self.status = StringVar(value=self.tr("initial_status", log=self.file_logger.latest_path))
        Label(top, textvariable=self.status, wraplength=980, justify="left").pack(anchor="w")

        btns = Frame(root)
        btns.pack(fill="x", padx=10, pady=5)

        self.btn_pick_dir = Button(btns, command=self.pick_dir)
        self.btn_pick_dir.pack(side="left")
        self.btn_map_video = Button(btns, command=self.make_map_video_for_selection)
        self.btn_map_video.pack(side="left", padx=10)
        self.btn_map_html = Button(btns, command=self.open_map_for_selection)
        self.btn_map_html.pack(side="left", padx=10)
        self.btn_export = Button(btns, command=self.export_selection)
        self.btn_export.pack(side="left", padx=10)

        self.quality_mode = StringVar(value="native")
        self.lbl_quality = Label(btns)
        self.lbl_quality.pack(side="left", padx=(18, 4))
        self.rb_native = Radiobutton(btns, variable=self.quality_mode, value="native")
        self.rb_native.pack(side="left")
        self.rb_reduced = Radiobutton(btns, variable=self.quality_mode, value="reduced")
        self.rb_reduced.pack(side="left", padx=6)

        self.map_overlay_enabled = BooleanVar(value=True)
        self.chk_map_overlay = ttk.Checkbutton(btns, variable=self.map_overlay_enabled)
        self.chk_map_overlay.pack(side="left", padx=(18, 0))

        self.lang_flags = Frame(btns)
        self.lang_flags.pack(side="left", padx=(18, 0))
        self.btn_lang_hu = Button(self.lang_flags, text="🇭🇺", command=lambda: self.set_language("hu"), width=3)
        self.btn_lang_hu.pack(side="left")
        self.btn_lang_en = Button(self.lang_flags, text="🇬🇧", command=lambda: self.set_language("en"), width=3)
        self.btn_lang_en.pack(side="left", padx=4)

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
        self.lbl_hint = Label(bottom, wraplength=980, justify="left")
        self.lbl_hint.pack(anchor="w")

        self.refresh_texts()

        self.log("=== MiVue GUI started ===")
        self.log(f"Session log: {self.file_logger.session_path}")
        self.log(f"Latest log:  {self.file_logger.latest_path}")
        self.log(f"ffmpeg: {self.ffmpeg}")
        self.log(f"ffprobe:{self.ffprobe}")

    def tr(self, msg_id: str, **kwargs) -> str:
        lang = self.lang.get() if self.lang.get() in I18N else "hu"
        text = I18N[lang].get(msg_id, msg_id)
        return text.format(**kwargs) if kwargs else text

    def tr_err(self, msg: str) -> str:
        if self.lang.get() != "en":
            return msg
        replacements = [
            (
                "A kiválasztott mappában nem találom az F/ és R/ könyvtárakat (Normal/F és Normal/R).",
                "The selected folder does not contain F/ and R/ directories (Normal/F and Normal/R).",
            ),
            (
                "Nem találtam használható $GPRMC koordinátákat ebben a .NMEA fájlban.",
                "No usable $GPRMC coordinates were found in this .NMEA file.",
            ),
            ("Nem tudom a hosszát: ", "Cannot determine duration: "),
            ("Kevés GPS pont a kombinált track-ben.", "Too few GPS points in the combined track."),
            ("Kevés GPS pont a kombinált track-ben (export map).", "Too few GPS points in the combined track (export map)."),
            ("Nem sikerült letölteni tile-t:", "Failed to download tile:"),
            (
                "Hiányzik a numpy vagy pillow. Telepítés: pip install numpy pillow",
                "numpy or pillow is missing. Install: pip install numpy pillow",
            ),
            ("ffmpeg overlay hiba", "ffmpeg overlay error"),
            ("ffmpeg concat hiba", "ffmpeg concat error"),
            ("ffmpeg map encode hiba", "ffmpeg map encode error"),
        ]
        out = msg
        for hu, en in replacements:
            out = out.replace(hu, en)
        return out

    def set_language(self, lang: str):
        if lang not in I18N:
            return
        if self.lang.get() == lang:
            return
        self.lang.set(lang)
        self.on_language_change()

    def on_language_change(self):
        self.refresh_texts()
        self.refresh_status()

    def refresh_texts(self):
        self.root.title(self.tr("app_title"))
        self.btn_pick_dir.config(text=self.tr("btn_pick_dir"))
        self.btn_map_video.config(text=self.tr("btn_map_video"))
        self.btn_map_html.config(text=self.tr("btn_map_html"))
        self.btn_export.config(text=self.tr("btn_export"))
        self.lbl_quality.config(text=self.tr("quality_label"))
        self.rb_native.config(text=self.tr("quality_native"))
        self.rb_reduced.config(text=self.tr("quality_reduced"))
        self.chk_map_overlay.config(text=self.tr("map_overlay_checkbox"))
        self.lbl_hint.config(text=self.tr("bottom_hint"))
        self._refresh_language_flag_state()

    def _refresh_language_flag_state(self):
        selected = self.lang.get()
        self.btn_lang_hu.config(relief="sunken" if selected == "hu" else "raised")
        self.btn_lang_en.config(relief="sunken" if selected == "en" else "raised")

    def refresh_status(self):
        if self.base_dir is None:
            self.set_status(self.tr("initial_status", log=self.file_logger.latest_path))
            return
        self.set_status(
            self.tr(
                "status_loaded",
                base_dir=self.base_dir,
                count=len(self.pairs),
                duration=fmt_duration(self.total_known_duration),
                log=self.file_logger.latest_path,
            )
        )

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
        d = filedialog.askdirectory(title=self.tr("pick_normal_title"))
        if not d:
            return
        self.base_dir = Path(d)
        self.set_status(self.tr("status_scanning"))
        self.root.update_idletasks()

        busy = BusyDialog(self.root, title=self.tr("busy_scan_title"))
        busy.set_label(self.tr("busy_scan_label"))
        busy.set_detail(self.tr("busy_scan_detail"))
        busy.set_progress(0)

        def on_progress(done: int, total: int):
            self.root.after(0, lambda d=done, t=total: busy.set_progress((d / max(1, t)) * 100.0))

        def on_detail(detail: str):
            self.root.after(0, lambda txt=detail: busy.set_detail(self.tr("busy_scan_detail_current", detail=txt)))

        def worker():
            try:
                pairs = scan_pairs(self.base_dir, self.ffprobe, progress_cb=on_progress, detail_cb=on_detail)
                total_known = sum((p.duration_s or 0.0) for p in pairs)

                def finish_ok():
                    try:
                        self.pairs = pairs
                        self.listbox.delete(0, END)
                        for pair in self.pairs:
                            marker = "✅" if pair.complete() else "⚠️"
                            dur = fmt_duration(pair.duration_s)
                            self.listbox.insert(
                                END,
                                self.tr(
                                    "clip_row",
                                    marker=marker,
                                    key=pair.key,
                                    dur=dur,
                                    fmp4="Y" if pair.front_mp4 else "N",
                                    rmp4="Y" if pair.rear_mp4 else "N",
                                    fnmea="Y" if pair.front_nmea else "N",
                                ),
                            )

                        self.total_known_duration = total_known
                        self.refresh_status()
                        self.log(f"Selected Normal dir: {self.base_dir}")
                    finally:
                        busy.close()

                self.root.after(0, finish_ok)
            except Exception as e:
                err = str(e)
                self.log(f"Pick dir error: {err}")
                self.root.after(
                    0,
                    lambda: (
                        busy.close(),
                        messagebox.showerror(self.tr("error_title"), self.tr_err(err)),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def get_selected_pairs(self) -> list[ClipPair]:
        if not self.pairs:
            return []
        idxs = list(self.listbox.curselection())
        return [self.pairs[i] for i in idxs] if idxs else []

    def open_map_for_selection(self):
        sel = self.get_selected_pairs()
        if not sel:
            messagebox.showinfo(self.tr("info_title"), self.tr("msg_select_at_least_one"))
            return

        nmeas = [p.front_nmea for p in sel if p.front_nmea]
        if not nmeas:
            messagebox.showerror(self.tr("error_title"), self.tr("msg_no_front_nmea"))
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
            messagebox.showerror(self.tr("error_title"), self.tr_err(str(e)))

    def make_map_video_for_selection(self):
        sel = self.get_selected_pairs()
        if not sel:
            messagebox.showinfo(self.tr("info_title"), self.tr("msg_select_at_least_one"))
            return

        clips = sorted([p for p in sel if p.complete()], key=lambda x: x.key)
        if not clips:
            messagebox.showerror(self.tr("error_title"), self.tr("msg_need_complete_clips_for_map"))
            return

        reduced = (self.quality_mode.get() == "reduced")
        follow = True

        pip_w, pip_h = compute_pip_size(self.ffprobe, clips[0].front_mp4, clips[0].rear_mp4, reduced=reduced)

        nmeas: list[Path] = []
        durations: list[float] = []
        for c in clips:
            dur = c.duration_s or ffprobe_duration(self.ffprobe, c.front_mp4) or 0.0
            if dur <= 0:
                messagebox.showerror(self.tr("error_title"), self.tr("msg_unknown_duration", path=c.front_mp4))
                return
            durations.append(dur)
            nmeas.append(c.front_nmea)

        total_dur = sum(durations)

        start_key = clips[0].key
        end_key = clips[-1].key
        mode_tag = "REDUCED_720P" if reduced else "NATIVE"
        follow_tag = "FOLLOW" if follow else "FIXED"
        out_mp4 = SCRIPT_DIR / f"MAP_{mode_tag}_{start_key}_to_{end_key}_{follow_tag}_{pip_w}x{pip_h}.mp4"

        busy = BusyDialog(self.root, title=self.tr("busy_map_title"))
        busy.set_label(self.tr("busy_map_label"))
        busy.set_detail(self.tr("busy_map_detail", segments=len(clips), w=pip_w, h=pip_h, duration=fmt_duration(total_dur)))
        busy.set_progress(0)

        self.log("=== Map video start (multi-segment) ===")
        self.log(f"Segments: {len(clips)}")
        self.log(f"Mode: {'reduced' if reduced else 'native'} follow={follow}")
        self.log(f"Output map mp4: {out_mp4}")

        def worker():
            try:
                track = parse_nmea_tracks_combined(nmeas, durations)
                if len(track) < 2:
                    raise ValueError("Kevés GPS pont a kombinált track-ben.")

                def progress_cb(ratio: float):
                    self.root.after(0, lambda r=ratio: busy.set_progress(r * 100))

                pip_w, pip_h = compute_pip_size(self.ffprobe, clips[0].front_mp4, clips[0].rear_mp4, reduced=reduced)

                render_map_video_from_track(
                    ffmpeg=self.ffmpeg,
                    track=track,
                    duration_s=total_dur,
                    out_mp4=out_mp4,
                    log_cb=self.log,
                    progress_cb=progress_cb,
                    pip_w=pip_w,
                    pip_h=pip_h,
                    fps=5,
                    follow=follow,
                    gps_ema_alpha=0.25,
                    camera_smooth_alpha=0.18,
                )

                self.log("=== Map video done (multi-segment) ===")
                self.root.after(0, lambda: (busy.close(), messagebox.showinfo(self.tr("done_title"), self.tr("msg_map_video_done", out_mp4=out_mp4, log=self.file_logger.latest_path))))
            except Exception as e:
                err = str(e)
                self.log(f"!!! Map video error (multi): {err}")
                self.root.after(0, lambda err=err: (busy.close(), messagebox.showerror(self.tr("error_title"), f"{self.tr_err(err)}\n\nLog: {self.file_logger.latest_path}")))

        threading.Thread(target=worker, daemon=True).start()

    def export_selection(self):
        sel = self.get_selected_pairs()
        if not sel:
            messagebox.showinfo(self.tr("info_title"), self.tr("msg_select_at_least_one"))
            return

        clips = sorted([p for p in sel if p.complete()], key=lambda x: x.key)
        if not clips:
            messagebox.showerror(self.tr("error_title"), self.tr("msg_no_exportable_clip"))
            return

        out_dir = filedialog.askdirectory(title=self.tr("pick_output_title"))
        if not out_dir:
            return
        out_dir = Path(out_dir)

        reduced = (self.quality_mode.get() == "reduced")
        want_map = bool(self.map_overlay_enabled.get())

        start_key = clips[0].key
        end_key = clips[-1].key
        mode_tag = "REDUCED_720P" if reduced else "NATIVE"
        out_mp4 = out_dir / f"COMBINED_{mode_tag}_{start_key}_to_{end_key}.mp4"
        out_nmea = out_dir / f"COMBINED_{mode_tag}_{start_key}_to_{end_key}.nmea"

        busy = BusyDialog(self.root, title=self.tr("busy_export_title"))
        busy.set_label(self.tr("busy_export_label"))
        mode_name = self.tr("busy_export_mode_reduced") if reduced else self.tr("busy_export_mode_native")
        busy.set_detail(self.tr("busy_export_detail", mode=mode_name, overlay="ON" if want_map else "OFF"))
        self.root.update_idletasks()

        self.log("=== Export start ===")
        self.log(f"Mode: {'reduced(720p+strong)' if reduced else 'native'}")
        self.log(f"Map overlay export: {want_map}")
        self.log(f"Output MP4: {out_mp4}")
        self.log(f"Output NMEA: {out_nmea}")

        def worker():
            try:
                workdir = Path(tempfile.mkdtemp(prefix="mivue_"))
                segments: list[Path] = []
                total_segments = len(clips)

                # Compute durations & offsets
                durations: list[float] = []
                offsets: list[float] = []
                acc = 0.0
                for c in clips:
                    dur = c.duration_s or ffprobe_duration(self.ffprobe, c.front_mp4) or 0.0
                    if dur <= 0:
                        raise RuntimeError(self.tr("msg_unknown_duration", path=c.front_mp4))
                    offsets.append(acc)
                    durations.append(dur)
                    acc += dur
                total_dur = acc

                # Create map video ONCE for full duration (optional)
                map_mp4: Path | None = None
                if want_map:
                    self.root.after(0, lambda: (busy.set_progress(0), busy.set_detail(self.tr("busy_generating_map"))))
                    pip_w, pip_h = compute_pip_size(self.ffprobe, clips[0].front_mp4, clips[0].rear_mp4, reduced=reduced)
                    map_mp4 = workdir / f"map_full_{mode_tag}_{start_key}_to_{end_key}_{pip_w}x{pip_h}.mp4"

                    self.log("=== Export map video start ===")
                    nmeas = [c.front_nmea for c in clips if c.front_nmea]
                    track = parse_nmea_tracks_combined(nmeas, durations)
                    if len(track) < 2:
                        raise RuntimeError("Kevés GPS pont a kombinált track-ben (export map).")

                    def map_progress_cb(ratio: float):
                        # map stage takes 0..15%
                        self.root.after(0, lambda r=ratio: busy.set_progress(r * 15.0))

                    render_map_video_from_track(
                        ffmpeg=self.ffmpeg,
                        track=track,
                        duration_s=total_dur,
                        out_mp4=map_mp4,
                        log_cb=self.log,
                        progress_cb=map_progress_cb,
                        pip_w=pip_w,
                        pip_h=pip_h,
                        fps=5,
                        follow=True,
                        gps_ema_alpha=0.25,
                        camera_smooth_alpha=0.18,
                    )
                    self.log(f"=== Export map video done: {map_mp4} ===")

                def make_progress_cb(done_segments: int):
                    def _cb(seg_ratio: float):
                        # overlay stage occupies 15..95%
                        overall = (done_segments + seg_ratio) / max(1, total_segments)
                        pct = 15.0 + overall * 80.0
                        self.root.after(0, lambda p=pct: busy.set_progress(p))
                    return _cb

                for i, pair in enumerate(clips, start=1):
                    done = i - 1
                    self.set_status(self.tr("status_export_overlay", i=i, total=total_segments, log=self.file_logger.latest_path))
                    self.log(f"--- Overlay {i}/{total_segments}: {pair.key} ---")

                    self.root.after(0, lambda i=i, ts=total_segments: busy.set_detail(self.tr("busy_segment", i=i, total=ts)))

                    seg_out = workdir / f"seg_{i:04d}_{pair.key}.mp4"

                    overlay_segment(
                        self.ffmpeg,
                        pair.front_mp4,
                        pair.rear_mp4,
                        seg_out,
                        self.log,
                        duration_s=durations[i - 1],
                        reduced=reduced,
                        progress_cb=make_progress_cb(done),
                        map_mp4=map_mp4,
                        map_offset_s=offsets[i - 1],
                        map_duration_s=durations[i - 1] if map_mp4 else None,
                        pip_w=pip_w,
                        pip_h=pip_h,
                    )

                    segments.append(seg_out)

                self.set_status(self.tr("status_export_concat", log=self.file_logger.latest_path))
                self.root.after(0, lambda: (busy.set_progress(96), busy.set_detail(self.tr("busy_concat"))))
                concat_mp4(self.ffmpeg, segments, out_mp4, self.log)

                self.set_status(self.tr("status_export_nmea", log=self.file_logger.latest_path))
                self.root.after(0, lambda: (busy.set_progress(98), busy.set_detail(self.tr("busy_nmea"))))
                concat_nmea([p.front_nmea for p in clips if p.front_nmea], out_nmea)

                self.root.after(0, lambda: busy.set_progress(100))
                self.set_status(self.tr("status_done", mp4=out_mp4.name, nmea=out_nmea.name, log=self.file_logger.latest_path))
                self.log("=== Export done ===")

                self.root.after(0, lambda: (busy.close(), messagebox.showinfo(self.tr("done_title"), self.tr("msg_export_saved", out_mp4=out_mp4, out_nmea=out_nmea, log=self.file_logger.latest_path))))

            except Exception as e:
                err = str(e)
                self.log(f"!!! Export error: {err}")
                self.root.after(0, lambda err=err: (busy.close(), messagebox.showerror(self.tr("error_title"), f"{self.tr_err(err)}\n\nLog: {self.file_logger.latest_path}")))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

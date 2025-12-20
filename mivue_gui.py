#!/usr/bin/env python3
import re
import subprocess
import tempfile
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from tkinter import (
    Tk, Frame, Button, Label, Listbox, Scrollbar, END,
    filedialog, messagebox, StringVar, Toplevel, Radiobutton
)
from tkinter import ttk

import folium

PAIR_RE = re.compile(r"^FILE(\d{6})-(\d{6})([FR])\.(MP4|NMEA)$", re.IGNORECASE)
SCRIPT_DIR = Path(__file__).resolve().parent

# Parse ffmpeg progress lines: frame= ... time=00:04:59.96 bitrate=...
TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


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


def open_in_browser(path: Path) -> None:
    try:
        p = subprocess.run(["xdg-open", str(path)], capture_output=True, text=True)
        if p.returncode == 0:
            return
    except Exception:
        pass
    subprocess.Popen(["firefox", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
        if ext == "MP4":
            pair.front_mp4 = p
        elif ext == "NMEA":
            pair.front_nmea = p

    # Rear: MP4 only
    for p in r_dir.iterdir():
        if not p.is_file():
            continue
        m = PAIR_RE.match(p.name)
        if not m:
            continue
        yymmdd, hhmmss, cam, ext = m.group(1), m.group(2), m.group(3).upper(), m.group(4).upper()
        if cam != "R" or ext != "MP4":
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
    rear PiP: scaled to 1/4 width
    """
    if reduced:
        # scale front to 720p height, keep aspect (width divisible by 2)
        filter_complex = (
            "[0:v]scale=-2:720[front720];"
            "[1:v]scale=main_w/4:-1[rear];"
            "[front720][rear]overlay=10:10:format=auto[v]"
        )
        crf = "26"
        preset = "slow"
        audio_bitrate = "64k"
    else:
        filter_complex = (
            "[1:v]scale=iw/4:-1[rear];"
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
        self.top.geometry("600x190")
        self.top.transient(root)
        self.top.grab_set()

        self.label_var = StringVar(value="Indítás…")
        Label(self.top, textvariable=self.label_var, wraplength=580, justify="left").pack(anchor="w", padx=12, pady=(12, 6))

        self.progress = ttk.Progressbar(self.top, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12, pady=6)

        self.detail_var = StringVar(value="")
        Label(self.top, textvariable=self.detail_var, wraplength=580, justify="left").pack(anchor="w", padx=12)

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
        Button(btns, text="Térkép (kijelölt)", command=self.open_map_for_selection).pack(side="left", padx=12)
        Button(btns, text="Egyesítés (több clip → 1 MP4 + 1 NMEA)", command=self.export_selection).pack(side="left", padx=12)

        # Quality selector (Native vs Reduced)
        self.quality_mode = StringVar(value="native")
        Label(btns, text="Minőség:").pack(side="left", padx=(20, 4))
        Radiobutton(btns, text="Nativ", variable=self.quality_mode, value="native").pack(side="left")
        Radiobutton(btns, text="Csökkentett (720p + erős tömörítés)", variable=self.quality_mode, value="reduced").pack(side="left", padx=6)

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
        Label(bottom, text="✅ exportálható. ⚠️ hiányos. (Hátsó kamera PiP: 1/4 méret)", wraplength=980, justify="left").pack(anchor="w")

        # initial log header
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
            self.log(f"Map error: {e}")
            messagebox.showerror("Hiba", str(e))

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
                        pct = overall * 90.0  # overlay phase = 0..90%
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

                # concat 90..97
                self.set_status(f"Export: concat… | Log: {self.file_logger.latest_path}")
                self.root.after(0, lambda: (busy.set_progress(92), busy.set_detail("Összefűzés (concat)…")))
                concat_mp4(self.ffmpeg, segments, out_mp4, self.log)
                self.root.after(0, lambda: busy.set_progress(97))

                # nmea 97..100
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

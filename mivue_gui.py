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
    filedialog, messagebox, StringVar, Toplevel, Text, Menu
)
from tkinter import ttk

import folium

PAIR_RE = re.compile(r"^FILE(\d{6})-(\d{6})([FR])\.(MP4|NMEA)$", re.IGNORECASE)
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


def run_and_stream(cmd: list[str], log_cb) -> int:
    log_cb(" ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        log_cb(line.rstrip("\n"))
    return p.wait()


def overlay_segment(ffmpeg: str, front_mp4: Path, rear_mp4: Path, out_mp4: Path, log_cb):
    filter_complex = (
        "[1:v]scale=iw/8:-1[rear];"
        "[0:v][rear]overlay=10:10:format=auto[v]"
    )

    cmd = [
        ffmpeg, "-y",
        "-i", str(front_mp4),
        "-i", str(rear_mp4),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(out_mp4)
    ]

    rc = run_and_stream(cmd, log_cb)
    if rc != 0:
        raise RuntimeError(f"ffmpeg overlay hiba (exit={rc}). Nézd a logot.")


def concat_mp4(ffmpeg: str, segments: list[Path], out_mp4: Path, log_cb):
    """
    Concatenate segments using concat demuxer (-c copy).
    IMPORTANT: Do NOT wrap paths in double quotes, ffmpeg may treat quotes as literal chars on concat demuxer.
    We write absolute paths, one per line.
    """
    list_file = out_mp4.with_suffix(".concat.txt")

    with list_file.open("w", newline="\n") as f:
        for seg in segments:
            # absolute path, no quotes
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

        # reset latest
        self.latest_path.write_text("", encoding="utf-8")

        self._lock = threading.Lock()

    def write(self, line: str):
        with self._lock:
            with self.latest_path.open("a", encoding="utf-8", errors="ignore") as f:
                f.write(line + "\n")
            with self.session_path.open("a", encoding="utf-8", errors="ignore") as f:
                f.write(line + "\n")


class LogWindow:
    def __init__(self, root: Tk):
        self.top = Toplevel(root)
        self.top.title("Log")
        self.top.geometry("920x420")

        container = Frame(self.top)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        self.text = Text(container, wrap="word")
        self.text.pack(side="left", fill="both", expand=True)

        sb = Scrollbar(container, command=self.text.yview)
        sb.pack(side="right", fill="y")
        self.text.config(yscrollcommand=sb.set)

        self.menu = Menu(self.top, tearoff=0)
        self.menu.add_command(label="Copy", command=self.copy_selection)
        self.menu.add_command(label="Select all", command=self.select_all)

        self.text.bind("<Button-3>", self._popup_menu)  # right click

        btns = Frame(self.top)
        btns.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left")
        ttk.Button(btns, text="Copy selection", command=self.copy_selection).pack(side="left", padx=6)
        ttk.Button(btns, text="Copy all", command=self.copy_all).pack(side="left", padx=6)
        ttk.Button(btns, text="Save log as…", command=self.save_as).pack(side="left", padx=6)
        ttk.Button(btns, text="Close", command=self.top.withdraw).pack(side="right")

        self.text.bind("<Control-a>", self._select_all)
        self.text.bind("<Control-A>", self._select_all)

    def _popup_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def write(self, msg: str):
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def clear(self):
        self.text.delete("1.0", "end")

    def select_all(self):
        self.text.tag_add("sel", "1.0", "end")

    def copy_selection(self):
        try:
            s = self.text.get("sel.first", "sel.last")
        except Exception:
            s = ""
        if not s:
            return
        self.top.clipboard_clear()
        self.top.clipboard_append(s)

    def copy_all(self):
        all_text = self.text.get("1.0", "end-1c")
        self.top.clipboard_clear()
        self.top.clipboard_append(all_text)

    def save_as(self):
        content = self.text.get("1.0", "end-1c")
        path = filedialog.asksaveasfilename(
            title="Log mentése",
            defaultextension=".log",
            filetypes=[("Log file", "*.log"), ("Text file", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        Path(path).write_text(content, encoding="utf-8", errors="ignore")

    def _select_all(self, _event=None):
        self.select_all()
        return "break"


class BusyDialog:
    def __init__(self, root: Tk, title="Dolgozom…"):
        self.top = Toplevel(root)
        self.top.title(title)
        self.top.geometry("520x140")
        self.top.transient(root)
        self.top.grab_set()

        self.label_var = StringVar(value="Indítás…")
        Label(self.top, textvariable=self.label_var, wraplength=500, justify="left").pack(anchor="w", padx=12, pady=(12, 8))

        self.progress = ttk.Progressbar(self.top, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12, pady=6)

        self.detail_var = StringVar(value="")
        Label(self.top, textvariable=self.detail_var, wraplength=500, justify="left").pack(anchor="w", padx=12)

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

        self.log_window = LogWindow(root)
        self.log_window.top.withdraw()

        self.ui_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._poll_ui_queue()

        top = Frame(root)
        top.pack(fill="x", padx=10, pady=10)

        self.status = StringVar(value="Válaszd ki a Normal mappát. Többet is kijelölhetsz (Ctrl/Shift).")
        Label(top, textvariable=self.status, wraplength=980, justify="left").pack(anchor="w")

        btns = Frame(root)
        btns.pack(fill="x", padx=10, pady=5)

        Button(btns, text="Mappa (Normal/)", command=self.pick_dir).pack(side="left")
        Button(btns, text="Log megnyitása", command=self.show_log).pack(side="left", padx=6)

        Button(btns, text="Térkép (kijelölt)", command=self.open_map_for_selection).pack(side="left", padx=18)
        Button(btns, text="Egyesítés (több clip → 1 MP4 + 1 NMEA)", command=self.export_selection).pack(side="left", padx=18)

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
        Label(
            bottom,
            text=f"✅ = exportálható. ⚠️ = hiányos. Log fájl: {self.file_logger.latest_path}",
            wraplength=980,
            justify="left"
        ).pack(anchor="w")

        # initial log header
        self.log(f"=== MiVue GUI started ===")
        self.log(f"Session log: {self.file_logger.session_path}")
        self.log(f"Latest log:  {self.file_logger.latest_path}")
        self.log(f"ffmpeg: {self.ffmpeg}")
        self.log(f"ffprobe:{self.ffprobe}")

    def show_log(self):
        self.log_window.top.deiconify()
        self.log_window.top.lift()

    def log(self, msg: str):
        # write to file immediately + to UI via queue
        self.file_logger.write(msg)
        self.ui_queue.put(("log", msg))

    def set_status(self, msg: str):
        self.ui_queue.put(("status", msg))

    def _poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self.log_window.write(payload)
                elif kind == "status":
                    self.status.set(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_ui_queue)

    def pick_dir(self):
        d = filedialog.askdirectory(title="Válaszd ki a Normal mappát")
        if not d:
            return
        try:
            self.base_dir = Path(d)
            self.set_status("Beolvasás… (fájlok + hossz lekérdezés ffprobe-bal)")
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
                f"Betöltve: {self.base_dir} | Clip-ek: {len(self.pairs)} | "
                f"Össz-idő (ismert): {fmt_duration(total_known)}"
            )
            self.log(f"Selected Normal dir: {self.base_dir}")

        except Exception as e:
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

        start_key = clips[0].key
        end_key = clips[-1].key
        out_mp4 = out_dir / f"COMBINED_{start_key}_to_{end_key}.mp4"
        out_nmea = out_dir / f"COMBINED_{start_key}_to_{end_key}.nmea"

        busy = BusyDialog(self.root, title="Export – dolgozom…")
        self.show_log()
        self.log("=== Export start ===")
        self.log(f"Output MP4: {out_mp4}")
        self.log(f"Output NMEA: {out_nmea}")

        def worker():
            try:
                workdir = Path(tempfile.mkdtemp(prefix="mivue_"))
                segments: list[Path] = []

                total_steps = len(clips) + 2
                step = 0

                for i, pair in enumerate(clips, start=1):
                    step += 1
                    pct = (step / total_steps) * 100.0
                    self.set_status(f"Export: overlay {i}/{len(clips)} …")
                    self.log(f"--- Overlay {i}/{len(clips)}: {pair.key} ---")
                    self.log(f"Front: {pair.front_mp4}")
                    self.log(f"Rear : {pair.rear_mp4}")

                    self.root.after(0, lambda p=pct, i=i: (busy.set_progress(p), busy.set_detail(f"Szegmens: {i}/{len(clips)}")))

                    seg_out = workdir / f"seg_{i:04d}_{pair.key}.mp4"
                    overlay_segment(self.ffmpeg, pair.front_mp4, pair.rear_mp4, seg_out, self.log)
                    segments.append(seg_out)

                step += 1
                pct = (step / total_steps) * 100.0
                self.set_status("Export: szegmensek összefűzése (concat)…")
                self.root.after(0, lambda p=pct: (busy.set_progress(p), busy.set_detail("Összefűzés (concat)…")))
                concat_mp4(self.ffmpeg, segments, out_mp4, self.log)

                step += 1
                pct = (step / total_steps) * 100.0
                self.set_status("Export: NMEA összefűzés…")
                self.root.after(0, lambda p=pct: (busy.set_progress(p), busy.set_detail("NMEA összefűzés…")))
                concat_nmea([p.front_nmea for p in clips if p.front_nmea], out_nmea)

                self.set_status(f"Kész! {out_mp4.name} + {out_nmea.name}")
                self.root.after(0, lambda: (busy.set_progress(100), busy.set_detail("Kész!")))

                self.log("=== Export done ===")
                self.root.after(0, lambda: (busy.close(), messagebox.showinfo("Kész", f"Mentve:\n{out_mp4}\n{out_nmea}\n\nLog: {self.file_logger.latest_path}")))

            except Exception as e:
                err = str(e)
                self.log(f"!!! Export error: {err}")
                self.root.after(0, lambda err=err: (self.show_log(), busy.close(), messagebox.showerror("Hiba", f"{err}\n\nLog: {self.file_logger.latest_path}")))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

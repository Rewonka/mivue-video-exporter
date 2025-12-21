# MiVue 798D + A30 – Simple GUI Exporter (WSL/Linux)

A small Python GUI tool to work with **Mio MiVue 798D** (front) + **MiVue A30** (rear) dashcam recordings.

It can:
- scan the camera SD-card folder structure,
- combine **front + rear** videos into one MP4 with a **rear PiP overlay**,
- generate a **moving OpenStreetMap (OSM) map video** from the front `.NMEA`,
- optionally burn the **map video overlay** into the exported combined video,
- concatenate multiple selected segments into one final MP4 and one combined NMEA.

Works well on Linux and on **WSL2** (GUI via WSLg / X server, Firefox for HTML maps).

---

## Folder structure expected

Select the **Normal/** directory (the one that contains `F/` and `R/`):

```
Normal/
  F/
    FILEYYMMDD-HHMMSSF.MP4
    FILEYYMMDD-HHMMSSF.NMEA
  R/
    FILEYYMMDD-HHMMSSR.MP4
```

Notes:
- The rear camera typically has **no NMEA** file (GPS comes from the front camera).
- The tool matches clips by timestamp `YYMMDD-HHMMSS`.

---

## Requirements

### System packages (Ubuntu / WSL Ubuntu)

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-pip python3-tk
```

### Python packages

```bash
pip install folium numpy pillow
```

---

## Run

```bash
python3 mivue_gui.py
```

Logs:
- `./logs/latest.log`
- `./logs/mivue_YYYYMMDD_HHMMSS.log`

Map tile cache:
- `./tiles_cache/`

---

## GUI features

### Load clips
**Mappa (Normal/)**  
Scans and lists clips:
- ✅ complete (exportable)
- ⚠️ incomplete

### Map HTML
**Térkép HTML (kijelölt)**  
Creates an interactive OpenStreetMap HTML preview.

### Map video
**Térkép videó (kijelölt)**  
Creates a moving map video for the selected segments:
- OSM tiles
- Follow mode
- 5 FPS
- GPS + camera smoothing

### Export (combine)
**Egyesítés (több clip → 1 MP4 + 1 NMEA)**

---

## Quality modes

### Native
- Original resolution
- H.264: CRF 20, veryfast
- Audio: AAC 128k

### Reduced
- Front scaled to 720p
- H.264: CRF 26, slow
- Audio: AAC 64k

---

## Notes

- GPS from front camera only
- Designed for MiVue naming scheme
- Personal use tool

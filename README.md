# Qc\_MartialArts\_NN

> **Martial arts pose tracking & analysis toolkit** built around AlphaPose output + your original video. Includes ID repair, distance measurements, selective frame export, and clean GUI launchers.

---

## ✨ Features

- **Pose overlay to video/images** from AlphaPose JSON
- **ID-aware highlighting** (single- or two-person focus)
- **Distance plotting** between two tracked IDs (per frame)
- **Frame range selector GUI** (time → frames → copy to folder)
- **ID Repair (**``**)** to smooth tracking and prevent spurious IDs
- **Windows-friendly** scripts and paths; works on macOS/Linux too

---

## 🗂 Project Layout

```
Qc_MartialArts_NN/
├─ main.py                 # Simple launcher for common workflows (GUI)
├─ reader.py               # Two-person pose plotter, optional distance label
├─ singleReader.py         # Single-subject pose plotter
├─ frameGUIandSelect.py    # GUI to pick a time range and copy the frames
├─ repair2.py              # Fix inconsistent track IDs across frames
├─ AlphaPose_Code/         # Outputs (images, compiled videos, selected_frames)
├─ Video_Outputs/          # Saved videos produced by readers
├─ otherTasks/             # Ideas / experimental scripts (not core pipeline)
└─ README.md               # This file
```

> **Tip:** Keep your AlphaPose JSON and source video together. Readers will ask you to pick them and create outputs in `AlphaPose_Code` / `Video_Outputs`.

---

## 🚀 Quickstart

### 1) Clone

```bash
git clone https://github.com/FranciscoJ201/Qc_MartialArts_NN.git
cd Qc_MartialArts_NN
```

### 2) Environment

Use Python 3.10–3.12. (3.13 also works for most scripts; if you’re using CUDA/PyTorch, prefer 3.10–3.12 for smoother wheels.)

**Conda (recommended):**

```bash
conda create -n qcmartial python=3.11 -y
conda activate qcmartial
```

**Pip requirements:**

```bash
pip install -r requirements.txt  # if present
# or install the core pieces manually
pip install opencv-python numpy tqdm pillow
```

> The GUIs use `tkinter` (bundled with most Python installs). On Linux you may need `sudo apt-get install python3-tk`.

### 3) AlphaPose outputs

Generate AlphaPose JSON for your video(s). Place the JSON alongside the matching video file. (This repo **consumes** AlphaPose output; it doesn’t run AlphaPose itself.)

---

## ▶️ Common Workflows

### A) Two‑person reader (with optional distance)

```bash
python main.py
```

- Choose **AlphaPose JSON** and **Video** when prompted.
- Toggle **“Plot distance”** if supported by your `reader.py` build.
- Produces frames under `AlphaPose_Code/` and a compiled video under `Video_Outputs/`.

**Direct call (advanced):**

```python
from reader import run_pose_plotter
run_pose_plotter(plot_distance=True)  # or False
```

### B) Single‑subject reader

Focus on one ID; other people are hidden.

```python
from singleReader import run_single_pose_plotter
run_single_pose_plotter()
```

### C) Frame range selector GUI

Turn times (seconds) into exact frame indices and copy only that slice.

```python
from frameGUIandSelect import frame_selector
frame_selector(json_path, video_path)
```

Result goes to `AlphaPose_Code/selected_frames/`.

### D) Repair inconsistent IDs

Stabilize IDs across frames before plotting.

```bash
python repair2.py
```

You’ll be prompted for the JSON to repair; output (e.g., `repaired.json`) is saved next to it. Then feed `repaired.json` to the readers.

**What it does** (high level):

- Keeps a short **pose history** to match people across frames
- Enforces a **maximum pixel jump** to reduce ID swaps
- **Avoids creating new IDs** mid‑sequence unless warranted
- Can **drop late-appearing detections** that don’t belong to the initial set

---

## 📦 Outputs

- **Images** with keypoints/lines, color‑coded per ID
- **Distance labels** (if enabled)
- **Videos** compiled from per‑frame images
- **Selected frames** copied by the frame selector

Suggested structure:

```
AlphaPose_Code/
  ├─ images/               # per-frame keypoint renders
  ├─ selected_frames/      # copied by frame selector GUI
  └─ logs/                 # run logs (optional)
Video_Outputs/
  └─ <your_output>.mp4
```

---

## ⚙️ Configuration Notes

- **Track IDs:** Readers expect AlphaPose “`idx`/track\_id\`” fields. For 3D readers or alternative formats, adapt the JSON parser.
- **Distance metric:** Pixel distance between chosen ID centers; convert to meters by calibrating with a known scale.
- **Performance:** If rendering is slow, reduce image size or skip every N frames for previews.

---

## 🧪 Example Snippets

**Plot two IDs and write a video:**

```python
from reader import run_pose_plotter
run_pose_plotter(plot_distance=True)
```

**Export a 4–7s slice as frames:**

```python
from frameGUIandSelect import frame_selector
frame_selector("path/to/repaired.json", "path/to/video.mp4")
# GUI: start=4, end=7 → copies frames to selected_frames/
```

---

## 🩹 Troubleshooting

- **TorchVision warnings**: “`pretrained` is deprecated; use `weights`.” → Harmless; future‑proof by switching to the `weights=` API if you customize models.
- **Tkinter messagebox errors**: If you see `messagebox.showinfo` raising errors, ensure the root Tk app is initialized before dialogs, or remove modal dialogs when running headless.
- **CUDA DLL not found (Windows)**: If you later add GPU inference (e.g., for AlphaPose), ensure matching **CUDA/cuDNN** versions are installed and on `PATH`.
- **FPS/time mismatch**: Make sure the FPS read from the video matches the one used by your frame selector.

---

## 🧭 Roadmap

- Pixel→meter conversion (camera calibration helpers)
- Batch processing for many JSON/video pairs
- Optional optical-flow continuity & ReID embeddings for better track reassignment
- 3D pose support notes / converters

---

## 🤝 Contributing

PRs are welcome! If you add features, please include a short demo, sample JSON, or a test video snippet showing the change.

---

## 📄 License

MIT (proposed). If you prefer a different license, update this section.

---

## 🙏 Acknowledgments

- **AlphaPose** for 2D pose estimation
- Early testers who helped find rough edges in the ID repair + GUI flows

---

## 📸 (Optional) Demo

Add GIFs or screenshots of:

- Two‑person distance overlay
- Single‑subject highlight
- Frame selector GUI

> Drop images in `docs/` and link them here once you have them.


# Qc_MartialArts_NN

## Overview
This project is a **pose analysis and frame selection toolkit** built around AlphaPose JSON output and original video footage.  
It provides tools to:

- Visualize and highlight skeleton keypoints from AlphaPose.
- Measure distances between selected tracked IDs in a video.
- Interactively select frame ranges for further analysis.
- Convert JSON keypoint data back into videos with overlaid poses.

---

## Project Structure
```
Qc_MartialArts_NN/
│
├── main.py                 # Entry point for running core scripts.
├── reader.py               # Pose plotting and distance measurement.
├── singleReader.py         # Variant of reader for single ID tracking.
├── frameGUIandSelect.py    # GUI to select time ranges and copy frames.
├── dataFIX/                # For files and notes that need fixing or revisiting later.
├── otherTasks/             # Potential projects or experiments to work on in the future.
├── AlphaPose_Code/         # Output plots, selected frames, etc.
└── README.md               # This file.
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/FranciscoJ201/Qc_MartialArts_NN.git
   cd Qc_MartialArts_NN
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include:*  
   `opencv-python`, `numpy`, `tkinter` (bundled with most Python installations).

---

## Usage

### 1. Run Pose Plotter
From `main.py`, the project launches the Pose Plotter GUI:
```bash
python main.py
```
Steps in the GUI:
- **Select JSON File** — AlphaPose output file.
- **Select Video File** — Original video corresponding to the JSON.
- **Enter Output Video Name** — Name for the generated video.
- Click **Run Pose Plotter** to:
  - Clear old outputs.
  - Render keypoints and distances into images.
  - Compile them into a video.

---

### 2. Select Frame Ranges
After generating an output, you can run:
```python
from frameGUIandSelect import frame_selector
frame_selector(json_path, video_path)
```
This opens a GUI where you:
- Enter start and end times (in seconds).
- Automatically get corresponding frame numbers.
- Copy only the frames in that range into `AlphaPose_Code/selected_frames/`.

---

### 3. Output Example
- **Gray skeletons** — All detected people.
- **Red / Blue skeletons** — Highlighted tracked IDs (`ID_A` and `ID_B`).
- **Distance line + label** — Pixel distance between centers of the two tracked people.

---

## Notes on Folders
- **`dataFIX/`**  
  For files, code snippets, or results that need further debugging, cleanup, or post-processing later.
- **`otherTasks/`**  
  For ideas, potential experiments, and “maybe later” scripts that aren’t part of the main pipeline yet.

---

## Future Improvements
- Integrate pose repair script to maintain consistent IDs when AlphaPose tracking changes unexpectedly.
- Add metric conversion for distances (pixels → meters).
- Batch processing for multiple JSON/video pairs.
- Optional optical flow continuity for smoother tracking.

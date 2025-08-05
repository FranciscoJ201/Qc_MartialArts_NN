# Qc\_MartialArts\_NN

This project processes pose estimation JSON output (e.g. from AlphaPose), visualizes human skeletons over time as individual frame plots, and compiles them into a final video for analysis of martial arts movements. It also supports selecting frame ranges using a GUI.

---

## 🔧 Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- Tkinter (built into most Python installations)

---

## 📁 Folder Setup (IMPORTANT)

Before using the code, make sure the following folders exist (create them if missing):

- `OpenPose_Code/newplots` — leave this empty
- `AlphaPose_Code/output_plots` — will store frame plots
- `AlphaPose_Code/selected_frames` — will store selected frame plots
- `Video_Outputs/` — created automatically for final videos

---

## 🧠 Files Overview

### `main.py`

Entry point. Launches the full process:

1. Opens a GUI to select:
   - JSON file of keypoints
   - Corresponding video
   - Output video name
2. Plots skeletons for each frame using `reader.py`
3. Lets you select a frame range using a second GUI (`frameGUIandSelect.py`)
4. Copies selected plots to a dedicated folder

### `reader.py`

- Draws pose skeletons from AlphaPose-style JSON output.
- Saves one plot per frame to `AlphaPose_Code/output_plots/`.
- Creates a final video from the plots using `videoCreator.py`.

### `frameGUIandSelect.py`

- GUI for entering start and end times (in seconds).
- Converts to frame range and copies corresponding plots to `AlphaPose_Code/selected_frames`.

### `videoCreator.py`

- Turns the plotted PNGs into an `.mp4` video using OpenCV.
- Saves to `Video_Outputs/{your_name}.mp4`.

### `folderclear.py`

- Empties `output_plots/` and `selected_frames/` before processing to avoid mixing runs.

---

## 🥚 Experimental Files

- `test2.py` (not currently used in the main pipeline) contains an example neural network directly from the PyTorch documentation. This is for learning purposes and will be adapted in the future.
- `OpenPose_Code` (not currently in use) the process will be the same as alphapose code file just needs to be reformated 

---

## 🧴 COCO 17 Keypoint Skeleton Edges

These are the connections between keypoints that the plotter uses:

```
(0, 1):  Nose → Left Eye
(0, 2):  Nose → Right Eye
(1, 3):  Left Eye → Left Ear
(2, 4):  Right Eye → Right Ear
(0, 5):  Nose → Left Shoulder
(0, 6):  Nose → Right Shoulder
(5, 7):  Left Shoulder → Left Elbow
(7, 9):  Left Elbow → Left Wrist
(6, 8):  Right Shoulder → Right Elbow
(8, 10): Right Elbow → Right Wrist
(5, 11): Left Shoulder → Left Hip
(6, 12): Right Shoulder → Right Hip
(11, 13): Left Hip → Left Knee
(13, 15): Left Knee → Left Ankle
(12, 14): Right Hip → Right Knee
(14, 16): Right Knee → Right Ankle
```

---

## ▶️ How to Run

1. **Run **``
2. **Select your JSON and video**
3. **Enter a name for the output video**
4. **Wait for plots to be generated**
5. **Input start and end times (in seconds) to select a portion**
6. **Review final output in **``

---
## 🧪 New Experimental Additions

### `repair.py` – *Fixes ID Reuse Issues*

- AlphaPose may reuse `"idx"` values when a person leaves and re-enters the frame.
- This script scans all frames in order and remaps reused `"idx"` values to consistent ones by assigning new IDs when a person disappears and later reappears.
- **Output:** a new JSON file called `result_recycled.json` with updated IDs.
- 🔧 **Status:** Early-stage: It handles recycling based on visibility but may misassign IDs if people overlap or occlude each other rapidly.

### `singleReader.py` – *Focus on One Person Only*

- Modified version of `reader.py` that only plots the skeleton of a specific `"idx"` (person).
- GUI shows available `"idx"` values (if tracking was enabled), and user can choose which person to visualize.
- Useful for isolating a single subject across long videos with many detected individuals.
- **Depends on:** consistent tracking from AlphaPose.
- 🔧 **Status:** In progress — if AlphaPose doesn't track reliably or recycles IDs, output may be inconsistent. Pair with `repair.py` for best (but still inaccurate) results.


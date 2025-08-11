# Qc_MartialArts_NN

## 📌 Overview
This project is a **pose analysis and frame selection toolkit** built around AlphaPose JSON output and original video footage.  
It provides tools to:

• Visualize and highlight skeleton keypoints from AlphaPose.
• Measure distances between selected tracked IDs in a video.
• Interactively select frame ranges for further analysis.
• Convert JSON keypoint data back into videos with overlaid poses.

---

## 📂 Project Structure
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

## ⚙️ Installation
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

## ▶️ Usage

### 1. Run Pose Plotter
From `main.py`, the project launches the Pose Plotter GUI:
```bash
python main.py
```
Steps in the GUI:
• **Select JSON File** — AlphaPose output file.
• **Select Video File** — Original video corresponding to the JSON.
• **Enter Output Video Name** — Name for the generated video.
• Click **Run Pose Plotter** to:
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
• Enter start and end times (in seconds).
• Automatically get corresponding frame numbers.
• Copy only the frames in that range into `AlphaPose_Code/selected_frames/`.

---

### 3. Output Example
• **Gray skeletons** — All detected people.
• **Red / Blue skeletons** — Highlighted tracked IDs (`ID_A` and `ID_B`).
• **Distance line + label** — Pixel distance between centers of the two tracked people.

---

## 🗂️ Notes on Folders
• **`dataFIX/`**  
  For files, code snippets, or results that need further debugging, cleanup, or post-processing later.
• **`otherTasks/`**  
  For ideas, potential experiments, and “maybe later” scripts that aren’t part of the main pipeline yet.

---

## 🚀 Future Improvements
• Add metric conversion for distances (pixels → meters).
• Batch processing for multiple JSON/video pairs.
• Optional optical flow continuity for smoother tracking.

---

## 📜 Script Descriptions

### **`main.py`**
Entry point for running the core scripts. Currently launches the `run_pose_plotter()` function from `reader.py` to start the Pose Plotter GUI.

### **`reader.py`**
• **Purpose:** Main tool for converting AlphaPose JSON output into visual skeleton plots and videos.
• **Features:**
  - Draws skeleton keypoints from the JSON data.
  - Highlights two specific tracked IDs in red and blue.
  - Calculates and displays the distance between the two selected IDs.
  - Saves each processed frame as an image and compiles them into a video.

### **`singleReader.py`**
• **Purpose:** A variant of `reader.py` for focusing on a single tracked person instead of two.
• **Use Case:** Useful when analyzing one subject in detail without other distractions.
  - Must be updated for new repair script

### **`frameGUIandSelect.py`**
• **Purpose:** GUI tool to select start and end times from a video and extract the corresponding frames.
• **Features:**
  - Reads FPS from the video.
  - Converts time ranges into frame numbers.
  - Copies only the frames in the selected range to a separate folder for further analysis.

### **`repair2.py`**
• **Purpose:** Post-processing tool for AlphaPose output to fix inconsistent person tracking IDs.
• **Features:**
  - Uses pose history to match people across frames.
  - Prevents unrealistic ID reassignments based on maximum allowed pixel jumps.
  - Maintains original starting IDs and avoids creating new IDs for people not present at the start.
  - Drops detections of new people appearing mid-video if they weren't in the initial frame set.

### **`dataFIX/`**
• Storage for files, code, and notes that need fixing or review later.

### **`otherTasks/`**
• Collection of scripts and ideas for future potential tasks.


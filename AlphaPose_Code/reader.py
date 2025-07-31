import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Step 1: Use non-GUI backend
import matplotlib.pyplot as plt
import cv2
from videoCreator import make_video
from folderclear import clear_all
import time
import sys
import tkinter as tk
from tkinter import filedialog

# GUI SETUP
root = tk.Tk()
root.withdraw()

# --- CONFIG ---
VIDEO_DIR    = 'videos'
OUTPUT_DIR   = 'AlphaPose_Code/output_plots'
dpi          = 100

# --- Get JSON File ---
json_path = filedialog.askopenfilename(
    title="Select JSON File",
    initialdir="json_input",
    filetypes=[("JSON files", "*.json")]
)
if not json_path:
    print("No JSON file selected. Exiting.")
    sys.exit()
JSON_PATH = json_path

# --- Get Video File ---
video_path = filedialog.askopenfilename(
    title="Select Video File",
    initialdir=VIDEO_DIR,
    filetypes=[("Video files", "*.mp4 *.avi *.mov")]
)
if not video_path:
    print("No video selected. Exiting.")
    sys.exit()

# 17‑point skeleton edges
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# --- Setup Code ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
start_time = time.perf_counter()

cap = cv2.VideoCapture(video_path)
w_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_res = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fig_w = w_res / dpi
fig_h = h_res / dpi

clear_all()

# --- Load JSON ---
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

frames = {}
for entry in data:
    fid = entry.get('image_id')
    if fid is not None:
        frames.setdefault(fid, []).append(entry)

def frame_num(fname):
    name, _ = os.path.splitext(os.path.basename(fname))
    return int(name)

sorted_fids = sorted(frames.keys(), key=frame_num)
total = len(sorted_fids)

# --- Plot Loop ---
for idx, fid in enumerate(sorted_fids):
    people = []
    for person in frames[fid]:
        kp = np.array(person['keypoints'], dtype=float)
        if kp.size % 3 == 0:
            people.append(kp.reshape(-1, 3))
    if not people:
        continue

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)  # Step 2: use ax instead of plt global
    for person in people:
        visible = person[:, 2] > 0
        ax.plot(person[visible, 0], person[visible, 1], 'o')

        for i, j in EDGES:
            if person[i, 2] > 0 and person[j, 2] > 0:
                ax.plot(
                    [person[i, 0], person[j, 0]],
                    [person[i, 1], person[j, 1]],
                    color='red'
                )

    ax.invert_yaxis()
    ax.set_xlim(0, w_res)
    ax.set_ylim(h_res, 0)
    ax.axis('off')  # Optional: hide axes completely

    out_path = os.path.join(OUTPUT_DIR, f'plot_{idx}.png')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    elapsed = time.perf_counter() - start_time
    sys.stdout.write(f"\rFrame {idx+1}/{total} • Elapsed: {elapsed:.2f}s")
    sys.stdout.flush()

# --- Done ---
end_time = time.perf_counter()
print("\nDone!")
print(f"Generated {len(sorted_fids)} plots in {end_time - start_time:.2f} seconds.")

name = input('What do you want to name the video:')
make_video(name)

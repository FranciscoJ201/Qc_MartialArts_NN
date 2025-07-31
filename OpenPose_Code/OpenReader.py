import cv2
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend for faster rendering
import matplotlib.pyplot as plt
import re
import os
import time
import sys
from OpenvideoCreator import make_video
from Openfolderclear import clear_all
import tkinter as tk
from tkinter import filedialog

# --- GUI Setup ---
root = tk.Tk()
root.withdraw()

# --- Folder Selection ---
demo_video_directory = filedialog.askdirectory(
    title="Select Folder Containing JSON Files",
    initialdir="OpenPose_Code/json_input"
)
if not demo_video_directory:
    print("No folder selected. Exiting.")
    exit()

# --- Video File Selection ---
video_path = filedialog.askopenfilename(
    title="Select Video File",
    initialdir="videos",
    filetypes=[("Video files", "*.mp4 *.avi *.mov")]
)
if not video_path:
    print("No video selected. Exiting.")
    exit()

# --- Setup ---
clear_all()

cap = cv2.VideoCapture(video_path)
width_resolution = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_resolution = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

dpi = 100
width = width_resolution / dpi
height = height_resolution / dpi

edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (10, 11), (11, 24),
    (11, 22), (22, 23), (8, 12), (12, 13),
    (13, 14), (14, 21), (14, 19), (19, 20),
    (0, 15), (15, 17), (0, 16), (16, 18)
]

# Extract numeric frame index from file name
def extract_frame_number(filename):
    match = re.search(r'.*?_(\d+)_keypoints\.json', filename)
    return int(match.group(1)) if match else -1

files = sorted(os.listdir(demo_video_directory), key=extract_frame_number)
total = len(files)
start_time = time.perf_counter()

for ite, cur_file in enumerate(files):
    json_path = os.path.join(demo_video_directory, cur_file)

    with open(json_path) as f:
        data_dict = json.load(f)

    people = []
    for person in data_dict['people']:
        keypoints = person['pose_keypoints_2d']
        np_keypoints = np.array(keypoints)
        people.append(np_keypoints.reshape(-1, 3))
    people = np.array(people)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    for person in people:
        for keypoint in person:
            if keypoint[2] != 0:
                ax.plot(keypoint[0], keypoint[1], 'o')

        for i, j in edges:
            node1 = person[i]
            node2 = person[j]
            if node1[2] != 0 and node2[2] != 0:
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='red')

    ax.invert_yaxis()
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis('off')  # optional: hides axis lines/ticks

    fig.savefig(f"OpenPose_Code/newplots/plot_{ite}.png")
    plt.close(fig)


    elapsed = time.perf_counter() - start_time
    sys.stdout.write(f"\rFrame {ite+1}/{total} • Elapsed: {elapsed:.2f}s")
    sys.stdout.flush()

end_time = time.perf_counter()
print("\nDone!")
print(f"Generated {total} plots in {end_time - start_time:.2f} seconds.")

make_video(os.path.basename(demo_video_directory))

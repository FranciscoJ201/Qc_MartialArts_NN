
import os
import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from videoCreator import make_video
from folderclear import clear_all
import time

def compute_keypoint_distance(p1, p2, keypoint_index):
    x1, y1, c1 = p1[keypoint_index]
    x2, y2, c2 = p2[keypoint_index]
    if c1 > 0 and c2 > 0:
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return None

def frame_num(fname):
    return int(os.path.splitext(os.path.basename(fname))[0])

def draw_skeleton(frame, keypoints, color):
    EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for i, j in EDGES:
        if keypoints[i, 2] > 0 and keypoints[j, 2] > 0:
            pt1 = tuple(keypoints[i, :2].astype(int))
            pt2 = tuple(keypoints[j, :2].astype(int))
            cv2.line(frame, pt1, pt2, color, 2)
    for i in range(len(keypoints)):
        if keypoints[i, 2] > 0:
            pt = tuple(keypoints[i, :2].astype(int))
            cv2.circle(frame, pt, 3, color, -1)

def convert_json_to_opencv_images(json_path, video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    w_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_res = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = {}
    for entry in data:
        fid = entry.get('image_id')
        if fid:
            frames.setdefault(fid, []).append(entry)

    sorted_fids = sorted(frames.keys(), key=frame_num)

    start_time = time.perf_counter()

    for idx, fid in enumerate(sorted_fids):
        people = []
        for person in frames[fid]:
            kp = np.array(person['keypoints'], dtype=float)
            if kp.size == 51:
                people.append(kp.reshape(-1, 3))
        if not people:
            continue

        frame = np.ones((h_res, w_res, 3), dtype=np.uint8) * 255
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255)]
        for i, person in enumerate(people):
            draw_skeleton(frame, person, colors[i % len(colors)])

        if len(people) >= 2:
            dist = compute_keypoint_distance(people[0], people[1], 0)
            if dist is not None:
                x1, y1 = people[0][0][:2].astype(int)
                x2, y2 = people[1][0][:2].astype(int)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2, lineType=cv2.LINE_AA)

        out_path = os.path.join(output_dir, f'plot_{idx}.png')
        cv2.imwrite(out_path, frame)

        elapsed = time.perf_counter() - start_time
        print(f"Frame {idx+1}/{len(sorted_fids)} • Elapsed: {elapsed:.2f}s")

    return output_dir

def run_pose_plotter():
    result = {"json": None, "video": None, "name": None}

    def browse_json():
        path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON Files", "*.json")]
        )
        json_path_var.set(path)

    def browse_video():
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        video_path_var.set(path)

    def run_processing():
        json_path = json_path_var.get()
        video_path = video_path_var.get()
        video_name = video_name_entry.get()

        if not json_path or not os.path.exists(json_path):
            messagebox.showerror("Missing JSON", "Please select a valid JSON file.")
            return
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Missing Video", "Please select a valid video file.")
            return
        if not video_name.strip():
            messagebox.showerror("Missing Name", "Please enter a video name.")
            return

        OUTPUT_DIR = 'AlphaPose_Code/output_plots'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clear_all()

        convert_json_to_opencv_images(json_path, video_path, OUTPUT_DIR)
        make_video(video_name, video_path)

        result["json"] = json_path
        result["video"] = video_path
        result["name"] = video_name

        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Pose Plotter GUI")

    json_path_var = tk.StringVar()
    video_path_var = tk.StringVar()

    tk.Label(root, text="JSON File:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=json_path_var, width=50).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_json).grid(row=0, column=2)

    tk.Label(root, text="Video File:").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=video_path_var, width=50).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_video).grid(row=1, column=2)

    tk.Label(root, text="Output Video Name:").grid(row=2, column=0, sticky="e")
    video_name_entry = tk.Entry(root)
    video_name_entry.grid(row=2, column=1)

    tk.Button(root, text="Run Pose Plotter", command=run_processing).grid(row=3, column=1, pady=10)
    root.mainloop()

    return result["json"], result["video"], result["name"]

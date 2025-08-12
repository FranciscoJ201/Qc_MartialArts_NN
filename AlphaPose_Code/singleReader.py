import os
import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from videoCreator import make_video
from folderclear import clear_all
import time

# --- Helpers (copied from reader.py) ---

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

def draw_axes(frame, step=100, grid_color=(200, 200, 200)):
    h, w = frame.shape[:2]
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), grid_color, 1)
        _put_text_with_outline(frame, str(x), (x + 2, 15))
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), grid_color, 1)
        _put_text_with_outline(frame, str(y), (2, max(12, y - 2)))
    cv2.line(frame, (0, 0), (w, 0), (0, 0, 0), 2)
    cv2.line(frame, (0, 0), (0, h), (0, 0, 0), 2)
    _put_text_with_outline(frame, "X", (w - 20, 20), scale=0.6)
    _put_text_with_outline(frame, "Y", (10, h - 10), scale=0.6)

def _put_text_with_outline(frame, text, org, scale=0.4):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

# --- Core ---

def convert_single_json_to_images(json_path, video_path, output_dir, target_id):
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
        frame = np.ones((h_res, w_res, 3), dtype=np.uint8) * 255
        draw_axes(frame, step=100)

        pose_drawn = False
        for person in frames[fid]:
            idx_val = person.get("idx")
            if idx_val != target_id:
                continue
            kp = np.array(person["keypoints"], dtype=float)
            if kp.size == 51:
                pose = kp.reshape(-1, 3)
                draw_skeleton(frame, pose, (0, 0, 255))
                if pose[0, 2] > 0:
                    x, y = pose[0, :2].astype(int)
                    _put_text_with_outline(frame, f"ID {idx_val}", (x, max(0, y - 10)), scale=0.6)
                pose_drawn = True

        if not pose_drawn:
            _put_text_with_outline(frame, f"ID {target_id} Missing", (20, 40), scale=0.9)

        out_path = os.path.join(output_dir, f'plot_{idx}.png')
        cv2.imwrite(out_path, frame)

        elapsed = time.perf_counter() - start_time
        print(f"Frame {idx+1}/{len(sorted_fids)} • Elapsed: {elapsed:.2f}s")

    return output_dir

def run_single_pose_plotter():
    result = {"json": None, "video": None, "name": None}

    def browse_json():
        path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON Files", "*.json")]
        )
        if not path:
            return
        json_path_var.set(path)
        try:
            with open(path, 'r') as f:
                raw_data = json.load(f)
            available_ids = sorted({entry.get("idx") for entry in raw_data if "idx" in entry})
            if available_ids:
                messagebox.showinfo("Available Person IDs", f"Detected person IDs: {available_ids}")
            else:
                messagebox.showerror("No IDs Found", "No 'idx' values found in JSON file.")
        except Exception as e:
            messagebox.showerror("JSON Error", f"Failed to read JSON: {str(e)}")

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

        try:
            selected_index = int(person_id_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for person ID.")
            return

        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        available_ids = sorted({entry.get("idx") for entry in raw_data if "idx" in entry})
        if selected_index not in available_ids:
            messagebox.showerror("Invalid ID", f"ID {selected_index} not found in JSON. Available: {available_ids}")
            return

        OUTPUT_DIR = 'AlphaPose_Code/output_plots'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clear_all()

        convert_single_json_to_images(json_path, video_path, OUTPUT_DIR, selected_index)
        make_video(video_name, video_path)

        result["json"] = json_path
        result["video"] = video_path
        result["name"] = video_name
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Single Person Pose Plotter")

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

    tk.Label(root, text="Person ID (idx):").grid(row=3, column=0, sticky="e")
    person_id_entry = tk.Entry(root)
    person_id_entry.grid(row=3, column=1)

    tk.Button(root, text="Run Pose Plotter", command=run_processing).grid(row=4, column=1, pady=10)
    root.mainloop()

    return result["json"], result["video"], result["name"]

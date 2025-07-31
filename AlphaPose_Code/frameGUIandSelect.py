import os
import json
import shutil
import cv2
import tkinter as tk
from tkinter import messagebox

# ----------------------------
# Frame Range Helper Functions
# ----------------------------

def extract_frame_number(image_id):
    base = os.path.splitext(image_id)[0]
    return int(base.split('_')[-1])

def frame_range_from_json(json_path, start_time, end_time, fps):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = set()
    for entry in data:
        fname = entry['image_id']
        frame_num = extract_frame_number(fname)
        frames.add(frame_num)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    valid_frames = sorted(f for f in frames if start_frame <= f < end_frame)
    return start_frame, end_frame - 1, valid_frames

def detect_fps_and_total(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

# ----------------------------
# GUI: Only Start and End Time Input
# ----------------------------

def launch_gui(json_path, video_path=None):
    result = {"start": None, "end": None}

    def compute_range():
        try:
            start_time = float(start_time_entry.get())
            end_time = float(end_time_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Start and end times must be numeric.")
            return

        if start_time >= end_time:
            messagebox.showerror("Invalid Range", "Start time must be less than end time.")
            return

        try:
            fps = detect_fps_and_total(video_path)[0] if video_path else 30
        except:
            messagebox.showerror("FPS Error", "Could not read FPS from video.")
            return

        try:
            s_frame, e_frame, valid = frame_range_from_json(json_path, start_time, end_time, fps)
            msg = (
                f"FPS used: {fps:.2f}\n"
                f"Start Time: {start_time}s → Frame {s_frame}\n"
                f"End Time: {end_time}s → Frame {e_frame}\n"
                f"Frames found in JSON: {valid}\n"
            )
            if video_path:
                _, total_frames = detect_fps_and_total(video_path)
                msg += f"Total Frames in Video: {total_frames}"

            messagebox.showinfo("Frame Info", msg)
            result["start"] = s_frame
            result["end"] = e_frame
            root.quit()
            root.destroy()

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    root = tk.Tk()
    root.title("Enter Time Range")

    tk.Label(root, text="Start Time (s):").grid(row=0, column=0, sticky="e")
    start_time_entry = tk.Entry(root)
    start_time_entry.grid(row=0, column=1)

    tk.Label(root, text="End Time (s):").grid(row=1, column=0, sticky="e")
    end_time_entry = tk.Entry(root)
    end_time_entry.grid(row=1, column=1)

    tk.Button(root, text="Get Frame Range", command=compute_range).grid(row=2, column=1, pady=10)

    root.mainloop()

    return result["start"], result["end"]

# ----------------------------
# Frame Copying Logic
# ----------------------------

# ✅ Insert your paths here manually

def frame_selector(json_path,video_path):
    # ✅ Run GUI to get frame range
    start, end = launch_gui(json_path, video_path)

    # ✅ Define directories
    SOURCE_DIR = 'AlphaPose_Code/output_plots'
    TARGET_DIR = 'AlphaPose_Code/selected_frames'
    os.makedirs(TARGET_DIR, exist_ok=True)

    # ✅ Copy selected frames
    copied = 0
    for frame in range(start, end + 1):
        filename = f"plot_{frame}.png"
        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(TARGET_DIR, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1

    print(f"Copied {copied} frames (plot_{start}.png to plot_{end}.png) to '{TARGET_DIR}'")

import json
import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

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

# GUI
def launch_gui():
    def browse_json():
        path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON Files", "*.json")]
        )
        json_path_var.set(path)

    def browse_video():
        path = filedialog.askopenfilename(
            title="Select Video File (optional for FPS detection)",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        video_path_var.set(path)

    def compute_range():
        json_path = json_path_var.get()
        video_path = video_path_var.get()
        total_frames = None

        try:
            start_time = float(start_time_entry.get())
            end_time = float(end_time_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Start and end times must be numeric.")
            return

        if not json_path:
            messagebox.showerror("Missing JSON", "Please select a JSON file.")
            return
        if start_time >= end_time:
            messagebox.showerror("Invalid Range", "Start time must be less than end time.")
            return

        # FPS: from video if provided, else fallback to 30
        try:
            if video_path:
                fps, total_frames = detect_fps_and_total(video_path)
            else:
                fps = 30  # fallback
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
            if total_frames is not None:
                msg += f"Total Frames in Video: {total_frames}"
            messagebox.showinfo("Frame Info", msg)
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    root = tk.Tk()
    root.title("JSON Frame Range (with Optional FPS Detection)")

    json_path_var = tk.StringVar()
    video_path_var = tk.StringVar()

    tk.Label(root, text="JSON File:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=json_path_var, width=50).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_json).grid(row=0, column=2)

    tk.Label(root, text="Video File (Optional for FPS Calibration):").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=video_path_var, width=50).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_video).grid(row=1, column=2)

    tk.Label(root, text="Start Time (s):").grid(row=2, column=0, sticky="e")
    start_time_entry = tk.Entry(root)
    start_time_entry.grid(row=2, column=1)

    tk.Label(root, text="End Time (s):").grid(row=3, column=0, sticky="e")
    end_time_entry = tk.Entry(root)
    end_time_entry.grid(row=3, column=1)

    tk.Button(root, text="Get Frame Range", command=compute_range).grid(row=4, column=1, pady=10)

    root.mainloop()

# Launch it
launch_gui()

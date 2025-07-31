import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

def get_frame_range(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    if start_frame >= total_frames or end_frame > total_frames:
        raise ValueError("Timestamps exceed video duration.")

    return start_frame, end_frame - 1, fps, total_frames

def launch_gui():
    def browse_file():
        path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        video_path_var.set(path)

    def show_frame_range():
        video_path = video_path_var.get()
        try:
            start_time = float(start_time_entry.get())
            end_time = float(end_time_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Start and end times must be numeric.")
            return

        if not video_path:
            messagebox.showerror("No File", "Please select a video file.")
            return
        if start_time >= end_time:
            messagebox.showerror("Invalid Range", "Start time must be less than end time.")
            return

        try:
            start_f, end_f, fps, total = get_frame_range(video_path, start_time, end_time)
            msg = (
                f"Video FPS: {fps:.2f}\n"
                f"Total Frames: {total}\n"
                f"Start Time: {start_time}s → Frame {start_f}\n"
                f"End Time: {end_time}s → Frame {end_f}"
            )
            messagebox.showinfo("Frame Range", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Frame Range Finder")

    video_path_var = tk.StringVar()

    tk.Label(root, text="Video File:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=video_path_var, width=50).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2)

    tk.Label(root, text="Start Time (s):").grid(row=1, column=0, sticky="e")
    start_time_entry = tk.Entry(root)
    start_time_entry.grid(row=1, column=1)

    tk.Label(root, text="End Time (s):").grid(row=2, column=0, sticky="e")
    end_time_entry = tk.Entry(root)
    end_time_entry.grid(row=2, column=1)

    tk.Button(root, text="Get Frame Range", command=show_frame_range).grid(row=3, column=1, pady=10)

    root.mainloop()

# Run the GUI
launch_gui()


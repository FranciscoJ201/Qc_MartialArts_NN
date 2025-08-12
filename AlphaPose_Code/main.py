# main.py
import os
import tkinter as tk
from tkinter import ttk, messagebox
import runpy

from reader import run_pose_plotter
from singleReader import run_single_pose_plotter
from frameGUIandSelect import frame_selector

def run_repair():
    try:
        runpy.run_path("repair2.py", run_name="__main__")
        if not os.path.exists("repaired.json"):
            raise FileNotFoundError("repaired.json was not created.")
        messagebox.showinfo(
            "Repair complete",
            "Wrote repaired.json.\nWhen prompted for a JSON file, choose repaired.json."
        )
        return True
    except Exception as e:
        messagebox.showerror("Repair failed", f"repair2.py error:\n{e}")
        return False

def launch():
    use_repair = repair_var.get()
    mode = mode_var.get()
    run_frames = frames_var.get()
    plot_dist = plot_distance_var.get()  # <— NEW

    root.destroy()

    if use_repair:
        if not run_repair():
            return

    try:
        if mode == "single":
            json_path, video_path, name = run_single_pose_plotter()
        elif mode == "reader":
            # pass the bool into reader
            json_path, video_path, name = run_pose_plotter(plot_distance=plot_dist)
        else:
            messagebox.showerror("No selection", "Choose Single or Two‑person view.")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run plotter:\n{e}")
        return

    if not json_path or not video_path:
        messagebox.showinfo("Cancelled", "No output produced (flow may have been cancelled).")
        return

    if run_frames:
        try:
            frame_selector(json_path, video_path)
        except Exception as e:
            messagebox.showerror("Frame Selector Error", f"Failed to run frame selector:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Pose Plotter Launcher")

    container = ttk.Frame(root, padding=16)
    container.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(container, text="View mode:").grid(row=0, column=0, sticky="w")
    mode_var = tk.StringVar(value="single")
    ttk.Radiobutton(container, text="Single Person", variable=mode_var, value="single").grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Radiobutton(container, text="Two‑person (reader)", variable=mode_var, value="reader").grid(row=2, column=0, sticky="w")

    ttk.Separator(container).grid(row=3, column=0, sticky="ew", pady=10)

    repair_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(container, text="Run repair first (repair2.py → repaired.json)", variable=repair_var).grid(row=4, column=0, sticky="w")

    # NEW: distance plotting toggle (only affects reader)
    plot_distance_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(container, text="Plot distance (reader only)", variable=plot_distance_var).grid(row=5, column=0, sticky="w", pady=(6, 0))

    frames_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(container, text="Run Frame Selector after plotting", variable=frames_var).grid(row=6, column=0, sticky="w", pady=(8, 0))

    btns = ttk.Frame(container)
    btns.grid(row=7, column=0, sticky="e", pady=(12, 0))
    ttk.Button(btns, text="Launch", command=launch).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btns, text="Cancel", command=root.destroy).grid(row=0, column=1)

    root.mainloop()

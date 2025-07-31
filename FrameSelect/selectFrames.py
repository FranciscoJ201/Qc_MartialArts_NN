import os
import shutil
from frames import launch_gui  # your GUI returns start and end frame

# Run GUI and get frame range
start, end = launch_gui()

# Settings
SOURCE_DIR = 'AlphaPose_Code/output_plots'  # where plots are currently saved
TARGET_DIR = 'FrameSelect/selected_frames'              # new directory to copy to

# Create target directory
os.makedirs(TARGET_DIR, exist_ok=True)

# Copy selected plots
copied = 0
for frame in range(start, end + 1):
    filename = f"plot_{frame}.png"
    src_path = os.path.join(SOURCE_DIR, filename)
    dst_path = os.path.join(TARGET_DIR, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied += 1

print(f"Copied {copied} frames (plot_{start}.png to plot_{end}.png) to '{TARGET_DIR}'")

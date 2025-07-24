import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from videoCreator import make_video
from folderclear import clear_all
import time
import sys


# --- CONFIG ---

VIDEO_DIR    = 'videos'
OUTPUT_DIR   = 'output_plots'
dpi          = 100
JSON_PATH    = f'json_input/{input('Enter Name of File in Json_input folder you are reading: ')}'
# 17‑point skeleton edges
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# --- Setup Code ---
os.makedirs(OUTPUT_DIR, exist_ok=True) #creates output if it doesnt exist
#the stuff underneath is still setup just basically matches video resolution to json file
video_ref = input('Video filename (in videos/): ')
start_time = time.perf_counter()
video_path = os.path.join(VIDEO_DIR, video_ref)
cap = cv2.VideoCapture(video_path)
w_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_res = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fig_w = w_res / dpi
fig_h = h_res / dpi

clear_all()

# --- Loading The Json File ---
with open(JSON_PATH, 'r') as f:
    data = json.load(f)   # data is a list of entries

# Group by image_id
frames = {}
#frames takes the form 
# {
#   '000001.jpg': [entryA, entryB, …],
#   '000002.jpg': [entryC, …],
#   … 
# }
# where each entryX is one person’s from that frame.

for entry in data:
    fid = entry.get('image_id')
    #skip any entries with no image id
    if fid is None:
        continue
    frames.setdefault(fid, []).append(entry)





# Sort by numeric frame number (strip off ".jpg")
def frame_num(fname):
    name, _ = os.path.splitext(os.path.basename(fname))
    return int(name) #convers 000123 to 123

#sorts the frame IDs for the next step
sorted_fids = sorted(frames.keys(), key=frame_num)
total = len(sorted_fids)  #for live timer



# --- Plotting for each frame ---
for idx, fid in enumerate(sorted_fids):
    people = []
    for person in frames[fid]:
        kp = np.array(person['keypoints'], dtype=float)
        # reshape into N×3 (should be 17×3)
        if kp.size % 3 == 0:
            # reshape into (N,3): N rows (keypoints), 3 cols (x, y, confidence)
            people.append(kp.reshape(-1, 3))
    if not people:
        continue

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    for person in people:
        visible = person[:, 2] > 0
        ax.plot(person[visible, 0], person[visible, 1], 'o')
        # draw skeleton edges (EDGES list of (i,j) index pairs from all the way at the top):

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
    ax.axis('on')

    out_path = os.path.join(OUTPUT_DIR, f'plot_{idx}.png')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    
    elapsed = time.perf_counter() - start_time
    sys.stdout.write(f"\rFrame {idx+1}/{total} • Elapsed: {elapsed:.2f}s")
    sys.stdout.flush()





end_time = time.perf_counter()
print("Done!")
print(f"Generated {len(sorted_fids)} plots in {end_time - start_time:.2f} seconds.")
#name video here
name = input('What do you want to name the video:')
make_video(name)


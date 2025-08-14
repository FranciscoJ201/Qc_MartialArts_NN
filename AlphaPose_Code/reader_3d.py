# reader3d.py — 3D pose anchored to 2D motion (per-frame translation & scale)
import os, json, time
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

from videoCreator import make_video
from folderclear import clear_all

# ----------------------------
# Skeleton edges
# ----------------------------
COCO17_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,11),(6,12),
    (11,13),(13,15),(12,14),(14,16)
]
SMPL24_EDGES = [
    (0,1),(1,4),(4,7),(7,10),
    (0,2),(2,5),(5,8),(8,11),
    (0,3),(3,6),(6,9),(9,12),(12,15),
    (12,13),(13,16),(16,18),(18,20),(20,22),
    (12,14),(14,17),(17,19),(19,21),(21,23)
]

def draw_axes(frame, step=100, grid_color=(200,200,200)):
    h,w = frame.shape[:2]
    for x in range(0,w,step):
        cv2.line(frame,(x,0),(x,h),grid_color,1)
        _txt(frame,str(x),(x+2,15))
    for y in range(0,h,step):
        cv2.line(frame,(0,y),(w,y),grid_color,1)
        _txt(frame,str(y),(2,max(12,y-2)))
    cv2.line(frame,(0,0),(w,0),(0,0,0),2)
    cv2.line(frame,(0,0),(0,h),(0,0,0),2)
    _txt(frame,"X",(w-20,20),0.6); _txt(frame,"Y",(10,h-10),0.6)

def _txt(frame,text,org,scale=0.4):
    cv2.putText(frame,text,org,cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(frame,text,org,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),1,cv2.LINE_AA)

def draw_skeleton_2d(img, pts, vis=None, color=(0,0,255), th=2):
    n = pts.shape[0]
    if n==24: edges = SMPL24_EDGES
    elif n==17: edges = COCO17_EDGES
    else: edges = [(i,j) for (i,j) in COCO17_EDGES if i<n and j<n]
    for i,j in edges:
        ok = (vis is None) or (vis[i] and vis[j])
        if ok:
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), color, th)
    for i in range(n):
        if vis is None or vis[i]:
            cv2.circle(img, tuple(pts[i].astype(int)), 3, color, -1)

# ----------------------------
# Parse helpers
# ----------------------------
def parse_3d(entry):
    if 'pred_xyz_jts' in entry:
        arr = np.array(entry['pred_xyz_jts'], dtype=float)
        X = arr.reshape(-1,3) if arr.ndim==1 else arr
        return X, None
    if 'keypoints_3d' in entry:
        arr = np.array(entry['keypoints_3d'], dtype=float)
        if arr.ndim==1 and arr.size%4==0:
            A = arr.reshape(-1,4); return A[:,:3], (A[:,3]>0)
        if arr.ndim==2 and arr.shape[1]>=3:
            X = arr[:,:3]; vis = (arr[:,3]>0) if arr.shape[1]>3 else None
            return X, vis
    return None, None

def parse_2d(entry):
    """AlphaPose 2D: flat [x,y,score,...] -> (N,3)"""
    if 'keypoints' not in entry: return None
    kp = np.array(entry['keypoints'], dtype=float).reshape(-1,3)
    return kp

def frame_num(fname):
    try:
        return int(os.path.splitext(os.path.basename(fname))[0])
    except:
        base=os.path.splitext(os.path.basename(fname))[0]
        for p in base.split('_')[::-1]:
            if p.isdigit(): return int(p)
        return 0

# ----------------------------
# 3D → 2D relative projection (no global recenter)
# ----------------------------
def project3d_relative(X, use_plane="xy"):
    """
    Returns 2D **relative** coords centered at the 3D pose centroid and
    normalized by its max XY span (so it's size-invariant before we scale to 2D).
    """
    if X is None or X.size==0:
        return np.zeros((0,2)), 1.0
    Xc = X - X.mean(axis=0, keepdims=True)       # remove translation (root-relative)
    Y = Xc[:, [0,1]] if use_plane=="xy" else Xc[:, [0,2]]
    # Do NOT invert Y here — keep dataset convention (your 2D already looks correct)
    span = np.maximum(Y.max(axis=0) - Y.min(axis=0), 1e-6)
    scale = span.max()
    Yrel = Y / scale                              # roughly within [-0.5, 0.5] box
    return Yrel, scale

# ----------------------------
# 2D anchor (center & scale) from 2D keypoints
# ----------------------------
def center_and_scale_2d(kp2d, conf_thr=0.05):
    """
    Returns (center_xy, size_px). Uses visible joints to estimate person box size on the frame.
    """
    if kp2d is None: return None, None
    m = kp2d[:,2] > conf_thr
    if not np.any(m): return None, None
    xy = kp2d[m,:2]
    center = xy.mean(axis=0)
    size = np.max(xy.max(axis=0) - xy.min(axis=0))  # max of width/height
    # Fallback minimum to avoid tiny/zero scale
    size = float(max(size, 40.0))
    return center, size

# ----------------------------
# Main conversion
# ----------------------------
def convert_json3d_to_images(json_path, video_path, output_dir,
                             highlight_ids=None, use_plane="xy", rel_to_2d_scale=1.0):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    w_res = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_res = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(json_path,'r') as f:
        data = json.load(f)

    frames = {}
    for e in data:
        fid = e.get('image_id')
        if fid: frames.setdefault(fid, []).append(e)
    sorted_fids = sorted(frames.keys(), key=frame_num)

    t0 = time.perf_counter()
    for fi, fid in enumerate(sorted_fids):
        frame = np.ones((h_res, w_res, 3), dtype=np.uint8) * 255
        draw_axes(frame, step=100)

        id_to_proj = {}
        people_sorted = sorted(frames[fid], key=lambda p: (p.get('idx') is None, p.get('idx', 1e9)))

        for person in people_sorted:
            X3, vis3 = parse_3d(person)
            kp2d   = parse_2d(person)

            # 1) 3D relative shape (no translation)
            Yrel, _ = project3d_relative(X3, use_plane=use_plane)  # ~[-0.5,0.5]

            # 2) 2D anchor: where to place & how big on THIS frame
            center2d, size2d = center_and_scale_2d(kp2d)

            if Yrel.size == 0 or center2d is None or size2d is None:
                # If missing either, just skip or draw a placeholder
                continue

            # 3) Scale relative shape to person's 2D size
            #    rel_to_2d_scale lets you tune how big the 3D skeleton is vs. 2D box
            s = rel_to_2d_scale * size2d
            P = np.column_stack([center2d[0] + s * Yrel[:,0],
                                 center2d[1] + s * Yrel[:,1]])

            idx_val = person.get('idx', None)
            id_to_proj[idx_val] = (P, vis3)

            # context draw
            draw_skeleton_2d(frame, P, vis3, color=(180,180,180), th=2)
            if idx_val is not None and len(P)>0:
                x,y = P[0].astype(int)
                _txt(frame, str(idx_val), (int(x), max(0,int(y)-10)), 0.6)

        # highlight two (optional)
        if highlight_ids is not None:
            a,b = highlight_ids
            if a in id_to_proj:
                Pa,Va = id_to_proj[a]; draw_skeleton_2d(frame, Pa, Va, (0,0,255), 2)
            if b in id_to_proj:
                Pb,Vb = id_to_proj[b]; draw_skeleton_2d(frame, Pb, Vb, (255,0,0), 2)

        cv2.imwrite(os.path.join(output_dir, f"plot_{fi}.png"), frame)
        print(f"Frame {fi+1}/{len(sorted_fids)} • Elapsed: {time.perf_counter()-t0:.2f}s")

    return output_dir

# ----------------------------
# GUI
# ----------------------------
def run_pose_plotter_3d():
    result = {"json": None, "video": None, "name": None}

    def browse_json():
        p = filedialog.askopenfilename(title="Select 3D JSON File", filetypes=[("JSON Files","*.json")])
        json_path_var.set(p)

    def browse_video():
        p = filedialog.askopenfilename(title="Select Source Video", filetypes=[("Video Files","*.mp4 *.avi *.mov")])
        video_path_var.set(p)

    def run_processing():
        json_path = json_path_var.get()
        video_path = video_path_var.get()
        video_name = video_name_entry.get()

        if not json_path or not os.path.exists(json_path):
            messagebox.showerror("Missing JSON","Please select a valid 3D JSON file."); return
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Missing Video","Please select a valid video file."); return
        if not video_name.strip():
            messagebox.showerror("Missing Name","Please enter a video name."); return

        OUTPUT_DIR = 'AlphaPose_Code/output_plots'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clear_all()

        # rel_to_2d_scale: tweak if you want the 3D bones thicker/larger vs the 2D person box (default 0.4)
        convert_json3d_to_images(json_path, video_path, OUTPUT_DIR,
                                 highlight_ids=None, use_plane="xy", rel_to_2d_scale=0.4)
        make_video(video_name, video_path)

        result.update({"json": json_path, "video": video_path, "name": video_name})
        root.quit(); root.destroy()

    root = tk.Tk(); root.title("3D Pose → 2D Motion (anchored)")
    json_path_var = tk.StringVar(); video_path_var = tk.StringVar()

    tk.Label(root, text="3D JSON File:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=json_path_var, width=50).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_json).grid(row=0, column=2)

    tk.Label(root, text="Video File:").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=video_path_var, width=50).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_video).grid(row=1, column=2)

    tk.Label(root, text="Output Video Name:").grid(row=2, column=0, sticky="e")
    video_name_entry = tk.Entry(root); video_name_entry.grid(row=2, column=1)

    tk.Button(root, text="Run 3D Pose Plotter", command=run_processing).grid(row=3, column=1, pady=10)
    root.mainloop()

    return result["json"], result["video"], result["name"]

if __name__ == "__main__":
    run_pose_plotter_3d()

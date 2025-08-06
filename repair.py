import json
import numpy as np
from collections import defaultdict

# Load AlphaPose JSON
with open("backupresults.json") as f:
    data = json.load(f)

# Organize by frame
frames_by_image = defaultdict(list)
for entry in data:
    frames_by_image[entry["image_id"]].append(entry)

def frame_number(k): return int(k.split('.')[0])
sorted_frame_keys = sorted(frames_by_image.keys(), key=frame_number)

# --- Helpers ---
def normalize_keypoints(kp):
    kp = kp.copy()
    visible = kp[:, 2] > 0
    if not np.any(visible):
        return kp
    min_xy = np.min(kp[visible, :2], axis=0)
    max_xy = np.max(kp[visible, :2], axis=0)
    size = max_xy - min_xy
    size[size == 0] = 1
    kp[:, :2] = (kp[:, :2] - min_xy) / size
    return kp

def pose_distance(kp1, kp2):
    kp1 = normalize_keypoints(kp1)
    kp2 = normalize_keypoints(kp2)
    mask = (kp1[:, 2] > 0) & (kp2[:, 2] > 0)
    if np.sum(mask) == 0:
        return np.inf
    dist = np.linalg.norm(kp1[mask, :2] - kp2[mask, :2], axis=1)
    conf = (kp1[mask, 2] + kp2[mask, 2]) / 2
    return np.sum(dist * conf) / np.sum(conf)

def get_center(kp):
    try:
        hips = kp[[11, 12]]
        if hips[:, 2].min() > 0:
            return np.mean(hips[:, :2], axis=0)
    except:
        pass
    return kp[0, :2] if kp[0, 2] > 0 else np.array([0, 0])

def spatial_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

# --- Fixed ID Pool (built from frame 0) ---
id_templates = {}  # fixed_id → (keypoints, center)
frame_0_key = sorted_frame_keys[0]
next_id = 0

# Assign initial IDs
for person in frames_by_image[frame_0_key]:
    kp = np.array(person["keypoints"]).reshape(-1, 3)
    cid = get_center(kp)
    id_templates[next_id] = (kp, cid)
    person["idx"] = next_id
    next_id += 1

print(f"🧠 Initialized with {len(id_templates)} fixed IDs from frame 0")

# --- Track with fixed ID pool ---
for frame_idx, frame in enumerate(sorted_frame_keys[1:], start=1):
    current_entries = frames_by_image[frame]

    used_ids = set()
    for person in current_entries:
        kp = np.array(person["keypoints"]).reshape(-1, 3)
        center = get_center(kp)

        # Match to fixed ID pool
        best_id = None
        best_score = float('inf')

        for fixed_id, (ref_kp, ref_center) in id_templates.items():
            if fixed_id in used_ids:
                continue
            pose_score = pose_distance(kp, ref_kp)
            center_score = spatial_distance(center, ref_center) / 100
            total_score = 0.7 * pose_score + 0.3 * center_score
            if total_score < best_score:
                best_score = total_score
                best_id = fixed_id

        if best_id is not None:
            person["idx"] = best_id
            id_templates[best_id] = (kp, center)  # update template
            used_ids.add(best_id)
            print(f"[Frame {frame_idx}] Assigned ID {best_id} (score={best_score:.2f})")
        else:
            print(f"[Frame {frame_idx}] ❌ No match found — dropping detection")

# Save output
with open("result_recycled.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Saved to result_recycled.json with fixed ID pool")

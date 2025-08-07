import json
import numpy as np
from collections import defaultdict, deque

# --- Parameters ---
POSE_HISTORY = 10           # Rolling memory size
MAX_CENTER_JUMP = 125      # Max pixels allowed between frames
POSE_SIM_THRESHOLD = 0.30  # Pose distance threshold
EMBED_SIM_THRESHOLD = 0.25  # Embedding distance threshold

# Optional dummy embedding function (you can replace with real model)
def pose_to_embedding(kp):
    visible = kp[:, 2] > 0
    if not np.any(visible): return np.zeros(10)
    flat = kp[visible, :2].flatten()
    return flat / np.linalg.norm(flat) if np.linalg.norm(flat) else flat

def pose_distance(kp1, kp2):
    mask = (kp1[:, 2] > 0) & (kp2[:, 2] > 0)
    if np.sum(mask) == 0: return np.inf
    return np.mean(np.linalg.norm(kp1[mask, :2] - kp2[mask, :2], axis=1))

def embedding_distance(e1, e2):
    return np.linalg.norm(e1 - e2)

def get_center(kp):
    try:
        hips = kp[[11, 12]]
        if hips[:, 2].min() > 0:
            return np.mean(hips[:, :2], axis=0)
    except:
        pass
    return kp[0, :2] if kp[0, 2] > 0 else np.array([0, 0])

# --- Load Data ---
with open("backupresults.json") as f:
    data = json.load(f)

frames_by_image = defaultdict(list)
for entry in data:
    frames_by_image[entry["image_id"]].append(entry)

def frame_number(k): return int(k.split('.')[0])
sorted_frame_keys = sorted(frames_by_image.keys(), key=frame_number)

# --- Initialize Fixed ID Pool from Frame 0 ---
frame0 = sorted_frame_keys[0]
id_pose_history = {}     # id → deque of poses
id_embeddings = {}       # id → fixed embedding
id_centers = {}          # id → last known center
next_id = 0

for person in frames_by_image[frame0]:
    kp = np.array(person["keypoints"]).reshape(-1, 3)
    cid = get_center(kp)
    person_id = next_id
    next_id += 1

    id_pose_history[person_id] = deque([kp], maxlen=POSE_HISTORY)
    id_embeddings[person_id] = pose_to_embedding(kp)
    id_centers[person_id] = cid
    person["idx"] = person_id

print(f"🧠 Initialized {len(id_pose_history)} fixed IDs from frame 0")

# --- Track Through Remaining Frames ---
for frame_idx, frame_key in enumerate(sorted_frame_keys[1:], start=1):
    current_entries = frames_by_image[frame_key]
    assigned_ids = set()

    for person in current_entries:
        kp = np.array(person["keypoints"]).reshape(-1, 3)
        center = get_center(kp)
        embed = pose_to_embedding(kp)

        best_id = None
        best_score = float('inf')

        # 1️⃣ Pose History Matching
        for pid in id_pose_history:
            if pid in assigned_ids:
                continue
            for prev_kp in id_pose_history[pid]:
                dist = pose_distance(kp, prev_kp)
                if dist < best_score and dist < POSE_SIM_THRESHOLD:
                    if np.linalg.norm(center - id_centers[pid]) < MAX_CENTER_JUMP:
                        best_id = pid
                        best_score = dist

        # 2️⃣ Embedding Matching (fallback)
        if best_id is None:
            for pid in id_embeddings:
                if pid in assigned_ids:
                    continue
                dist = embedding_distance(embed, id_embeddings[pid])
                if dist < EMBED_SIM_THRESHOLD:
                    if np.linalg.norm(center - id_centers[pid]) < MAX_CENTER_JUMP:
                        best_id = pid
                        best_score = dist
                        print(f"[{frame_key}] 🧬 Embedding matched ID {pid} (score={dist:.2f})")

        # 3️⃣ Assign or Skip
        if best_id is not None:
            person["idx"] = best_id
            id_pose_history[best_id].append(kp)
            id_centers[best_id] = center
            assigned_ids.add(best_id)
            print(f"[{frame_key}] ✅ Assigned ID {best_id}")
        else:
            person["skip"] = True  # flag to skip in output
            print(f"[{frame_key}] ⏩ Skipped unmatched person")

data = [entry for entry in data if not entry.get("skip")]
# --- Save Output ---
with open("result_recycled.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Saved to result_recycled.json with fixed IDs only")

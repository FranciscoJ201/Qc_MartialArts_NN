import json
import numpy as np
from collections import defaultdict, deque

# --- Load AlphaPose JSON ---
with open("backupresults.json") as f:
    data = json.load(f)

# --- Organize entries by frame ---
frames_by_image = defaultdict(list)
for entry in data:
    frames_by_image[entry["image_id"]].append(entry)

def frame_number(k): return int(k.split('.')[0])
sorted_frame_keys = sorted(frames_by_image.keys(), key=frame_number)

# --- Helper Functions ---

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
    # Use mid-hip as rough center
    try:
        center_points = kp[[11, 12]]  # left and right hip
        if center_points[:, 2].min() > 0:
            return np.mean(center_points[:, :2], axis=0)
    except:
        pass
    # fallback to nose
    return kp[0, :2] if kp[0, 2] > 0 else np.array([0, 0])

def spatial_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

# --- Tracking Structures ---
next_id = 0
id_map = {}                  # AlphaPose ID → new recycled ID
active_ids = set()
available_ids = []
pose_history = defaultdict(lambda: deque(maxlen=5))  # recycled_id → last 5 poses
center_history = {}          # recycled_id → last known center
frame_seen = {}              # recycled_id → last frame index

# --- Main Loop ---
for frame_idx, frame in enumerate(sorted_frame_keys):
    current_ids = set()
    used_ids = set()
    people = frames_by_image[frame]

    for person in people:
        alpha_id = person["idx"]
        keypoints = np.array(person["keypoints"]).reshape(-1, 3)
        center = get_center(keypoints)

        matched_id = None
        best_score = float('inf')

        # Try match to active + available IDs (recently disappeared)
        candidate_ids = list(set(pose_history.keys()) - used_ids)
        for cid in candidate_ids:
            history_poses = pose_history[cid]
            avg_pose_dist = np.mean([pose_distance(keypoints, past_kp) for past_kp in history_poses])
            spatial_dist = spatial_distance(center, center_history.get(cid, center))
            time_penalty = frame_idx - frame_seen.get(cid, 0)

            score = 0.6 * avg_pose_dist + 0.3 * spatial_dist / 100 + 0.1 * time_penalty
            if score < best_score and score < 0.8:  # <-- threshold
                best_score = score
                matched_id = cid

        # Assign new ID if no good match
        if matched_id is None:
            matched_id = next_id
            next_id += 1
            print(f"[NEW ID] Assigned {matched_id} to original AlphaPose ID {alpha_id}")
        else:
            print(f"[REUSE] Frame {frame_idx}: AlphaPose ID {alpha_id} → Recycled ID {matched_id} (score={best_score:.2f})")

        id_map[alpha_id] = matched_id
        person["idx"] = matched_id
        current_ids.add(alpha_id)
        used_ids.add(matched_id)

        # Update histories
        pose_history[matched_id].append(keypoints)
        center_history[matched_id] = center
        frame_seen[matched_id] = frame_idx

    # Update active IDs
    disappeared = active_ids - current_ids
    for aid in disappeared:
        available_ids.append(id_map[aid])
    active_ids = current_ids

# --- Save Output ---
with open("result_recycled.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Final result saved to result_recycled.json")

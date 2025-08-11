import json
import numpy as np
from collections import defaultdict, deque

# ----------------------------
# Tunables
# ----------------------------
POSE_HISTORY = 5            # how many past poses per ID to keep
MAX_CENTER_JUMP = 150.0     # max pixels a person can move between frames
POSE_SIM_THRESHOLD = 0.9  # lower = stricter (0 ~ identical after normalization)
CENTER_WEIGHT = 0.3         # blend center distance into the score
POSE_WEIGHT = 0.7           # blend pose distance into the score
OUTPUT_JSON = "repaired.json"

# ----------------------------
# Helpers
# ----------------------------
def frame_number(k: str) -> int:
    # turns "000123.jpg" -> 123, "123.png" -> 123
    base = k.split('.')[0]
    try:
        return int(base)
    except:
        # fall back: try underscores like "img_123.jpg"
        for part in base.split('_')[::-1]:
            if part.isdigit():
                return int(part)
        return 0

def arr_from_keypoints(entry):
    """AlphaPose keypoints: flat list [x1,y1,score1, x2,y2,score2, ...]."""
    kp = np.array(entry["keypoints"], dtype=float).reshape(-1, 3)
    return kp

def visible_mask(kp, conf_thresh=0.05):
    return kp[:, 2] > conf_thresh

def center_of(kp):
    m = visible_mask(kp)
    if not np.any(m):
        return None
    xy = kp[m, :2]
    return xy.mean(axis=0)

def normalize_pose(kp):
    """Center to mean(x,y), scale by RMS distance to center to be size/position invariant."""
    m = visible_mask(kp)
    if not np.any(m):
        return None
    xy = kp[m, :2]
    c = xy.mean(axis=0, keepdims=True)
    xy0 = xy - c
    scale = np.sqrt((xy0**2).sum(axis=1).mean()) + 1e-6
    xy_norm = xy0 / scale
    out = kp.copy()
    out[m, :2] = xy_norm
    return out, c.ravel(), scale

def pose_distance(kp_a, kp_b):
    """Mean L2 over joints visible in both, after per-pose normalization."""
    norm_a = normalize_pose(kp_a)
    norm_b = normalize_pose(kp_b)
    if norm_a is None or norm_b is None:
        return np.inf
    na, _, _ = norm_a
    nb, _, _ = norm_b
    ma = visible_mask(na)
    mb = visible_mask(nb)
    both = ma & mb
    if not np.any(both):
        return np.inf
    diffs = na[both, :2] - nb[both, :2]
    return float(np.linalg.norm(diffs, axis=1).mean())

def center_distance(kp_a, kp_b):
    ca = center_of(kp_a)
    cb = center_of(kp_b)
    if ca is None or cb is None:
        return np.inf
    return float(np.linalg.norm(ca - cb))

# ----------------------------
# Load AlphaPose JSON
# ----------------------------
with open("dataFIX/backupresults.json", "r") as f:
    data = json.load(f)

frames_by_image = defaultdict(list)
for entry in data:
    frames_by_image[entry["image_id"]].append(entry)

sorted_frame_keys = sorted(frames_by_image.keys(), key=frame_number)
if not sorted_frame_keys:
    raise RuntimeError("No frames found in backupresults.json")

first_frame = sorted_frame_keys[0]

# ----------------------------
# Lock the ID universe from frame 0
# ----------------------------
initial_people = frames_by_image[first_frame]
# The exact IDs that appeared initially (we will only ever reuse these)
initial_ids = [p.get("idx") for p in initial_people if "idx" in p]
# If idx missing, create a stable set [0..N-1]
if len(initial_ids) != len(initial_people) or any(i is None for i in initial_ids):
    initial_ids = list(range(len(initial_people)))

id_set = list(initial_ids)  # fixed ordering
id_to_history = {pid: deque(maxlen=POSE_HISTORY) for pid in id_set}
id_to_last_center = {pid: None for pid in id_set}
id_present_flag = {pid: True for pid in id_set}  # start as True (present) as you requested

# Initialize histories from first frame (best-effort greedy by current idx match)
# Build a map idx->kp for first frame
idx_to_kp_first = {}
for person in initial_people:
    kp = arr_from_keypoints(person)
    pid = person.get("idx")
    if pid is None:
        # assign by order fallback
        continue
    idx_to_kp_first[pid] = kp

# If we had to synthesize IDs, assign in listed order
if not idx_to_kp_first and initial_people:
    for pid, person in zip(id_set, initial_people):
        idx_to_kp_first[pid] = arr_from_keypoints(person)

for pid in id_set:
    if pid in idx_to_kp_first:
        kp0 = idx_to_kp_first[pid]
        id_to_history[pid].append(kp0)
        id_to_last_center[pid] = center_of(kp0)

# ----------------------------
# Pass: repair across frames
# ----------------------------
repaired_entries = []  # same schema as input, but with repaired 'idx'

for frame_key in sorted_frame_keys:
    detections = frames_by_image[frame_key]

    # Reset presence flags this frame
    for pid in id_set:
        id_present_flag[pid] = False

    # Precompute kp for all detections in this frame
    det_kps = [arr_from_keypoints(d) for d in detections]
    det_used = [False] * len(det_kps)

    # Build candidate scores (pid, det_idx) with constraints
    # Score blends normalized pose distance and (scaled) center distance
    candidates = []
    for pid in id_set:
        # Get a reference pose for this pid: prefer recent history average
        if len(id_to_history[pid]) == 0:
            ref_kp = None
        elif len(id_to_history[pid]) == 1:
            ref_kp = id_to_history[pid][-1]
        else:
            # average last K normalized poses back to original coords is messy,
            # so we simply pick the most recent as reference (robust + cheap)
            ref_kp = id_to_history[pid][-1]

        last_c = id_to_last_center[pid]

        for j, kp in enumerate(det_kps):
            if det_used[j]:
                continue

            # center gate: don't allow impossible jumps
            if (last_c is not None) and (center_distance(kp, ref_kp if ref_kp is not None else kp) == np.inf):
                # if ref pose invalid but we have last center, fall through to jump test vs kp center
                pass

            if last_c is not None:
                c = center_of(kp)
                if c is None:
                    continue
                jump = float(np.linalg.norm(c - last_c))
                if jump > MAX_CENTER_JUMP:
                    continue  # impossible teleport -> reject

            # pose distance (if no ref, set to a moderate value to allow initial reacquire)
            if ref_kp is None:
                pdist = 0.5  # permissive when we have no history
            else:
                pdist = pose_distance(ref_kp, kp)
                if pdist > POSE_SIM_THRESHOLD:
                    # too dissimilar to this ID -> reject
                    continue

            # center contribution (smaller is better); if no last center, treat as 0 weight
            if id_to_last_center[pid] is None:
                cdist = 0.0
            else:
                cdist = center_distance(ref_kp if ref_kp is not None else kp, kp)
                if not np.isfinite(cdist):
                    cdist = MAX_CENTER_JUMP  # degrade gracefully

            score = POSE_WEIGHT * pdist + CENTER_WEIGHT * (cdist / max(1.0, MAX_CENTER_JUMP))
            candidates.append((score, pid, j))

    # Greedy assignment: lowest score wins per (pid, det)
    candidates.sort(key=lambda x: x[0])
    assigned_pid = set()
    assigned_det = set()
    for score, pid, j in candidates:
        if pid in assigned_pid or j in assigned_det or det_used[j]:
            continue
        # assign
        assigned_pid.add(pid)
        assigned_det.add(j)
        det_used[j] = True
        id_present_flag[pid] = True

        # update history/center
        kp = det_kps[j]
        id_to_history[pid].append(kp)
        id_to_last_center[pid] = center_of(kp)

        # write repaired entry (force idx to locked pid)
        fixed = dict(detections[j])  # shallow copy original dict
        fixed["idx"] = pid
        repaired_entries.append(fixed)

    # Any leftover detections are *not* from our initial set -> drop them for this frame.
    # Any IDs not assigned this frame remain absent (flag False). We do not fabricate entries.

# ----------------------------
# Save
# ----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(repaired_entries, f)
print(f"Wrote {len(repaired_entries)} repaired entries to {OUTPUT_JSON}")

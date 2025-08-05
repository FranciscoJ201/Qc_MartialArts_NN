import json
from collections import defaultdict

# --- Load the AlphaPose result file ---
with open("alphapose-results.json") as f:
    data = json.load(f)

# --- Group entries by frame ---
frames_by_image = defaultdict(list)
for entry in data:
    frames_by_image[entry["image_id"]].append(entry)

# --- Sort frames numerically by filename ---
sorted_frame_keys = sorted(frames_by_image.keys(), key=lambda k: int(k.split('.')[0]))

# --- ID Recycling Logic ---
id_map = {}          # Maps original AlphaPose IDs to new ones
available_ids = []   # Freed IDs for reuse
next_id = 0          # Next new ID if no recycled one available
active_ids = set()   # IDs currently visible

# --- Process frames in order ---
for frame in sorted_frame_keys:
    current_ids = set()
    for person in frames_by_image[frame]:
        alpha_id = person["idx"]

        # Assign a new or recycled ID
        if alpha_id not in id_map:
            if available_ids:
                id_map[alpha_id] = available_ids.pop(0)
            else:
                id_map[alpha_id] = next_id
                next_id += 1

        current_ids.add(alpha_id)
        person["idx"] = id_map[alpha_id]

    # Detect which IDs disappeared
    disappeared = active_ids - current_ids
    for old_id in disappeared:
        available_ids.append(id_map[old_id])

    active_ids = current_ids

# --- Save to a new file ---
with open("result_recycled.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Recycled ID JSON saved to result_recycled.json")

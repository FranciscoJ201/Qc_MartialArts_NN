import json
import numpy as np
from collections import defaultdict, deque



#CONSTANTS 
INPUT_PATH = 'example2.json'
OUTPUT_PATH = 'Test.json'
def frame_number(k: str) -> int:
    # turns "000123.jpg" -> 123, "123.png" -> 123, "img_123.jpg" -> 123
    base = k.split('.')[0]
    try:
        return int(base)
    except:
        for part in base.split('_')[::-1]:
            if part.isdigit():
                return int(part)
        return 0
    

with open(INPUT_PATH, "r") as f:
        data = json.load(f)
framedata = defaultdict(list)
#save as a dict so that it doesnt go back and forth searching through the file 
initial_ids = set() #ids for each person in a set
for entry in data:
    framedata[entry["image_id"]].append(entry)
    initial_ids.add(entry['idx'])
#sorted keys for easy access later
sorted_frame_keys = sorted(framedata.keys(), key=frame_number)
if not sorted_frame_keys:
    raise RuntimeError("No frames found in the selected JSON.")
#self explanatory
first_frame = sorted_frame_keys[0]

pose_dict = {}
for id in initial_ids:
    pose_dict[id] = []

for key in sorted_frame_keys:
    for entry in framedata[key]:
        idx = entry['idx']
        pose_dict[idx].append(entry['keypoints'])
#each keypoint entry is already ordered because the JSON its drawing from is ordered so pose_dict[1][0]
#is equivalent to keypoints for idx 1 at frame 0


with open(OUTPUT_PATH, "w") as f:
    json.dump(pose_dict, f)
    
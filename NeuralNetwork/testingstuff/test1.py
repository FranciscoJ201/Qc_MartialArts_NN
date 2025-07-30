import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.pyplot as plt



json_path = f'json_input/test.json'
# Open and load the JSON file
with open(json_path, 'r') as file:
    data = json.load(file)
    
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (11, 24),
    (11, 22), (22, 23),
    (8, 12), (12, 13), (13, 14), (14, 21),
    (14, 19), (19, 20),
    (0, 15), (15, 17),
    (0, 16), (16, 18)
]

keypoints = []
for keypoint in data['keypoints']:
    keypoints.append(keypoint)

x_vals = []
y_vals = []
conf_vals = []

# Iterate over keypoints in steps of 3
for i in range(0, len(keypoints), 3):
    x_vals.append(keypoints[i])
    y_vals.append(keypoints[i+1])
    conf_vals.append(keypoints[i+2])


plt.figure(figsize=(10, 10))

# Plot all keypoints as dots
plt.scatter(x_vals, y_vals, c='red', s=20)

# Draw lines only for defined edges
for (i, j) in edges:
    if i < len(x_vals) and j < len(x_vals):
        x_line = [x_vals[i], x_vals[j]]
        y_line = [y_vals[i], y_vals[j]]
        plt.plot(x_line, y_line, color='blue', linewidth=2)

plt.xlim(min(x_vals) - 50, max(x_vals) + 50)
plt.ylim(min(y_vals) - 50, max(y_vals) + 50)
plt.gca().invert_yaxis()  # Invert y-axis to match image coords
plt.title('Skeleton keypoints with connected edges')
plt.grid(True)
plt.show()
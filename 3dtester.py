# reader3d_single_frame.py
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# Skeleton edges (SMPL24 or COCO17 depending on your JSON)
# ----------------------------
SMPL24_EDGES = [
    (0,1),(1,4),(4,7),(7,10),
    (0,2),(2,5),(5,8),(8,11),
    (0,3),(3,6),(6,9),(9,12),(12,15),
    (12,13),(13,16),(16,18),(18,20),(20,22),
    (12,14),(14,17),(17,19),(19,21),(21,23)
]

def plot_single_frame_3d(json_file, frame_index=0, use_edges=True):
    # Load AlphaPose or SMPL JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    # Pick a frame
    entries = [d for d in data if d.get("image_id") == f"{frame_index}.jpg"]
    if not entries:
        raise ValueError(f"No data for frame {frame_index}")

    # Just take the first person for now
    person = entries[0]

    # AlphaPose 3D output is usually "pred_xyz_jts"
    if "pred_xyz_jts" not in person:
        raise ValueError("No 3D joints (pred_xyz_jts) found in JSON")

    keypoints = np.array(person["pred_xyz_jts"])  # shape: (num_joints, 3)
    x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter points
    ax.scatter3D(x, y, z, c=z, cmap="viridis", s=50)

    # Draw skeleton edges
    if use_edges:
        for i, j in SMPL24_EDGES:
            if i < len(x) and j < len(x):
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c="black")

    scale = 3.0  # multiplier to zoom out
    ax.set_xlim([x.min()*scale, x.max()*scale])
    ax.set_ylim([y.min()*scale, y.max()*scale])
    ax.set_zlim([z.min()*scale, z.max()*scale])


    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.title(f"3D Pose Frame {frame_index}")
    plt.show()

if __name__ == "__main__":
    # Example usage:
    plot_single_frame_3d("repaired.json", frame_index=0)

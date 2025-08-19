# reader3d_video.py
import json, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib.widgets import Button, Slider



#CONSTS ---------
lim = (-1.5,1.5)
# lim = None
#----------------

# ----------------------------
# Skeleton edges (COCO-17 default; swap to SMPL24 if you prefer)
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

def frame_number(k: str) -> int:
    base = os.path.splitext(k)[0]
    try:
        return int(base)
    except:
        # Fallback: grab last underscore-separated int
        for part in base.split('_')[::-1]:
            if part.isdigit():
                return int(part)
        return 0

def load_frames(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Group by image_id
    frames = {}
    for d in data:
        frames.setdefault(d["image_id"], []).append(d)
    # Sort by frame number
    keys = sorted(frames.keys(), key=frame_number)
    return keys, frames

def select_person_entry(entries, target_idx=None):
    if not entries:
        return None
    if target_idx is None:
        return entries[0]  # first person in frame
    for e in entries:
        if e.get("idx") == target_idx:
            return e
    return None  # not found this frame

def get_xyz_from_entry(entry):
    """
    Returns (x, y, z) arrays or (None, None, None) if missing.
    Expects 'pred_xyz_jts' shape (J, 3).
    """
    if entry is None:
        return None, None, None
    kj = entry.get("pred_xyz_jts")
    if kj is None:
        return None, None, None
    kp = np.array(kj, dtype=float)
    if kp.ndim != 2 or kp.shape[1] < 3:
        return None, None, None
    return kp[:, 0], kp[:, 1], kp[:, 2]

class Pose3DPlayer:
    def __init__(
        self,
        json_path,
        target_idx=None,           # track id to follow, or None for first person
        edges=SMPL24_EDGES,
        fps=15,
        fixed_limits= lim,         # e.g., (-1000,1000) to force all axes same range
        auto_scale_margin=1.2,     # margin factor if not using fixed_limits
        point_size=40
    ):
        self.keys, self.frames = load_frames(json_path)
        if not self.keys:
            raise RuntimeError("No frames found in JSON.")
        self.fps = max(1, int(fps))
        self.target_idx = target_idx
        self.edges = edges
        self.interval = int(1000 / self.fps)
        self.fixed_limits = fixed_limits
        self.auto_scale_margin = auto_scale_margin
        self.point_size = point_size

        # state
        self.i = 0
        self.playing = False

        # figure + axes
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("3D Pose Player")

        # initial data
        x, y, z = self._get_xyz(self.i)
        if x is None:
            # Try to find a frame with data
            for k in range(len(self.keys)):
                x, y, z = self._get_xyz(k)
                if x is not None:
                    self.i = k
                    break

        if x is None:
            raise RuntimeError("Could not find any frame with 'pred_xyz_jts' data.")

        # artists
        self.scat = self.ax.scatter3D(x, y, z, s=self.point_size)
        self.lines = []
        for (a, b) in self.edges:
            if a < len(x) and b < len(x):
                ln, = self.ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
                self.lines.append((ln, a, b))
        # axes limits
        self._set_limits(x, y, z)

        # UI: buttons + slider
        self._add_widgets()

        # keyboard bindings
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # timer for playback
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._on_timer)

    # ---------- Data helpers ----------
    def _get_xyz(self, idx):
        key = self.keys[idx]
        entries = self.frames[key]
        entry = select_person_entry(entries, self.target_idx)
        return get_xyz_from_entry(entry)

    def _set_limits(self, x, y, z):
        if self.fixed_limits is not None:
            lo, hi = self.fixed_limits
            self.ax.set_xlim(lo, hi)
            self.ax.set_ylim(lo, hi)
            self.ax.set_zlim(lo, hi)
        else:
            # autoscale with margin
            xs = np.array(x); ys = np.array(y); zs = np.array(z)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            zmin, zmax = zs.min(), zs.max()

            # make cubic-ish box so rotations look nice
            cmin = min(xmin, ymin, zmin)
            cmax = max(xmax, ymax, zmax)
            span = (cmax - cmin) * self.auto_scale_margin
            center = (cmax + cmin) / 2.0
            lo = center - span / 2.0
            hi = center + span / 2.0

            self.ax.set_xlim(lo, hi)
            self.ax.set_ylim(lo, hi)
            self.ax.set_zlim(lo, hi)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    # ---------- UI ----------
    def _add_widgets(self):
        # layout: reserve space at bottom
        plt.subplots_adjust(bottom=0.18)  # or a bit more if you need space

        ax_prev   = plt.axes([0.12, 0.10, 0.10, 0.06])
        ax_play   = plt.axes([0.24, 0.10, 0.12, 0.06])
        ax_next   = plt.axes([0.38, 0.10, 0.10, 0.06])

        # ⬇️ Move the frame slider to span the bottom
        ax_slider = plt.axes([0.12, 0.04, 0.76, 0.04])

        # Keep FPS just above or beside buttons (your call)
        ax_fps    = plt.axes([0.55, 0.10, 0.38, 0.06])

        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_play = Button(ax_play, "Play")
        self.btn_next = Button(ax_next, "Next")
        self.slider = Slider(ax_slider, "Frame", 0, len(self.keys) - 1, valinit=self.i, valstep=1)

        self.fps_slider = Slider(ax_fps, "FPS", 1, 60, valinit=self.fps, valstep=1)

        self.btn_prev.on_clicked(lambda evt: self.step(-1))
        self.btn_next.on_clicked(lambda evt: self.step(1))
        self.btn_play.on_clicked(lambda evt: self.toggle_play())
        self.slider.on_changed(self._on_slider)
        self.fps_slider.on_changed(self._on_fps_changed)  # NEW

    # ---------- Events ----------
    def _on_key(self, event):
        if event.key == " ":
            self.toggle_play()
        elif event.key == "left":
            self.step(-1)
        elif event.key == "right":
            self.step(1)
        elif event.key == "r":
            self.ax.view_init(elev=20, azim=-60)
            self.fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(self.fig)

    def _on_slider(self, val):
        self.i = int(val)
        self._draw_frame(self.i)

    def _on_fps_changed(self, val):
        self.fps = int(val)
        self.interval = int(1000 / self.fps)
        # restart timer if currently playing so new interval takes effect
        if self.playing:
            self.timer.stop()
            self.timer = self.fig.canvas.new_timer(interval=self.interval)
            self.timer.add_callback(self._on_timer)
            self.timer.start()
        # (Optional) reflect FPS in the title
        # self._set_title(self.i)

    def _on_timer(self):
        if not self.playing:
            return
        self.i = (self.i + 1) % len(self.keys)
        self.slider.set_val(self.i)  # also triggers _draw_frame

    # ---------- Controls ----------
    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def step(self, delta):
        self.playing = False
        self.btn_play.label.set_text("Play")
        self.timer.stop()
        self.i = (self.i + delta) % len(self.keys)
        self.slider.set_val(self.i)  # updates plot

    # ---------- Drawing ----------
    def _draw_frame(self, i):
        x, y, z = self._get_xyz(i)
        if x is None:
            # Hide if missing data on this frame
            self.scat._offsets3d = ([], [], [])
            for ln, a, b in self.lines:
                ln.set_data_3d([], [], [])
            self.fig.canvas.draw_idle()
            return

        # update scatter
        self.scat._offsets3d = (x, y, z)

        # update lines
        n = len(x)
        for ln, a, b in self.lines:
            if a < n and b < n:
                ln.set_data_3d([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
            else:
                ln.set_data_3d([], [], [])

        # update limits (comment out if you want them fixed from frame 0)
        if self.fixed_limits is None:
            self._set_limits(x, y, z)

        # update title
        frame_label = self.keys[i]
        who = f"idx={self.target_idx}" if self.target_idx is not None else "first person"
        self.ax.set_title(f"3D Pose Player — {frame_label} ({who})")

        self.fig.canvas.draw_idle()

    # ---------- Run ----------
    def run(self):
        self._draw_frame(self.i)
        plt.show()

if __name__ == "__main__":
    # Example usage:
    # - Set json_path to your AlphaPose 3D output that contains 'pred_xyz_jts'
    # - Optionally set target_idx to a specific track id after running your repair step
    json_path = "/Users/franciscojimenez/Desktop/repaired.json"
    # json_path = " "
    viewer = Pose3DPlayer(
        json_path=json_path,
        target_idx=2,        # or an integer track id, e.g., 0 or 1
        edges=SMPL24_EDGES,
        fps=30,
        fixed_limits= lim,     
        auto_scale_margin=1.3,  # enlarge the autoscaled cube a bit
        point_size=40
    )
    viewer.run()

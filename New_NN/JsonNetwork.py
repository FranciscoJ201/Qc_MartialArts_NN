import os, json, glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---- Settings ----
batch_size = 8
epochs = 25
learning_rate = 1e-3
num_classes = 3
NUM_KPTS = 17        # change if your JSONs have a different count
USE_XY_ONLY = True   # if your JSON has [x,y,score], set True to use XY only

# ---- Utilities: extract & normalize keypoints ----
def extract_keypoints(obj):
    """
    Returns a flat list of keypoints for ONE person.
    Supports a few common shapes:
      - {"keypoints": [x1,y1,s1, x2,y2,s2, ...]}
      - {"people":[{"pose_keypoints_2d":[...]}]}  (OpenPose)
      - {"preds":[[x,y,conf], ...]} or {"keypoints":[[x,y,conf], ...]} (AlphaPose-like)
    """
    if "keypoints" in obj and isinstance(obj["keypoints"], list):
        kp = obj["keypoints"]
        # could be flat or [[x,y,c],...]
        if len(kp) == NUM_KPTS * 3 and isinstance(kp[0], (int, float)):
            return kp
        if isinstance(kp[0], list):
            return [v for trip in kp for v in trip]  # flatten
    if "people" in obj and obj["people"]:
        kp = obj["people"][0].get("pose_keypoints_2d", [])
        return kp
    if "preds" in obj and obj["preds"]:
        kp = obj["preds"]
        return [v for trip in kp for v in trip]  # flatten
    raise ValueError("Unsupported JSON format")

def normalize_xy_flat(flat):
    """
    flat: [x1,y1,score1, x2,y2,score2, ...] or [x1,y1, x2,y2, ...]
    1) keep XY (drop scores if present)
    2) center at mid-hip (avg of L/R hip if available, else first point)
    3) scale by shoulder distance (or overall std fallback) for size invariance
    """
    stride = 3 if not USE_XY_ONLY and len(flat) == NUM_KPTS*3 else (3 if len(flat)==NUM_KPTS*3 else 2)
    # get xy only
    xy = []
    if stride == 3:
        for i in range(0, len(flat), 3):
            xy.extend([flat[i], flat[i+1]])
    else:
        xy = flat[:]

    import numpy as np
    arr = np.array(xy, dtype="float32").reshape(NUM_KPTS, 2)

    # indices (COCO17): L hip=11, R hip=12, L shoulder=5, R shoulder=6
    def safe_idx(i): return i if 0 <= i < NUM_KPTS else None

    lhip, rhip = safe_idx(11), safe_idx(12)
    if lhip is not None and rhip is not None:
        root = (arr[lhip] + arr[rhip]) / 2.0
    else:
        root = arr[0]

    arr = arr - root  # center

    lsh, rsh = safe_idx(5), safe_idx(6)
    if lsh is not None and rsh is not None:
        shoulder_dist = float(((arr[lsh] - arr[rsh])**2).sum() ** 0.5)
    else:
        shoulder_dist = float(arr.std())  # fallback

    scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0
    arr = arr / scale

    return arr.reshape(-1).astype("float32")

# ---- Dataset for JSON keypoints in class folders ----
class KeypointsFolder(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # class subfolders
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.samples = []  # (json_path, class_idx)
        for ci, cname in enumerate(self.classes):
            for p in glob.glob(os.path.join(root_dir, cname, "*.json")):
                self.samples.append((p, ci))

        # input dimension (XY only or XYZ/conf kept)
        self.in_dim = NUM_KPTS * (2 if USE_XY_ONLY else 3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, "r") as f:
            obj = json.load(f)
        flat = extract_keypoints(obj)

        if USE_XY_ONLY:
            x = normalize_xy_flat(flat)
        else:
            # keep xyz/score but still center/scale XY; append score back
            # simple route: normalize with XY-only then re-attach scores as-is
            xy_norm = normalize_xy_flat(flat)
            scores = []
            if len(flat) == NUM_KPTS * 3:
                scores = [flat[i] for i in range(2, len(flat), 3)]
            import numpy as np
            x = np.concatenate([xy_norm, np.array(scores, dtype="float32")]) if scores else xy_norm

        return torch.from_numpy(x), label

# ---- Build datasets & loaders ----
train_data = KeypointsFolder(root_dir='New_NN/dataset/train')
test_data  = KeypointsFolder(root_dir='New_NN/dataset/test')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size)

# ---- Model (MLP on flattened keypoints) ----
INPUT_SIZE = len(train_data[0][0])  # inferred from dataset

class PoseNet(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = PoseNet(INPUT_SIZE, num_classes)

# ---- Loss/Opt ----
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ---- Training / Testing (same as your loops) ----
train_losses, test_accuracies = [], []

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total = 0.0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        total += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 10 == 0:
            print(f"loss: {loss.item():.4f}")
    train_losses.append(total / len(dataloader))

@torch.no_grad()
def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss, correct = 0.0, 0
    size = len(dataloader.dataset)
    for X, y in dataloader:
        pred = model(X)
        total_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).sum().item()
    acc = 100.0 * correct / size
    test_accuracies.append(acc)
    print(f"Test Accuracy: {acc:.2f}% | Avg Loss: {total_loss / len(dataloader):.4f}")

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)

# ---- Single prediction demo ----
label_map = train_data.classes
sample, label = test_data[0]
model.eval()
with torch.no_grad():
    pred = model(sample.unsqueeze(0))
    pred_label = label_map[pred.argmax(1).item()]
print(f"\nPredicted: {pred_label}, Actual: {label_map[label]}")

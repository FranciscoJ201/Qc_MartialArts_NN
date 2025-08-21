import os, json, glob, collections

# Point this to a file OR folder OR glob pattern
PATH = r"C:/Users/Francisco Jimenez/Desktop/3d.json"   # e.g. r"C:/.../repaired.json" or r"C:/.../*.json" or a folder

# ------------ Helpers ------------
def analyze_entry(e):
    info = {}
    # 2D keypoints
    if isinstance(e, dict) and "keypoints" in e and isinstance(e["keypoints"], list):
        L = len(e["keypoints"])
        info["keypoints_len"] = L
        info["keypoints_joints_xys"] = (L // 2) if L % 2 == 0 else None
        info["keypoints_joints_xyscore"] = (L // 3) if L % 3 == 0 else None
    # 3D joints
    if isinstance(e, dict) and "pred_xyz_jts" in e and isinstance(e["pred_xyz_jts"], list):
        J = len(e["pred_xyz_jts"])
        K = len(e["pred_xyz_jts"][0]) if J and isinstance(e["pred_xyz_jts"][0], list) else "?"
        info["pred_xyz_jts_shape"] = (J, K)
    return info

def inspect_file(file):
    print(f"\n=== {file} ===")
    try:
        obj = json.load(open(file, "r"))
    except Exception as ex:
        print(f"  [ERROR] Could not parse JSON: {ex}")
        return

    # Case A: list of entries (AlphaPose-style)
    if isinstance(obj, list):
        if not obj:
            print("  [WARN] Empty list.")
            return
        kp_len_counter = collections.Counter()
        j3d_counter = collections.Counter()
        has_kp = has_xyz = 0

        # sample some entries (first 100 or all if small)
        sample = obj[:min(100, len(obj))]
        for e in sample:
            info = analyze_entry(e)
            if "keypoints_len" in info:
                has_kp += 1
                kp_len_counter[info["keypoints_len"]] += 1
            if "pred_xyz_jts_shape" in info:
                has_xyz += 1
                j3d_counter[info["pred_xyz_jts_shape"]] += 1

        print(f"  Entries inspected: {len(sample)} / {len(obj)}")
        if has_kp:
            print(f"  keypoints present in {has_kp} / {len(sample)} entries")
            for L, cnt in kp_len_counter.items():
                joints2 = L//2 if L%2==0 else None
                joints3 = L//3 if L%3==0 else None
                shape = f"{L} (xy={joints2} joints, xy+score={joints3} joints)"
                print(f"    - len={shape}: {cnt} entries")
        else:
            print("  No 'keypoints' found in sampled entries.")

        if has_xyz:
            print(f"  pred_xyz_jts present in {has_xyz} / {len(sample)} entries")
            for shape, cnt in j3d_counter.items():
                J, K = shape
                hint = " (SMPL-24)" if J == 24 and K >= 3 else ""
                print(f"    - shape={J}x{K}{hint}: {cnt} entries")
        else:
            print("  No 'pred_xyz_jts' found in sampled entries.")

        # Show one concrete example
        e0 = sample[0]
        ex = analyze_entry(e0)
        print("  Example (first sampled entry):", ex)

    # Case B: single object with fields (per-frame file, etc.)
    elif isinstance(obj, dict):
        info = analyze_entry(obj)
        if not info:
            print("  No 'keypoints' or 'pred_xyz_jts' keys found.")
        else:
            print("  Info:", info)
            if "keypoints_len" in info:
                L = info["keypoints_len"]
                print(f"  len(keypoints) = {L}  → xy={L//2 if L%2==0 else 'n/a'} joints, xy+score={L//3 if L%3==0 else 'n/a'} joints")
            if "pred_xyz_jts_shape" in info:
                J, K = info["pred_xyz_jts_shape"]
                hint = " (SMPL-24)" if J == 24 and (K == 3 or K == "?") else ""
                print(f"  pred_xyz_jts shape = {J} x {K}{hint}")

    else:
        print(f"  [WARN] Unsupported JSON top-level type: {type(obj)}")

# ------------ Resolve PATH ------------
files = []
if os.path.isdir(PATH):
    files = glob.glob(os.path.join(PATH, "*.json"))
elif any(ch in PATH for ch in "*?[]"):
    files = glob.glob(PATH)
else:
    files = [PATH]

if not files:
    print("[ERROR] No JSON files found. Check PATH.")
else:
    # Inspect up to 5 files to keep output readable
    for f in files[:5]:
        inspect_file(f)
    if len(files) > 5:
        print(f"\n[Note] {len(files)-5} more files not shown.")

# make_spatialvid_metadata_kaggle.py
import os
import json
import pandas as pd
from pathlib import Path

# --- INPUT (read-only) ---
INPUT_DATASET_PATH = "/kaggle/input/group1-first50/SmallDataset"   # ← your dataset

# --- OUTPUT (writable) ---
OUTPUT_CSV_PATH = "/kaggle/working/SpatialVID_HQ_metadata.csv"     # ← saved here!

videos_dir = Path(INPUT_DATASET_PATH) / "videos"

rows = []

print("Scanning videos and loading captions...")

for video_path in videos_dir.rglob("*.mp4"):
    rel_path = video_path.relative_to(INPUT_DATASET_PATH)   # e.g. videos/group_0001/abc123.mp4
    
    group = video_path.parts[-3]   # group_0001
    uuid = video_path.stem       # abc123...

    caption_file = Path(INPUT_DATASET_PATH) / "annotations" / group / uuid / "caption.json"
    
    if caption_file.exists():
        try:
            with open(caption_file, "r", encoding="utf-8") as f:
                caption = json.load(f)["caption"]
        except:
            caption = "A detailed 3D object rotating on a turntable"
    else:
        caption = "A detailed 3D object rotating on a turntable"

    rows.append({
        "video": str(rel_path),
        "prompt": caption.strip() + ", photorealistic, sharp focus, studio lighting, 8k, turntable rotation"
    })

# Save CSV to writable directory
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Success! Created metadata with {len(df)} videos")
print(f"→ Saved to: {OUTPUT_CSV_PATH}")
print("\nFirst few rows:")
print(df.head())
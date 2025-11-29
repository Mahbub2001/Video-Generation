# make_spatialvid_metadata_kaggle_FIXED.py
import json
import pandas as pd
from pathlib import Path

INPUT_DATASET_PATH = "/kaggle/input/group1-first50/SmallDataset"
OUTPUT_CSV_PATH = "/kaggle/working/SpatialVID_HQ_metadata.csv"

videos_dir = Path(INPUT_DATASET_PATH) / "videos"
rows = []

print("Scanning videos and loading REAL captions...")

for video_path in videos_dir.rglob("*.mp4"):
    rel_path = video_path.relative_to(INPUT_DATASET_PATH)
    uuid_clean = video_path.stem                     # e.g. 0a00f99d-9d9a-5265-9548-e97a34c1302c

    # Try three possible folder names (covers all known SpatialVID versions)
    possible_folders = [
        uuid_clean,           # normal
        uuid_clean + "_",     # ← most common bug
        uuid_clean + "__",    # rare
    ]

    caption = "A photorealistic 3D object rotating on a turntable, studio lighting, 8k"
    found = False

    for folder_name in possible_folders:
        caption_file = Path(INPUT_DATASET_PATH) / "annotations" / video_path.parts[-3] / folder_name / "caption.json"
        if caption_file.exists():
            try:
                data = json.load(open(caption_file))
                raw_caption = data.get("caption") or data.get("text") or "A beautiful 3D render"
                caption = raw_caption.strip()
                found = True
                break
            except:
                pass

    # Final polished prompt
    final_prompt = caption + ", extremely detailed, sharp focus, cinematic studio lighting, 8k, smooth 360-degree rotation, turntable shot"

    rows.append({
        "video": str(rel_path),
        "prompt": final_prompt
    })

# Save
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\nSuccess! {len(df)} videos → real captions loaded: {sum('turntable' not in r['prompt'].lower() for r in rows)}")
print(f"CSV saved → {OUTPUT_CSV_PATH}")
print("\nFirst 5 REAL captions:")
print(df.head())
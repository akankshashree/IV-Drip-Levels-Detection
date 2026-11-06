
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "near_duplicate_review" / "candidate_pairs.csv"
OUT_DIR = ROOT / "results" / "near_duplicate_review" / "cross_class"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[df["class_1"] != df["class_2"]].copy()
df = df.sort_values("distance")

print("Cross-class candidate pairs:", len(df))

def load_image(relative_path):
    image = cv2.imread(str(ROOT / relative_path))
    if image is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    return cv2.resize(image, (320, 240))

def labeled_image(path, class_name):
    image = load_image(path)
    bar = np.full((55, 320, 3), 245, dtype=np.uint8)

    cv2.putText(
        bar, f"Class: {class_name}", (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )
    cv2.putText(
        bar, Path(path).name[:43], (5, 43),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
    )

    return np.vstack([image, bar])

pairs_per_sheet = 10

for start in range(0, len(df), pairs_per_sheet):
    batch = df.iloc[start:start + pairs_per_sheet]
    rows = []

    for _, pair in batch.iterrows():
        left = labeled_image(pair["image_1"], pair["class_1"])
        right = labeled_image(pair["image_2"], pair["class_2"])
        combined = np.hstack([left, right])

        header = np.full((32, combined.shape[1], 3), 230, dtype=np.uint8)
        text = f"Distance: {pair['distance']}"
        cv2.putText(
            header, text, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        rows.append(np.vstack([header, combined]))

    sheet = np.vstack(rows)
    number = start // pairs_per_sheet + 1
    path = OUT_DIR / f"cross_class_{number:02d}.jpg"
    cv2.imwrite(str(path), sheet)
    print("Saved:", path)

print("Finished. Original images were not changed.")

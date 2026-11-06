
from pathlib import Path
from collections import defaultdict

import cv2
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "mobilenetv2_split.csv"

df = pd.read_csv(MANIFEST)

def perceptual_hash(path):
    image = cv2.imread(str(path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return bits

records = []

print("Hashing images...")

for row in df.itertuples(index=False):
    path = Path(row.path)
    if not path.is_absolute():
        path = ROOT / path

    bits = perceptual_hash(path)
    if bits is not None:
        records.append((str(path), row.label, row.split, bits))

print("Images hashed:", len(records))
print("Comparing images across different splits...")

cross_split_pairs = []
for i in range(len(records)):
    path_a, label_a, split_a, hash_a = records[i]

    for j in range(i + 1, len(records)):
        path_b, label_b, split_b, hash_b = records[j]

        if split_a == split_b:
            continue

        distance = int(np.count_nonzero(hash_a != hash_b))

        if distance <= 5:
            cross_split_pairs.append(
                (distance, split_a, label_a, split_b, label_b,
                 path_a, path_b)
            )

print("\nNear-duplicate pairs across splits:", len(cross_split_pairs))
print("Pairs with different labels:",
      sum(a[2] != a[4] for a in cross_split_pairs))

output = ROOT / "results" / "near_duplicate_pairs.csv"
pd.DataFrame(
    cross_split_pairs,
    columns=[
        "hash_distance", "split_a", "label_a", "split_b",
        "label_b", "path_a", "path_b"
    ]
).to_csv(output, index=False)

print("Saved report:", output)

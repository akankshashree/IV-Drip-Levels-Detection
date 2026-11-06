
from pathlib import Path
from collections import Counter
import csv
import hashlib
import random

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results" / "dataset_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIRS = [
    ROOT / "datasets" / "dataset_1",
    ROOT / "datasets" / "dataset_2",
]

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

SEED = 42
MAX_HASH_DISTANCE = 5
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def perceptual_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, (32, 32))
    dct = cv2.dct(np.float32(image))
    low = dct[:8, :8]
    median = np.median(low.flatten()[1:])

    result = 0
    for bit in (low > median).flatten():
        result = (result << 1) | int(bit)
    return result


def hamming_distance(a, b):
    return (a ^ b).bit_count()


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return

        if self.rank[a] < self.rank[b]:
            a, b = b, a

        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


# Read all images and retain their folder labels.
records = []

for dataset_dir in DATASET_DIRS:
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            records.append({
                "path": path,
                "relative_path": str(path.relative_to(ROOT)),
                "label": path.parent.name,
                "dataset": dataset_dir.name,
            })

records.sort(key=lambda item: item["relative_path"])
print("Images found:", len(records))

if not records:
    raise SystemExit("No images found. Check dataset folders.")

labels = {record["label"] for record in records}
expected = {"sal_data_100", "sal_data_80", "sal_data_50", "sal_data_empty"}

if labels != expected:
    raise SystemExit(f"Unexpected class folders: {sorted(labels)}")

print("Calculating perceptual hashes...")
hashes = []

for i, record in enumerate(records):
    hashes.append(perceptual_hash(record["path"]))

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(records)}")

if any(value is None for value in hashes):
    raise SystemExit("Some images could not be read. Resolve this before splitting.")

# Connect all images whose perceptual hashes are sufficiently similar.
# Connected images remain together in the same split.
print("Grouping near-duplicate candidates...")
uf = UnionFind(len(records))

for i in range(len(records)):
    for j in range(i + 1, len(records)):
        if hamming_distance(hashes[i], hashes[j]) <= MAX_HASH_DISTANCE:
            uf.union(i, j)

components = {}
for i in range(len(records)):
    root = uf.find(i)
    components.setdefault(root, []).append(i)

groups = list(components.values())
print("Near-duplicate groups:", len(groups))
print("Largest group size:", max(map(len, groups)))

# Summarize group labels before making any split.
mixed_groups = []
for group in groups:
    group_labels = {records[i]["label"] for i in group}
    if len(group_labels) > 1:
        mixed_groups.append(group)

print("Groups containing multiple labels:", len(mixed_groups))

# Assign whole groups to splits, aiming for 70/15/15 overall.
# Greedy balancing is used; this is not guaranteed to be class-stratified.
random.Random(SEED).shuffle(groups)
split_counts = {name: Counter() for name in RATIOS}
split_sizes = Counter()
assignments = {}

targets = {
    name: len(records) * ratio for name, ratio in RATIOS.items()
}

for group in sorted(groups, key=len, reverse=True):
    group_label_counts = Counter(records[i]["label"] for i in group)

    def cost(split_name):
        size_after = split_sizes[split_name] + len(group)
        size_cost = abs(size_after - targets[split_name]) / max(1, targets[split_name])

        class_cost = 0
        for label in expected:
            desired = sum(r["label"] == label for r in records) * RATIOS[split_name]
            after = split_counts[split_name][label] + group_label_counts[label]
            class_cost += abs(after - desired) / max(1, desired)

        return size_cost + class_cost

    chosen = min(RATIOS, key=cost)

    for i in group:
        assignments[i] = chosen

    split_sizes[chosen] += len(group)
    split_counts[chosen].update(group_label_counts)

# Save a manifest; no image files are copied or changed.
manifest_path = OUTPUT_DIR / "split_manifest.csv"

with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label", "dataset", "split", "group_id"])

    for group_number, group in enumerate(components.values(), start=1):
        for i in group:
            writer.writerow([
                records[i]["relative_path"],
                records[i]["label"],
                records[i]["dataset"],
                assignments[i],
                group_number,
            ])

print("\nSplit summary:")
for split_name in RATIOS:
    print(f"\n{split_name}: {split_sizes[split_name]} images")
    for label in sorted(expected):
        print(f"  {label}: {split_counts[split_name][label]}")

print("\nManifest saved to:", manifest_path)
print("Original images were not modified.")
print(
    "\nIMPORTANT: Review mixed-label groups and group sizes before "
    "using this split for model training."
)

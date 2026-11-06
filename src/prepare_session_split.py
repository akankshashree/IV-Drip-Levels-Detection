
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import random
import re

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results" / "session_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIRS = [
    ROOT / "datasets" / "dataset_1",
    ROOT / "datasets" / "dataset_2",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEED = 42

# Provisional: start a new group when adjacent captures are
# more than 60 seconds apart, within the same dataset and class.
MAX_GAP_SECONDS = 60

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

TIMESTAMP_PATTERN = re.compile(r"IMG(\d{14})")


def get_timestamp(path):
    match = TIMESTAMP_PATTERN.search(path.stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


records = []

for dataset_dir in DATASET_DIRS:
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            timestamp = get_timestamp(path)
            records.append({
                "path": path,
                "relative_path": str(path.relative_to(ROOT)),
                "dataset": dataset_dir.name,
                "label": path.parent.name,
                "timestamp": timestamp,
            })

records.sort(
    key=lambda r: (
        r["dataset"],
        r["label"],
        r["timestamp"] or datetime.min,
        r["relative_path"],
    )
)

if not records:
    raise SystemExit("No images found.")

print("Images found:", len(records))
print("Creating provisional timestamp groups...")

# Group adjacent images only within the same dataset and class.
groups = []
current_group = []
previous = None

for record in records:
    same_source = (
        previous is not None
        and record["dataset"] == previous["dataset"]
        and record["label"] == previous["label"]
        and record["timestamp"] is not None
        and previous["timestamp"] is not None
    )

    within_gap = (
        same_source
        and (record["timestamp"] - previous["timestamp"]).total_seconds()
        <= MAX_GAP_SECONDS
    )

    if current_group and not within_gap:
        groups.append(current_group)
        current_group = []

    current_group.append(record)
    previous = record

if current_group:
    groups.append(current_group)

print("Provisional groups:", len(groups))
print("Largest group:", max(len(g) for g in groups))
print("Groups containing missing timestamps:",
      sum(any(r["timestamp"] is None for r in g) for g in groups))

# Greedy assignment of whole groups to target split sizes and class counts.
rng = random.Random(SEED)
rng.shuffle(groups)

labels = sorted({r["label"] for r in records})
targets = {
    split: len(records) * ratio
    for split, ratio in SPLIT_RATIOS.items()
}
class_totals = Counter(r["label"] for r in records)

split_sizes = Counter()
split_classes = {s: Counter() for s in SPLIT_RATIOS}
assignments = {}

for group in sorted(groups, key=len, reverse=True):
    group_counts = Counter(r["label"] for r in group)

    def cost(split):
        size_after = split_sizes[split] + len(group)
        size_cost = abs(size_after - targets[split]) / max(1, targets[split])

        class_cost = 0
        for label in labels:
            target_class = class_totals[label] * SPLIT_RATIOS[split]
            after = split_classes[split][label] + group_counts[label]
            class_cost += abs(after - target_class) / max(1, target_class)

        return size_cost + class_cost

    chosen = min(SPLIT_RATIOS, key=cost)

    for record in group:
        assignments[record["relative_path"]] = chosen

    split_sizes[chosen] += len(group)
    split_classes[chosen].update(group_counts)

manifest = OUTPUT_DIR / "session_split_manifest.csv"

with manifest.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "image_path", "dataset", "label", "timestamp",
        "provisional_group_id", "split",
    ])

    for group_id, group in enumerate(groups, start=1):
        for record in group:
            writer.writerow([
                record["relative_path"],
                record["dataset"],
                record["label"],
                record["timestamp"].isoformat()
                if record["timestamp"] else "",
                group_id,
                assignments[record["relative_path"]],
            ])

print("\nSplit summary:")
for split in SPLIT_RATIOS:
    print(f"\n{split}: {split_sizes[split]} images")
    for label in labels:
        print(f"  {label}: {split_classes[split][label]}")

print("\nManifest saved to:", manifest)
print("Original images were not modified.")
print(
    "\nWARNING: Timestamp groups are provisional. "
    "Review them before using this manifest for model evaluation."
)

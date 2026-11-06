
from pathlib import Path
from collections import Counter
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
OUTPUT = ROOT / "results" / "temporal_split.csv"

CLASSES = [
    "sal_data_100",
    "sal_data_80",
    "sal_data_50",
    "sal_data_empty",
]

def get_timestamp(path):
    match = re.search(r"IMG(\d{14})", path.name)
    if not match:
        return None
    return pd.to_datetime(
        match.group(1), format="%Y%m%d%H%M%S", errors="coerce"
    )

records = []

for class_name in CLASSES:
    images = []

    for path in DATASETS.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and path.parent.name == class_name
        ):
            timestamp = get_timestamp(path)
            if timestamp is not None and pd.notna(timestamp):
                images.append((path, timestamp))

    if not images:
        raise RuntimeError(f"No timestamped images found for {class_name}")

    # Keep every image sharing a timestamp in the same partition.
    unique_times = sorted({timestamp for _, timestamp in images})
    n = len(unique_times)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    time_to_split = {}
    for i, timestamp in enumerate(unique_times):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "validation"
        else:
            split = "test"
        time_to_split[timestamp] = split

    for path, timestamp in images:
        records.append({
            "path": str(path.resolve()),
            "label": class_name,
            "split": time_to_split[timestamp],
            "timestamp": timestamp,
        })

df = pd.DataFrame(records)

if df["path"].duplicated().any():
    raise RuntimeError("Duplicate file paths found in the manifest.")

# Verify that no timestamp occurs in more than one split within a class.
checks = df.groupby(["label", "timestamp"])["split"].nunique()
if (checks > 1).any():
    raise RuntimeError("A timestamp was assigned to multiple splits.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Temporal split saved:", OUTPUT)
print("\nImages per class and split:")
print(pd.crosstab(df["label"], df["split"], margins=True))

print("\nTimestamp ranges:")
for (label, split), group in df.groupby(["label", "split"]):
    print(
        f"{label} | {split}: {len(group)} images | "
        f"{group['timestamp'].min()} to {group['timestamp'].max()}"
    )

print("\nNote: this is a chronological holdout, not an independent-bottle test.")

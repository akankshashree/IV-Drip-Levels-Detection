
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import statistics

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
PATTERN = re.compile(r"IMG(\d{14})")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

timestamps = defaultdict(list)

for path in DATASETS.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue

    match = PATTERN.search(path.stem)
    if not match:
        continue

    timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    key = (path.parts[-5], path.parent.name)
    timestamps[key].append((timestamp, path.name))

print("TIMESTAMP GAP SUMMARY")

for key, items in sorted(timestamps.items()):
    items.sort()
    gaps = [
        (items[i][0] - items[i - 1][0]).total_seconds()
        for i in range(1, len(items))
    ]

    if not gaps:
        continue

    gaps.sort()

    def percentile(values, p):
        return values[round((len(values) - 1) * p)]

    print(f"\nDataset: {key[0]} | Class: {key[1]}")
    print(f"Images: {len(items)}")
    print(f"Median gap: {statistics.median(gaps):.1f} seconds")
    print(f"90th percentile: {percentile(gaps, 0.90):.1f} seconds")
    print(f"Gaps <= 5 seconds: {sum(g <= 5 for g in gaps)}")
    print(f"Gaps > 30 seconds: {sum(g > 30 for g in gaps)}")
    print(f"Gaps > 60 seconds: {sum(g > 60 for g in gaps)}")

print("\nAnalysis complete. No files were modified.")

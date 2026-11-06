
from pathlib import Path
from collections import Counter
import re

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

suffix_counts = Counter()
examples = Counter()
time_ranges = {}

pattern = re.compile(r"IMG(\d{14})(?:_(\d+))?$")

for path in DATASETS.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue

    match = pattern.search(path.stem)

    if not match:
        suffix_counts["unmatched"] += 1
        continue

    timestamp = match.group(1)
    suffix = match.group(2)

    suffix_counts[suffix or "no_suffix"] += 1
    examples[suffix or "no_suffix"] += 1

    key = (path.parts[-5], path.parent.name)

    if key not in time_ranges:
        time_ranges[key] = [timestamp, timestamp]
    else:
        time_ranges[key][0] = min(time_ranges[key][0], timestamp)
        time_ranges[key][1] = max(time_ranges[key][1], timestamp)

print("FILENAME SUFFIX COUNTS")
for suffix, count in suffix_counts.most_common():
    print(f"{suffix}: {count}")

print("\nTIMESTAMP RANGES")
for key, (start, end) in sorted(time_ranges.items()):
    print(f"{key}: {start} to {end}")

print("\nNo files were modified.")

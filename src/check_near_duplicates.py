
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIRS = [
    PROJECT_DIR / "datasets" / "dataset_1",
    PROJECT_DIR / "datasets" / "dataset_2",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def perceptual_hash(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (32, 32))
    dct = cv2.dct(np.float32(image))
    low_frequency = dct[:8, :8]

    median = np.median(low_frequency.flatten()[1:])
    bits = low_frequency > median

    result = 0
    for bit in bits.flatten():
        result = (result << 1) | int(bit)

    return result


def hamming_distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()


images = []

for folder in DATASET_DIRS:
    images.extend(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

print(f"Images to analyse: {len(images)}")
print("Calculating perceptual hashes...")

records = []

for index, path in enumerate(images, start=1):
    image_hash = perceptual_hash(path)

    if image_hash is not None:
        records.append((path, image_hash))

    if index % 250 == 0:
        print(f"Processed {index}/{len(images)}")

print("\nComparing images...")
print("This may take a few minutes.")

# A distance of 0 means identical perceptual hashes.
# A distance of 0-5 is a candidate near-duplicate match.
matches = []

for i in range(len(records)):
    path1, hash1 = records[i]

    for j in range(i + 1, len(records)):
        path2, hash2 = records[j]

        distance = hamming_distance(hash1, hash2)

        if distance <= 5:
            matches.append((distance, path1, path2))

matches.sort(key=lambda item: item[0])

print("\n" + "=" * 60)
print("NEAR-DUPLICATE CANDIDATES")
print("=" * 60)
print(f"Images successfully hashed: {len(records)}")
print(f"Candidate pairs found: {len(matches)}")

for distance, path1, path2 in matches[:50]:
    print(f"\nDistance: {distance}")
    print(f"  {path1.relative_to(PROJECT_DIR)}")
    print(f"  {path2.relative_to(PROJECT_DIR)}")

if len(matches) > 50:
    print(f"\nShowing first 50 of {len(matches)} candidate pairs.")

print("\nNo files were modified or deleted.")

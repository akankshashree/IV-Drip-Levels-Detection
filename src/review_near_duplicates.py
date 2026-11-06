
from pathlib import Path
import csv
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "results" / "near_duplicate_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIRS = [
    PROJECT_DIR / "datasets" / "dataset_1",
    PROJECT_DIR / "datasets" / "dataset_2",
]

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

MAX_DISTANCE = 5
MAX_PAIRS_PER_SHEET = 20


def perceptual_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

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


def class_name(path):
    return path.parent.name


def make_thumbnail(path, width=320, height=280):
    image = cv2.imread(str(path))

    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    image = cv2.resize(image, (width, height))
    return image


def add_label(image, lines):
    bar_height = 65
    bar = np.full(
        (bar_height, image.shape[1], 3),
        245,
        dtype=np.uint8
    )

    y = 22
    for line in lines:
        cv2.putText(
            bar,
            line[:48],
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA
        )
        y += 21

    return np.vstack([image, bar])


print("Scanning image files...")

images = []
for dataset_dir in DATASET_DIRS:
    images.extend(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

images = sorted(images)
print(f"Found {len(images)} images.")

print("Calculating perceptual hashes...")
records = []

for index, path in enumerate(images, start=1):
    image_hash = perceptual_hash(path)

    if image_hash is not None:
        records.append((path, image_hash))

    if index % 500 == 0:
        print(f"Processed {index}/{len(images)}")

print("Comparing image hashes...")
pairs = []

for i in range(len(records)):
    path1, hash1 = records[i]

    for j in range(i + 1, len(records)):
        path2, hash2 = records[j]
        distance = hamming_distance(hash1, hash2)

        if distance <= MAX_DISTANCE:
            pairs.append((distance, path1, path2))

pairs.sort(key=lambda item: item[0])

csv_path = OUTPUT_DIR / "candidate_pairs.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "distance",
        "image_1",
        "class_1",
        "image_2",
        "class_2",
    ])

    for distance, path1, path2 in pairs:
        writer.writerow([
            distance,
            str(path1.relative_to(PROJECT_DIR)),
            class_name(path1),
            str(path2.relative_to(PROJECT_DIR)),
            class_name(path2),
        ])

print(f"Saved all candidate pairs to: {csv_path}")
print(f"Total candidate pairs: {len(pairs)}")

print("Generating contact sheets...")

for start in range(0, len(pairs), MAX_PAIRS_PER_SHEET):
    batch = pairs[start:start + MAX_PAIRS_PER_SHEET]
    rows = []

    for distance, path1, path2 in batch:
        left = add_label(
            make_thumbnail(path1),
            [
                f"A: {path1.name}",
                f"Class: {class_name(path1)}",
            ]
        )

        right = add_label(
            make_thumbnail(path2),
            [
                f"B: {path2.name}",
                f"Class: {class_name(path2)}",
            ]
        )

        pair_image = np.hstack([left, right])

        header = np.full(
            (32, pair_image.shape[1], 3),
            230,
            dtype=np.uint8
        )

        cv2.putText(
            header,
            f"Pair {start + len(rows) + 1} | Hash distance: {distance}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA
        )

        rows.append(np.vstack([header, pair_image]))

    sheet = np.vstack(rows)
    sheet_number = start // MAX_PAIRS_PER_SHEET + 1
    sheet_path = OUTPUT_DIR / f"contact_sheet_{sheet_number:02d}.jpg"

    if not cv2.imwrite(str(sheet_path), sheet):
        print(f"WARNING: Could not save {sheet_path.name}")
    else:
        print(f"Saved {sheet_path.name}")

print("\nFinished.")
print(f"Review folder: {OUTPUT_DIR}")
print("Original dataset images were not modified.")

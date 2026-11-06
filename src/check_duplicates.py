
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIRS = [
    PROJECT_DIR / "datasets" / "dataset_1",
    PROJECT_DIR / "datasets" / "dataset_2",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


hash_to_paths = defaultdict(list)
total = 0

for dataset_dir in DATASET_DIRS:
    print(f"\nScanning: {dataset_dir}")

    images = [
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Images found: {len(images)}")

    for index, path in enumerate(images, start=1):
        hash_to_paths[file_hash(path)].append(path)
        total += 1

        if index % 250 == 0:
            print(f"  Hashed {index}/{len(images)} files")

duplicate_groups = [
    paths for paths in hash_to_paths.values()
    if len(paths) > 1
]

print("\n" + "=" * 55)
print("EXACT DUPLICATE CHECK")
print("=" * 55)
print(f"Total images checked: {total}")
print(f"Unique file contents: {len(hash_to_paths)}")
print(f"Duplicate groups: {len(duplicate_groups)}")

for group in duplicate_groups[:20]:
    print("\nDuplicate group:")
    for path in group:
        print(f"  {path.relative_to(PROJECT_DIR)}")

if len(duplicate_groups) > 20:
    print(f"\n...and {len(duplicate_groups) - 20} more groups")

print("\nNo files were changed or deleted.")

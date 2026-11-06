
from pathlib import Path
from collections import Counter

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "datasets"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def inspect_dataset(dataset_path):
    print("\n" + "=" * 65)
    print(f"DATASET: {dataset_path}")
    print("=" * 65)

    if not dataset_path.exists():
        print("ERROR: Dataset folder does not exist.")
        return

    images = sorted(
        path for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Total images found: {len(images)}")

    if not images:
        print("No supported images found. Check the extracted folders.")
        return

    # Count images by their immediate parent folder.
    folder_counts = Counter(path.parent.name for path in images)

    print("\nImage counts by folder:")
    for folder_name, count in sorted(folder_counts.items()):
        print(f"  {folder_name}: {count}")

    readable = 0
    unreadable = []

    print("\nChecking image readability and dimensions...")

    for index, path in enumerate(images, start=1):
        image = cv2.imread(str(path))

        if image is None:
            unreadable.append(path)
        else:
            readable += 1

            if index <= 10:
                height, width = image.shape[:2]
                print(
                    f"  {path.relative_to(dataset_path)} "
                    f"| width={width}, height={height}"
                )

        if index % 500 == 0:
            print(f"  Checked {index}/{len(images)} images...")

    print("\nRESULTS")
    print(f"Readable images: {readable}")
    print(f"Unreadable images: {len(unreadable)}")

    if unreadable:
        print("\nFirst 20 unreadable files:")
        for path in unreadable[:20]:
            print(f"  {path.relative_to(dataset_path)}")


if __name__ == "__main__":
    inspect_dataset(DATASETS_DIR / "dataset_1")
    inspect_dataset(DATASETS_DIR / "dataset_2")

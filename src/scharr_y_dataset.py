import argparse
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path):
    # np.fromfile + imdecode works with spaces and non-ASCII characters on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if ok:
        encoded.tofile(str(path))
    return ok


def resize_max(image, max_size):
    if max_size <= 0:
        return image
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )



def scharr_y(image, blur):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        # Kernel size must be odd.
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    # dx=0, dy=1 -> derivative along the y axis only (responds to horizontal edges).
    grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    # Absolute value, scaled to 8-bit for saving.
    return cv2.convertScaleAbs(grad_y)


def main():
    parser = argparse.ArgumentParser(
        description="Apply a Scharr filter (y axis only) to every image in a dataset."
    )
    parser.add_argument("input_dir", type=Path, help="Dataset root folder")
    parser.add_argument("output_dir", type=Path, help="Folder to save filtered images")
    parser.add_argument(
        "--flatten-by-class",
        action="store_true",
        help="Save as output/<class_folder>/file instead of copying the full "
        "folder structure (merges dataset_1 and dataset_2 per class).",
    )
    parser.add_argument(
        "--blur", type=int, default=5,
        help="Gaussian blur kernel size before Scharr (0 = off, default 5).",
    )
    parser.add_argument(
        "--max-size", type=int, default=1024,
        help="Downscale so the longest side is at most this many pixels "
        "(0 = keep original size, default 1024).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess images that already exist in the output folder.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and output_dir not in p.resolve().parents
    )

    if not files:
        raise SystemExit("No images found.")

    print(f"Found {len(files)} images.")

    done, skipped, existing = 0, 0, 0
    for i, path in enumerate(files, 1):
        if args.flatten_by_class:
            out_path = output_dir / path.parent.name / path.name
            if out_path.exists() and not args.overwrite:
                # Same filename from another dataset folder: keep both.
                source_tag = path.parents[2].name if len(path.parents) > 2 else "src"
                out_path = out_path.with_name(f"{source_tag}_{path.name}")
        else:
            out_path = output_dir / path.relative_to(input_dir)

        if out_path.exists() and not args.overwrite:
            existing += 1
            continue

        image = read_image(path)
        if image is None:
            print(f"Skipped (unreadable): {path}")
            skipped += 1
            continue

        image = resize_max(image, args.max_size)
        result = scharr_y(image, args.blur)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if write_image(out_path, result):
            done += 1
        else:
            print(f"Skipped (write failed): {path}")
            skipped += 1

        if i % 100 == 0:
            print(f"{i}/{len(files)} processed")

    print(
        f"Done. Saved {done}, already existed {existing}, skipped {skipped}. "
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
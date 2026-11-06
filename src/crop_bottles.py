
from pathlib import Path

import cv2
import pandas as pd

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = ROOT / "results" / "temporal_split.csv"
OUTPUT_DIR = ROOT / "datasets" / "bottle_crops"
REPORT_PATH = ROOT / "results" / "bottle_crop_report.csv"

PADDING = 20
MIN_CONTOUR_AREA = 500

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------- BOTTLE DETECTION ----------------

def detect_bottle(image):
    """
    Experimental foreground-contour detector.

    Returns (x1, y1, x2, y2) or None if no plausible
    tall foreground region is detected.

    Note: this is not a trained or validated bottle detector.
    """

    height, width = image.shape[:2]

    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Estimate background colour from the image borders.
    border_pixels = [
        image[0, :, :],
        image[-1, :, :],
        image[:, 0, :],
        image[:, -1, :],
    ]

    border_pixels = cv2.vconcat([
        pixels.reshape(-1, 1, 3)
        for pixels in border_pixels
    ])

    background = border_pixels.mean(axis=0).reshape(1, 1, 3)

    # FIX: match image data type and dimensions before absdiff.
    background = background.astype(blurred.dtype)
    background = cv2.resize(
        background,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    difference = cv2.absdiff(blurred, background)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Reduce small noise and close gaps.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Filter for reasonably tall regions.
        if h > 1.15 * w and h > height * 0.12:
            candidates.append((area, x, y, w, h))

    if not candidates:
        return None

    # Select the largest qualifying contour.
    candidates.sort(key=lambda item: item[0], reverse=True)

    _, x, y, w, h = candidates[0]

    # Add padding without exceeding image boundaries.
    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)
    x2 = min(width, x + w + PADDING)
    y2 = min(height, y + h + PADDING)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ---------------- PROCESS DATASET ----------------

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\n"
            "Run: python src\\prepare_temporal_split.py"
        )

    df = pd.read_csv(MANIFEST)

    required_columns = {"path", "label", "split"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Manifest must contain columns: {required_columns}"
        )

    if df.empty:
        raise RuntimeError("The manifest contains no images.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(df)

    for index, row in df.iterrows():
        source = Path(row["path"])
        label = str(row["label"])
        split = str(row["split"])

        record = {
            "source": str(source),
            "label": label,
            "split": split,
            "status": "",
        }

        if not source.is_file():
            record["status"] = "source_missing"
            records.append(record)
            continue

        if source.suffix.lower() not in VALID_EXTENSIONS:
            record["status"] = "unsupported_extension"
            records.append(record)
            continue

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

        if image is None:
            record["status"] = "read_failed"
            records.append(record)
            continue

        box = detect_bottle(image)

        if box is None:
            record["status"] = "no_bottle_detected"
            records.append(record)
            continue

        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            record["status"] = "empty_crop"
            records.append(record)
            continue

        destination_dir = OUTPUT_DIR / split / label
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / f"{source.stem}_crop.jpg"

        saved = cv2.imwrite(str(destination), crop)

        if not saved:
            record["status"] = "save_failed"
            records.append(record)
            continue

        record.update({
            "crop_path": str(destination),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "crop_width": crop.shape[1],
            "crop_height": crop.shape[0],
            "status": "cropped",
        })

        records.append(record)

        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"Processed {index + 1}/{total} images")

    # Save report even if some images could not be processed.
    report = pd.DataFrame(records)
    report.to_csv(REPORT_PATH, index=False)

    print("\n========== CROP SUMMARY ==========")
    print(f"Total manifest rows: {total}")

    print("\nStatuses:")
    print(report["status"].value_counts().to_string())

    print("\nStatuses by class:")
    print(
        pd.crosstab(
            report["label"],
            report["status"],
            margins=True,
        ).to_string()
    )

    print("\nStatuses by split:")
    print(
        pd.crosstab(
            report["split"],
            report["status"],
            margins=True,
        ).to_string()
    )

    print("\nCrops directory:", OUTPUT_DIR)
    print("Report saved:", REPORT_PATH)

    print(
        "\nIMPORTANT: visually inspect the crops before training. "
        "This experimental detector may select the wrong object, "
        "crop part of a bottle, or miss transparent bottle edges."
    )


if __name__ == "__main__":
    main()

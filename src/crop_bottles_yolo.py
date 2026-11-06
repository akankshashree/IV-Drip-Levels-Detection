
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "temporal_split.csv"
OUTPUT_DIR = ROOT / "datasets" / "yolo_bottle_crops"
REPORT = ROOT / "results" / "yolo_bottle_crop_report.csv"

MODEL_NAME = "yolo11n.pt"
CONFIDENCE = 0.15
PADDING_FRACTION = 0.08

# COCO class IDs: cup=41, bottle=39, vase=75.
# IV bags are not a dedicated COCO class, so use bottle and vase only.
ALLOWED_CLASSES = {39, 75}

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    model = YOLO(MODEL_NAME)
    rows = []

    for index, row in df.iterrows():
        source = Path(row["path"])
        label = row["label"]
        split = row["split"]

        record = {
            "source": str(source),
            "label": label,
            "split": split,
        }

        # Never process generated crops as source images.
        if "bottle_crops" in source.parts or "yolo_bottle_crops" in source.parts:
            record["status"] = "generated_crop_skipped"
            rows.append(record)
            continue

        image = cv2.imread(str(source))
        if image is None:
            record["status"] = "read_failed"
            rows.append(record)
            continue

        height, width = image.shape[:2]

        result = model.predict(
            source=image,
            conf=CONFIDENCE,
            verbose=False,
        )[0]

        candidates = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                if class_id not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w <= 0 or box_h <= 0:
                    continue

                aspect_ratio = box_h / box_w
                area_fraction = (
                    box_w * box_h / (width * height)
                )

                # Reject small or non-tall detections.
                if aspect_ratio < 1.15 or area_fraction < 0.03:
                    continue

                candidates.append(
                    (confidence, x1, y1, x2, y2, class_id)
                )

        if not candidates:
            record["status"] = "no_suitable_detection"
            rows.append(record)
            continue

        # Highest confidence among filtered detections.
        candidates.sort(reverse=True)
        confidence, x1, y1, x2, y2, class_id = candidates[0]

        pad_x = int((x2 - x1) * PADDING_FRACTION)
        pad_y = int((y2 - y1) * PADDING_FRACTION)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        crop = image[y1:y2, x1:x2]

        out_dir = OUTPUT_DIR / split / label
        out_dir.mkdir(parents=True, exist_ok=True)

        destination = out_dir / f"{source.stem}_crop.jpg"

        if not cv2.imwrite(str(destination), crop):
            record["status"] = "save_failed"
            rows.append(record)
            continue

        record.update({
            "crop_path": str(destination),
            "confidence": confidence,
            "detected_class_id": class_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "status": "cropped",
        })
        rows.append(record)

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)}")

    report = pd.DataFrame(rows)
    report.to_csv(REPORT, index=False)

    print("\n=== Crop status ===")
    print(report["status"].value_counts())

    print("\n=== Status by class ===")
    print(pd.crosstab(report["label"], report["status"]))

    print("\nCrops:", OUTPUT_DIR)
    print("Report:", REPORT)
    print(
        "\nInspect the resulting crops manually before training. "
        "COCO detections are not guaranteed to be correct for IV bottles."
    )

if __name__ == "__main__":
    main()

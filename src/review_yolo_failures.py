
from pathlib import Path
import random

import cv2
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "yolo_bottle_crop_report.csv"
OUTPUT = ROOT / "results" / "yolo_failure_review.png"

df = pd.read_csv(REPORT)
random.seed(42)

classes = [
    "sal_data_100",
    "sal_data_80",
    "sal_data_50",
    "sal_data_empty",
]

samples = []

for label in classes:
    failed = df[
        (df["label"] == label)
        & (df["status"] == "no_suitable_detection")
    ]
    successful = df[
        (df["label"] == label)
        & (df["status"] == "cropped")
    ]

    if not failed.empty:
        samples.extend(
            failed.sample(min(2, len(failed)), random_state=42)
            .to_dict("records")
        )

    if not successful.empty:
        samples.extend(
            successful.sample(min(2, len(successful)), random_state=43)
            .to_dict("records")
        )

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for ax, row in zip(axes, samples):
    path = Path(row["source"])
    image = cv2.imread(str(path))

    if image is None:
        ax.set_title("Could not read image")
        ax.axis("off")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)

    status = row["status"]
    ax.set_title(
        f"{row['label']}\n{status}",
        fontsize=9,
    )
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=130, bbox_inches="tight")
plt.close(fig)

print("Review image saved:", OUTPUT)
print("Images displayed:", len(samples))

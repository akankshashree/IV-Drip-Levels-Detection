
from pathlib import Path
import random

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "bottle_crop_report.csv"
OUTPUT = ROOT / "results" / "bottle_crop_review.png"

SAMPLE_SIZE = 12
SEED = 42

if not REPORT.exists():
    raise FileNotFoundError(
        "Crop report not found. Run python src\\crop_bottles.py first."
    )

df = pd.read_csv(REPORT)

if "status" not in df.columns:
    raise ValueError("The crop report has no status column.")

successful = df[df["status"] == "cropped"].copy()

if successful.empty:
    print("No successful crops to review.")
    print(df["status"].value_counts())
    raise SystemExit(1)

sample = successful.sample(
    n=min(SAMPLE_SIZE, len(successful)),
    random_state=SEED,
)

fig, axes = plt.subplots(3, 4, figsize=(14, 12))
axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for ax, (_, row) in zip(axes, sample.iterrows()):
    path = Path(row["crop_path"])

    try:
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(
            f"True label: {row['label']}\n"
            f"Split: {row['split']}",
            fontsize=9,
        )
        ax.axis("off")

    except Exception as exc:
        ax.text(
            0.5, 0.5, f"Failed to open crop:\n{exc}",
            ha="center", va="center",
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Review sheet saved:", OUTPUT)
print(f"Successful crops available: {len(successful)}")
print(f"Images displayed: {len(sample)}")

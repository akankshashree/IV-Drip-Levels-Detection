
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
OUTPUT = ROOT / "results" / "yolo_bottle_test.png"

CLASSES = [
    "sal_data_100",
    "sal_data_80",
    "sal_data_50",
    "sal_data_empty",
]

# Find a small sample of original images.
images = []
for path in DATASETS.rglob("*"):
    if (
        path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and path.parent.name in CLASSES
    ):
        images.append(path)

if not images:
    raise RuntimeError("No dataset images found.")

random.seed(42)
sample = random.sample(images, min(8, len(images)))

# Load a small pretrained YOLO model.
model = YOLO("yolo11n.pt")

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

for ax, path in zip(axes, sample):
    image = cv2.imread(str(path))

    if image is None:
        ax.set_title("Image could not be read")
        ax.axis("off")
        continue

    # COCO-pretrained YOLO may not recognize transparent IV bottles.
    result = model.predict(
        source=image,
        conf=0.15,
        verbose=False,
    )[0]

    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    ax.imshow(annotated)
    ax.set_title(f"{path.parent.name}\n{path.name}", fontsize=8)
    ax.axis("off")

for ax in axes[len(sample):]:
    ax.axis("off")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Test image saved:", OUTPUT)
print("Images tested:", len(sample))
print("Inspect the detections before proceeding.")

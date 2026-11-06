
from pathlib import Path
import sys
import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hog_linear_svc.joblib"

if len(sys.argv) < 2:
    print("Usage: python src\\predict.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_absolute():
    image_path = Path.cwd() / image_path

if not image_path.is_file():
    print("Error: Image file not found:", image_path)
    sys.exit(1)

saved = joblib.load(MODEL_PATH)
model = saved["model"]
image_size = tuple(saved["image_size"])

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Could not read the image.")
    sys.exit(1)

image = cv2.resize(image, image_size)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hog = cv2.HOGDescriptor(
    (96, 96),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)

features = hog.compute(gray).flatten().reshape(1, -1)
prediction = model.predict(features)[0]

level_names = {
    "sal_data_100": "100% filled",
    "sal_data_80": "80% filled",
    "sal_data_50": "50% filled",
    "sal_data_empty": "Empty"
}

print("Image:", image_path.name)
print("Predicted class:", prediction)
print("Estimated fluid category:", level_names.get(prediction, prediction))
print("\nPrototype prediction only; verify visually.")


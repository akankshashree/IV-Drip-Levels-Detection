
from pathlib import Path
import json

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent.parent
candidates = list((ROOT / "results").glob("test_bottle.*"))
if not candidates:
    raise FileNotFoundError("No test_bottle image found in results/")
IMAGE_PATH = candidates[0]

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Put your test image here: {IMAGE_PATH}")

image = ImageOps.exif_transpose(Image.open(IMAGE_PATH)).convert("RGB")
classes = ["sal_data_100", "sal_data_80", "sal_data_50", "sal_data_empty"]

print("Testing:", IMAGE_PATH)
print("Image size:", image.size)

# Original MobileNetV2
for filename, labels_file in [
    ("mobilenetv2_iv.keras", "mobilenetv2_labels.json"),
    ("temporal_mobilenetv2_iv.keras", "temporal_mobilenetv2_labels.json"),
]:
    path = ROOT / "models" / filename
    label_path = ROOT / "models" / labels_file

    if not path.exists() or not label_path.exists():
        print(f"\nSkipping missing model: {filename}")
        continue

    model = tf.keras.models.load_model(path)
    labels = json.loads(label_path.read_text(encoding="utf-8"))

    arr = np.asarray(image.resize((224, 224)), dtype=np.float32)
    arr = np.expand_dims(arr / 127.5 - 1.0, axis=0)
    scores = model.predict(arr, verbose=0)[0]

    print(f"\n{filename}")
    for i in np.argsort(scores)[::-1]:
        print(f"  {labels[i]}: {scores[i]:.4f}")

# HOG baseline: use the exact saved model's feature settings
hog_path = ROOT / "models" / "hog_linear_svc.joblib"
if hog_path.exists():
    saved = joblib.load(hog_path)
    svm = saved["model"]
    width, height = saved["image_size"]

    import cv2
    resized = cv2.resize(np.asarray(image), (width, height))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    descriptor = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)
    features = descriptor.compute(gray).flatten().reshape(1, -1)

    print("\nHOG + Linear SVM:", svm.predict(features)[0])

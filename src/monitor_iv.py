
from pathlib import Path
import sys
import cv2
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hog_linear_svc.joblib"

IMAGE_SIZE = (96, 96)
LOW_LEVEL_THRESHOLD = 50

LEVELS = {
    "sal_data_100": 100,
    "sal_data_80": 80,
    "sal_data_50": 50,
    "sal_data_empty": 0,
}

def predict_level(image_path):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor(
        (96, 96), (16, 16), (8, 8), (8, 8), 9
    )

    features = hog.compute(gray).flatten().reshape(1, -1)
    predicted_class = model.predict(features)[0]

    return predicted_class, LEVELS[predicted_class]

def main():
    if len(sys.argv) < 2:
        print('Usage: python src\\monitor_iv.py "path\\to\\image.jpg"')
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    if not image_path.is_file():
        print("ERROR: Image not found:", image_path)
        return

    try:
        predicted_class, level = predict_level(image_path)
    except (ValueError, KeyError, OSError) as error:
        print("ERROR:", error)
        return

    print("\n=== IV FLUID MONITOR ===")
    print("Image:", image_path.name)
    print(f"Estimated category: {level}%")
    print("Predicted class:", predicted_class)

    if level == 0:
        print("ALERT: EMPTY CATEGORY DETECTED")
    elif level <= LOW_LEVEL_THRESHOLD:
        print("WARNING: LOW CATEGORY DETECTED")
    else:
        print("STATUS: ABOVE DEMONSTRATION THRESHOLD")

    print("\nDemonstration only — verify the bottle visually.")
    print("Not validated for clinical use.")

if __name__ == "__main__":
    main()

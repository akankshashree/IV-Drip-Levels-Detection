
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (96, 96)
SEED = 42

# Discover images using their class-folder names.
records = []

for path in DATASETS.rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        label = path.parent.name

        if label in {"sal_data_100", "sal_data_80",
                     "sal_data_50", "sal_data_empty"}:
            records.append({"path": str(path), "label": label})

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No labelled images found. Check the dataset folders.")

print("Images found:", len(df))
print("\nClass counts:")
print(df["label"].value_counts().sort_index())

# Provisional stratified split: 70% train, 15% validation, 15% test.
train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temporary_df["label"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

manifest = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)
manifest.to_csv(RESULTS / "baseline_split_manifest.csv", index=False)

# HOG features provide a lightweight computer-vision baseline.
hog = cv2.HOGDescriptor(
    (96, 96),       # window size
    (16, 16),       # block size
    (8, 8),         # block stride
    (8, 8),         # cell size
    9               # orientation bins
)

def extract_features(frame):
    features = []

    for filename in frame["path"]:
        image = cv2.imread(filename)

        if image is None:
            raise RuntimeError(f"Could not read image: {filename}")

        image = cv2.resize(image, IMAGE_SIZE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.append(hog.compute(gray).flatten())

    return np.asarray(features, dtype=np.float32)

print("\nExtracting training features...")
X_train = extract_features(train_df)
y_train = train_df["label"].to_numpy()

print("Extracting validation features...")
X_val = extract_features(val_df)
y_val = val_df["label"].to_numpy()

print("Extracting test features...")
X_test = extract_features(test_df)
y_test = test_df["label"].to_numpy()

# Scale features and train a linear SVM.
model = make_pipeline(
    StandardScaler(),
    LinearSVC(random_state=SEED, max_iter=5000)
)

print("\nTraining baseline model...")
model.fit(X_train, y_train)

print("\nVALIDATION RESULTS")
val_predictions = model.predict(X_val)
print(classification_report(y_val, val_predictions, zero_division=0))

print("TEST RESULTS")
test_predictions = model.predict(X_test)
print(classification_report(y_test, test_predictions, zero_division=0))

labels = sorted(df["label"].unique())
matrix = confusion_matrix(y_test, test_predictions, labels=labels)

pd.DataFrame(
    matrix, index=labels, columns=labels
).to_csv(RESULTS / "baseline_confusion_matrix.csv")

joblib.dump(
    {"model": model, "image_size": IMAGE_SIZE, "labels": labels},
    MODELS / "hog_linear_svc.joblib"
)

print("\nSaved model:", MODELS / "hog_linear_svc.joblib")
print("Saved split:", RESULTS / "baseline_split_manifest.csv")
print("Saved confusion matrix:", RESULTS / "baseline_confusion_matrix.csv")
print("\nBaseline complete.")
print("IMPORTANT: random splitting may leak related recording frames.")
print("Treat test metrics as preliminary, not clinical validation.")

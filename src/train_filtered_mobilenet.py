import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# =====================================================================
# CONFIGURATION: edit this section only
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent

# Folder that contains the class folders (searched recursively).
# For Scharr-filtered images, point this at your filtered output folder.
DATA_ROOT = Path(r"datasets_scharr")

# Prefix for every output file. Change it for each experiment so nothing
# is overwritten (for example "scharr_blocked" and "gray_blocked").
RUN_NAME = "scharr_blocked"

# Class folder names. Folders named sal_data_* are mapped automatically.
CLASSES = ["sal_data_100", "sal_data_80", "sal_data_50", "sal_data_empty"]
ALIASES = {
    "sal_data_100": "sal_data_100",
    "sal_data_80": "sal_data_80",
    "sal_data_50": "sal_data_50",
    "sal_data_empty": "sal_data_empty",
}

# Set True first: tiny run (about 240 images, 1 epoch per phase) to catch
# errors in a few minutes before the full run.
QUICK_TEST = False

IMAGE_SIZE = (224, 224)      # (160, 160) is roughly 2x faster on CPU
BATCH_SIZE = 16
SEED = 42

HEAD_EPOCHS = 10
FINE_TUNE = True
FINE_TUNE_EPOCHS = 5
UNFREEZE_LAST = 30

# Stretch each image so this percentile maps to 255 (filtered images are dark).
# Use None for unfiltered images.
STRETCH_PERCENTILE = 99.5

# 70:15:15 split, applied per class using contiguous time blocks.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15             # test = the remainder
MAX_BLOCK = 30              # max consecutive images kept together
GAP_SECONDS = 30            # a time gap larger than this starts a new block
PURGE_SECONDS = 10          # drop lower-priority frames this close to another split

# =====================================================================

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CACHE_PATH = RESULTS / f"{RUN_NAME}_cache_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.npz"
MANIFEST_PATH = RESULTS / f"{RUN_NAME}_split_manifest.csv"
HEAD_PATH = MODELS / f"{RUN_NAME}_head.keras"
FINETUNE_PATH = MODELS / f"{RUN_NAME}_finetuned.keras"
MODEL_PATH = MODELS / f"{RUN_NAME}_mobilenetv2.keras"
LABELS_PATH = MODELS / f"{RUN_NAME}_labels.json"
META_PATH = RESULTS / f"{RUN_NAME}_run_info.json"
MATRIX_PATH = RESULTS / f"{RUN_NAME}_confusion_matrix.csv"
REPORT_PATH = RESULTS / f"{RUN_NAME}_classification_report.csv"
PREDICTIONS_PATH = RESULTS / f"{RUN_NAME}_test_predictions.csv"
MISCLASSIFIED_PATH = RESULTS / f"{RUN_NAME}_misclassified.csv"
HISTORY_PATH = RESULTS / f"{RUN_NAME}_history.csv"
CURVES_PATH = RESULTS / f"{RUN_NAME}_curves.png"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

T_START = time.time()
timings = {}

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

# ---------------- COLLECT IMAGES ----------------

TS_RE = re.compile(r"(\d{8})(\d{6})")


def parse_timestamp(stem):
    match = TS_RE.search(stem)
    if not match:
        return None
    try:
        return datetime.strptime(
            match.group(1) + match.group(2), "%Y%m%d%H%M%S"
        ).timestamp()
    except ValueError:
        return None


def collect_images():
    rows = []
    for path in sorted(DATA_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(DATA_ROOT)
        label = None
        for part in reversed(relative.parts[:-1]):  # nearest folder first
            name = ALIASES.get(part, part)
            if name in CLASSES:
                label = name
                break
        if label is None:
            continue
        rows.append(
            {"path": str(path), "label": label, "ts": parse_timestamp(path.stem)}
        )
    return pd.DataFrame(rows)


if not DATA_ROOT.is_dir():
    raise SystemExit(f"DATA_ROOT not found: {DATA_ROOT}")

df = collect_images()
if df.empty:
    raise SystemExit(f"No images found under {DATA_ROOT} for classes {CLASSES}")

if df["ts"].isna().any():
    example = df.loc[df["ts"].isna(), "path"].iloc[0]
    raise SystemExit(
        f"{int(df['ts'].isna().sum())} filenames have no capture timestamp "
        f"(expected like IMG20200208160211). Example: {example}"
    )

if df["path"].duplicated().any():
    raise SystemExit("Duplicate paths found.")

print("\nImages found per class:")
print(df["label"].value_counts().reindex(CLASSES))

# ---------------- PURGED, TIME-BLOCKED 70:15:15 SPLIT ----------------

SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}


def make_blocks(times):
    blocks, current = [], [0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > GAP_SECONDS or len(current) >= MAX_BLOCK:
            blocks.append(current)
            current = [i]
        else:
            current.append(i)
    blocks.append(current)
    return blocks


def split_class(frame, rng):
    frame = frame.sort_values("ts").reset_index(drop=True)
    blocks = make_blocks(frame["ts"].to_numpy())
    n = len(frame)
    targets = {
        "train": TRAIN_FRAC * n,
        "validation": VAL_FRAC * n,
        "test": (1.0 - TRAIN_FRAC - VAL_FRAC) * n,
    }
    counts = {name: 0 for name in targets}
    assigned = np.empty(n, dtype=object)
    for b in rng.permutation(len(blocks)):
        idx = blocks[b]
        name = max(targets, key=lambda k: (targets[k] - counts[k]) / targets[k])
        assigned[idx] = name
        counts[name] += len(idx)
    frame["split"] = assigned
    return frame


def purge_class(frame):
    """Drop frames within PURGE_SECONDS of a frame in a higher-priority split.

    Priority: test > validation > train. This removes near-identical
    consecutive shots that straddle a block boundary.
    """
    ts = frame["ts"].to_numpy()
    sp = frame["split"].map(SPLIT_RANK).to_numpy()
    lo = np.searchsorted(ts, ts - PURGE_SECONDS, side="left")
    hi = np.searchsorted(ts, ts + PURGE_SECONDS, side="right")
    drop = np.zeros(len(frame), dtype=bool)
    for i in range(len(frame)):
        if (sp[lo[i]:hi[i]] > sp[i]).any():
            drop[i] = True
    return frame[~drop], int(drop.sum())


rng = np.random.default_rng(SEED)
parts, dropped_total = [], 0
for class_name in CLASSES:
    class_frame = df[df["label"] == class_name]
    if class_frame.empty:
        raise SystemExit(f"No images for class {class_name}")
    class_frame = split_class(class_frame, rng)
    class_frame, n_dropped = purge_class(class_frame)
    dropped_total += n_dropped
    parts.append(class_frame)

df = pd.concat(parts, ignore_index=True)
df.to_csv(MANIFEST_PATH, index=False)

print(f"\nPurged {dropped_total} frames that sat too close to another split.")
coverage = pd.crosstab(df["label"], df["split"]).reindex(CLASSES)
print("\nImages per class and split:")
print(coverage)

for class_name in CLASSES:
    for split_name in SPLIT_RANK:
        if coverage.loc[class_name, split_name] == 0:
            raise SystemExit(f"No {class_name} images in {split_name} split.")

print("\nSplit share:")
print((df["split"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")

if QUICK_TEST:
    df = df.groupby(["label", "split"], group_keys=False).head(20).reset_index(drop=True)
    HEAD_EPOCHS, FINE_TUNE_EPOCHS = 1, 1
    print(f"\nQUICK_TEST on: using {len(df)} images, 1 epoch per phase.")

# ---------------- LOAD IMAGES INTO A CACHE (ONCE) ----------------


def load_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(
        image, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA
    )
    if STRETCH_PERCENTILE:
        high = float(np.percentile(image, STRETCH_PERCENTILE))
        if high > 0:
            image = np.clip(
                image.astype(np.float32) * (255.0 / high), 0, 255
            ).astype(np.uint8)
    return image


def load_cache(paths):
    t0 = time.time()
    if CACHE_PATH.exists():
        saved = np.load(CACHE_PATH, allow_pickle=True)
        saved_paths = saved["paths"].tolist()
        lookup = {p: i for i, p in enumerate(saved_paths)}
        if all(p in lookup for p in paths):
            print(f"\nUsing cached images: {CACHE_PATH.name}")
            arrays = saved["images"]
            return np.stack([arrays[lookup[p]] for p in paths]), time.time() - t0

    print(f"\nBuilding image cache for {len(paths)} images (one time only)...")
    images = np.empty((len(paths), IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8)
    for i, p in enumerate(paths):
        image = load_gray(p)
        if image is None:
            raise SystemExit(f"Unreadable image: {p}")
        images[i] = image
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)}  ({time.time() - t0:.0f}s)")
    if not QUICK_TEST:
        np.savez(CACHE_PATH, images=images, paths=np.array(paths, dtype=object))
    return images, time.time() - t0


paths_all = df["path"].tolist()
images_all, timings["cache_s"] = load_cache(paths_all)
print(f"Cache ready in {timings['cache_s']:.0f}s. Array: {images_all.nbytes / 1e6:.0f} MB")

class_to_id = {name: i for i, name in enumerate(CLASSES)}
labels_all = np.array([class_to_id[n] for n in df["label"]], dtype=np.int32)

masks = {s: (df["split"] == s).to_numpy() for s in SPLIT_RANK}
x_train, y_train = images_all[masks["train"]], labels_all[masks["train"]]
x_val, y_val = images_all[masks["validation"]], labels_all[masks["validation"]]
x_test, y_test = images_all[masks["test"]], labels_all[masks["test"]]
test_df = df[masks["test"]].reset_index(drop=True)

AUTOTUNE = tf.data.AUTOTUNE


def prepare(image, label):
    image = tf.cast(image[..., tf.newaxis], tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    return image, label


def make_dataset(x, y, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(prepare, num_parallel_calls=AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


train_ds = make_dataset(x_train, y_train, training=True)
val_ds = make_dataset(x_val, y_val)
test_ds = make_dataset(x_test, y_test)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=y_train,
)
class_weights = {i: float(w) for i, w in enumerate(weights)}
print("Class weights:", {CLASSES[i]: round(w, 3) for i, w in class_weights.items()})

# ---------------- MODEL ----------------

print("\nLoading pretrained MobileNetV2...")

base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)   # never flip vertically
x = tf.keras.layers.RandomRotation(0.025)(x)
x = tf.keras.layers.RandomZoom(0.05)(x)
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.durations = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.durations.append(time.time() - self._start)
        print(f"  epoch time: {self.durations[-1]:.0f}s")


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

head_timer = EpochTimer()
head_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(HEAD_PATH), monitor="val_loss", save_best_only=True
    ),
    head_timer,
]

print("\nPhase 1: training classification head...")
t0 = time.time()
head_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=head_callbacks,
)
timings["head_s"] = time.time() - t0
head_model = tf.keras.models.load_model(HEAD_PATH)

# ---------------- FINE-TUNING (OPTIONAL) ----------------

fine_history = None
fine_model = None

if FINE_TUNE:
    print("\nPhase 2: fine-tuning upper backbone layers...")
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_timer = EpochTimer()
    fine_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(FINETUNE_PATH), monitor="val_loss", save_best_only=True
        ),
        fine_timer,
    ]

    t0 = time.time()
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=fine_callbacks,
    )
    timings["fine_tune_s"] = time.time() - t0
    fine_model = tf.keras.models.load_model(FINETUNE_PATH)

# ---------------- HISTORY AND CURVES ----------------

history_parts = [pd.DataFrame(head_history.history).assign(phase="head")]
if fine_history is not None:
    history_parts.append(pd.DataFrame(fine_history.history).assign(phase="finetune"))
history_df = pd.concat(history_parts, ignore_index=True)
history_df.to_csv(HISTORY_PATH, index_label="epoch_overall")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_df["loss"], label="train")
    axes[0].plot(history_df["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history_df["accuracy"], label="train")
    axes[1].plot(history_df["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=120)
    plt.close(fig)
except ImportError:
    print("matplotlib not installed; skipped the curve plot (CSV was saved).")

# ---------------- MODEL SELECTION (VALIDATION ONLY) ----------------

head_val_loss = float(head_model.evaluate(val_ds, verbose=0)[0])
print(f"\nHead-only validation loss: {head_val_loss:.4f}")

if fine_model is not None:
    fine_val_loss = float(fine_model.evaluate(val_ds, verbose=0)[0])
    print(f"Fine-tuned validation loss: {fine_val_loss:.4f}")
    if head_val_loss <= fine_val_loss:
        best_model, selected_stage = head_model, "head_only"
    else:
        best_model, selected_stage = fine_model, "fine_tuned"
else:
    best_model, selected_stage = head_model, "head_only"

best_model.save(MODEL_PATH)
LABELS_PATH.write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
print("Selected checkpoint:", selected_stage)

# ---------------- FINAL TEST EVALUATION (ONCE) ----------------

print("\nEvaluating the test set...")
t0 = time.time()
probabilities = best_model.predict(test_ds, verbose=1)
timings["test_s"] = time.time() - t0

predicted_ids = np.argmax(probabilities, axis=1)
overall_acc = float((predicted_ids == y_test).mean())
print(
    f"\nOverall test accuracy: {overall_acc:.4f} "
    f"({int((predicted_ids == y_test).sum())}/{len(y_test)})"
)

label_ids = list(range(len(CLASSES)))
print("\nTEST CLASSIFICATION REPORT")
print(
    classification_report(
        y_test, predicted_ids, labels=label_ids, target_names=CLASSES, zero_division=0
    )
)

matrix = confusion_matrix(y_test, predicted_ids, labels=label_ids)
matrix_df = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
print("Confusion matrix (rows = actual, columns = predicted):")
print(matrix_df)

matrix_df.to_csv(MATRIX_PATH)
pd.DataFrame(
    classification_report(
        y_test,
        predicted_ids,
        labels=label_ids,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
).transpose().to_csv(REPORT_PATH)

prediction_df = test_df[["path", "label"]].copy()
prediction_df["predicted_label"] = [CLASSES[i] for i in predicted_ids]
prediction_df["correct"] = predicted_ids == y_test
for i, class_name in enumerate(CLASSES):
    prediction_df[f"score_{class_name}"] = probabilities[:, i]
prediction_df.to_csv(PREDICTIONS_PATH, index=False)
prediction_df[~prediction_df["correct"]].to_csv(MISCLASSIFIED_PATH, index=False)

# ---------------- TIMING SUMMARY ----------------

timings["total_s"] = time.time() - T_START
head_epochs_run = len(head_timer.durations)
fine_epochs_run = len(fine_timer.durations) if FINE_TUNE else 0

META_PATH.write_text(
    json.dumps(
        {
            "run_name": RUN_NAME,
            "data_root": str(DATA_ROOT),
            "selected_stage": selected_stage,
            "test_accuracy": overall_acc,
            "images": {s: int(m.sum()) for s, m in masks.items()},
            "purged_frames": dropped_total,
            "image_size": list(IMAGE_SIZE),
            "batch_size": BATCH_SIZE,
            "epochs_run": {"head": head_epochs_run, "fine_tune": fine_epochs_run},
            "seconds": {k: round(v, 1) for k, v in timings.items()},
            "quick_test": QUICK_TEST,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\n================ TIMING ================")
print(f"Image cache:       {timings['cache_s'] / 60:6.1f} min")
print(f"Head training:     {timings['head_s'] / 60:6.1f} min  ({head_epochs_run} epochs)")
if FINE_TUNE:
    print(f"Fine-tuning:       {timings['fine_tune_s'] / 60:6.1f} min  ({fine_epochs_run} epochs)")
print(f"Test prediction:   {timings['test_s'] / 60:6.1f} min")
print(f"TOTAL:             {timings['total_s'] / 60:6.1f} min")

print("\nSaved model:", MODEL_PATH)
print("Saved labels:", LABELS_PATH)
print("Saved split manifest:", MANIFEST_PATH)
print("Saved results to:", RESULTS)
print(
    "\nCaution: this split removes near-duplicate leakage between consecutive "
    "frames, but each class was still recorded in its own session and "
    "background. A good score here does not show that the model reads the "
    "liquid level on new bottles or backgrounds."
)
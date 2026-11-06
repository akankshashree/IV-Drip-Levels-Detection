
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Use the chronological manifest created earlier.
SPLIT_PATH = RESULTS / "temporal_split.csv"

# Keep this experiment separate from the original model.
HEAD_PATH = MODELS / "temporal_mobilenetv2_head.keras"
FINETUNE_PATH = MODELS / "temporal_mobilenetv2_finetuned.keras"
MODEL_PATH = MODELS / "temporal_mobilenetv2_iv.keras"
LABELS_PATH = MODELS / "temporal_mobilenetv2_labels.json"

MATRIX_PATH = RESULTS / "temporal_mobilenetv2_confusion_matrix.csv"
REPORT_PATH = RESULTS / "temporal_mobilenetv2_classification_report.csv"
PREDICTIONS_PATH = RESULTS / "temporal_mobilenetv2_test_predictions.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

CLASSES = [
    "sal_data_100",
    "sal_data_80",
    "sal_data_50",
    "sal_data_empty",
]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

# ---------------- LOAD AND VALIDATE MANIFEST ----------------

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Temporal manifest not found: {SPLIT_PATH}\n"
        "Run: python src\\prepare_temporal_split.py"
    )

df = pd.read_csv(SPLIT_PATH)

required = {"path", "label", "split"}
if not required.issubset(df.columns):
    raise ValueError(f"Manifest must contain columns: {required}")

if df.empty:
    raise ValueError("The temporal manifest is empty.")

if df["path"].duplicated().any():
    raise ValueError("Duplicate image paths found in the manifest.")

if not set(df["label"]).issubset(set(CLASSES)):
    raise ValueError("Unexpected class label in manifest.")

EXPECTED_SPLITS = {"train", "validation", "test"}
if set(df["split"]) != EXPECTED_SPLITS:
    raise ValueError(
        f"Expected splits {EXPECTED_SPLITS}, found {set(df['split'])}"
    )

missing = [p for p in df["path"] if not Path(p).is_file()]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} manifest images are missing. Example: {missing[0]}"
    )

# Ensure every class is represented in every partition.
coverage = pd.crosstab(df["label"], df["split"])
for split in EXPECTED_SPLITS:
    if split not in coverage.columns:
        raise ValueError(f"Missing partition: {split}")

for class_name in CLASSES:
    for split in EXPECTED_SPLITS:
        if coverage.loc[class_name, split] == 0:
            raise ValueError(f"No {class_name} images in {split} split.")

print("\nImages per class and split:")
print(coverage.reindex(CLASSES))

train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
val_df = df[df["split"] == "validation"].copy().reset_index(drop=True)
test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

print(
    f"\nUsing temporal split: train={len(train_df)}, "
    f"validation={len(val_df)}, test={len(test_df)}"
)

class_to_id = {name: i for i, name in enumerate(CLASSES)}

# ---------------- TF.DATA PIPELINE ----------------

AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

def make_dataset(frame, training=False):
    paths = frame["path"].to_numpy()
    labels = np.array(
        [class_to_id[name] for name in frame["label"]],
        dtype=np.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(AUTOTUNE)

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# Calculate class weights using training labels only.
y_train = np.array([class_to_id[name] for name in train_df["label"]])
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=y_train,
)
class_weights = {i: float(w) for i, w in enumerate(weights)}

# ---------------- MODEL ----------------

print("\nLoading pretrained MobileNetV2...")

base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.025)(x)
x = tf.keras.layers.RandomZoom(0.05)(x)
x = tf.keras.layers.RandomContrast(0.10)(x)

x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    len(CLASSES),
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

head_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(HEAD_PATH),
        monitor="val_loss",
        save_best_only=True,
    ),
]

print("\nPhase 1: training classification head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=head_callbacks,
)

# Keep the best head-only model safe as a separate checkpoint.
head_model = tf.keras.models.load_model(HEAD_PATH)

# ---------------- FINE-TUNING ----------------

print("\nPhase 2: fine-tuning upper backbone layers...")

base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fine_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(FINETUNE_PATH),
        monitor="val_loss",
        save_best_only=True,
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=fine_callbacks,
)

# Evaluate both checkpoints on validation data only.
# The test set is not used to choose the final model.
fine_model = tf.keras.models.load_model(FINETUNE_PATH)

head_val_loss = float(head_model.evaluate(val_ds, verbose=0)[0])
fine_val_loss = float(fine_model.evaluate(val_ds, verbose=0)[0])

print(f"\nHead-only validation loss: {head_val_loss:.4f}")
print(f"Fine-tuned validation loss: {fine_val_loss:.4f}")

if head_val_loss <= fine_val_loss:
    best_model = head_model
    selected_stage = "head_only"
else:
    best_model = fine_model
    selected_stage = "fine_tuned"

best_model.save(MODEL_PATH)

LABELS_PATH.write_text(
    json.dumps(CLASSES, indent=2),
    encoding="utf-8",
)

print("Selected checkpoint:", selected_stage)

# ---------------- FINAL TEST EVALUATION ----------------

# Test is evaluated once after selecting the model using validation loss.
print("\nEvaluating chronological test set...")

probabilities = best_model.predict(test_ds, verbose=1)
predicted_ids = np.argmax(probabilities, axis=1)
actual_ids = np.array(
    [class_to_id[name] for name in test_df["label"]],
)

report_dict = classification_report(
    actual_ids,
    predicted_ids,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)

print("\nCHRONOLOGICAL TEST CLASSIFICATION REPORT")
print(classification_report(
    actual_ids,
    predicted_ids,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0,
))

matrix = confusion_matrix(
    actual_ids,
    predicted_ids,
    labels=list(range(len(CLASSES))),
)

pd.DataFrame(
    matrix,
    index=CLASSES,
    columns=CLASSES,
).to_csv(MATRIX_PATH)

pd.DataFrame(report_dict).transpose().to_csv(REPORT_PATH)

prediction_df = test_df[["path", "label"]].copy()
prediction_df["actual_id"] = actual_ids
prediction_df["predicted_label"] = [
    CLASSES[i] for i in predicted_ids
]

for i, class_name in enumerate(CLASSES):
    prediction_df[f"score_{class_name}"] = probabilities[:, i]

prediction_df.to_csv(PREDICTIONS_PATH, index=False)

print("\nSelected model saved:", MODEL_PATH)
print("Labels saved:", LABELS_PATH)
print("Confusion matrix saved:", MATRIX_PATH)
print("Classification report saved:", REPORT_PATH)
print("Per-image predictions saved:", PREDICTIONS_PATH)
print("\nTemporal training experiment complete.")
print(
    "Caution: chronological splitting reduces one leakage risk, but "
    "does not establish performance on independent bottles or real-world "
    "clinical settings."
)

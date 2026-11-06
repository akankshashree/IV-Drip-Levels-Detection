from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_iv.keras"
LABELS_PATH = ROOT / "models" / "mobilenetv2_labels.json"

LEVELS = {
    "sal_data_100": 100,
    "sal_data_80": 80,
    "sal_data_50": 50,
    "sal_data_empty": 0,
}

LEVEL_LABELS = {
    "sal_data_100": "100% filled",
    "sal_data_80": "80% filled",
    "sal_data_50": "50% filled",
    "sal_data_empty": "Empty",
}

IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="IV Fluid Monitoring System",
    page_icon="💧",
    layout="wide",
)


# ---------------- LOAD MOBILENETV2 ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MobileNetV2 model not found: {MODEL_PATH}")

    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Model labels not found: {LABELS_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    return model, labels


# ---------------- PREDICTION ----------------

def predict_image(pil_image, model, labels):
    image = pil_image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # The model includes its Rescaling(1/127.5, offset=-1) layer.
    # Do not normalize the image again here.
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = labels[predicted_index]
    score = float(predictions[predicted_index])

    return predicted_class, LEVELS[predicted_class], score, predictions


# ---------------- DASHBOARD UI ----------------

st.title("💧 IV Fluid Monitoring System")
st.caption("MobileNetV2-based IV fluid category classification — research prototype")

st.warning(
    "Research prototype only. Predictions may be incorrect. "
    "Do not use this system for clinical decisions or as a replacement "
    "for direct observation by trained healthcare staff."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Low-level warning threshold (%)",
        min_value=0,
        max_value=80,
        value=50,
        step=10,
    )
    st.info("This threshold controls the demonstration warning only.")

st.subheader("Input IV bottle image")

input_method = st.radio(
    "Choose input method",
    ["Upload image", "Use camera"],
    horizontal=True,
)

uploaded_image = None

if input_method == "Upload image":
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

else:
    camera_file = st.camera_input("Capture IV bottle image")

    if camera_file is not None:
        uploaded_image = Image.open(camera_file).convert("RGB")


if uploaded_image is not None:
    left, right = st.columns(2)

    with left:
        st.image(
            uploaded_image,
            caption="Input IV bottle image",
            use_container_width=True,
        )

    with right:
        st.subheader("Image information")
        st.write(f"Width: {uploaded_image.width} pixels")
        st.write(f"Height: {uploaded_image.height} pixels")

    if st.button("Analyze Fluid Level", type="primary"):
        try:
            model, labels = load_model()

            with st.spinner("Analyzing with MobileNetV2..."):
                predicted_class, level, score, all_scores = predict_image(
                    uploaded_image,
                    model,
                    labels,
                )

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated category",
                    LEVEL_LABELS[predicted_class],
                )

            with col2:
                st.metric(
                    "Model score",
                    f"{score * 100:.2f}%",
                )

            st.write("Predicted class:", predicted_class)

            if level == 0:
                st.error("DEMONSTRATION ALERT: Empty category predicted")
            elif level <= threshold:
                st.warning("DEMONSTRATION WARNING: Low category predicted")
            else:
                st.success("Above demonstration threshold")

            st.progress(level / 100)

            st.caption(
                "The displayed percentage is a class category, not a direct "
                "measurement of remaining fluid volume. Model scores are not "
                "guaranteed to represent calibrated confidence."
            )

            with st.expander("Scores for all classes"):
                for label, value in zip(labels, all_scores):
                    st.write(f"{label}: {float(value) * 100:.2f}%")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

else:
    st.info("Upload or capture an image to begin.")

st.divider()
st.caption(
    "IV Fluid Monitoring Prototype | Not clinically validated | "
    "Always verify the fluid level independently."
)
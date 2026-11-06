"""
dashboard_improved.py - Clinical Bedside IV Fluid Monitoring Dashboard

Hospital-Grade Frontend Refinements:
1. Professional Healthcare Interface: Minimal, clean, institutional styling
   with dignified typography, balanced whitespace, and subtle slate/neutral tones.
2. Hospital Status System: Restrained, medical-grade indicators for Normal (Green 700),
   Low (Amber 700), and Empty (Red 700) without flashy animations or excessive emojis.
3. Reorganized Clinical Sidebar: Logical grouping of Navigation, Patient Records,
   Camera Hardware Link, and Adaptive Timing with compact inputs and crisp section headers.
4. Independent Video & Inference: Non-blocking foreground video rendering (~25 FPS)
   coupled with asynchronous background MobileNetV2 inference via ThreadPoolExecutor.
5. Adaptive Inference Schedule:
   - > 50% & stable: 30s power/thermal-saving baseline
   - <= 50%: 15s vigilant surveillance
   - Rapid drop (>= 30%): 7s accelerated monitoring
   - First suspected empty: 4s confirmation check
   - Replenished: automatically resets state to baseline
6. Complete Functional Integrity:
   - Only displays the 4 true model classes: 100%, 80%, 50%, Empty.
   - Low confidence (< 0.60) guard prevents spurious empty confirmations.
   - Retains last valid prediction with clear staleness indicator if an error occurs.
   - Reuses MobileNetV2 model and CameraFeedManager in memory across script reruns.
   - Preserves diagnostic upload and manual camera input modes.
"""

import concurrent.futures
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Optional OpenCV and Picamera2 imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False

# Optional integration with backend alerts engine if present
try:
    backend_path = Path(__file__).resolve().parent.parent / "Dashboard" / "iv_drip_dashboard" / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    from alerts import evaluate_reading_for_alert
    BACKEND_ALERTS_AVAILABLE = True
except Exception:
    BACKEND_ALERTS_AVAILABLE = False


# ---------------- CONFIGURATION & MODEL DEFINITIONS ----------------

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
    "sal_data_100": "100% Filled",
    "sal_data_80": "80% Filled",
    "sal_data_50": "50% Filled",
    "sal_data_empty": "Empty",
}

IMAGE_SIZE = (224, 224)

# Adaptive interval constants
DEFAULT_INTERVAL_HIGH_STABLE = 30   # Fluid > 50% & stable
DEFAULT_INTERVAL_LOW = 15           # Fluid <= 50%
DEFAULT_INTERVAL_RAPID_DROP = 7     # Fluid dropped >= 30%
DEFAULT_INTERVAL_CONFIRMATION = 4   # First suspected empty (fast 2nd check)
DEFAULT_MIN_CONFIDENCE = 0.60       # Safe confidence threshold


# ---------------- PAGE CONFIGURATION & MEDICAL CSS ----------------

st.set_page_config(
    page_title="IV Fluid Monitoring System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Application background & clean typography */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Refined Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-heading {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #475569;
        margin-top: 14px;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 1px solid #f1f5f9;
    }
    .sidebar-heading-nav,
    .sidebar-nav-title,
    .sidebar-mode-title {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #0f172a !important;
    }

    /* Hospital App Bar */
    .hospital-appbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 14px;
    }
    .appbar-title {
        font-size: 19px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.2px;
        margin: 0;
    }
    .appbar-subtitle {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
        margin: 2px 0 0 0;
    }
    .appbar-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    .appbar-badge.active {
        background-color: #f0fdf4;
        color: #15803d;
        border-color: #bbf7d0;
    }
    .badge-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
    }
    .badge-dot.idle {
        background-color: #94a3b8;
    }

    /* Patient Demographics Strip */
    .patient-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 18px;
    }
    .patient-cell {
        display: flex;
        flex-direction: column;
    }
    .patient-cell-label {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-bottom: 2px;
    }
    .patient-cell-value {
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
    }

    /* Panel Framing */
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }
    .panel-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #334155;
    }
    .panel-meta {
        font-size: 11px;
        color: #64748b;
        font-weight: 500;
    }
    .panel-body {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 6px;
        border-bottom-right-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Fluid Status Banners */
    .status-banner {
        padding: 14px 16px;
        border-radius: 4px;
        margin-bottom: 14px;
        border: 1px solid transparent;
    }
    .status-banner-normal {
        background-color: #f0fdf4;
        border-color: #bbf7d0;
        color: #14532d;
    }
    .status-banner-low {
        background-color: #fffbeb;
        border-color: #fde68a;
        color: #78350f;
    }
    .status-banner-empty {
        background-color: #fef2f2;
        border-color: #fecaca;
        color: #7f1d1d;
    }
    .status-banner-title {
        font-size: 17px;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin-bottom: 4px;
    }
    .status-banner-sub {
        font-size: 12px;
        font-weight: 500;
        opacity: 0.9;
    }

    /* Metrics 2x2 Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 12px;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 10px 12px;
    }
    .metric-box-label {
        font-size: 10px;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-box-value {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 2px;
    }

    /* Timing / Telemetry Card */
    .timing-info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 12px;
        color: #334155;
        line-height: 1.6;
        margin-top: 10px;
    }
    .timing-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }
    .timing-row:last-child {
        margin-bottom: 0;
    }

    /* Stale Data Warning */
    .stale-notice {
        background-color: #fff7ed;
        border: 1px solid #ffedd5;
        color: #9a3412;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 8px;
    }

    /* Class Probability Table */
    .prob-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        margin-top: 6px;
    }
    .prob-table th {
        text-align: left;
        padding: 6px 8px;
        background-color: #f8fafc;
        color: #475569;
        font-weight: 600;
        border-bottom: 1px solid #e2e8f0;
    }
    .prob-table td {
        padding: 6px 8px;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
    }

    /* Clean Button Styling */
    .stButton button {
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
        padding: 6px 14px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING & THREAD-SAFE PREDICTION ----------------

_predict_lock = threading.Lock()

@st.cache_resource
def load_model():
    """Loads and caches the MobileNetV2 model in memory permanently."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MobileNetV2 model file missing at {MODEL_PATH}")

    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Model labels mapping missing at {LABELS_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    return model, labels


def predict_image(pil_image, model, labels):
    """
    Executes MobileNetV2 inference on a PIL snapshot.
    Preserves existing preprocessing: input 224x224 RGB, no extra rescaling
    because the model incorporates embedded Rescaling layers.
    """
    image = pil_image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    with _predict_lock:
        predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = labels[predicted_index]
    score = float(predictions[predicted_index])

    return predicted_class, LEVELS[predicted_class], score, predictions


# ---------------- ADAPTIVE INTERVAL ENGINE ----------------

class AdaptiveIntervalEngine:
    """
    Implements intelligent snapshot interval scheduling based on observed fluid trend:
    - Fluid > 50% and stable: 30 seconds (reduces computational load and thermal heat)
    - Fluid <= 50%: 15 seconds (closer monitoring as fluid depletes)
    - Rapid decrease (>= 30 percentage points): 7 seconds (rapid tracking)
    - First suspected empty condition: 4 seconds for confirmation
    - Fluid replenished: resets adaptive state and smoothly returns to baseline
    - Low-confidence readings (< 0.60): safely handled without triggering empty alerts
    """

    def __init__(
        self,
        interval_high_stable=DEFAULT_INTERVAL_HIGH_STABLE,
        interval_low=DEFAULT_INTERVAL_LOW,
        interval_rapid_drop=DEFAULT_INTERVAL_RAPID_DROP,
        interval_confirmation=DEFAULT_INTERVAL_CONFIRMATION,
        min_confidence=DEFAULT_MIN_CONFIDENCE,
    ):
        self.interval_high_stable = interval_high_stable
        self.interval_low = interval_low
        self.interval_rapid_drop = interval_rapid_drop
        self.interval_confirmation = interval_confirmation
        self.min_confidence = min_confidence

        self.last_valid_level = None
        self.last_valid_class = None
        self.consecutive_empty_count = 0
        self.current_interval = interval_high_stable
        self.reason = "Baseline: Fluid level > 50% and stable (30s interval)"
        self.is_confirmed_empty = False

    def update(self, predicted_class, confidence):
        level_percent = LEVELS.get(predicted_class, 0)

        # 1. Low confidence safe handling (< 0.60)
        if confidence < self.min_confidence:
            self.current_interval = self.interval_rapid_drop
            self.reason = (
                f"Low confidence ({confidence * 100:.1f}% < {self.min_confidence * 100:.0f}%): "
                f"scheduling verification snapshot in {self.current_interval}s"
            )
            return self.current_interval, self.reason, self.is_confirmed_empty

        # 2. Empty Condition Handling
        if predicted_class == "sal_data_empty" or level_percent == 0:
            if self.consecutive_empty_count == 0:
                # First empty reading -> Suspected empty condition
                self.consecutive_empty_count = 1
                self.current_interval = self.interval_confirmation  # 4 seconds
                self.reason = (
                    f"First suspected Empty detected: taking confirmation snapshot in {self.interval_confirmation}s"
                )
                self.is_confirmed_empty = False
            else:
                # Second or subsequent consecutive empty reading -> Confirmed empty!
                self.consecutive_empty_count += 1
                self.current_interval = self.interval_low  # 15 seconds
                self.reason = (
                    f"Confirmed Empty (reading #{self.consecutive_empty_count}): continuous surveillance every {self.interval_low}s"
                )
                self.is_confirmed_empty = True

            self.last_valid_level = 0
            self.last_valid_class = predicted_class
            return self.current_interval, self.reason, self.is_confirmed_empty

        # 3. Non-Empty Levels (100%, 80%, 50%)
        prev = self.last_valid_level
        self.consecutive_empty_count = 0
        self.is_confirmed_empty = False

        # Fluid replenishment detection
        if prev == 0 and level_percent > 0:
            self.reason = f"Fluid replenished ({level_percent}%): adaptive state reset to baseline"
            self.current_interval = self.interval_low if level_percent <= 50 else self.interval_high_stable
            self.last_valid_level = level_percent
            self.last_valid_class = predicted_class
            return self.current_interval, self.reason, self.is_confirmed_empty

        # Rapid decrease detection (drop >= 30 percentage points, e.g. 100% -> 50% or 80% -> 50%)
        if prev is not None and (prev - level_percent) >= 30:
            self.current_interval = self.interval_rapid_drop  # 7 seconds
            self.reason = (
                f"Rapid decrease detected ({prev}% -> {level_percent}%, drop >= 30%): accelerated interval {self.interval_rapid_drop}s"
            )
        elif level_percent <= 50:
            self.current_interval = self.interval_low  # 15 seconds
            self.reason = f"Fluid level at or below 50% ({level_percent}%): monitoring every {self.interval_low}s"
        else:
            self.current_interval = self.interval_high_stable  # 30 seconds
            self.reason = f"Fluid level > 50% and stable ({level_percent}%): power-saving interval {self.interval_high_stable}s"

        self.last_valid_level = level_percent
        self.last_valid_class = predicted_class
        return self.current_interval, self.reason, self.is_confirmed_empty


# ---------------- CAMERA HARDWARE & STREAM MANAGER ----------------

class CameraFeedManager:
    """
    Manages continuous live video feed from:
    1. Raspberry Pi Picamera2 (native on Pi hardware)
    2. MJPEG Stream URL (e.g. http://localhost:8080/video_feed from pi_camera_service.py)
    3. OpenCV VideoCapture device (e.g. index 0 or /dev/video0)
    4. Diagnostic Simulation Feed (for evaluation panel demos when hardware is unplugged)
    Prevents repeated camera reinitialization across Streamlit script reruns.
    """

    def __init__(self):
        self.cap = None
        self.picam2 = None
        self.active_source = None
        self.is_open = False
        self.simulation_step = 0
        self.simulated_level = 80
        self.error_message = None

    def initialize(self, source_type, stream_url="http://localhost:8080/video_feed", device_index=0):
        target_source = (source_type, stream_url, device_index)
        if self.is_open and self.active_source == target_source:
            return True

        self.release()
        self.error_message = None

        if source_type == "picamera2":
            if not PICAMERA2_AVAILABLE:
                self.error_message = "Picamera2 is not installed in this environment (requires Raspberry Pi OS with python3-picamera2)."
                return False
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                self.is_open = True
                self.active_source = target_source
                return True
            except Exception as e:
                self.error_message = f"Picamera2 hardware error: {e}"
                return False

        elif source_type == "stream":
            if not CV2_AVAILABLE:
                self.error_message = "OpenCV (cv2) is required to connect to network video streams."
                return False
            try:
                self.cap = cv2.VideoCapture(stream_url)
                if not self.cap.isOpened():
                    self.error_message = f"Unable to reach live stream at {stream_url}. Ensure pi_camera_service.py is running."
                    self.cap = None
                    return False
                self.is_open = True
                self.active_source = target_source
                return True
            except Exception as e:
                self.error_message = f"Stream connection failed: {e}"
                return False

        elif source_type == "opencv":
            if not CV2_AVAILABLE:
                self.error_message = "OpenCV (cv2) is required for local device capture."
                return False
            try:
                self.cap = cv2.VideoCapture(device_index)
                if not self.cap.isOpened():
                    self.error_message = f"Unable to open camera device index {device_index}."
                    self.cap = None
                    return False
                self.is_open = True
                self.active_source = target_source
                return True
            except Exception as e:
                self.error_message = f"OpenCV device error: {e}"
                return False

        elif source_type == "simulation":
            self.is_open = True
            self.active_source = target_source
            return True

        return False

    def get_frame(self):
        if not self.is_open:
            return None

        # 1. Picamera2 native capture
        if self.picam2 is not None:
            try:
                frame = self.picam2.capture_array("main")
                if frame is not None:
                    return frame
            except Exception as e:
                self.error_message = f"Picamera2 capture error: {e}"
                return None

        # 2. OpenCV stream / device capture
        if self.cap is not None:
            try:
                ret, frame_bgr = self.cap.read()
                if ret and frame_bgr is not None:
                    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    self.error_message = "Live camera stream timed out or disconnected."
                    return None
            except Exception as e:
                self.error_message = f"Camera read exception: {e}"
                return None

        # 3. Diagnostic simulation generator
        if self.active_source and self.active_source[0] == "simulation":
            return self._generate_simulation_frame()

        return None

    def _generate_simulation_frame(self):
        """Generates a continuous synthetic IV bottle video frame for demonstration."""
        self.simulation_step += 1
        width, height = 640, 480
        img = np.full((height, width, 3), (248, 250, 252), dtype=np.uint8)

        if not CV2_AVAILABLE:
            return img

        # Subtle medical grid background
        img[::40, :, :] = (235, 240, 245)
        img[:, ::40, :] = (235, 240, 245)

        # Saline bottle outline
        bx, by, bw, bh = 220, 75, 200, 330
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (71, 85, 105), 3)
        # Hanger ring & neck
        cv2.ellipse(img, (bx + 100, by - 25), (20, 15), 0, 0, 360, (71, 85, 105), 2)
        cv2.rectangle(img, (bx + 70, by - 15), (bx + 130, by), (51, 65, 85), -1)

        # Fluid level meniscus
        wave = int(np.sin(self.simulation_step * 0.12) * 2)
        fluid_pct = self.simulated_level
        y_empty = by + bh - 10
        y_full = by + 60
        fluid_top = int(y_empty - ((y_empty - y_full) * (fluid_pct / 100.0))) + wave

        if fluid_pct > 0:
            cv2.rectangle(img, (bx + 4, fluid_top), (bx + bw - 4, by + bh - 4), (186, 230, 253), -1)
            cv2.line(img, (bx + 4, fluid_top), (bx + bw - 4, fluid_top), (14, 165, 233), 3)

        # Bottle graduation markers
        for y_off, pct in [(by + 60, "100%"), (by + 125, "80%"), (by + 210, "50%"), (by + 310, "Empty")]:
            cv2.line(img, (bx + 8, y_off), (bx + 32, y_off), (71, 85, 105), 2)
            cv2.putText(img, pct, (bx + 38, y_off + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (71, 85, 105), 1)

        # Live overlay watermark & timecode
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"BEDSIDE PI CAMERA [LIVE STREAM] | {time_str}", (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (15, 118, 110), 2)
        cv2.putText(img, f"FRAME: {self.simulation_step} | RES: 640x480 RGB", (18, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 116, 139), 1)

        return img

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
        self.is_open = False
        self.active_source = None


# ---------------- INITIALIZE SESSION STATE ----------------

if "camera_manager" not in st.session_state:
    st.session_state.camera_manager = CameraFeedManager()

if "timing_engine" not in st.session_state:
    st.session_state.timing_engine = AdaptiveIntervalEngine()

if "executor" not in st.session_state:
    st.session_state.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "last_snapshot_time" not in st.session_state:
    st.session_state.last_snapshot_time = 0.0

if "last_snapshot_image" not in st.session_state:
    st.session_state.last_snapshot_image = None

if "inference_future" not in st.session_state:
    st.session_state.inference_future = None

if "is_stale" not in st.session_state:
    st.session_state.is_stale = False

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ---------------- ASYNCHRONOUS INFERENCE WORKER ----------------

def _run_background_inference(pil_img, model, labels):
    """Executes MobileNetV2 prediction on a snapshot copy in a background thread."""
    t0 = time.monotonic()
    pred_class, level, score, all_scores = predict_image(pil_img, model, labels)
    elapsed = time.monotonic() - t0
    timestamp_str = datetime.now().strftime("%H:%M:%S")
    return pred_class, level, score, all_scores, pil_img, timestamp_str, elapsed


# ---------------- REDESIGNED SIDEBAR CONFIGURATION ----------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-heading sidebar-heading-nav" style="color: #0f172a !important;">'
        '<span class="sidebar-nav-title" style="color: #0f172a !important;">Navigation</span> & '
        '<span class="sidebar-mode-title" style="color: #0f172a !important;">Operating Mode</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    monitoring_mode = st.radio(
        "Select Operating Mode",
        [
            "Live Bedside Video Stream",
            "Diagnostic Image Upload",
            "Manual Camera Snapshot",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-heading">Patient Assignment</div>', unsafe_allow_html=True)
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        patient_id = st.text_input("Patient ID", value="IVP002")
    with c_p2:
        bed_number = st.text_input("Bed Number", value="Bed B-02")

    c_p3, c_p4 = st.columns(2)
    with c_p3:
        patient_name = st.text_input("Patient Name", value="Anjana")
    with c_p4:
        assigned_nurse = st.text_input("Nurse Assigned", value="Nurse Nada")

    if monitoring_mode == "Live Bedside Video Stream":
        st.markdown('<div class="sidebar-heading">Camera Hardware Interface</div>', unsafe_allow_html=True)

        cam_source_options = ["MJPEG Network Stream (/video_feed)"]
        if PICAMERA2_AVAILABLE:
            cam_source_options.insert(0, "Picamera2 Hardware (Native)")
        cam_source_options.extend(["OpenCV Device Index", "Diagnostic Simulation Feed"])

        cam_choice = st.selectbox(
            "Hardware Source",
            cam_source_options,
            index=0,
            label_visibility="collapsed",
        )

        if "Picamera2" in cam_choice:
            resolved_source = "picamera2"
            stream_url = ""
            dev_idx = 0
        elif "MJPEG" in cam_choice:
            resolved_source = "stream"
            stream_url = st.text_input("Stream URL", value="http://localhost:8080/video_feed")
            dev_idx = 0
        elif "OpenCV" in cam_choice:
            resolved_source = "opencv"
            stream_url = ""
            dev_idx = st.number_input("Device Index", min_value=0, max_value=5, value=0)
        else:
            resolved_source = "simulation"
            stream_url = ""
            dev_idx = 0
            sim_lvl = st.select_slider(
                "Demonstration Saline Level",
                options=[100, 80, 50, 0],
                value=80,
                format_func=lambda x: f"{x}% Filled" if x > 0 else "Empty",
            )
            st.session_state.camera_manager.simulated_level = sim_lvl
    else:
        resolved_source = None
        stream_url = ""
        dev_idx = 0

    st.markdown('<div class="sidebar-heading">Clinical Parameters</div>', unsafe_allow_html=True)
    threshold = st.slider("Low-Fluid Warning Level (%)", 0, 80, 50, 10)

    with st.expander("Adaptive Timing Intervals", expanded=False):
        cfg_high = st.number_input("Fluid > 50% Stable (sec)", min_value=15, max_value=120, value=DEFAULT_INTERVAL_HIGH_STABLE)
        cfg_low = st.number_input("Fluid <= 50% (sec)", min_value=5, max_value=60, value=DEFAULT_INTERVAL_LOW)
        cfg_rapid = st.number_input("Rapid Drop >= 30% (sec)", min_value=3, max_value=30, value=DEFAULT_INTERVAL_RAPID_DROP)
        cfg_confirm = st.number_input("Suspected Empty Check (sec)", min_value=2, max_value=15, value=DEFAULT_INTERVAL_CONFIRMATION)

        st.session_state.timing_engine.interval_high_stable = cfg_high
        st.session_state.timing_engine.interval_low = cfg_low
        st.session_state.timing_engine.interval_rapid_drop = cfg_rapid
        st.session_state.timing_engine.interval_confirmation = cfg_confirm

    st.markdown("""
    <div style="font-size: 11px; color: #94a3b8; margin-top: 24px; text-align: center;">
        Inpatient Telemetry Interface v2.2<br>
        MobileNetV2 Inference Service
    </div>
    """, unsafe_allow_html=True)


# ---------------- MAIN APPLICATION BAR & DEMOGRAPHICS ----------------

is_active = st.session_state.monitoring_active
status_badge_class = "active" if is_active else ""
status_dot_class = "" if is_active else "idle"
status_text = "MONITORING ACTIVE" if is_active else "SYSTEM STANDBY"

st.markdown(f"""
<div class="hospital-appbar">
    <div class="appbar-brand">
        <h1 class="appbar-title">IV Fluid Monitoring System</h1>
        <p class="appbar-subtitle">Inpatient Telemetry & Autonomous Fluid-Level Surveillance</p>
    </div>
    <div class="appbar-badge {status_badge_class}">
        <span class="badge-dot {status_dot_class}"></span>
        <span>{status_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="patient-strip">
    <div class="patient-cell">
        <span class="patient-cell-label">Patient ID</span>
        <span class="patient-cell-value">{patient_id}</span>
    </div>
    <div class="patient-cell">
        <span class="patient-cell-label">Patient Name</span>
        <span class="patient-cell-value">{patient_name}</span>
    </div>
    <div class="patient-cell">
        <span class="patient-cell-label">Bed Location</span>
        <span class="patient-cell-value">{bed_number}</span>
    </div>
    <div class="patient-cell">
        <span class="patient-cell-label">Assigned Staff</span>
        <span class="patient-cell-value">{assigned_nurse}</span>
    </div>
    <div class="patient-cell">
        <span class="patient-cell-label">Operating Mode</span>
        <span class="patient-cell-value" style="color: #0f766e;">{monitoring_mode.split()[0]}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- MODEL INITIALIZATION ----------------

try:
    model, labels = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load MobileNetV2 model: {e}")
    model_loaded = False


# ---------------- MODE 1: LIVE BEDSIDE VIDEO STREAM ----------------

if monitoring_mode == "Live Bedside Video Stream":
    left_col, right_col = st.columns([1.12, 1.0])

    with left_col:
        st.markdown("""
        <div class="panel-header">
            <span class="panel-title">Bedside Live Video Feed</span>
            <span class="panel-meta">Continuous Display (640x480)</span>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            video_placeholder = st.empty()

            c_ctrl1, c_ctrl2 = st.columns([1, 1])
            with c_ctrl1:
                if not st.session_state.monitoring_active:
                    if st.button("Start Live Monitoring", type="primary", use_container_width=True):
                        ok = st.session_state.camera_manager.initialize(resolved_source, stream_url, dev_idx)
                        if ok:
                            st.session_state.monitoring_active = True
                            st.session_state.last_snapshot_time = 0.0
                            st.session_state.is_stale = False
                            st.rerun()
                        else:
                            st.error(st.session_state.camera_manager.error_message or "Unable to initialize camera source.")
                else:
                    if st.button("Stop Live Monitoring", type="secondary", use_container_width=True):
                        st.session_state.monitoring_active = False
                        st.session_state.camera_manager.release()
                        st.rerun()

            with c_ctrl2:
                force_snap = st.button("Capture Snapshot Now", use_container_width=True, disabled=not st.session_state.monitoring_active)

            snapshot_box = st.empty()

    with right_col:
        st.markdown("""
        <div class="panel-header">
            <span class="panel-title">Fluid Level Analysis</span>
            <span class="panel-meta">MobileNetV2 Inference</span>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            status_box = st.empty()
            metrics_box = st.empty()
            timing_box = st.empty()
            details_box = st.empty()

    def render_prediction_panel(pred, snap_img, countdown, active_interval, is_stale, last_err):
        if pred is None:
            status_box.markdown("""
            <div class="status-banner status-banner-normal">
                <div class="status-banner-title">System Standby</div>
                <div class="status-banner-sub">Awaiting initial camera snapshot for MobileNetV2 fluid-level inference.</div>
            </div>
            """, unsafe_allow_html=True)
            timing_box.markdown("""
            <div class="timing-info-card">
                <div class="timing-row">
                    <span>Sampling Interval</span>
                    <strong>30 seconds (Default Baseline)</strong>
                </div>
                <div class="timing-row">
                    <span>Inference State</span>
                    <span>Ready</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        level_val = pred["level"]
        label_text = pred["label"]
        confidence_val = pred["confidence"]
        is_empty = pred["is_confirmed_empty"]
        consecutive_count = pred["consecutive_empty"]
        reason_text = pred["reason"]
        t_stamp = pred["timestamp"]

        # 1. Status Indicator Banner
        if is_empty or level_val == 0:
            status_html = f"""
            <div class="status-banner status-banner-empty">
                <div class="status-banner-title">CRITICAL ALERT: CONTAINER EMPTY</div>
                <div class="status-banner-sub">
                    Confirmed empty fluid condition (reading #{consecutive_count}). Immediate replacement required.
                </div>
            </div>
            """
        elif level_val <= threshold:
            status_html = f"""
            <div class="status-banner status-banner-low">
                <div class="status-banner-title">LOW FLUID ATTENTION ({label_text})</div>
                <div class="status-banner-sub">
                    Fluid level at or below warning threshold. Prepare replacement container.
                </div>
            </div>
            """
        else:
            status_html = f"""
            <div class="status-banner status-banner-normal">
                <div class="status-banner-title">NORMAL INFUSION ({label_text})</div>
                <div class="status-banner-sub">
                    Fluid volume within safe clinical range. Infusion proceeding normally.
                </div>
            </div>
            """
        status_box.markdown(status_html, unsafe_allow_html=True)

        # 2. Key Metrics 2x2 Grid
        with metrics_box.container():
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-box-label">Fluid Level Category</div>
                    <div class="metric-box-value">{label_text}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Model Confidence</div>
                    <div class="metric-box-value">{confidence_val * 100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Sampling Interval</div>
                    <div class="metric-box-value">{active_interval}s</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-label">Last Analyzed</div>
                    <div class="metric-box-value">{t_stamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(max(0.0, min(1.0, level_val / 100.0)))

        # 3. Adaptive Timing & Schedule Detail Card
        stale_html = ""
        if is_stale:
            stale_html = f"""
            <div class="stale-notice">
                Warning: Last inference attempt encountered an issue ({last_err or 'Timeout'}).
                Displaying last valid reading from {t_stamp} (STALE DATA).
            </div>
            """

        timing_box.markdown(f"""
        <div class="timing-info-card">
            <div class="timing-row">
                <span>Active Capture Interval</span>
                <strong>{active_interval} seconds</strong>
            </div>
            <div class="timing-row">
                <span>Next Automatic Inference</span>
                <strong>in {countdown:.1f}s</strong>
            </div>
            <div class="timing-row">
                <span>Inference Latency</span>
                <span>{pred.get('elapsed_ms', 0):.0f} ms</span>
            </div>
            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #f1f5f9; color: #64748b; font-size: 11px;">
                <strong>Reason:</strong> {reason_text}
            </div>
        </div>
        {stale_html}
        """, unsafe_allow_html=True)

        # 4. Snapshot Thumbnail
        if snap_img is not None:
            snapshot_box.image(
                snap_img,
                caption=f"Latest Analyzed Snapshot — Sampled at {t_stamp} (224x224 RGB)",
                use_container_width=True,
            )

        # 5. Class Probability Distribution Table
        with details_box.container():
            with st.expander("Probability Distribution (Model Softmax Head)", expanded=False):
                rows_html = "".join([
                    f"<tr><td><strong>{LEVEL_LABELS.get(cls_name, cls_name)}</strong> ({cls_name})</td><td style='text-align: right;'><strong>{float(prob) * 100:.2f}%</strong></td></tr>"
                    for cls_name, prob in zip(labels, pred["all_scores"])
                ])
                st.markdown(f"""
                <table class="prob-table">
                    <thead>
                        <tr>
                            <th>Class Category</th>
                            <th style="text-align: right;">Softmax Output</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                """, unsafe_allow_html=True)

    # ---------------- CONTINUOUS STREAMING & INFERENCE LOOP ----------------
    if st.session_state.monitoring_active and model_loaded:
        while st.session_state.monitoring_active:
            # Step A: Capture latest live frame
            frame = st.session_state.camera_manager.get_frame()
            if frame is None:
                video_placeholder.error(
                    f"Camera stream unreachable: {st.session_state.camera_manager.error_message or 'No frame received'}"
                )
                st.session_state.monitoring_active = False
                break

            # Step B: Render live frame immediately (smooth continuous video, ~25 FPS)
            video_placeholder.image(frame, channels="RGB", use_container_width=True)

            now = time.monotonic()

            # Step C: Check for completed background inference
            if st.session_state.inference_future is not None:
                if st.session_state.inference_future.done():
                    try:
                        pred_class, level, score, all_scores, pil_img, t_stamp, elapsed = st.session_state.inference_future.result()
                        next_interval, reason, is_confirmed_empty = st.session_state.timing_engine.update(pred_class, score)

                        st.session_state.latest_prediction = {
                            "class": pred_class,
                            "level": level,
                            "label": LEVEL_LABELS.get(pred_class, pred_class),
                            "confidence": score,
                            "all_scores": all_scores,
                            "timestamp": t_stamp,
                            "elapsed_ms": elapsed * 1000.0,
                            "next_interval": next_interval,
                            "reason": reason,
                            "is_confirmed_empty": is_confirmed_empty,
                            "consecutive_empty": st.session_state.timing_engine.consecutive_empty_count,
                        }
                        st.session_state.is_stale = False
                        st.session_state.last_error = None

                        # Automatic backend alert evaluation (50% low warning & confirmed empty)
                        if BACKEND_ALERTS_AVAILABLE:
                            try:
                                if is_confirmed_empty:
                                    evaluate_reading_for_alert(
                                        patient_id=patient_id,
                                        level_label="Empty",
                                        confidence=score,
                                        captured_at=datetime.now(timezone.utc).isoformat(),
                                    )
                                elif level == 50:
                                    evaluate_reading_for_alert(
                                        patient_id=patient_id,
                                        level_label="50%",
                                        confidence=score,
                                        captured_at=datetime.now(timezone.utc).isoformat(),
                                    )
                                elif level in (80, 100):
                                    evaluate_reading_for_alert(
                                        patient_id=patient_id,
                                        level_label=f"{level}%",
                                        confidence=score,
                                        captured_at=datetime.now(timezone.utc).isoformat(),
                                    )
                            except Exception:
                                pass

                    except Exception as exc:
                        st.session_state.is_stale = True
                        st.session_state.last_error = str(exc)
                    finally:
                        st.session_state.inference_future = None

            # Step D: Determine interval & check if snapshot is due
            active_interval = (
                st.session_state.latest_prediction["next_interval"]
                if st.session_state.latest_prediction
                else st.session_state.timing_engine.current_interval
            )
            time_since_snapshot = now - st.session_state.last_snapshot_time

            should_snapshot = (
                (time_since_snapshot >= active_interval or st.session_state.last_snapshot_time == 0.0)
                and st.session_state.inference_future is None
            )

            if should_snapshot:
                pil_snapshot = Image.fromarray(frame)
                st.session_state.last_snapshot_image = pil_snapshot
                st.session_state.last_snapshot_time = now

                # Submit to background thread so video playback is NEVER blocked
                st.session_state.inference_future = st.session_state.executor.submit(
                    _run_background_inference, pil_snapshot, model, labels
                )

            # Step E: Update UI telemetry & countdown
            countdown = max(0.0, active_interval - (now - st.session_state.last_snapshot_time))
            render_prediction_panel(
                st.session_state.latest_prediction,
                st.session_state.last_snapshot_image,
                countdown,
                active_interval,
                st.session_state.is_stale,
                st.session_state.last_error,
            )

            time.sleep(0.035)

    elif not st.session_state.monitoring_active:
        # Paused / Standby state: render last captured telemetry if available
        if st.session_state.last_snapshot_image is not None:
            video_placeholder.image(
                st.session_state.last_snapshot_image,
                caption="Monitoring Paused — Last Sampled Frame",
                use_container_width=True,
            )
        else:
            video_placeholder.info("Camera standby. Click 'Start Live Monitoring' to begin continuous video surveillance.")

        active_int = (
            st.session_state.latest_prediction["next_interval"]
            if st.session_state.latest_prediction
            else st.session_state.timing_engine.current_interval
        )
        render_prediction_panel(
            st.session_state.latest_prediction,
            st.session_state.last_snapshot_image,
            0.0,
            active_int,
            st.session_state.is_stale,
            st.session_state.last_error,
        )


# ---------------- MODE 2: DIAGNOSTIC IMAGE UPLOAD ----------------

elif monitoring_mode == "Diagnostic Image Upload":
    st.markdown("""
    <div class="panel-header">
        <span class="panel-title">Diagnostic Image Evaluation</span>
        <span class="panel-meta">Offline File Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        uploaded_file = st.file_uploader(
            "Select IV bottle image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            col_img, col_pred = st.columns([1, 1])

            with col_img:
                st.image(uploaded_image, caption="Uploaded Container Image", use_container_width=True)
                st.caption(f"Image Resolution: {uploaded_image.width} × {uploaded_image.height} pixels")

            with col_pred:
                if st.button("Run Fluid Analysis", type="primary"):
                    try:
                        with st.spinner("Classifying with MobileNetV2..."):
                            pred_class, level, score, all_scores = predict_image(uploaded_image, model, labels)

                        if level == 0:
                            st.markdown("""
                            <div class="status-banner status-banner-empty">
                                <div class="status-banner-title">CRITICAL: EMPTY CONTAINER DETECTED</div>
                                <div class="status-banner-sub">Immediate fluid replenishment required.</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif level <= threshold:
                            st.markdown(f"""
                            <div class="status-banner status-banner-low">
                                <div class="status-banner-title">LOW FLUID ATTENTION ({LEVEL_LABELS[pred_class]})</div>
                                <div class="status-banner-sub">Container volume below threshold.</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="status-banner status-banner-normal">
                                <div class="status-banner-title">NORMAL FLUID LEVEL ({LEVEL_LABELS[pred_class]})</div>
                                <div class="status-banner-sub">Container level is within normal clinical limits.</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-box-label">Category</div>
                                <div class="metric-box-value">{LEVEL_LABELS[pred_class]}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-box-label">Confidence</div>
                                <div class="metric-box-value">{score * 100:.2f}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.progress(level / 100.0)

                        with st.expander("Probability Distribution (Model Softmax Head)", expanded=False):
                            rows_html = "".join([
                                f"<tr><td><strong>{LEVEL_LABELS.get(lbl, lbl)}</strong> ({lbl})</td><td style='text-align: right;'><strong>{float(val) * 100:.2f}%</strong></td></tr>"
                                for lbl, val in zip(labels, all_scores)
                            ])
                            st.markdown(f"""
                            <table class="prob-table">
                                <thead>
                                    <tr>
                                        <th>Class Category</th>
                                        <th style="text-align: right;">Softmax Output</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rows_html}
                                </tbody>
                            </table>
                            """, unsafe_allow_html=True)

                    except Exception as err:
                        st.error(f"Inference error: {err}")
        else:
            st.info("Select or drop an IV container image file above to perform diagnostic classification.")


# ---------------- MODE 3: MANUAL CAMERA SNAPSHOT ----------------

else:
    st.markdown("""
    <div class="panel-header">
        <span class="panel-title">Manual Camera Snapshot</span>
        <span class="panel-meta">Single-Frame Capture</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        camera_file = st.camera_input("Capture single IV container image")

        if camera_file is not None:
            captured_image = Image.open(camera_file).convert("RGB")
            col_c1, col_c2 = st.columns([1, 1])

            with col_c1:
                st.image(captured_image, caption="Captured Snapshot", use_container_width=True)

            with col_c2:
                if st.button("Analyze Snapshot", type="primary"):
                    try:
                        with st.spinner("Classifying with MobileNetV2..."):
                            pred_class, level, score, all_scores = predict_image(captured_image, model, labels)

                        if level == 0:
                            st.markdown("""
                            <div class="status-banner status-banner-empty">
                                <div class="status-banner-title">CRITICAL: EMPTY CONTAINER DETECTED</div>
                                <div class="status-banner-sub">Immediate fluid replenishment required.</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif level <= threshold:
                            st.markdown(f"""
                            <div class="status-banner status-banner-low">
                                <div class="status-banner-title">LOW FLUID ATTENTION ({LEVEL_LABELS[pred_class]})</div>
                                <div class="status-banner-sub">Container volume below threshold.</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="status-banner status-banner-normal">
                                <div class="status-banner-title">NORMAL FLUID LEVEL ({LEVEL_LABELS[pred_class]})</div>
                                <div class="status-banner-sub">Container level is within normal clinical limits.</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-box-label">Category</div>
                                <div class="metric-box-value">{LEVEL_LABELS[pred_class]}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-box-label">Confidence</div>
                                <div class="metric-box-value">{score * 100:.2f}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.progress(level / 100.0)

                        with st.expander("Probability Distribution (Model Softmax Head)", expanded=False):
                            rows_html = "".join([
                                f"<tr><td><strong>{LEVEL_LABELS.get(lbl, lbl)}</strong> ({lbl})</td><td style='text-align: right;'><strong>{float(val) * 100:.2f}%</strong></td></tr>"
                                for lbl, val in zip(labels, all_scores)
                            ])
                            st.markdown(f"""
                            <table class="prob-table">
                                <thead>
                                    <tr>
                                        <th>Class Category</th>
                                        <th style="text-align: right;">Softmax Output</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rows_html}
                                </tbody>
                            </table>
                            """, unsafe_allow_html=True)

                    except Exception as err:
                        st.error(f"Inference error: {err}")


# ---------------- CLINICAL DISCLAIMER & FOOTER ----------------

st.markdown("""
<div style="color: #64748b; font-size: 11px; line-height: 1.5; margin-top: 24px; padding-top: 14px; border-top: 1px solid #e2e8f0;">
    <strong>Clinical Monitoring Disclaimer:</strong> This technological surveillance system provides auxiliary optical level classification to assist healthcare staff.
    Estimations represent discrete categorical classes (100% Filled, 80% Filled, 50% Filled, Empty) and do not replace direct physical evaluation by trained medical personnel.
    <br>
    <em>Operating Scope:</em> Real-time video rendering, snapshot sampling, and inference execution are active while this Streamlit session is active.
    For continuous headless 24/7 background monitoring without a browser session, run the system service <code>pi_camera_service.py</code>.
</div>
""", unsafe_allow_html=True)

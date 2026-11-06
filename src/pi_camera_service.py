"""
pi_camera_service.py - Unified Raspberry Pi Camera & Monitoring Service

This service runs on the Raspberry Pi:
1. Manages a SINGLE Picamera2 instance safely (prevents camera hardware contention).
2. Serves a live MJPEG video stream on port 8080 (/video_feed) for the Flutter Nurse Dashboard.
3. Automatically grabs frames every 5 seconds, classifies the IV fluid level (either
   using local MobileNetV2 or by delegating inference to the laptop FastAPI backend),
   and posts authenticated readings to the FastAPI backend with the patient device key.
"""

import argparse
import io
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple

# Optional imports for local inference
try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False


# Default configuration
DEFAULT_STREAM_PORT = 8080
DEFAULT_BACKEND_URL = "http://LAPTOP-C51RSUMS:8000"
DEFAULT_PATIENT_ID = "IVP002"
DEFAULT_INTERVAL = 5
FRAME_SIZE = (640, 480)
MODEL_SIZE = (224, 224)

LEVELS = {
    "sal_data_100": "100%",
    "sal_data_80": "80%",
    "sal_data_50": "50%",
    "sal_data_empty": "Empty",
}


class CameraManager:
    """Thread-safe camera manager holding a single camera instance."""

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.camera = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_pil = None
        self.capture_thread = None

    def start(self):
        if not PICAMERA2_AVAILABLE:
            print("[CameraManager] WARNING: Picamera2 library not found. Running in mock/fallback mode.")
            self._start_mock_loop()
            return

        try:
            print(f"[CameraManager] Initializing Picamera2 ({self.width}x{self.height})...")
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            self.running = True

            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            print("[CameraManager] Picamera2 initialized and capturing successfully.")
        except Exception as error:
            print(f"[CameraManager] ERROR initializing Picamera2: {error}")
            self.stop()
            self._start_mock_loop()

    def _capture_loop(self):
        while self.running and self.camera:
            try:
                frame = self.camera.capture_array("main")
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Process image into RGB and JPEG
                if frame.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if cv2 else None
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if cv2 else None
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if cv2 else None
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) if cv2 else None
                else:
                    # In Picamera2 RGB888 mode
                    frame_rgb = frame
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if cv2 else None

                if cv2 and frame_bgr is not None:
                    _, jpeg_bytes = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    jpeg_data = jpeg_bytes.tobytes()
                elif Image and frame_rgb is not None:
                    pil_img = Image.fromarray(frame_rgb)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="JPEG", quality=80)
                    jpeg_data = buffer.getvalue()
                else:
                    jpeg_data = b""

                pil_img = Image.fromarray(frame_rgb) if Image and frame_rgb is not None else None

                with self.lock:
                    self.latest_jpeg = jpeg_data
                    self.latest_pil = pil_img

                time.sleep(0.04)  # ~25 FPS
            except Exception as e:
                time.sleep(0.05)

    def _start_mock_loop(self):
        """Fallback mock frame generator if camera hardware is unavailable."""
        self.running = True

        def mock_loop():
            # Generate a 640x480 placeholder image with timestamp
            while self.running:
                if Image:
                    img = Image.new("RGB", (self.width, self.height), color=(30, 41, 59))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    with self.lock:
                        self.latest_jpeg = buf.getvalue()
                        self.latest_pil = img
                time.sleep(0.2)

        t = threading.Thread(target=mock_loop, daemon=True)
        t.start()

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_latest_pil(self):
        with self.lock:
            return self.latest_pil

    def stop(self):
        self.running = False
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.close()
            except Exception:
                pass
            self.camera = None
        print("[CameraManager] Camera safely stopped and closed.")


class StreamRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving MJPEG video feed and health status."""

    camera_manager = None

    def log_message(self, format, *args):
        # Suppress routine GET logs for clean console output
        return

    def do_GET(self):
        if self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    frame = self.camera_manager.get_latest_jpeg()
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.06)  # ~16 FPS
            except (ConnectionResetError, BrokenPipeError):
                pass
            return

        elif self.path in ("/", "/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {
                "status": "online",
                "camera_connected": self.camera_manager.camera is not None if PICAMERA2_AVAILABLE else False,
                "streaming_url": "/video_feed",
                "timestamp": time.time(),
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


PERCENT_MAP = {
    "100%": 100,
    "80%": 80,
    "50%": 50,
    "Empty": 0,
    "sal_data_100": 100,
    "sal_data_80": 80,
    "sal_data_50": 50,
    "sal_data_empty": 0,
}


class AdaptiveTimingManager:
    """
    Dynamically optimizes MobileNetV2 capture and inference frequency
    on the Raspberry Pi based on current fluid level and observed rate of change:
    - Level > 50% and stable: 30 seconds (reduces computational load & heat)
    - Level <= 50%: 15 seconds (closer monitoring as bottle empties)
    - Rapidly decreasing level (drop >= 30%): 7 seconds (configurable 5-10s)
    - Suspected empty condition: 4 seconds (fast re-capture for 2nd confirmation)
    - When stabilized or refilled: smoothly returns to normal monitoring interval
    """

    def __init__(
        self,
        interval_high_stable: int = 30,
        interval_low: int = 15,
        interval_rapid_drop: int = 7,
        interval_confirmation: int = 4,
        min_confidence: float = 0.60,
    ):
        self.interval_high_stable = interval_high_stable
        self.interval_low = interval_low
        self.interval_rapid_drop = interval_rapid_drop
        self.interval_confirmation = interval_confirmation
        self.min_confidence = min_confidence

        self.last_valid_level: Optional[str] = None
        self.last_valid_percent: Optional[int] = None
        self.consecutive_empty_count: int = 0
        self.current_interval: int = interval_high_stable

    def calculate_next_interval(self, level_label: str, confidence: float) -> Tuple[int, str]:
        """
        Calculates the next capture interval in seconds and returns (interval, reason).
        """
        # 1. Low Confidence Safe Handling (< 0.60)
        # Prediction is ambiguous (glare/occlusion); do not trigger state changes,
        # schedule a fast re-check to re-verify safely.
        if confidence < self.min_confidence:
            interval = self.interval_rapid_drop
            reason = (
                f"Low confidence ({confidence * 100:.1f}% < {self.min_confidence * 100:.0f}%); "
                f"scheduling fast verification check in {interval}s"
            )
            self.current_interval = interval
            return interval, reason

        # 2. Suspected / Confirmed Empty Condition
        if level_label == "Empty":
            if self.consecutive_empty_count == 0:
                # First Empty detection: Suspected empty condition.
                # Fast capture (3-5s) to confirm or refute before triggering alerts.
                self.consecutive_empty_count = 1
                self.last_valid_level = level_label
                self.last_valid_percent = 0
                interval = self.interval_confirmation
                reason = f"Suspected empty condition; fast confirmation capture scheduled in {interval}s"
            else:
                self.consecutive_empty_count += 1
                interval = self.interval_low
                reason = (
                    f"Empty condition active (consecutive reading #{self.consecutive_empty_count}); "
                    f"monitoring every {interval}s"
                )
            self.current_interval = interval
            return interval, reason

        # 3. Non-Empty Levels (100%, 80%, 50%)
        curr_percent = PERCENT_MAP.get(level_label, 50)
        prev_percent = self.last_valid_percent
        self.consecutive_empty_count = 0  # Reset empty counter upon valid non-empty reading

        # Check for rapid decrease (drop >= 30%)
        if prev_percent is not None and (prev_percent - curr_percent) >= 30:
            interval = self.interval_rapid_drop
            reason = (
                f"Rapid fluid drop detected ({prev_percent}% -> {curr_percent}%); "
                f"increased capture frequency to {interval}s"
            )
        elif curr_percent <= 50:
            interval = self.interval_low
            reason = f"Fluid level at or below 50% ({curr_percent}%); monitoring every {interval}s"
        else:
            # Fluid level > 50% and stable
            interval = self.interval_high_stable
            reason = f"Fluid level > 50% and stable ({curr_percent}%); power-saving interval {interval}s"

        self.last_valid_level = level_label
        self.last_valid_percent = curr_percent
        self.current_interval = interval
        return interval, reason


class MonitoringLoop:
    """
    Runs adaptive IV fluid classification and reading submission.
    Keeps the loaded MobileNetV2 model in memory across all captures to avoid reloading overhead.
    Transmits capture_interval with each reading to enable dynamic device status windows.
    """

    def __init__(
        self,
        camera_manager,
        backend_url,
        patient_id,
        device_key,
        timing_manager: Optional[AdaptiveTimingManager] = None,
        fixed_interval: Optional[int] = None,
    ):
        self.camera_manager = camera_manager
        self.backend_url = backend_url.rstrip("/")
        self.patient_id = patient_id
        self.device_key = device_key
        self.timing_manager = timing_manager or AdaptiveTimingManager()
        self.fixed_interval = fixed_interval
        self.current_interval = self.fixed_interval or self.timing_manager.current_interval
        self.running = False
        self.local_model = None
        self.labels = None
        self.thread = None

    def load_local_model_if_available(self):
        try:
            import tensorflow as tf
            root = Path(__file__).resolve().parent
            model_path = root / "models" / "mobilenetv2_iv.keras"
            labels_path = root / "models" / "mobilenetv2_labels.json"

            if model_path.is_file() and labels_path.is_file():
                print(f"[MonitoringLoop] Found local model at {model_path}. Loading into memory once...")
                self.local_model = tf.keras.models.load_model(model_path)
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                print("[MonitoringLoop] Local MobileNetV2 model loaded and cached in memory successfully.")
                return True
        except Exception as error:
            print(f"[MonitoringLoop] Local model load failed ({error}). Using remote backend inference.")
        return False

    def predict_local(self, pil_image):
        image = pil_image.convert("RGB").resize(MODEL_SIZE)
        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = self.local_model.predict(image_array, verbose=0)[0]
        idx = int(np.argmax(predictions))
        pred_class = self.labels[idx]
        level_label = LEVELS.get(pred_class, pred_class)
        confidence = float(predictions[idx])
        return level_label, confidence

    def predict_remote(self, jpeg_bytes):
        url = f"{self.backend_url}/api/predict"
        boundary = "----WebKitFormBoundary" + os.urandom(16).hex()
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(b'Content-Disposition: form-data; name="file"; filename="pi_frame.jpg"\r\n')
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(jpeg_bytes)
        body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        payload = body.getvalue()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(payload)),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pred_class = data.get("label")
            level_label = LEVELS.get(pred_class, pred_class)
            confidence = float(data.get("confidence", 0.0))
            return level_label, confidence

    def send_reading(self, level_label: str, confidence: float, capture_interval: int):
        url = f"{self.backend_url}/api/patients/{self.patient_id}/readings"
        data = {
            "level_label": level_label,
            "confidence": confidence,
            "capture_interval": capture_interval,
        }
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Device-Key": self.device_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def start(self):
        self.load_local_model_if_available()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        mode_str = f"Fixed {self.fixed_interval}s" if self.fixed_interval else "Adaptive Timing Enabled"
        print(f"[MonitoringLoop] Monitoring started for Patient {self.patient_id} ({mode_str})")

    def _run(self):
        time.sleep(2)  # Allow camera to warm up
        next_interval = self.fixed_interval or self.current_interval

        while self.running:
            cycle_start = time.monotonic()
            try:
                jpeg_frame = self.camera_manager.get_latest_jpeg()
                pil_frame = self.camera_manager.get_latest_pil()

                if not jpeg_frame:
                    time.sleep(1)
                    continue

                if self.local_model and pil_frame:
                    level_label, confidence = self.predict_local(pil_frame)
                else:
                    level_label, confidence = self.predict_remote(jpeg_frame)

                # Send reading with current active interval to backend
                result = self.send_reading(level_label, confidence, next_interval)

                # Determine next capture interval
                if self.fixed_interval is not None:
                    next_interval = self.fixed_interval
                    reason = f"Fixed interval {next_interval}s"
                else:
                    next_interval, reason = self.timing_manager.calculate_next_interval(
                        level_label, confidence
                    )

                self.current_interval = next_interval
                print(
                    f"[{time.strftime('%H:%M:%S')}] Patient: {self.patient_id} | "
                    f"Level: {level_label} | "
                    f"Confidence: {confidence * 100:.1f}% | "
                    f"Next Capture: {next_interval}s ({reason}) | "
                    f"Backend: {result.get('message', 'Saved')}"
                )
            except urllib.error.HTTPError as error:
                print(f"[MonitoringLoop] Backend HTTP error {error.code}: {error.reason}")
            except urllib.error.URLError as error:
                print(f"[MonitoringLoop] Cannot connect to laptop at {self.backend_url}: {error.reason}")
            except Exception as error:
                print(f"[MonitoringLoop] Error in cycle: {error}")

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.5, next_interval - elapsed))

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Unified Raspberry Pi Camera & Adaptive Monitoring Service")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP stream bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_STREAM_PORT, help="HTTP stream port (default 8080)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL, help="FastAPI backend URL (e.g. http://192.168.1.46:8000)")
    parser.add_argument("--patient-id", default=DEFAULT_PATIENT_ID, help="Patient ID to monitor (default IVP002)")
    parser.add_argument("--device-key", default="", help="Patient device key for X-Device-Key header")
    parser.add_argument("--interval", type=int, default=None, help="Fixed capture interval override in seconds (disables adaptive timing)")
    parser.add_argument("--interval-high-stable", type=int, default=int(os.getenv("IV_INTERVAL_HIGH_STABLE", "30")),
                        help="Adaptive interval when fluid > 50%% and stable in seconds (default 30)")
    parser.add_argument("--interval-low", type=int, default=int(os.getenv("IV_INTERVAL_LOW", "15")),
                        help="Adaptive interval when fluid <= 50%% in seconds (default 15)")
    parser.add_argument("--interval-rapid-drop", type=int, default=int(os.getenv("IV_INTERVAL_RAPID_DROP", "7")),
                        help="Adaptive interval when rapid drop detected in seconds (default 7)")
    parser.add_argument("--interval-confirmation", type=int, default=int(os.getenv("IV_INTERVAL_CONFIRMATION", "4")),
                        help="Fast re-capture interval for suspected empty confirmation in seconds (default 4)")
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("IV_MIN_CONFIDENCE", "0.60")),
                        help="Minimum confidence threshold (default 0.60)")
    parser.add_argument("--no-predict", action="store_true", help="Run stream only without posting predictions")
    args = parser.parse_args()

    # Fall back to environment variables if arguments not provided
    backend = os.getenv("IV_BACKEND_URL", args.backend)
    patient_id = os.getenv("IV_PATIENT_ID", args.patient_id)
    device_key = os.getenv("IV_DEVICE_KEY", args.device_key)

    # Initialize shared camera manager
    camera_manager = CameraManager(width=FRAME_SIZE[0], height=FRAME_SIZE[1])
    camera_manager.start()

    # Start HTTP stream server
    StreamRequestHandler.camera_manager = camera_manager
    server = ThreadingHTTPServer((args.host, args.port), StreamRequestHandler)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[PiService] Live MJPEG video feed running at: http://{args.host}:{args.port}/video_feed")

    # Start adaptive monitoring loop if device key is provided
    monitor = None
    if not args.no_predict and device_key:
        timing_manager = AdaptiveTimingManager(
            interval_high_stable=args.interval_high_stable,
            interval_low=args.interval_low,
            interval_rapid_drop=args.interval_rapid_drop,
            interval_confirmation=args.interval_confirmation,
            min_confidence=args.min_confidence,
        )
        monitor = MonitoringLoop(
            camera_manager=camera_manager,
            backend_url=backend,
            patient_id=patient_id,
            device_key=device_key,
            timing_manager=timing_manager,
            fixed_interval=args.interval,
        )
        monitor.start()
    elif not device_key:
        print("[PiService] NOTE: No --device-key provided. Video stream is active, but readings will not be sent.")
        print("          Provide --device-key <key> to start automatic IV fluid predictions.")

    # Graceful shutdown handler
    def shutdown_signal(sig, frame):
        print("\n[PiService] Shutting down service cleanly...")
        if monitor:
            monitor.stop()
        server.shutdown()
        camera_manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_signal)
    signal.signal(signal.SIGTERM, shutdown_signal)

    print("[PiService] Service running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

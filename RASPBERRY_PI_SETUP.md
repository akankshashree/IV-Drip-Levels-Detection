# Raspberry Pi Setup & Camera Integration Guide

This guide provides step-by-step instructions for setting up the Raspberry Pi camera as the **primary camera source** for your IV Fluid Monitoring System.

---

## 1. Files to Copy to Your Raspberry Pi

You only need a few lightweight files on your Raspberry Pi. You do **not** need to copy the entire Flutter project or the backend database to the Pi.

### Required Files:
1. `pi_camera_service.py` — The unified camera and monitoring service.
2. `models/mobilenetv2_iv.keras` *(Optional)* — The MobileNetV2 model if you wish to run predictions locally on the Pi.
3. `models/mobilenetv2_labels.json` *(Optional)* — The class labels file.

### Folder Structure on Raspberry Pi:
Create a folder named `iv_monitoring` in your Pi user home directory (`/home/pi` or `/home/<username>`):

```text
/home/pi/iv_monitoring/
├── pi_camera_service.py
└── models/
    ├── mobilenetv2_iv.keras       (Optional)
    └── mobilenetv2_labels.json    (Optional)
```

> [!TIP]
> **Two Inference Modes Supported**:
> `pi_camera_service.py` automatically detects whether TensorFlow is installed on your Pi:
> - **Lightweight Mode (Recommended)**: If TensorFlow is not installed, the Pi captures camera frames and delegates the classification to your laptop's FastAPI backend (`/api/predict`). This is extremely fast, uses minimal Pi CPU/RAM, and requires no complicated TensorFlow setup on the Pi!
> - **Local Inference Mode**: If you install TensorFlow and copy the model files, the Pi runs the classification on its own CPU.

---

## 2. Enabling the Raspberry Pi Camera

Execute these commands in the **Raspberry Pi terminal**:

### Step 2.1: Enable Camera Interface
```bash
sudo raspi-config
```
1. Use arrow keys to navigate to **Interface Options**.
2. Select **Camera** (or **Legacy Camera** depending on your Raspberry Pi OS version) and choose **Enable**.
3. Select **Finish** and reboot if prompted:
```bash
sudo reboot
```

### Step 2.2: Test the Camera Hardware
Run this quick command to verify your camera sensor is detected:
```bash
# On newer Raspberry Pi OS (Bullseye / Bookworm):
rpicam-hello -t 3000

# On legacy Raspberry Pi OS:
libcamera-hello -t 3000
```
If a camera preview appears on your Pi screen for 3 seconds, your camera hardware is connected and working!

---

## 3. Installing Dependencies on Raspberry Pi

Open a **Raspberry Pi terminal** and run:

```bash
# 1. Update system package index
sudo apt update

# 2. Install required Python and camera libraries
sudo apt install -y python3-pip python3-picamera2 python3-pil python3-opencv python3-numpy
```

---

## 4. Connecting the Pi to Your Windows Laptop

Both your Raspberry Pi and your Windows laptop must be connected to the **same Wi-Fi network**.

### Step 4.1: Find your Laptop's IP Address
On your **Windows laptop**, open PowerShell and run:
```powershell
ipconfig
```
Look for **Wireless LAN adapter Wi-Fi** -> **IPv4 Address** (for example: `192.168.1.46`).

### Step 4.2: Find your Raspberry Pi's IP Address
On your **Raspberry Pi terminal**, run:
```bash
hostname -I
```
This prints your Pi's local IP address (for example: `192.168.1.15`).

### Step 4.3: Test Connection from the Pi to your Laptop
On your **Raspberry Pi terminal**, ping your laptop:
```bash
ping -c 3 192.168.1.46
```
If you see replies (`bytes from 192.168.1.46...`), the network path is open!

> [!NOTE]
> If `ping` times out, check **Windows Defender Firewall** on your laptop and ensure port `8000` is permitted for inbound local traffic.

---

## 5. Starting the Unified Pi Camera Service

On your **Raspberry Pi terminal**, navigate to your folder and start the service:

```bash
cd /home/pi/iv_monitoring

# Standard execution with Adaptive Timing (Recommended for Bedside Monitoring):
python3 pi_camera_service.py \
  --backend http://192.168.1.46:8000 \
  --patient-id IVP002 \
  --device-key boFeqCEcQiYphs1v0Nx7BcJf1G61s9oZto7LebVQHMc \
  --port 8080
```

> **Adaptive Inference Frequency (Bedside Optimization):**
> - **Level > 50% & Stable**: Inference runs every **30 seconds** (minimizes Pi CPU load and thermal heat).
> - **Level <= 50%**: Inference runs every **15 seconds** (closer surveillance as bottle empties).
> - **Rapid Drop (>= 30%)**: Capture interval accelerates to **7 seconds** (fast tracking of sudden fluctuations).
> - **Suspected Empty**: Performs a fast re-capture in **4 seconds** for second confirmation before triggering emergency alerts.
> - **Custom intervals**: Optionally adjust with `--interval-high-stable 30`, `--interval-low 15`, `--interval-rapid-drop 7`, `--interval-confirmation 4`, or set a fixed rate with `--interval <seconds>`.

### Expected Output on the Raspberry Pi:
```text
[CameraManager] Initializing Picamera2 (640x480)...
[CameraManager] Picamera2 initialized and capturing successfully.
[PiService] Live MJPEG video feed running at: http://0.0.0.0:8080/video_feed
[MonitoringLoop] Monitoring started for Patient IVP002 (Adaptive Timing Enabled)
[PiService] Service running. Press Ctrl+C to stop.
[14:32:05] Patient: IVP002 | Level: 80% | Confidence: 91.4% | Next Capture: 30s (Fluid level > 50% and stable (80%); power-saving interval 30s) | Backend: Saved
[14:32:35] Patient: IVP002 | Level: 50% | Confidence: 89.1% | Next Capture: 7s (Rapid fluid drop detected (80% -> 50%); increased capture frequency to 7s) | Backend: Saved
[14:32:42] Patient: IVP002 | Level: 50% | Confidence: 90.2% | Next Capture: 15s (Fluid level at or below 50% (50%); monitoring every 15s) | Backend: Saved
```

---

## 6. Viewing the Pi Camera in the Flutter Dashboard

1. On your laptop, open the Flutter Nurse Dashboard in Chrome.
2. Log in as a nurse (`nada@mailinator.com` or `priya.nurse@mailinator.com`).
3. Click **Live Camera** on Patient `IVP002`.
4. Ensure **Bedside Pi Camera (Bedside Monitor)** is selected.
5. If your Pi IP address differs from `raspberrypi.local:8080`, click **Change Pi Address**, enter your Pi's address (e.g. `192.168.1.15:8080`), and click **Apply**.
6. The live bedside camera feed and automated adaptive IV fluid predictions will display directly inside the dashboard!

---

## 7. Optional: Run Pi Service Automatically on Boot

To run `pi_camera_service.py` as a background system service that restarts automatically:

Create a systemd service file:
```bash
sudo nano /etc/systemd/system/iv-camera.service
```

Paste the following configuration:
```ini
[Unit]
Description=IV Fluid Monitoring Camera Service
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/iv_monitoring
ExecStart=/usr/bin/python3 /home/pi/iv_monitoring/pi_camera_service.py --backend http://192.168.1.46:8000 --patient-id IVP002 --device-key boFeqCEcQiYphs1v0Nx7BcJf1G61s9oZto7LebVQHMc --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable iv-camera.service
sudo systemctl start iv-camera.service
```

Check service status anytime:
```bash
sudo systemctl status iv-camera.service
```

---

## 8. Raspberry Pi Troubleshooting Guide

### 1. "Device or resource busy" / Camera failed to start
- **Cause**: Another program (like Streamlit, Motion, or an earlier Python process) is already using the camera.
- **Solution**:
  Find and kill any competing Python camera processes:
  ```bash
  sudo killall python3
  ```
  Then restart `pi_camera_service.py`.

### 2. "Picamera2 library not found"
- **Solution**: Install the system Picamera2 package:
  ```bash
  sudo apt install -y python3-picamera2
  ```

### 3. "Cannot connect to laptop at http://...: Connection refused / No route to host"
- **Causes**:
  1. The laptop IP changed (DHCP reassignment).
  2. The FastAPI backend is not running on the laptop.
  3. Windows Firewall is blocking inbound connections on port 8000.
- **Solutions**:
  1. Verify FastAPI is running on the laptop (`uvicorn main:app --host 0.0.0.0 --port 8000`).
  2. Confirm your laptop's current IPv4 address with `ipconfig`.
  3. Allow port 8000 in Windows Firewall:
     Run PowerShell as Administrator on your laptop:
     ```powershell
     New-NetFirewallRule -DisplayName "FastAPI Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
     ```

---

## 9. Raspberry Pi Testing Checklist

- [ ] Camera hardware ribbon cable secured firmly in the CAMERA port.
- [ ] `rpicam-hello` runs and displays preview.
- [ ] `python3-picamera2` installed.
- [ ] Pi can ping laptop IP address.
- [ ] `pi_camera_service.py` starts without camera busy errors.
- [ ] Laptop browser can open `http://<pi_ip>:8080/video_feed`.
- [ ] Dashboard displays live Pi stream and updates IV readings every 5 seconds.

# IV Fluid Monitoring System — Complete System Guide

A real-time, IoT-enabled IV fluid level monitoring and automated emergency notification system:
- **Bedside Monitoring Device**: Raspberry Pi Camera (Picamera2 with adaptive MobileNetV2 inference scheduling and MJPEG streaming).
- **Evaluation Demonstration Mode**: Laptop Webcam console allowing the evaluation panel to observe real-time inference and fluid-level transitions at configurable capture intervals (5s default, 10s option).

Both streams utilize the same pre-trained **MobileNetV2** deep learning model and **FastAPI** backend, reporting to the centralized **Flutter Nurse Dashboard**.
---

## 1. System Architecture

```text
                               +-----------------------------------+
                               |     Flutter Nurse Dashboard       |
                               |    (Web / Desktop / Chrome)       |
                               +-----------------+-----------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
              [Bedside Monitor (Hardware)]                            [Demonstration Mode (Evaluation)]
                       |                                                   |
                       v                                                   v
           Raspberry Pi Camera                                        Laptop Webcam
       (pi_camera_service.py :8080)                           (Authorized Nurse Console)
       - Authenticates with X-Device-Key                          - Authenticates via Nurse JWT Session
       - Serves /video_feed (MJPEG)                               - Direct canvas frame capture
       - Auto 5s IV prediction                                    - Auto 5s IV prediction
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |      FastAPI Backend (:8000)      |
                               |  - Dual Authentication (JWT / Key)|
                               |  - MobileNetV2 Inference          |
                               |  - Empty Detection & Alerts Engine|
                               |  - Email (SMTP) & SMS (Twilio)    |
                               |  - SQLite (users.db)              |
                               +-----------------------------------+
```

---

## 2. Authentication Architecture: Why Nurses Do Not Need Device Keys

A key security improvement in this system is the clear separation of **human user authentication** from **hardware machine authentication**:

### A. Hardware Device Authentication (Raspberry Pi)
- The Raspberry Pi is a physical appliance placed by the patient's bedside.
- It authenticates with the FastAPI backend using a **Device Key** via the `X-Device-Key` HTTP header.
- The device key is provisioned by the Administrator and stored securely on the Pi (via environment variable `IV_DEVICE_KEY` or CLI argument).
- The key's SHA-256 hash is verified against `patients.device_key_hash`.

### B. Nurse Session Authentication (Laptop Webcam)
- A nurse is an authenticated human user who logs in with her own email and password.
- Upon login, the nurse receives a secure JSON Web Token (**JWT**).
- When the nurse opens the laptop webcam console for her assigned patient, her active session token is passed automatically to the webcam console (`?patientId=...&token=...`).
- When sending readings, the console sends `Authorization: Bearer <token>`.
- The backend verifies that:
  1. The token is valid and not expired.
  2. The authenticated user is assigned to that specific patient (or is an admin).
- **The nurse never needs to know, copy, paste, or see a device key.**

---

## 3. Automatic Alerts & Persistent 15-Minute Reminder Engine

The system includes an automated detection and notification engine to immediately alert the assigned nurse when a patient's IV container is empty, and automatically send reminders until the container is replaced.

### A. Confirmation Rule
- MobileNetV2 classifies IV levels into: `100%`, `80%`, `50%`, and `Empty` (0%).
- **Confirmation Threshold**: Requires **2 consecutive empty readings** (configurable via `IV_EMPTY_CONFIRMATION_COUNT`) before raising an official alert.

### B. 15-Minute Reminder Schedule
1. **Immediate Initial Alert**: When an IV fluid container is confirmed empty, an alert email is sent immediately to the nurse assigned to that patient.
2. **15-Minute Reminders**: While the container remains empty, subsequent reminder emails are sent every 15 minutes.
3. **Replenishment Reset**: When the saline bottle is replaced and the fluid is no longer empty (`50%`, `80%`, or `100%`), reminders stop immediately.
4. **Subsequent Empty Episodes**: If the container becomes empty again later, a fresh alert and reminder cycle starts automatically.
5. **Independent Patient Timing**: Each patient maintains their own independent reminder timer.
6. **SQLite Persistence**: Timestamps (`last_email_sent_at`) and reminder counts are stored in the SQLite database, ensuring backend restarts do not reset schedules or trigger duplicate emails.
7. **Safe Failure Handling**: Temporary SMTP failures are retried without spamming nurses or disrupting live in-app alerts.

### C. Simplified, Professional Email Format
All email notifications use a clean, concise layout that nurses can read instantly on mobile phones:

```text
Subject: URGENT: IV Fluid Level Empty

Dear Nurse,

The IV fluid level is empty. Immediate attention is required.

Patient Details:
- Patient Name: [Actual Patient Name]
- Patient ID: [Actual Patient ID]
- Bed Number: [Actual Bed Number]
- Fluid Level: 0% (Empty)
- Status: CRITICAL

Please check and replace the IV fluid container immediately.

Automated IV Fluid Monitoring System
```

*(No credentials, test wording, API responses, or error traces are ever included in the email.)*

---

## 4. Configuring Gmail SMTP via backend/.env

Automated email alerts are dispatched directly to the assigned nurse. All SMTP credentials are stored safely in:
```text
C:\Users\Anjana\IV-Drip-Detection-main - Copy\Dashboard\iv_drip_dashboard\backend\.env
```
*(A template is also available at `Dashboard\iv_drip_dashboard\backend\.env.example`)*

### A. How to Create a Google App Password (Step-by-Step)
For security, Google requires a **16-character App Password**:
1. Log into your dedicated project sender Gmail account.
2. Go to **Google Account Settings** (`https://myaccount.google.com/`).
3. Click **Security** on the left menu.
4. Under "How you sign in to Google", ensure **2-Step Verification** is **ON**.
5. Go to [App passwords](https://myaccount.google.com/apppasswords).
6. Enter an app name (e.g., `IV Fluid Monitoring`) and click **Create**.
7. Google will display a 16-character password (e.g., `abcd efgh ijkl mnop`).

### B. Configure `backend/.env`
Open `Dashboard\iv_drip_dashboard\backend\.env` and set:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=iv.project.alerts@gmail.com
SMTP_PASSWORD=abcdefghijklmnop
ALERT_EMAIL_FROM=iv.project.alerts@gmail.com
ALERT_EMAIL_TO=your.personal.email@gmail.com
```

- `ALERT_EMAIL_TO`: Your personal email where you can also receive copy of alerts for verification.

---

## 5. Dedicated Notifications Tab in Nurse Dashboard

The Nurse Dashboard includes a persistent, dedicated **Notifications Tab** located on the bottom navigation bar:

### Features:
- **Unread Badge Counter**: Displays live unread alert counts directly on the navigation icon.
- **Complete Persistent History**: All notifications remain stored in SQLite across app restarts (supports 100+ notifications).
- **Filter Chips**:
  - `All`: Complete history of detections.
  - `Unread / New`: Actionable alerts requiring nurse attendance.
  - `Acknowledged`: Alerts currently attended by nursing staff.
  - `Resolved`: Containers replaced and marked closed.
- **Notification Details Displayed**:
  - Patient Name & ID, Bed Number
  - Severity: `CRITICAL`
  - Alert Message: Saline level 0% (Empty)
  - Local Timestamp
  - User-Friendly Email Status (e.g. `Alert Email: Dispatched to Nurse` or `Alert Email: Delivery Pending`)
  - Direct Action Buttons: **Acknowledge** and **Resolve**
- **Clean Interface**: All debugging text, raw Python exception traces, and unnecessary configuration details have been removed for a clean, clinical experience.

---

## 6. Key Paths and Project Locations

All operations on your Windows laptop take place in:
```text
C:\Users\Anjana\IV-Drip-Detection-main - Copy
```

Key directories:
- **Python Virtual Environment**: `.venv\`
- **FastAPI Backend**: `Dashboard\iv_drip_dashboard\backend\`
- **Database**: `Dashboard\iv_drip_dashboard\backend\users.db`
- **Webcam Interface**: `Dashboard\iv_drip_dashboard\web\camera_patient_trial.html`
- **Flutter Application**: `Dashboard\iv_drip_dashboard\`
- **Unified Pi Service**: `pi_camera_service.py`
- **Automated Test Suite**: `tests\test_camera_integration.py`

---

## 7. Step-by-Step Startup Sequence (Windows Laptop)

### Step 1: Start the FastAPI Backend

Open a **Windows PowerShell** terminal:

```powershell
# 1. Navigate to the backend folder
cd "C:\Users\Anjana\IV-Drip-Detection-main - Copy\Dashboard\iv_drip_dashboard\backend"

# 2. Start Uvicorn using the project virtual environment
& "..\..\..\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

### Step 2: Start the Flutter Dashboard

Open a **second Windows PowerShell** terminal:

```powershell
# 1. Navigate to the Flutter project folder
cd "C:\Users\Anjana\IV-Drip-Detection-main - Copy\Dashboard\iv_drip_dashboard"

# 2. Add Flutter to your session PATH
$env:Path = "C:\Users\Anjana\develop\flutter\bin;" + $env:Path

# 3. Launch the dashboard in Google Chrome
flutter run -d chrome
```

Chrome will open automatically displaying the login page.

---

## 8. How to Monitor Patients & Test Alerts

### Step 1: Log in as Nurse Nada
- **Email**: `nada@mailinator.com`
- **Password**: Your existing password
- Assigned Patient: **IVP002 (Anjana)**, Bed `B-02`.

### Step 2: Use the Laptop Webcam (Backup Source)
1. On Patient **IVP002**'s card, click **Live Camera (Demo Webcam)**.
2. Select **Laptop Webcam (Demonstration Mode)**.
3. Click **Open Demonstration Console**.
4. The webcam console opens with your verified nurse session.
5. Click **Start monitoring** and allow browser camera permissions.
6. The webcam immediately starts sending predictions every 5 seconds without asking for any device key!

### Step 3: Test the Empty-Fluid Alert System
You have two easy ways to test alerts:
1. **Via Dashboard Button**:
   - On Patient IVP002's card, click the yellow alert icon (`+!`).
   - A confirmed test alert will fire immediately!
2. **Via Webcam**:
   - Point the webcam at an empty bottle or cover the bottle neck so MobileNetV2 predicts `Empty`.
   - On the 2nd consecutive empty prediction, the alert triggers!
3. **In the Nurse Dashboard**:
   - The urgent red banner: `URGENT ACTION REQUIRED — IV FLUID EMPTY` appears immediately.
   - Click **Acknowledge Alert** to confirm you are attending to the patient.
   - Once the saline bottle is replaced, click **Resolve** to close the alert.

---

## 9. Clinical Safety Notice

> [!IMPORTANT]
> **Safety Disclaimer**:
> This IV Fluid Monitoring System is an auxiliary clinical decision-support prototype. Automated empty-fluid alerts are designed to support timely nursing attention and reduce response times. They do **not** guarantee container status and do **not** replace mandatory regular direct observation of infusion sets and patients by qualified healthcare professionals.

---

## 10. Automated Test Suites

You can run the comprehensive verification test suites at any time using the virtual environment:

### 1. Camera & Core Architecture Verification (All 10 Stages):
```powershell
cd "C:\Users\Anjana\IV-Drip-Detection-main - Copy"
.\.venv\Scripts\python.exe tests\test_camera_integration.py
```
**Expected Result:**
```text
ALL 10 AUTOMATED VERIFICATION STAGES PASSED FLAWLESSLY!
```

### 2. Email Reminders (15-Min) & Simplified Body Verification (All 9 Stages):
```powershell
cd "C:\Users\Anjana\IV-Drip-Detection-main - Copy"
.\.venv\Scripts\python.exe tests\test_email_reminders_and_simplified_body.py
```
**Expected Result:**
```text
ALL 9 VERIFICATION STAGES PASSED SUCCESSFULLY!
```

### 3. High-Volume Notifications & Persistence Verification (All 5 Stages):
```powershell
cd "C:\Users\Anjana\IV-Drip-Detection-main - Copy"
.\.venv\Scripts\python.exe tests\test_email_and_notifications_tab.py
```
**Expected Result:**
```text
ALL GMAIL & NOTIFICATION SUITE TESTS PASSED SUCCESSFULLY!
```


---

## 10. Model Architecture & Fluid Level Transition Analysis

### Discrete Categorical Classes
Inspection of `models/mobilenetv2_labels.json` and the trained weights confirms the network classifies frames into exactly four discrete categories:
1. `sal_data_100` (100% full)
2. `sal_data_80` (80% full)
3. `sal_data_50` (50% full)
4. `sal_data_empty` (Empty bottle / 0%)

### Underlying Dataset Ground Truth
The model was trained on Mendeley Saline Bottle Datasets 1 & 2. The physical images collected and labeled in these datasets consist solely of bottles at 100%, 80%, 50%, and Empty. No intermediate images (such as 25% or 10%) exist in the source dataset.

### The "50% to Empty" Transition
Because the classification head utilizes a 4-class Softmax layer (`Dense(4, activation="softmax")`), the network assigns incoming visual evidence to the highest-probability discrete class. As the fluid drops below 50%, predictions continue to yield `sal_data_50` until the visual features cross the learned boundary into `sal_data_empty`. 

Artificially interpolating intermediate numbers (like 25%) from softmax outputs without ground truth would be scientifically invalid and misleading. 

### Adaptive Timing & Clinical Safeguards
To handle this transition reliably and safely without overloading the Raspberry Pi:
1. **Adaptive Inference Rate**:
   - `> 50% & Stable`: Captures every **30 seconds** (saving CPU/power).
   - `<= 50%`: Accelerates to every **15 seconds** as the fluid depletes.
   - `Rapid Drop (>= 30%)`: Instantly drops to **7 seconds** to capture transient fluctuations.
   - `Suspected Empty`: When the first Empty reading is observed, a fast confirmation capture is scheduled in **4 seconds**.
2. **False Empty Alert Prevention**:
   - The backend requires **2 consecutive confirmed Empty readings** before triggering emergency notifications.
   - Low-confidence predictions (`< 0.60`) are automatically ignored for alert dispatch.
   - An independent persistent 15-minute reminder engine notifies the assigned nurse while the empty condition persists.

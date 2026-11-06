"""
alerts.py - Automatic Empty-Fluid Detection, Gmail SMTP Alerts & 15-Minute Reminder Engine

Handles:
1. Database table 'alerts' creation and lifecycle management (supports 100+ notifications).
2. Configurable confirmation threshold for 'Empty' IV fluid detection (2 consecutive readings).
3. Persistent 15-minute reminder schedule per patient stored in SQLite (alerts.last_email_sent_at).
4. Immediate alert email on confirmed empty, followed by reminders every 15 minutes while empty.
5. Reminders stop when fluid is replenished or condition resolved; subsequent episodes start fresh.
6. Simplified, professional email body without technical logs, error traces, or credentials.
7. Resilient delivery with automatic retry on temporary SMTP failure without duplicate alerts.
8. Role-based API routes for listing, acknowledging, and resolving alerts.
"""

from datetime import datetime, timezone
from pathlib import Path
import email.message
import json
import logging
import os
import smtplib
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth import get_current_user

logger = logging.getLogger("iv_alerts")

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "users.db"

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


def load_env_file():
    """
    Loads environment variables from backend/.env and project root .env
    without requiring third-party libraries.
    """
    backend_env = BASE_DIR / ".env"
    root_env = BASE_DIR.parents[2] / ".env"

    for env_file in (backend_env, root_env):
        if env_file.is_file():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception as e:
                logger.warning(f"Failed to read env file {env_file}: {e}")


# Load environment on module import
load_env_file()

# Configuration via environment variables
EMPTY_CONFIRMATION_THRESHOLD = int(os.getenv("IV_EMPTY_CONFIRMATION_COUNT", "2"))

# In-memory tracking of consecutive empty readings per patient for initial confirmation
_consecutive_empty_counts = {}

# In-memory tracking to prevent duplicate 50% in-app notifications
_patient_50_triggered = {}


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_alerts_database():
    """Initializes the alerts table, reminder columns, and indexes in SQLite."""
    load_env_file()
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                nurse_id INTEGER,
                patient_name TEXT,
                bed_number TEXT,
                alert_type TEXT NOT NULL DEFAULT 'IV_EMPTY',
                severity TEXT NOT NULL DEFAULT 'CRITICAL',
                fluid_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('new', 'acknowledged', 'resolved')),
                email_status TEXT DEFAULT 'not_configured',
                sms_status TEXT DEFAULT 'not_configured',
                created_at TEXT NOT NULL,
                last_email_sent_at TEXT,
                email_count INTEGER DEFAULT 1,
                acknowledged_at TEXT,
                acknowledged_by INTEGER,
                resolved_at TEXT,
                resolved_by INTEGER,
                message TEXT,
                FOREIGN KEY (nurse_id) REFERENCES users(id)
            )
        """)

        cols = [r["name"] for r in connection.execute("PRAGMA table_info(alerts)").fetchall()]
        if "severity" not in cols:
            connection.execute("ALTER TABLE alerts ADD COLUMN severity TEXT DEFAULT 'CRITICAL'")
        if "last_email_sent_at" not in cols:
            connection.execute("ALTER TABLE alerts ADD COLUMN last_email_sent_at TEXT")
        if "email_count" not in cols:
            connection.execute("ALTER TABLE alerts ADD COLUMN email_count INTEGER DEFAULT 1")
        if "message" not in cols:
            connection.execute("ALTER TABLE alerts ADD COLUMN message TEXT")

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_nurse_status
            ON alerts (nurse_id, status)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_patient_status
            ON alerts (patient_id, status)
        """)


def reset_50_notification_trigger(patient_id: str):
    """Resets the 50% in-app notification trigger upon fluid replenishment to 80% or 100%."""
    global _patient_50_triggered
    _patient_50_triggered[patient_id] = False


def is_50_trigger_armed(patient_id: str, connection: sqlite3.Connection) -> bool:
    """
    Checks if a 50% in-app notification should fire for patient_id.
    Prevents duplicate notifications while fluid remains at 50%.
    Allows a new notification once fluid is replenished to 80% or 100%.
    Resilient across backend restarts by checking SQLite history.
    """
    global _patient_50_triggered

    if _patient_50_triggered.get(patient_id) is True:
        return False

    # Check database in case server restarted while patient was already at 50%
    latest_50_alert = connection.execute("""
        SELECT created_at FROM alerts
        WHERE patient_id = ? AND (alert_type = 'LOW_FLUID' OR fluid_level = '50%')
        ORDER BY id DESC LIMIT 1
    """, (patient_id,)).fetchone()

    if not latest_50_alert:
        _patient_50_triggered[patient_id] = False
        return True

    # Check if any reading of 80% or 100% occurred AFTER this 50% alert
    alert_created_at = latest_50_alert["created_at"]
    replenished = connection.execute("""
        SELECT id FROM iv_readings
        WHERE patient_id = ?
          AND level_label IN ('80%', '100%')
          AND captured_at > ?
        LIMIT 1
    """, (patient_id, alert_created_at)).fetchone()

    if replenished:
        _patient_50_triggered[patient_id] = False
        return True
    else:
        _patient_50_triggered[patient_id] = True
        return False


def format_email_body(patient_name: str, patient_id: str, bed_number: str) -> str:
    """
    Constructs the simplified, professional email alert body exactly as required.
    Does NOT include credentials, passwords, debug info, or technical logs.
    """
    p_name = str(patient_name).strip() if patient_name else "N/A"
    p_id = str(patient_id).strip() if patient_id else "N/A"
    b_num = str(bed_number).strip() if bed_number else "N/A"

    return (
        "Dear Nurse,\n\n"
        "The IV fluid level is empty. Immediate attention is required.\n\n"
        "Patient Details:\n"
        f"- Patient Name: {p_name}\n"
        f"- Patient ID: {p_id}\n"
        f"- Bed Number: {b_num}\n"
        "- Fluid Level: 0% (Empty)\n"
        "- Status: CRITICAL\n\n"
        "Please check and replace the IV fluid container immediately.\n\n"
        "Automated IV Fluid Monitoring System\n"
    )


def send_email_notification(
    nurse_email: Optional[str],
    patient_name: str,
    patient_id: str,
    bed_number: str,
) -> str:
    """
    Sends an automated, simplified email alert via Gmail SMTP (STARTTLS port 587)
    to the nurse assigned to the respective patient (and ALERT_EMAIL_TO if configured).
    Returns delivery status string: 'sent to ...', 'skipped (...)', or 'failed: ...'.
    """
    load_env_file()

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("ALERT_EMAIL_FROM") or smtp_user or "alerts@ivdrip.local"

    if not smtp_user or not smtp_password:
        return "skipped (SMTP credentials not configured in backend/.env)"

    # Determine recipient(s): assigned nurse + configured receiver (if set in .env)
    recipients = []
    if nurse_email and "@" in nurse_email:
        recipients.append(nurse_email.strip())

    alert_to = os.getenv("ALERT_EMAIL_TO")
    if alert_to and "@" in alert_to and not alert_to.strip().upper().startswith("MY_RECEIVER"):
        cleaned_alert_to = alert_to.strip()
        if cleaned_alert_to not in recipients:
            recipients.append(cleaned_alert_to)

    if not recipients:
        return "skipped (No recipient email configured)"

    subject = "URGENT: IV Fluid Level Empty"
    body = format_email_body(patient_name, patient_id, bed_number)

    try:
        msg = email.message.EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return f"sent to {', '.join(recipients)}"
    except Exception as exc:
        logger.warning(f"Email notification delivery failed: {exc}")
        return f"failed: {str(exc)}"


def send_sms_notification(
    recipient_phone: Optional[str],
    patient_name: str,
    patient_id: str,
    bed_number: str,
) -> str:
    """
    Optional SMS alert via Twilio REST API if configured.
    Does NOT require Twilio; gracefully skips if unconfigured.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not account_sid or not auth_token or not from_number:
        return "skipped (Twilio not configured)"

    phone = recipient_phone or os.getenv("DEFAULT_ALERT_PHONE")
    if not phone:
        return "skipped (No recipient phone registered)"

    message_body = (
        f"URGENT: IV Fluid EMPTY for {patient_name} (ID: {patient_id}, Bed: {bed_number}). "
        f"Please attend immediately."
    )

    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        data = urllib.parse.urlencode({
            "From": from_number,
            "To": phone,
            "Body": message_body,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        import base64
        auth_header = base64.b64encode(f"{account_sid}:{auth_token}".encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {auth_header}")

        with urllib.request.urlopen(req, timeout=8) as response:
            resp_data = json.loads(response.read().decode("utf-8"))
            return f"sent (SID: {resp_data.get('sid', 'ok')})"
    except Exception as exc:
        return f"failed: {str(exc)}"


def evaluate_reading_for_alert(
    patient_id: str,
    level_label: str,
    confidence: float,
    captured_at: str,
) -> Optional[dict]:
    """
    Evaluates incoming patient reading:
    1. Rejects/ignores low-confidence predictions (< 0.60) to prevent false alerts.
    2. Replenishment (80%, 100%): Resets empty count and resets 50% notification trigger.
    3. Low Fluid Level (50%): Automatically creates an in-app Low Fluid Warning notification
       on first occurrence without sending emails. Suppresses duplicates while fluid remains at 50%.
    4. Confirmed Empty: Confirms after 2 consecutive readings, dispatches immediate email alert
       to assigned nurse, and maintains persistent 15-minute reminders until resolved or refilled.
    """
    global _consecutive_empty_counts, _patient_50_triggered

    reminder_interval_sec = int(os.getenv("ALERT_REMINDER_INTERVAL_SECONDS", str(int(os.getenv("ALERT_REMINDER_INTERVAL_MINUTES", "15")) * 60)))
    retry_interval_sec = int(os.getenv("ALERT_RETRY_INTERVAL_SECONDS", "60"))

    # Reject or safely ignore low-confidence predictions (< 0.60) to prevent false alerts
    min_alert_confidence = float(os.getenv("IV_MIN_ALERT_CONFIDENCE", "0.60"))
    if confidence < min_alert_confidence:
        logger.info(
            f"Alert evaluation skipped for patient {patient_id}: "
            f"confidence {confidence:.2f} is below minimum threshold {min_alert_confidence:.2f}"
        )
        return None

    norm_level = level_label.strip() if level_label else ""

    # ---------------- 1. Replenishment: 80% or 100% ----------------
    if norm_level in ("80%", "100%", "sal_data_80", "sal_data_100", "80% filled", "100% filled"):
        # Reset consecutive empty counter (stops empty reminders)
        _consecutive_empty_counts[patient_id] = 0
        # Reset 50% notification trigger so subsequent drops to 50% fire a new notification
        _patient_50_triggered[patient_id] = False
        return None

    # ---------------- 2. Low Fluid Warning: 50% In-App Notification ----------------
    if norm_level in ("50%", "sal_data_50", "50% filled"):
        # Reset empty counter since fluid is not empty
        _consecutive_empty_counts[patient_id] = 0

        with get_connection() as connection:
            if not is_50_trigger_armed(patient_id, connection):
                # Duplicate prevention: patient already in 50% state without replenishment
                return None

            patient = connection.execute("""
                SELECT p.patient_id, p.name AS patient_name, p.bed_number, p.assigned_nurse_id,
                       u.name AS nurse_name, u.email AS nurse_email, u.phone AS nurse_phone
                FROM patients p
                LEFT JOIN users u ON p.assigned_nurse_id = u.id
                WHERE p.patient_id = ?
            """, (patient_id,)).fetchone()

            if not patient:
                return None

            patient_name = patient["patient_name"]
            bed_number = patient["bed_number"]
            nurse_id = patient["assigned_nurse_id"]

            alert_msg = "IV fluid level has reached 50%. Please monitor the fluid level and prepare for replacement."

            # Insert in-app notification into alerts table (NO EMAIL SENT)
            cursor = connection.execute("""
                INSERT INTO alerts (
                    patient_id, nurse_id, patient_name, bed_number,
                    alert_type, severity, fluid_level, confidence, status,
                    email_status, sms_status, created_at, last_email_sent_at, email_count, message
                )
                VALUES (?, ?, ?, ?, 'LOW_FLUID', 'LOW WARNING', '50%', ?, 'new', 'in_app_only', 'not_applicable', ?, NULL, 0, ?)
            """, (
                patient_id,
                nurse_id,
                patient_name,
                bed_number,
                confidence,
                captured_at,
                alert_msg,
            ))
            alert_id = cursor.lastrowid
            _patient_50_triggered[patient_id] = True

            logger.info(
                f"Low fluid 50% in-app notification created for Patient {patient_id} ({patient_name}, Bed {bed_number}): Alert #{alert_id}"
            )

            return {
                "alert_id": alert_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "bed_number": bed_number,
                "severity": "LOW WARNING",
                "fluid_level": "50%",
                "alert_type": "LOW_FLUID",
                "status": "new",
                "message": alert_msg,
                "email_status": "in_app_only",
                "sms_status": "not_applicable",
            }

    # ---------------- 3. Empty Fluid: 2-Consecutive Confirmation & Email Alerts ----------------
    if norm_level not in ("Empty", "sal_data_empty", "0%"):
        return None

    # Increment consecutive empty count for this patient
    count = _consecutive_empty_counts.get(patient_id, 0) + 1
    _consecutive_empty_counts[patient_id] = count

    # Require confirmation threshold before firing alert (reduces false alarms)
    if count < EMPTY_CONFIRMATION_THRESHOLD:
        return None

    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()

    with get_connection() as connection:
        # Check if an alert is already active (new or acknowledged) for this patient
        existing_active = connection.execute("""
            SELECT id, patient_id, nurse_id, patient_name, bed_number,
                   status, email_status, sms_status, created_at,
                   last_email_sent_at, email_count
            FROM alerts
            WHERE patient_id = ? AND status IN ('new', 'acknowledged') AND alert_type = 'IV_EMPTY'
            ORDER BY id DESC LIMIT 1
        """, (patient_id,)).fetchone()

        # Fetch patient & assigned nurse information from database
        patient = connection.execute("""
            SELECT p.patient_id, p.name AS patient_name, p.bed_number, p.assigned_nurse_id,
                   u.name AS nurse_name, u.email AS nurse_email, u.phone AS nurse_phone
            FROM patients p
            LEFT JOIN users u ON p.assigned_nurse_id = u.id
            WHERE p.patient_id = ?
        """, (patient_id,)).fetchone()

        if not patient:
            return None

        patient_name = patient["patient_name"]
        bed_number = patient["bed_number"]
        nurse_id = patient["assigned_nurse_id"]
        nurse_name = patient["nurse_name"] or "Staff Nurse"
        nurse_email = patient["nurse_email"]
        nurse_phone = patient["nurse_phone"]

        if existing_active:
            # Alert is already active in dashboard. Check 15-minute reminder / retry schedule
            last_sent_str = existing_active["last_email_sent_at"] or existing_active["created_at"]
            try:
                last_sent_dt = datetime.fromisoformat(last_sent_str.replace("Z", "+00:00"))
                if last_sent_dt.tzinfo is None:
                    last_sent_dt = last_sent_dt.replace(tzinfo=timezone.utc)
            except Exception:
                last_sent_dt = now_dt

            elapsed_seconds = (now_dt - last_sent_dt).total_seconds()
            prev_status = str(existing_active["email_status"] or "")

            is_due = False
            is_retry = False

            if prev_status.startswith("failed"):
                # Prior attempt failed; retry after retry interval without duplicate alerts
                if elapsed_seconds >= retry_interval_sec:
                    is_due = True
                    is_retry = True
            else:
                # Prior attempt was sent or skipped; send reminder after 15 minutes
                if elapsed_seconds >= reminder_interval_sec:
                    is_due = True

            if is_due:
                new_email_status = send_email_notification(
                    nurse_email=nurse_email,
                    patient_name=patient_name,
                    patient_id=patient_id,
                    bed_number=bed_number,
                )
                curr_count = int(existing_active["email_count"] or 1)
                new_count = curr_count + (1 if not is_retry or new_email_status.startswith("sent") else 0)

                connection.execute("""
                    UPDATE alerts
                    SET last_email_sent_at = ?, email_count = ?, email_status = ?
                    WHERE id = ?
                """, (now_iso, new_count, new_email_status, existing_active["id"]))

                logger.info(
                    f"Empty alert reminder for Patient {patient_id} ({patient_name}): "
                    f"dispatch #{new_count}, status: {new_email_status}"
                )

            # Avoid generating duplicate in-app alert cards
            return None

        # ---------------- Initial Confirmed Alert Dispatch ----------------
        email_status = send_email_notification(
            nurse_email=nurse_email,
            patient_name=patient_name,
            patient_id=patient_id,
            bed_number=bed_number,
        )

        sms_status = send_sms_notification(
            recipient_phone=nurse_phone,
            patient_name=patient_name,
            patient_id=patient_id,
            bed_number=bed_number,
        )

        empty_msg = "The IV fluid level is empty. Immediate container replacement required."
        cursor = connection.execute("""
            INSERT INTO alerts (
                patient_id, nurse_id, patient_name, bed_number,
                alert_type, severity, fluid_level, confidence, status,
                email_status, sms_status, created_at, last_email_sent_at, email_count, message
            )
            VALUES (?, ?, ?, ?, 'IV_EMPTY', 'CRITICAL', 'Empty', ?, 'new', ?, ?, ?, ?, 1, ?)
        """, (
            patient_id,
            nurse_id,
            patient_name,
            bed_number,
            confidence,
            email_status,
            sms_status,
            captured_at,
            now_iso,
            empty_msg,
        ))
        alert_id = cursor.lastrowid

        return {
            "alert_id": alert_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "bed_number": bed_number,
            "severity": "CRITICAL",
            "fluid_level": "Empty",
            "alert_type": "IV_EMPTY",
            "status": "new",
            "message": empty_msg,
            "email_status": email_status,
            "sms_status": sms_status,
        }


# ---------------- API ROUTER ENDPOINTS ----------------

@router.get("")
def list_alerts(
    status: Optional[str] = Query(None, description="Filter by status: all, new, unread, acknowledged, resolved, or active"),
    limit: int = Query(500, description="Max notifications to retrieve (supports 100+)"),
    current_user: dict = Depends(get_current_user),
):
    """
    Lists alerts with role-based filtering:
    - Admin: can view alerts across all patients.
    - Nurse: can only view alerts for patients assigned to her.
    Supports filtering and retrieving complete history (500+ items).
    """
    if hasattr(status, "default"):
        status = status.default
    if hasattr(limit, "default"):
        limit = limit.default
    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = 500

    with get_connection() as connection:
        query = "SELECT * FROM alerts"
        conditions = []
        params = []

        if current_user["role"] == "nurse":
            conditions.append("nurse_id = ?")
            params.append(current_user["id"])

        if status and status != "all":
            if status in ("unread", "new"):
                conditions.append("status = 'new'")
            elif status == "active":
                conditions.append("status IN ('new', 'acknowledged')")
            elif status in ("acknowledged", "resolved"):
                conditions.append("status = ?")
                params.append(status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = connection.execute(query, params).fetchall()

    return [dict(r) for r in rows]


@router.put("/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Marks an alert as acknowledged by the assigned nurse or admin."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        alert = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        if current_user["role"] == "nurse" and alert["nurse_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="This alert belongs to another nurse's patient")

        connection.execute("""
            UPDATE alerts
            SET status = 'acknowledged', acknowledged_at = ?, acknowledged_by = ?
            WHERE id = ?
        """, (now, current_user["id"], alert_id))

    return {"message": "Alert acknowledged successfully", "alert_id": alert_id, "status": "acknowledged"}


@router.put("/{alert_id}/resolve")
def resolve_alert(
    alert_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Marks an alert as resolved after replacing the IV fluid."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        alert = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        if current_user["role"] == "nurse" and alert["nurse_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="This alert belongs to another nurse's patient")

        connection.execute("""
            UPDATE alerts
            SET status = 'resolved', resolved_at = ?, resolved_by = ?
            WHERE id = ?
        """, (now, current_user["id"], alert_id))

        # Reset in-memory empty count for this patient
        _consecutive_empty_counts[alert["patient_id"]] = 0

    return {"message": "Alert resolved successfully", "alert_id": alert_id, "status": "resolved"}


@router.post("/send-test-email")
def send_test_email(current_user: dict = Depends(get_current_user)):
    """
    Backend test email endpoint.
    Maintained for automated verification and health checks without leaking secrets.
    """
    load_env_file()

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("ALERT_EMAIL_FROM") or smtp_user
    recipient = os.getenv("ALERT_EMAIL_TO") or current_user.get("email")

    if not smtp_user or not smtp_password or not recipient:
        return {
            "success": False,
            "message": "SMTP credentials not configured in backend/.env",
            "details": {
                "host": smtp_host,
                "port": smtp_port,
                "has_sender_user": bool(smtp_user),
                "has_password": bool(smtp_password),
                "has_recipient": bool(recipient),
            }
        }

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = "URGENT: IV Fluid Level Empty"
    body = format_email_body(
        patient_name="System Verification Patient",
        patient_id="IVP-TEST",
        bed_number="Bed-Test",
    )

    try:
        msg = email.message.EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = recipient
        msg.set_content(body)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return {
            "success": True,
            "message": f"Test email sent successfully to {recipient}!",
            "recipient": recipient,
            "timestamp": now_str
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"SMTP delivery failed: {str(exc)}",
        }


class TestAlertRequest(BaseModel):
    patient_id: str
    confidence: Optional[float] = 0.95
    level_label: Optional[str] = "Empty"


@router.post("/test")
def trigger_test_alert(
    data: TestAlertRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Triggers an initial alert for a patient to verify in-dashboard,
    email, and notification delivery without waiting for camera prediction.
    Supports level_label: 'Empty' (critical with email) or '50%' (in-app low warning).
    """
    now = datetime.now(timezone.utc).isoformat()
    norm_lvl = (data.level_label or "Empty").strip()

    if norm_lvl in ("50%", "sal_data_50"):
        _patient_50_triggered[data.patient_id] = False
        result = evaluate_reading_for_alert(
            patient_id=data.patient_id,
            level_label="50%",
            confidence=data.confidence,
            captured_at=now,
        )
    else:
        # Force confirmation threshold for Empty test
        _consecutive_empty_counts[data.patient_id] = EMPTY_CONFIRMATION_THRESHOLD
        result = evaluate_reading_for_alert(
            patient_id=data.patient_id,
            level_label="Empty",
            confidence=data.confidence,
            captured_at=now,
        )

    if not result:
        return {
            "message": "An active alert already exists for this patient, duplicate suppressed, or patient ID is invalid.",
            "patient_id": data.patient_id,
        }

    return {
        "message": "Test alert generated successfully",
        "alert": result,
    }

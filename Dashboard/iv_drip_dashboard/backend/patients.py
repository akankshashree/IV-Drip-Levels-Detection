import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import secrets
import hmac
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel, Field

from auth import get_current_user, require_role, verify_token_and_get_user
from alerts import evaluate_reading_for_alert

router = APIRouter(prefix="/api/patients", tags=["Patients"])

DATABASE_PATH = Path(__file__).resolve().parent / "users.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_patient_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                bed_number TEXT NOT NULL,
                assigned_nurse_id INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(patients)").fetchall()
        ]
        if "device_key_hash" not in columns:
            connection.execute(
                "ALTER TABLE patients ADD COLUMN device_key_hash TEXT"
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS iv_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                level_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                captured_at TEXT NOT NULL
            )
        """)
        reading_cols = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(iv_readings)").fetchall()
        ]
        if "source" not in reading_cols:
            connection.execute(
                "ALTER TABLE iv_readings ADD COLUMN source TEXT DEFAULT 'device'"
            )
        if "capture_interval" not in reading_cols:
            connection.execute(
                "ALTER TABLE iv_readings ADD COLUMN capture_interval INTEGER DEFAULT NULL"
            )


class PatientCreate(BaseModel):
    patient_id: str = Field(min_length=2, max_length=30)
    name: str = Field(min_length=2, max_length=100)
    bed_number: str = Field(min_length=1, max_length=20)


@router.post("")
def create_patient(
    patient: PatientCreate,
    current_user: dict = Depends(require_role("admin"))
):
    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO patients
                (patient_id, name, bed_number, assigned_nurse_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    patient.patient_id.strip(),
                    patient.name.strip(),
                    patient.bed_number.strip(),
                    None,
                    datetime.now(timezone.utc).isoformat()
                )
            )
        return {"message": "Patient registered successfully."}
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="Patient ID already exists."
        )


@router.get("")
def list_patients(current_user: dict = Depends(get_current_user)):
    with get_connection() as connection:
        if current_user["role"] == "admin":
            rows = connection.execute(
                "SELECT * FROM patients ORDER BY id DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM patients
                WHERE assigned_nurse_id = ?
                ORDER BY id DESC
                """,
                (current_user["id"],)
            ).fetchall()

    return [dict(row) for row in rows]


class NurseAssignment(BaseModel):
    nurse_id: int


class ReadingCreate(BaseModel):
    level_label: str = Field(
        pattern="^(100%|80%|50%|Empty)$"
    )
    confidence: float = Field(ge=0, le=1)
    capture_interval: Optional[int] = Field(None, ge=1, le=300)
    captured_at: Optional[str] = None


@router.get("/nurses")
def list_nurses(
    current_user: dict = Depends(require_role("admin"))
):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, email, phone
            FROM users
            WHERE role = 'nurse'
            ORDER BY name
            """
        ).fetchall()

    return [dict(row) for row in rows]


@router.put("/{patient_id}/assign")
def assign_patient(
    patient_id: str,
    assignment: NurseAssignment,
    current_user: dict = Depends(require_role("admin"))
):
    with get_connection() as connection:
        nurse = connection.execute(
            "SELECT id FROM users WHERE id = ? AND role = 'nurse'",
            (assignment.nurse_id,)
        ).fetchone()

        if nurse is None:
            raise HTTPException(
                status_code=404,
                detail="Nurse not found."
            )

        result = connection.execute(
            """
            UPDATE patients
            SET assigned_nurse_id = ?
            WHERE patient_id = ?
            """,
            (assignment.nurse_id, patient_id)
        )

        if result.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Patient not found."
            )

    return {"message": "Patient assigned to nurse successfully."}


@router.get("/{patient_id}/monitoring")
def get_patient_monitoring(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    with get_connection() as connection:
        patient = connection.execute(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,)
        ).fetchone()

        if patient is None:
            raise HTTPException(
                status_code=404,
                detail="Patient not found."
            )

        if (
            current_user["role"] == "nurse"
            and patient["assigned_nurse_id"] != current_user["id"]
        ):
            raise HTTPException(
                status_code=403,
                detail="This patient is not assigned to you."
            )

        readings = connection.execute(
            """
            SELECT id, patient_id, level_label, confidence, captured_at, source, capture_interval
            FROM iv_readings
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (patient_id,)
        ).fetchall()

        # Check for active alert for this patient
        active_alert = connection.execute(
            """
            SELECT id, alert_type, severity, fluid_level, confidence, status, created_at, acknowledged_at, message
            FROM alerts
            WHERE patient_id = ? AND status IN ('new', 'acknowledged')
            ORDER BY id DESC LIMIT 1
            """,
            (patient_id,)
        ).fetchone()

    latest_reading = dict(readings[0]) if readings else None

    device_status = "offline"

    if latest_reading:
        captured_at = datetime.fromisoformat(
            latest_reading["captured_at"]
        )

        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)

        # Dynamic online window based on reported capture interval:
        # Prevents false offline status during long adaptive intervals (e.g., 30s stable).
        reported_interval = latest_reading.get("capture_interval")
        timeout_seconds = max(25, int(reported_interval) * 2 + 10) if reported_interval else 45

        if datetime.now(timezone.utc) - captured_at <= timedelta(seconds=timeout_seconds):
            device_status = "online"

    return {
        "patient": dict(patient),
        "device_status": device_status,
        "latest_reading": latest_reading,
        "active_alert": dict(active_alert) if active_alert else None,
        "recent_readings": [dict(reading) for reading in readings]
    }


@router.post("/{patient_id}/device-key")
def generate_device_key(
    patient_id: str,
    current_user: dict = Depends(require_role("admin"))
):
    """
    Generates a secure 32-byte URL-safe hardware device key.
    Only the Admin has access to provision or regenerate device keys.
    The key hash is securely stored in SQLite.
    """
    device_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(
        device_key.encode()
    ).hexdigest()

    with get_connection() as connection:
        result = connection.execute(
            """
            UPDATE patients
            SET device_key_hash = ?
            WHERE patient_id = ?
            """,
            (key_hash, patient_id)
        )

        if result.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Patient not found."
            )

    return {
        "message": "Device key generated successfully.",
        "patient_id": patient_id,
        "device_key": device_key
    }


@router.post("/{patient_id}/readings")
def receive_reading(
    patient_id: str,
    reading: ReadingCreate,
    x_device_key: Optional[str] = Header(None, alias="X-Device-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    token: Optional[str] = Query(None, description="Nurse JWT session token if not in headers"),
):
    """
    Accepts IV fluid monitoring readings from two authorized sources:
    1. Hardware Device (e.g. Raspberry Pi): Authenticates via X-Device-Key header.
    2. Nurse Session (e.g. Laptop Webcam): Authenticates via nurse Bearer JWT token.
       Validates that the nurse is assigned to this patient. The nurse never needs a device key!
    """
    if hasattr(x_device_key, "default"):
        x_device_key = None
    if hasattr(authorization, "default"):
        authorization = None
    if hasattr(token, "default"):
        token = None

    with get_connection() as connection:
        patient = connection.execute(
            """
            SELECT patient_id, name, bed_number, assigned_nurse_id, device_key_hash
            FROM patients
            WHERE patient_id = ?
            """,
            (patient_id,)
        ).fetchone()

        if patient is None:
            raise HTTPException(
                status_code=404,
                detail="Patient not found."
            )

        source = "device"
        authenticated = False

        # Mode 1: Hardware Device Authentication (Raspberry Pi)
        if x_device_key:
            if not patient["device_key_hash"]:
                raise HTTPException(
                    status_code=403,
                    detail="No device key registered for this patient."
                )

            received_hash = hashlib.sha256(
                x_device_key.encode()
            ).hexdigest()

            if not hmac.compare_digest(received_hash, patient["device_key_hash"]):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid device key."
                )

            authenticated = True
            source = "device"

        # Mode 2: Nurse Session Authentication (Laptop Webcam)
        elif authorization or token:
            raw_token = token
            if authorization:
                parts = authorization.split()
                if len(parts) == 2 and parts[0].lower() == "bearer":
                    raw_token = parts[1]
                else:
                    raw_token = authorization

            if not raw_token:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authorization token."
                )

            # Decodes and validates JWT token
            user = verify_token_and_get_user(raw_token)

            # Verify that the nurse is authorized for this patient
            if user["role"] == "nurse":
                if patient["assigned_nurse_id"] != user["id"]:
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: This patient is not assigned to you."
                    )
            elif user["role"] != "admin":
                raise HTTPException(
                    status_code=403,
                    detail="Unauthorized role for submitting patient readings."
                )

            authenticated = True
            source = "webcam"

        if not authenticated:
            raise HTTPException(
                status_code=401,
                detail="Authentication required: Provide X-Device-Key header (for hardware devices) or Bearer token (for nurse sessions)."
            )

        captured_at = reading.captured_at or datetime.now(timezone.utc).isoformat()

        cursor = connection.execute(
            """
            INSERT INTO iv_readings
            (patient_id, level_label, confidence, captured_at, source, capture_interval)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                reading.level_label,
                reading.confidence,
                captured_at,
                source,
                reading.capture_interval
            )
        )

        reading_id = cursor.lastrowid

    # Automatically evaluate for confirmed empty-fluid alert
    alert_info = evaluate_reading_for_alert(
        patient_id=patient_id,
        level_label=reading.level_label,
        confidence=reading.confidence,
        captured_at=captured_at
    )

    return {
        "message": "IV reading saved successfully.",
        "reading_id": reading_id,
        "patient_id": patient_id,
        "level_label": reading.level_label,
        "confidence": reading.confidence,
        "capture_interval": reading.capture_interval,
        "captured_at": captured_at,
        "source": source,
        "alert": alert_info
    }

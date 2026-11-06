"""
tests/test_50_percent_in_app_notification.py - Comprehensive Test Suite for 50% In-App Notification

Tests:
1. Single in-app notification creation on valid 50% prediction.
2. Verification of all notification fields:
   - Patient name
   - Patient ID
   - Bed number
   - Fluid level: 50%
   - Alert message: "IV fluid level has reached 50%. Please monitor the fluid level and prepare for replacement."
   - Date and time (created_at)
   - Severity: "LOW WARNING" / alert_type: "LOW_FLUID"
3. Strict verification: NO EMAIL IS SENT at 50% (SMTP not triggered, email_status='in_app_only', email_count=0).
4. Duplicate prevention: repeated 50% predictions do not generate duplicate notifications.
5. Notification status lifecycle:
   - Starts with status='new' (increments unread notification count).
   - Nurse acknowledges notification (status='acknowledged').
   - Nurse resolves notification (status='resolved').
6. Replenishment reset:
   - Replenishing to 80% or 100% resets the 50% notification trigger.
   - A subsequent drop back to 50% fires a new in-app notification.
7. Confirmed Empty alert preservation:
   - Two consecutive Empty readings still confirm Empty condition.
   - Dispatches email alert to nurse with critical severity and initiates 15-minute reminders.
8. Backend restart resilience:
   - Server restart while fluid is at 50% does not fire duplicate notification.
9. Independent patient isolation:
   - 50% alerts for different patients operate independently.
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parent.parent / "Dashboard" / "iv_drip_dashboard" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import alerts
import patients
from auth import get_connection, create_access_token


class Test50PercentInAppNotification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alerts.initialize_alerts_database()
        patients.initialize_patient_database()

    def setUp(self):
        alerts.load_env_file()
        # Clean test alerts for test patients
        with get_connection() as con:
            con.execute("DELETE FROM alerts WHERE patient_id IN ('IVP001', 'IVP002', 'IVP_TEST50')")
            con.execute("DELETE FROM iv_readings WHERE patient_id IN ('IVP001', 'IVP002', 'IVP_TEST50')")

        alerts._consecutive_empty_counts.clear()
        alerts._patient_50_triggered.clear()

    def test_01_single_notification_generated_on_first_50_percent(self):
        """Test 1: Valid 50% prediction generates exactly one in-app notification with all required fields."""
        now = datetime.now(timezone.utc).isoformat()

        # Patch send_email_notification to guarantee it is NEVER called for 50%
        with patch("alerts.send_email_notification") as mock_email:
            result = alerts.evaluate_reading_for_alert(
                patient_id="IVP002",
                level_label="50%",
                confidence=0.92,
                captured_at=now,
            )

            # 1. Alert must be created
            self.assertIsNotNone(result, "50% reading must generate an alert dictionary")
            self.assertEqual(result["patient_id"], "IVP002")
            self.assertEqual(result["patient_name"], "Anjana")
            self.assertEqual(result["bed_number"], "B-02")
            self.assertEqual(result["fluid_level"], "50%")
            self.assertEqual(result["severity"], "LOW WARNING")
            self.assertEqual(result["alert_type"], "LOW_FLUID")
            self.assertEqual(result["status"], "new")
            self.assertEqual(
                result["message"],
                "IV fluid level has reached 50%. Please monitor the fluid level and prepare for replacement.",
            )

            # 2. Strict verification: NO EMAIL MUST BE SENT
            mock_email.assert_not_called()
            self.assertEqual(result["email_status"], "in_app_only")

            # 3. Verify database persistence
            with get_connection() as con:
                row = con.execute("SELECT * FROM alerts WHERE id = ?", (result["alert_id"],)).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["patient_name"], "Anjana")
                self.assertEqual(row["bed_number"], "B-02")
                self.assertEqual(row["fluid_level"], "50%")
                self.assertEqual(row["severity"], "LOW WARNING")
                self.assertEqual(row["status"], "new")
                self.assertEqual(row["email_status"], "in_app_only")
                self.assertIsNone(row["last_email_sent_at"])
                self.assertEqual(row["email_count"], 0)
                self.assertEqual(
                    row["message"],
                    "IV fluid level has reached 50%. Please monitor the fluid level and prepare for replacement.",
                )

    def test_02_duplicate_suppression_on_repeated_50_percent_predictions(self):
        """Test 2: Repeated 50% predictions do not generate duplicate notifications."""
        now = datetime.now(timezone.utc).isoformat()

        with patch("alerts.send_email_notification") as mock_email:
            # First 50% reading -> Alert created
            alert1 = alerts.evaluate_reading_for_alert("IVP002", "50%", 0.95, now)
            self.assertIsNotNone(alert1)

            # Repeated 50% readings -> Must be suppressed
            alert2 = alerts.evaluate_reading_for_alert("IVP002", "50%", 0.94, now)
            self.assertIsNone(alert2, "Second consecutive 50% reading must NOT generate a duplicate alert")

            alert3 = alerts.evaluate_reading_for_alert("IVP002", "50%", 0.96, now)
            self.assertIsNone(alert3, "Third consecutive 50% reading must NOT generate a duplicate alert")

            # Check database: Exactly 1 alert must exist for IVP002
            with get_connection() as con:
                count = con.execute("SELECT COUNT(*) as cnt FROM alerts WHERE patient_id = 'IVP002'").fetchone()["cnt"]
                self.assertEqual(count, 1, "Only 1 alert should exist in SQLite despite multiple 50% predictions")

            mock_email.assert_not_called()

    def test_03_unread_count_and_nurse_acknowledgement_and_resolution(self):
        """Test 3: Unread count increases on 50% alert, nurse can acknowledge and resolve using existing controls."""
        now = datetime.now(timezone.utc).isoformat()
        nurse_user = {"id": 4, "role": "nurse", "name": "Nada", "email": "nada@mailinator.com"}

        # Initial alert count for Nurse Nada (assigned to IVP002)
        initial_alerts = alerts.list_alerts(status="all", current_user=nurse_user)
        initial_unread = len([a for a in initial_alerts if a["status"] == "new"])

        # Generate 50% alert
        created = alerts.evaluate_reading_for_alert("IVP002", "50%", 0.95, now)
        alert_id = created["alert_id"]

        # Check unread count increment
        after_create = alerts.list_alerts(status="all", current_user=nurse_user)
        after_unread = len([a for a in after_create if a["status"] == "new"])
        self.assertEqual(after_unread, initial_unread + 1, "Unread notification count must increment by 1")

        # Nurse acknowledges alert
        ack_res = alerts.acknowledge_alert(alert_id, current_user=nurse_user)
        self.assertEqual(ack_res["status"], "acknowledged")

        with get_connection() as con:
            row = con.execute("SELECT status, acknowledged_by FROM alerts WHERE id = ?", (alert_id,)).fetchone()
            self.assertEqual(row["status"], "acknowledged")
            self.assertEqual(row["acknowledged_by"], 4)

        # Nurse resolves alert
        res_res = alerts.resolve_alert(alert_id, current_user=nurse_user)
        self.assertEqual(res_res["status"], "resolved")

        with get_connection() as con:
            row = con.execute("SELECT status, resolved_by FROM alerts WHERE id = ?", (alert_id,)).fetchone()
            self.assertEqual(row["status"], "resolved")
            self.assertEqual(row["resolved_by"], 4)

    def test_04_replenishment_resets_trigger_for_subsequent_50_percent(self):
        """Test 4: Fluid replenishment to 80% or 100% resets trigger; next drop to 50% generates a new alert."""
        now = datetime.now(timezone.utc).isoformat()
        patient_id = "IVP002"

        # 1. First 50% event
        alert1 = alerts.evaluate_reading_for_alert(patient_id, "50%", 0.95, now)
        self.assertIsNotNone(alert1)
        first_alert_id = alert1["alert_id"]

        # 2. Fluid remains at 50% -> Duplicate suppressed
        self.assertIsNone(alerts.evaluate_reading_for_alert(patient_id, "50%", 0.95, now))

        # 3. Fluid is replenished to 100%
        # Simulate inserting reading in iv_readings and evaluating
        KNOWN_PI_DEVICE_KEY = "boFeqCEcQiYphs1v0Nx7BcJf1G61s9oZto7LebVQHMc"
        patients.receive_reading(
            patient_id=patient_id,
            reading=patients.ReadingCreate(level_label="100%", confidence=0.98),
            x_device_key=KNOWN_PI_DEVICE_KEY,
        )

        # 4. Fluid drops to 80%
        patients.receive_reading(
            patient_id=patient_id,
            reading=patients.ReadingCreate(level_label="80%", confidence=0.95),
            x_device_key=KNOWN_PI_DEVICE_KEY,
        )

        # 5. Fluid drops back down to 50% -> Fresh notification MUST be generated!
        res_reading = patients.receive_reading(
            patient_id=patient_id,
            reading=patients.ReadingCreate(level_label="50%", confidence=0.93),
            x_device_key=KNOWN_PI_DEVICE_KEY,
        )

        self.assertIsNotNone(res_reading["alert"], "A subsequent drop to 50% after replenishment must fire a new alert!")
        second_alert_id = res_reading["alert"]["alert_id"]
        self.assertNotEqual(first_alert_id, second_alert_id, "New alert ID must be assigned")

        # 6. Another 50% reading without replenishment -> Duplicate suppressed again
        res_dup = patients.receive_reading(
            patient_id=patient_id,
            reading=patients.ReadingCreate(level_label="50%", confidence=0.91),
            x_device_key=KNOWN_PI_DEVICE_KEY,
        )
        self.assertIsNone(res_dup["alert"], "Repeated 50% reading must be suppressed")

    def test_05_confirmed_empty_still_triggers_email_and_critical_alert(self):
        """Test 5: Confirmed empty condition still triggers email alert and does not interfere with 50% alerts."""
        now = datetime.now(timezone.utc).isoformat()
        patient_id = "IVP002"

        # 1. Drop to 50% first -> In-app alert created, no email
        with patch("alerts.send_email_notification") as mock_email:
            alert_50 = alerts.evaluate_reading_for_alert(patient_id, "50%", 0.95, now)
            self.assertIsNotNone(alert_50)
            mock_email.assert_not_called()

        # 2. Fluid depletes to Empty
        # Reading 1 (Empty): Threshold = 1, no alert yet
        self.assertIsNone(alerts.evaluate_reading_for_alert(patient_id, "Empty", 0.96, now))

        # Reading 2 (Empty): Confirmed! Must fire CRITICAL Empty alert with email
        with patch("alerts.send_email_notification", return_value="sent to nurse@hospital.com") as mock_email:
            alert_empty = alerts.evaluate_reading_for_alert(patient_id, "Empty", 0.97, now)
            self.assertIsNotNone(alert_empty)
            self.assertEqual(alert_empty["severity"], "CRITICAL")
            self.assertEqual(alert_empty["fluid_level"], "Empty")
            self.assertEqual(alert_empty["alert_type"], "IV_EMPTY")
            mock_email.assert_called_once()

    def test_06_backend_restart_resilience_for_50_percent(self):
        """Test 6: Simulating backend restart while patient is at 50% does not produce duplicate alert."""
        now = datetime.now(timezone.utc).isoformat()
        patient_id = "IVP002"
        KNOWN_PI_DEVICE_KEY = "boFeqCEcQiYphs1v0Nx7BcJf1G61s9oZto7LebVQHMc"

        # Patient receives 50% reading through normal route
        r1 = patients.receive_reading(
            patient_id=patient_id,
            reading=patients.ReadingCreate(level_label="50%", confidence=0.95),
            x_device_key=KNOWN_PI_DEVICE_KEY,
        )
        self.assertIsNotNone(r1["alert"])

        # Simulate backend crash / restart: clear in-memory caches
        alerts._patient_50_triggered.clear()
        alerts._consecutive_empty_counts.clear()

        # Another 50% reading arrives after restart without any 80%/100% reading
        r2 = patients.receive_reading(
            patient_id=patient_id,
            reading=patients.ReadingCreate(level_label="50%", confidence=0.95),
            x_device_key=KNOWN_PI_DEVICE_KEY,
        )
        self.assertIsNone(r2["alert"], "Post-restart 50% reading must NOT produce a duplicate alert")

    def test_07_independent_patient_isolation(self):
        """Test 7: 50% alerts for different patients operate independently."""
        now = datetime.now(timezone.utc).isoformat()

        # Trigger 50% alert for Patient IVP002 (Nurse Nada)
        a2 = alerts.evaluate_reading_for_alert("IVP002", "50%", 0.95, now)
        self.assertIsNotNone(a2)

        # Trigger 50% alert for Patient IVP001 (Nurse Priya)
        a1 = alerts.evaluate_reading_for_alert("IVP001", "50%", 0.95, now)
        self.assertIsNotNone(a1)

        self.assertNotEqual(a1["alert_id"], a2["alert_id"])
        self.assertEqual(a1["patient_id"], "IVP001")
        self.assertEqual(a2["patient_id"], "IVP002")


if __name__ == "__main__":
    unittest.main()

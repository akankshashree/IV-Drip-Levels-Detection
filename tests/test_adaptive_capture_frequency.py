"""
test_adaptive_capture_frequency.py - Verification suite for adaptive inference frequency,
demonstration mode intervals, dynamic online timeout calculation, and false alert prevention.
"""

import os
import sys
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Dashboard" / "iv_drip_dashboard" / "backend"))

from pi_camera_service import AdaptiveTimingManager
from auth import create_access_token
from patients import (
    initialize_patient_database,
    get_connection as get_patient_conn,
    ReadingCreate,
    receive_reading,
    get_patient_monitoring,
)
from alerts import (
    initialize_alerts_database,
    get_connection as get_alert_conn,
    evaluate_reading_for_alert,
    _consecutive_empty_counts,
)


class TestAdaptiveCaptureFrequency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_patient_database()
        initialize_alerts_database()
        cls._cleanup_test_records()

    @staticmethod
    def _cleanup_test_records():
        with get_patient_conn() as conn:
            conn.execute("DELETE FROM alerts WHERE patient_id LIKE 'TEST_ADAPTIVE_%'")
            conn.execute("DELETE FROM iv_readings WHERE patient_id LIKE 'TEST_ADAPTIVE_%'")
            conn.execute("DELETE FROM patients WHERE patient_id LIKE 'TEST_ADAPTIVE_%'")

    def setUp(self):
        _consecutive_empty_counts.clear()
        self._cleanup_test_records()

    def tearDown(self):
        self._cleanup_test_records()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_test_records()

    def test_01_adaptive_timing_manager_transitions(self):
        """Tests that AdaptiveTimingManager properly adapts intervals across fluid lifecycle."""
        mgr = AdaptiveTimingManager(
            interval_high_stable=30,
            interval_low=15,
            interval_rapid_drop=7,
            interval_confirmation=4,
            min_confidence=0.60,
        )

        # 1. Starting at 100% (stable > 50%) -> 30s
        interval, reason = mgr.calculate_next_interval("100%", 0.95)
        self.assertEqual(interval, 30)
        self.assertIn("power-saving", reason.lower())

        # 2. Continues at 100% -> 30s
        interval, _ = mgr.calculate_next_interval("100%", 0.92)
        self.assertEqual(interval, 30)

        # 3. Drops to 80% (drop of 20 < 30, still > 50%) -> 30s
        interval, _ = mgr.calculate_next_interval("80%", 0.90)
        self.assertEqual(interval, 30)

        # 4. Rapid drop to 50% (drop of 30 >= 30) -> 7s (rapid drop interval)
        interval, reason = mgr.calculate_next_interval("50%", 0.88)
        self.assertEqual(interval, 7)
        self.assertIn("rapid fluid drop", reason.lower())

        # 5. Stays at 50% (stable at 50% <= 50) -> 15s (low fluid monitoring)
        interval, reason = mgr.calculate_next_interval("50%", 0.87)
        self.assertEqual(interval, 15)
        self.assertIn("at or below 50%", reason.lower())

        # 6. First Empty detection: Suspected empty condition -> 4s (fast confirmation check)
        interval, reason = mgr.calculate_next_interval("Empty", 0.94)
        self.assertEqual(interval, 4)
        self.assertIn("suspected empty", reason.lower())

        # 7. Second consecutive Empty: Confirmed empty -> 15s (ongoing empty monitoring)
        interval, reason = mgr.calculate_next_interval("Empty", 0.96)
        self.assertEqual(interval, 15)
        self.assertIn("empty condition active", reason.lower())

        # 8. Bottle replaced (Replenished to 100%) -> smoothly returns to 30s
        interval, reason = mgr.calculate_next_interval("100%", 0.97)
        self.assertEqual(interval, 30)
        self.assertIn("power-saving", reason.lower())

    def test_02_low_confidence_safe_handling(self):
        """Tests that low-confidence predictions (< 0.60) do not alter empty state and trigger fast re-check."""
        mgr = AdaptiveTimingManager(
            interval_high_stable=30,
            interval_low=15,
            interval_rapid_drop=7,
            interval_confirmation=4,
            min_confidence=0.60,
        )

        mgr.calculate_next_interval("100%", 0.95)

        # Ambiguous prediction with low confidence 0.45 (< 0.60)
        interval, reason = mgr.calculate_next_interval("Empty", 0.45)
        self.assertEqual(interval, 7)
        self.assertIn("low confidence", reason.lower())
        # Empty count should remain 0
        self.assertEqual(mgr.consecutive_empty_count, 0)

    def test_03_backend_reading_persistence_with_interval(self):
        """Tests that backend receives and persists capture_interval in iv_readings."""
        patient_id = "TEST_ADAPTIVE_P01"
        now_iso = datetime.now(timezone.utc).isoformat()

        with get_patient_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patients (patient_id, name, bed_number, device_key_hash, created_at) "
                "VALUES (?, 'Adaptive Test', 'B101', 'fakehash', ?)",
                (patient_id, now_iso),
            )

        # Submit reading via internal receive_reading
        reading = ReadingCreate(
            level_label="80%",
            confidence=0.92,
            capture_interval=30,
        )
        admin_token = create_access_token(user_id=1, role="admin")

        resp = receive_reading(
            patient_id=patient_id,
            reading=reading,
            authorization=f"Bearer {admin_token}",
        )
        self.assertEqual(resp["capture_interval"], 30)
        self.assertEqual(resp["level_label"], "80%")

        # Check DB
        with get_patient_conn() as conn:
            row = conn.execute(
                "SELECT capture_interval, level_label FROM iv_readings WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["capture_interval"], 30)
            self.assertEqual(row["level_label"], "80%")

    def test_04_dynamic_device_online_status_for_30s_interval(self):
        """
        Tests that when a device reports capture_interval=30s,
        a reading captured 25 seconds ago is correctly recognized as ONLINE
        (whereas under the old fixed 15s window it would have falsely reported offline).
        """
        patient_id = "TEST_ADAPTIVE_P02"
        # Reading captured 25 seconds ago
        captured_time = (datetime.now(timezone.utc) - timedelta(seconds=25)).isoformat()

        with get_patient_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patients (patient_id, name, bed_number, created_at) "
                "VALUES (?, 'Online Status Test', 'B102', ?)",
                (patient_id, captured_time),
            )
            conn.execute(
                "INSERT INTO iv_readings (patient_id, level_label, confidence, captured_at, source, capture_interval) "
                "VALUES (?, '100%', 0.95, ?, 'device', 30)",
                (patient_id, captured_time),
            )

        admin_user = {"id": 1, "username": "admin", "role": "admin"}
        mon_data = get_patient_monitoring(patient_id, current_user=admin_user)

        self.assertEqual(mon_data["device_status"], "online",
                         "Pi monitoring at 30s interval should remain ONLINE 25s after last capture")

        # But if captured 80 seconds ago (exceeding max(25, 30*2+10) = 70s):
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=80)).isoformat()
        with get_patient_conn() as conn:
            conn.execute(
                "INSERT INTO iv_readings (patient_id, level_label, confidence, captured_at, source, capture_interval) "
                "VALUES (?, '100%', 0.95, ?, 'device', 30)",
                (patient_id, old_time),
            )
        mon_data_old = get_patient_monitoring(patient_id, current_user=admin_user)
        self.assertEqual(mon_data_old["device_status"], "offline")

    def test_05_alerts_reject_low_confidence_empty_readings(self):
        """Tests that evaluate_reading_for_alert ignores Empty readings with confidence < 0.60."""
        patient_id = "TEST_ADAPTIVE_P03"
        now_iso = datetime.now(timezone.utc).isoformat()

        with get_patient_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patients (patient_id, name, bed_number, created_at) "
                "VALUES (?, 'Low Conf Alert Test', 'B103', ?)",
                (patient_id, now_iso),
            )

        # 1. First low confidence reading (0.40) -> should be skipped
        alert1 = evaluate_reading_for_alert(
            patient_id=patient_id,
            level_label="Empty",
            confidence=0.40,
            captured_at=now_iso,
        )
        self.assertIsNone(alert1)
        self.assertEqual(_consecutive_empty_counts.get(patient_id, 0), 0)

        # 2. First high confidence reading (0.85) -> increments count to 1, no alert yet
        alert2 = evaluate_reading_for_alert(
            patient_id=patient_id,
            level_label="Empty",
            confidence=0.85,
            captured_at=now_iso,
        )
        self.assertIsNone(alert2)
        self.assertEqual(_consecutive_empty_counts.get(patient_id), 1)

        # 3. Second high confidence reading (0.91) -> confirmed alert fired!
        alert3 = evaluate_reading_for_alert(
            patient_id=patient_id,
            level_label="Empty",
            confidence=0.91,
            captured_at=now_iso,
        )
        self.assertIsNotNone(alert3)
        self.assertEqual(alert3["severity"], "CRITICAL")
        self.assertEqual(alert3["status"], "new")
        self.assertEqual(alert3["patient_id"], patient_id)


if __name__ == "__main__":
    unittest.main()

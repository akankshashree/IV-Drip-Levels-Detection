"""
tests/test_camera_integration.py - Full Automated Test Suite

Tests:
1. Backend startup and database initialization (users, patients, readings, alerts).
2. MobileNetV2 model loading and verification.
3. Nurse authentication and role-based patient access.
4. Laptop Webcam reading submission using Nurse JWT token (NO device key required!).
5. Authorization enforcement: Nurse cannot submit reading for another nurse's patient (HTTP 403).
6. Hardware device authentication: Raspberry Pi authenticates using X-Device-Key.
7. Automated Empty-Fluid detection confirmation rule (requires consecutive empty readings).
8. Anti-duplicate alert prevention during continuous empty state.
9. Alert lifecycle: In-dashboard listing, nurse acknowledgement, and resolution.
10. Notification resilience: Graceful fallback when SMTP/SMS are unconfigured (no crash).
11. Reset cycle: Fluid replenishment resets counter and allows future alerts.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Adjust paths
BACKEND_DIR = Path(__file__).resolve().parent.parent / "Dashboard" / "iv_drip_dashboard" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image
import numpy as np


def run_tests():
    print("=" * 70)
    print("IV FLUID MONITORING SYSTEM - COMPREHENSIVE E2E VERIFICATION")
    print("=" * 70)

    import main
    import auth
    import patients
    import alerts
    from auth import get_connection, create_access_token, verify_token_and_get_user

    # -------------------------------------------------------------
    # 1. Startup & Model Verification
    # -------------------------------------------------------------
    print("\n[Step 1] Initializing backend and MobileNetV2...")
    main.startup()
    assert main.model is not None, "Model failed to load!"
    assert len(main.labels) == 4, f"Expected 4 labels, got: {main.labels}"
    print(f" -> OK: Model verified with labels: {main.labels}")

    # -------------------------------------------------------------
    # 2. Database & User Verification
    # -------------------------------------------------------------
    print("\n[Step 2] Verifying users and assigned patients...")
    with get_connection() as con:
        admin_user = con.execute("SELECT * FROM users WHERE role = 'admin'").fetchone()
        nada_nurse = con.execute("SELECT * FROM users WHERE email = 'nada@mailinator.com'").fetchone()
        priya_nurse = con.execute("SELECT * FROM users WHERE email = 'priya.nurse@mailinator.com'").fetchone()

        assert admin_user is not None, "Admin user missing!"
        assert nada_nurse is not None, "Nurse Nada missing!"
        assert priya_nurse is not None, "Nurse Priya missing!"

        # Patient IVP002 is assigned to Nada (id 4), IVP001 is assigned to Priya (id 2)
        p_ivp002 = con.execute("SELECT * FROM patients WHERE patient_id = 'IVP002'").fetchone()
        p_ivp001 = con.execute("SELECT * FROM patients WHERE patient_id = 'IVP001'").fetchone()

        assert p_ivp002 is not None, "Patient IVP002 missing!"
        assert p_ivp002["assigned_nurse_id"] == nada_nurse["id"], "IVP002 not assigned to Nada!"
        assert p_ivp002["device_key_hash"] is not None, "IVP002 has no device key hash!"

    print(f" -> OK: Nurse Nada (ID {nada_nurse['id']}) assigned to Patient IVP002 ({p_ivp002['name']})")
    print(f" -> OK: Nurse Priya (ID {priya_nurse['id']}) assigned to Patient IVP001 ({p_ivp001['name']})")

    # Generate test tokens
    nada_token = create_access_token(nada_nurse["id"], "nurse")
    priya_token = create_access_token(priya_nurse["id"], "nurse")
    admin_token = create_access_token(admin_user["id"], "admin")

    # Verify token decoding
    verified = verify_token_and_get_user(nada_token)
    assert verified["email"] == "nada@mailinator.com"
    print(" -> OK: JWT tokens generated and verified successfully.")

    # -------------------------------------------------------------
    # 3. Nurse Webcam Reading Submission (NO DEVICE KEY!)
    # -------------------------------------------------------------
    print("\n[Step 3] Testing Nurse Webcam Submission (NO device key required)...")
    reading_payload = patients.ReadingCreate(level_label="80%", confidence=0.92)

    # Nurse Nada submits for her assigned patient IVP002 using Bearer token only
    res_webcam = patients.receive_reading(
        patient_id="IVP002",
        reading=reading_payload,
        x_device_key=None,  # No device key!
        authorization=f"Bearer {nada_token}",
    )
    assert res_webcam["patient_id"] == "IVP002"
    assert res_webcam["source"] == "webcam"
    assert res_webcam["level_label"] == "80%"
    print(f" -> OK: Nurse Nada successfully submitted webcam reading for IVP002 (source: {res_webcam['source']})")

    # -------------------------------------------------------------
    # 4. Nurse Access Control Enforcement
    # -------------------------------------------------------------
    print("\n[Step 4] Testing Access Control (Nurse cannot submit for another nurse's patient)...")
    try:
        # Nurse Priya tries to submit a reading for Patient IVP002 (assigned to Nada)
        patients.receive_reading(
            patient_id="IVP002",
            reading=reading_payload,
            x_device_key=None,
            authorization=f"Bearer {priya_token}",
        )
        assert False, "Should have raised HTTP 403 Forbidden!"
    except patients.HTTPException as e:
        assert e.status_code == 403, f"Expected 403, got {e.status_code}"
        print(f" -> OK: Cross-patient access correctly blocked with HTTP 403: {e.detail}")

    # -------------------------------------------------------------
    # 5. Hardware Device Authentication (Raspberry Pi)
    # -------------------------------------------------------------
    print("\n[Step 5] Testing Hardware Device Authentication (Raspberry Pi X-Device-Key)...")
    KNOWN_PI_DEVICE_KEY = "boFeqCEcQiYphs1v0Nx7BcJf1G61s9oZto7LebVQHMc"

    res_device = patients.receive_reading(
        patient_id="IVP002",
        reading=patients.ReadingCreate(level_label="100%", confidence=0.98),
        x_device_key=KNOWN_PI_DEVICE_KEY,  # Pi device key
        authorization=None,
    )
    assert res_device["patient_id"] == "IVP002"
    assert res_device["source"] == "device"
    print(f" -> OK: Raspberry Pi device key accepted (source: {res_device['source']})")

    # Verify invalid device key is rejected
    try:
        patients.receive_reading(
            patient_id="IVP002",
            reading=reading_payload,
            x_device_key="invalid-device-key-sample",
            authorization=None,
        )
        assert False, "Should have raised HTTP 401 Unauthorized!"
    except patients.HTTPException as e:
        assert e.status_code == 401
        print(" -> OK: Invalid device key correctly rejected with HTTP 401.")

    # -------------------------------------------------------------
    # 6. Automatic Empty-Fluid Detection & Confirmation Rule
    # -------------------------------------------------------------
    print("\n[Step 6] Testing Empty-Fluid Detection Confirmation Threshold...")
    # Clean previous alerts for clean test
    with get_connection() as con:
        con.execute("DELETE FROM alerts WHERE patient_id = 'IVP002'")
    alerts._consecutive_empty_counts["IVP002"] = 0

    empty_payload = patients.ReadingCreate(level_label="Empty", confidence=0.88)

    # 1st Empty reading: Should NOT trigger alert yet (threshold = 2)
    res_empty_1 = patients.receive_reading(
        patient_id="IVP002",
        reading=empty_payload,
        x_device_key=KNOWN_PI_DEVICE_KEY,
    )
    assert res_empty_1["alert"] is None, "Alert should NOT fire on 1st reading (needs 2 consecutive)"
    print(" -> Reading 1 (Empty): Captured. Confirmation count = 1. No alert yet (correct).")

    # 2nd Empty reading: Confirmed! Alert MUST fire!
    res_empty_2 = patients.receive_reading(
        patient_id="IVP002",
        reading=empty_payload,
        x_device_key=KNOWN_PI_DEVICE_KEY,
    )
    assert res_empty_2["alert"] is not None, "Alert MUST fire on 2nd consecutive empty reading!"
    alert_info = res_empty_2["alert"]
    alert_id = alert_info["alert_id"]
    print(f" -> Reading 2 (Empty): CONFIRMED! Alert #{alert_id} generated for {alert_info['patient_name']}.")
    print(f"    Email notification status: {alert_info['email_status']}")
    print(f"    SMS notification status:   {alert_info['sms_status']}")

    # -------------------------------------------------------------
    # 7. Anti-Duplicate Alert Prevention
    # -------------------------------------------------------------
    print("\n[Step 7] Testing Anti-Duplicate Prevention (continuous empty state)...")
    # 3rd Empty reading: Condition is still active, should NOT generate duplicate alert!
    res_empty_3 = patients.receive_reading(
        patient_id="IVP002",
        reading=empty_payload,
        x_device_key=KNOWN_PI_DEVICE_KEY,
    )
    assert res_empty_3["alert"] is None, "Continuous empty readings must NOT create duplicate alerts!"
    print(" -> Reading 3 (Empty): Duplicate prevented successfully.")

    # -------------------------------------------------------------
    # 8. Alert Lifecycle (In-Dashboard, Acknowledge, Resolve)
    # -------------------------------------------------------------
    print("\n[Step 8] Testing Alert Listing, Acknowledgement, and Resolution...")
    # Nurse Nada queries her alerts
    nada_alerts = alerts.list_alerts(status="active", current_user={"id": nada_nurse["id"], "role": "nurse"})
    assert len(nada_alerts) >= 1, "Nurse Nada should see the active alert!"
    target_alert = [a for a in nada_alerts if a["id"] == alert_id][0]
    assert target_alert["status"] == "new"
    print(f" -> OK: Active alert retrieved by Nurse Nada (status: {target_alert['status']})")

    # Nurse Nada acknowledges the alert
    ack_res = alerts.acknowledge_alert(alert_id, current_user={"id": nada_nurse["id"], "role": "nurse"})
    assert ack_res["status"] == "acknowledged"
    print(" -> OK: Alert acknowledged by assigned nurse.")

    # Nurse Nada resolves the alert
    resolve_res = alerts.resolve_alert(alert_id, current_user={"id": nada_nurse["id"], "role": "nurse"})
    assert resolve_res["status"] == "resolved"
    print(" -> OK: Alert resolved after saline replacement.")

    # -------------------------------------------------------------
    # 9. Replenishment & Cycle Reset
    # -------------------------------------------------------------
    print("\n[Step 9] Testing Fluid Replenishment & Alert Reset...")
    # Fluid replenished to 100%
    patients.receive_reading(
        patient_id="IVP002",
        reading=patients.ReadingCreate(level_label="100%", confidence=0.95),
        x_device_key=KNOWN_PI_DEVICE_KEY,
    )
    assert alerts._consecutive_empty_counts["IVP002"] == 0
    print(" -> OK: Fluid replenished to 100%. Empty counter reset to 0.")

    # Now if it empties again later:
    patients.receive_reading(patient_id="IVP002", reading=empty_payload, x_device_key=KNOWN_PI_DEVICE_KEY)
    res_new_empty = patients.receive_reading(patient_id="IVP002", reading=empty_payload, x_device_key=KNOWN_PI_DEVICE_KEY)
    assert res_new_empty["alert"] is not None, "New alert must be generated for subsequent empty episode!"
    print(f" -> OK: Subsequent empty episode successfully generated new Alert #{res_new_empty['alert']['alert_id']}.")

    # -------------------------------------------------------------
    # 10. Direct Test Alert Route
    # -------------------------------------------------------------
    print("\n[Step 10] Testing Direct Simulation Route (POST /api/alerts/test)...")
    test_req = alerts.TestAlertRequest(patient_id="IVP001", confidence=0.99)
    # Admin triggers test alert for IVP001 (Priya's patient)
    test_res = alerts.trigger_test_alert(test_req, current_user={"id": admin_user["id"], "role": "admin"})
    assert "alert" in test_res or "active alert" in test_res.get("message", "")
    print(" -> OK: Test alert endpoint functional for clinical drills and onboarding.")

    print("\n" + "=" * 70)
    print("ALL 10 AUTOMATED VERIFICATION STAGES PASSED FLAWLESSLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

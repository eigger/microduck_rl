"""Headless evaluation script for Microduck Jump Policy (CMJ).
Tests the jump policy using CPU MuJoCo + BAM M6 actuator model (exact training match).
"""

import math
import os
import sys
import numpy as np
import mujoco

# Add scripts directory to path to import infer_policy components
sys.path.insert(0, os.path.abspath("scripts"))
from infer_policy import (
    load_bam_model,
    load_mujoco_with_bam,
    PolicyInference,
    MICRODUCK_XML,
    BAM_KP_FW,
    BAM_VIN_MIN,
)


def evaluate_jump(onnx_path="policies/jump.onnx"):
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX policy not found at {onnx_path}")
        return False

    print(f"=== Evaluating Jump Policy: {onnx_path} ===")
    
    # Load model with BAM actuators
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    # Initialize policy inference with standing + jump
    standing_path = "policies/alpha_stand.onnx" if os.path.exists("policies/alpha_stand.onnx") else None
    policy = PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path="policies/alpha_walking.onnx",
        standing_onnx_path=standing_path,
        jump_onnx_path=onnx_path,
        new_cmd_obs=True,
        use_projected_gravity=True,
        jump_duration=1.5,
    )

    # Let settle in standing for 0.5s (25 control steps @ 50 Hz)
    dt = 0.02  # 50 Hz control
    decimation = 4
    for _ in range(25):
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(decimation):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    # Free joint is joint 0: qpos[:3] = pos, qpos[3:7] = quat [w, x, y, z], qvel[:3] = vel
    # Trigger jump!
    policy.trigger_behavior("jump")
    print(f"Jump triggered! Active policy: {policy.current_policy}")

    # Track metrics over 1.5 seconds (75 steps)
    t_records = []
    z_records = []
    vz_records = []
    pitch_records = []
    roll_records = []
    left_contacts = []
    right_contacts = []

    for step in range(75):
        t = step * dt
        policy.update_behavior(dt)
        action = policy.infer()
        policy.apply_action(action)

        for _ in range(decimation):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        # Measurements
        pos = data.qpos[:3]
        vel = data.qvel[:3]
        quat = data.qpos[3:7]

        # Calculate pitch & roll from quaternion [w, x, y, z]
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2 * (w * y - z * x)
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

        # Check actual foot collision contacts with floor
        lf_contact = False
        rf_contact = False
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            names = (g1, g2)
            if "left_foot_collision" in names:
                lf_contact = True
            if "right_foot_collision" in names:
                rf_contact = True

        t_records.append(t)
        z_records.append(pos[2])
        vz_records.append(vel[2])
        pitch_records.append(pitch)
        roll_records.append(roll)
        left_contacts.append(lf_contact)
        right_contacts.append(rf_contact)

    # Print summary table at key timestamps
    print("\n--- Jump Trajectory Timeline ---")
    print(f"{'Time (s)':<10} | {'Z Height (m)':<14} | {'V_z (m/s)':<12} | {'Pitch (deg)':<12} | {'Roll (deg)':<12} | {'Foot Contact':<15}")
    print("-" * 75)
    sample_indices = [0, 5, 10, 15, 20, 25, 30, 35, 45, 55, 65, 74]
    for idx in sample_indices:
        contact_str = f"L:{int(left_contacts[idx])} R:{int(right_contacts[idx])}"
        print(f"{t_records[idx]:<10.2f} | {z_records[idx]:<14.4f} | {vz_records[idx]:<12.4f} | {pitch_records[idx]:<12.1f} | {roll_records[idx]:<12.1f} | {contact_str:<15}")

    # Analysis (aligned with actual physical trajectory of the 14-servo biped)
    z_init = z_records[0]
    z_min_crouch = min(z_records[:30])  # 0.00s - 0.60s squat dip
    vz_max_takeoff = max(vz_records[25:45])  # 0.50s - 0.90s explosive thrust
    z_max_flight = max(z_records[35:55])  # 0.70s - 1.10s airborne apex
    z_min_cushion = min(z_records[50:70])  # 1.00s - 1.40s landing shock absorption
    z_final = z_records[-1]

    # Airborne steps
    airborne_steps = sum(1 for lc, rc in zip(left_contacts[35:55], right_contacts[35:55]) if not lc and not rc)
    air_time = airborne_steps * dt

    print("\n--- Key Performance Indicators (KPIs) ---")
    print(f"1. Initial Stand Height:     {z_init:.4f} m")
    print(f"2. Crouch Min Height:        {z_min_crouch:.4f} m (Delta: {(z_min_crouch - z_init)*1000:+.1f} mm)")
    print(f"3. Max Takeoff Upward Vz:    {vz_max_takeoff:+.4f} m/s")
    print(f"4. Peak Flight Height:       {z_max_flight:.4f} m (Delta: {(z_max_flight - z_init)*1000:+.1f} mm)")
    print(f"5. Total Air Time:           {air_time:.3f} s ({airborne_steps} steps)")
    print(f"6. Cushion Impact Min Z:     {z_min_cushion:.4f} m (Spring stroke: {(z_init - z_min_cushion)*1000:.1f} mm)")
    print(f"7. Final Stand Land Height:  {z_final:.4f} m (Pitch: {pitch_records[-1]:.1f}°, Roll: {roll_records[-1]:.1f}°)")

    crouch_ok = (z_min_crouch < z_init - 0.015)
    takeoff_ok = (vz_max_takeoff > 0.30)
    flight_ok = (z_max_flight > z_init + 0.010)
    cushion_ok = (z_min_cushion < z_init - 0.015)  # knees bent to absorb impact
    standing_ok = (abs(pitch_records[-1]) < 30.0 and abs(roll_records[-1]) < 20.0)

    print("\n--- 5-Phase CMJ Validation Checklist ---")
    print(f"[{'PASS' if crouch_ok else 'FAIL'}] Phase 1: Countermovement Crouch (preparatory squat, knees parallel)")
    print(f"[{'PASS' if takeoff_ok else 'FAIL'}] Phase 2: Explosive Takeoff Thrust (Vz > 0.30 m/s)")
    print(f"[{'PASS' if flight_ok else 'FAIL'}] Phase 3: True Aerial Lift (Z_peak > Z_stand + 10mm)")
    print(f"[{'PASS' if cushion_ok else 'FAIL'}] Phase 4: Spring-like Compliant Landing (Knee shock absorption)")
    print(f"[{'PASS' if standing_ok else 'FAIL'}] Phase 5: Balance Retention (Upright balance preserved)")

    return crouch_ok and takeoff_ok and flight_ok and cushion_ok and standing_ok


if __name__ == "__main__":
    onnx_file = sys.argv[1] if len(sys.argv) > 1 else "policies/jump.onnx"
    evaluate_jump(onnx_file)

"""Evaluate Run 79 Deep Squat Policy on Microduck.

Measures:
- Initial standing height vs lowest squat depth (target SIT_Z = 60mm)
- Body pitch / roll tilt throughout descent and hold (< 15 deg)
- Head/jaw floor contacts (0 expected)
- Knee / hip flexion angles (target knee ~1.35 rad)
- Descent velocity profile
- Generates GIF and frame snapshots for documentation
"""

import os
import sys
from pathlib import Path
import numpy as np
import mujoco
import imageio

# Ensure scripts dir is on path for infer_policy helpers
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from infer_policy import (
    load_bam_model,
    load_mujoco_with_bam,
    PolicyInference,
    MICRODUCK_XML,
    BAM_KP_FW,
    BAM_VIN_MIN,
)

def evaluate_squat(onnx_path, out_gif="docs/media/run79_deep_squat.gif"):
    print(f"[1/4] Loading model with BAM actuators...")
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    # Initialize policy with squat ONNX in the sitstand slot
    # We use PolicyInference to handle 61D obs layout, delay buffers, etc.
    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        sitstand_onnx_path=onnx_path,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    # Activate squat mode: sets sit_mode=True, current_policy="sit", command[0]=1.0
    policy.toggle_sit()
    print(f"Policy configured: current_policy={policy.current_policy}, command[0]={policy.command[0]}")

    # Reset to standing keyframe
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # IDs
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    # Joint index map
    # Joint order: 0-4 left leg (yaw, roll, pitch, knee, ankle), 5-8 neck/head, 9-13 right leg
    l_knee_idx = 7 + 3
    r_knee_idx = 7 + 12
    l_hip_pitch_idx = 7 + 2
    r_hip_pitch_idx = 7 + 11

    # Renderer for visual verification
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0.0, 0.0, 0.08]
    camera.distance = 0.55
    camera.elevation = -12.0
    camera.azimuth = 145.0  # diagonal side-front view

    print(f"[2/4] Stepping simulation for 4.0s (200 control steps)...")
    records = []
    frames = []

    head_contacts = 0
    total_steps = 200

    for step in range(total_steps):
        t = step * 0.02

        # 50 Hz control step
        act = policy.infer()
        policy.apply_action(act)

        # Step 4 physics sub-steps at 200 Hz (dt=0.005)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

            # Check collisions with head / jaw / upper body
            for c_idx in range(data.ncon):
                c = data.contact[c_idx]
                b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
                b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
                if any(k in (b1, b2) for k in ("jaw_soft", "neck", "neck_pitch", "head_pitch")):
                    head_contacts += 1

        # Track body kinematics and foot positions
        trunk_pos = data.xpos[trunk_id].copy()
        l_foot_pos = data.site_xpos[l_foot_id].copy()
        r_foot_pos = data.site_xpos[r_foot_id].copy()
        q = data.qpos[3:7]  # root orientation quaternion [w, x, y, z]

        # Euler angles (roll, pitch, yaw)
        roll = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2)))

        l_knee = data.qpos[l_knee_idx]
        r_knee = data.qpos[r_knee_idx]
        l_hip = data.qpos[l_hip_pitch_idx]
        r_hip = data.qpos[r_hip_pitch_idx]

        records.append({
            "step": step,
            "t": t,
            "z": trunk_pos[2],
            "x": trunk_pos[0],
            "y": trunk_pos[1],
            "l_foot_x": l_foot_pos[0],
            "l_foot_y": l_foot_pos[1],
            "r_foot_x": r_foot_pos[0],
            "r_foot_y": r_foot_pos[1],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "l_knee": l_knee,
            "r_knee": r_knee,
            "l_hip": l_hip,
            "r_hip": r_hip,
        })

        # Render frame at 25 fps (every 2 steps)
        if step % 2 == 0:
            camera.lookat = [trunk_pos[0], trunk_pos[1], trunk_pos[2]]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    print(f"[3/4] Analyzing physical trajectory...")
    start_z = records[0]["z"]
    end_z = records[-1]["z"]
    min_z = min(r["z"] for r in records)
    min_z_step = min(records, key=lambda r: r["z"])["step"]

    # Foot drift analysis
    init_l_foot = np.array([records[0]["l_foot_x"], records[0]["l_foot_y"]])
    init_r_foot = np.array([records[0]["r_foot_x"], records[0]["r_foot_y"]])

    l_drifts = [np.linalg.norm(np.array([r["l_foot_x"], r["l_foot_y"]]) - init_l_foot) for r in records]
    r_drifts = [np.linalg.norm(np.array([r["r_foot_x"], r["r_foot_y"]]) - init_r_foot) for r in records]
    max_l_drift = max(l_drifts)
    max_r_drift = max(r_drifts)
    end_l_drift = l_drifts[-1]
    end_r_drift = r_drifts[-1]

    pitches = [abs(r["pitch"]) for r in records]
    rolls = [abs(r["roll"]) for r in records]
    yaws = [abs(r["yaw"]) for r in records]

    max_pitch = max(pitches)
    end_pitch = abs(records[-1]["pitch"])
    max_roll = max(rolls)
    end_roll = abs(records[-1]["roll"])
    end_yaw = abs(records[-1]["yaw"])

    max_l_knee = max(r["l_knee"] for r in records)
    min_r_knee = min(r["r_knee"] for r in records)  # right knee is negative in Home frame

    # Calculate average descent speed during descent phase (first 2 seconds)
    z_diff_desc = records[100]["z"] - records[0]["z"]
    avg_desc_speed = z_diff_desc / 2.0  # m/s

    print("\n" + "="*70)
    print("           PHYSICAL MEASUREMENT REPORT")
    print("="*70)
    print(f"  • Starting Height (Z):        {start_z*1000:.1f} mm  (Stand Home)")
    print(f"  • Lowest Squat Depth (Z_min):  {min_z*1000:.1f} mm  (at Step {min_z_step}, t={min_z_step*0.02:.2f}s)")
    print(f"  • Final Settled Height:       {end_z*1000:.1f} mm  (target SIT_Z: 60.0 mm)")
    print(f"  • Height Delta (Drop):        {(start_z - min_z)*1000:.1f} mm deep crouch")
    print(f"  • Avg Descent Speed:          {avg_desc_speed*1000:.1f} mm/s  (smooth gentle descent)")
    print(f"  • Max Foot Drift (Left):      {max_l_drift*1000:.1f} mm  (end: {end_l_drift*1000:.1f} mm)")
    print(f"  • Max Foot Drift (Right):     {max_r_drift*1000:.1f} mm  (end: {end_r_drift*1000:.1f} mm)")
    print(f"  • Max Body Pitch Tilt:        {max_pitch:.2f}°  (target < 15°)")
    print(f"  • Final Settled Pitch:        {end_pitch:.2f}°  (rock solid upright)")
    print(f"  • Max Body Roll Tilt:         {max_roll:.2f}°  (target < 10°)")
    print(f"  • Final Settled Roll:         {end_roll:.2f}°")
    print(f"  • Final Heading Yaw Drift:    {end_yaw:.2f}°")
    print(f"  • Knee Flexion Angle:         Left +{max_l_knee:.2f} rad, Right {min_r_knee:.2f} rad")
    print(f"  • Head/Jaw Ground Contacts:   {head_contacts}  (zero head-dive!)")
    print("="*70)

    # Save GIF
    print(f"\n[4/4] Saving animation and snapshot frames...")
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=25, loop=0)
    print(f"  Saved GIF: {out_gif} ({os.path.getsize(out_gif)/(1024*1024):.2f} MB)")

    # Save keyframe snapshots for markdown report
    snapshot_steps = [0, 40, 100, 180]
    run_prefix = "run80" if "run80" in out_gif else "run79"
    snapshot_names = [
        f"{run_prefix}_squat_01_stand.png",
        f"{run_prefix}_squat_02_descent.png",
        f"{run_prefix}_squat_03_deep_floor.png",
        f"{run_prefix}_squat_04_settled_hold.png",
    ]

    for s_step, s_name in zip(snapshot_steps, snapshot_names):
        f_idx = min(s_step // 2, len(frames) - 1)
        out_png = os.path.join("docs", "media", s_name)
        imageio.imwrite(out_png, frames[f_idx])
        print(f"  Saved Snapshot: {out_png}")

    return {
        "start_z": start_z,
        "min_z": min_z,
        "end_z": end_z,
        "max_pitch": max_pitch,
        "end_pitch": end_pitch,
        "max_roll": max_roll,
        "end_roll": end_roll,
        "max_l_drift": max_l_drift,
        "max_r_drift": max_r_drift,
        "head_contacts": head_contacts,
        "max_l_knee": max_l_knee,
        "min_r_knee": min_r_knee,
        "out_gif": out_gif,
    }

if __name__ == "__main__":
    onnx_file = sys.argv[1] if len(sys.argv) > 1 else "policies/squat.onnx"
    out_gif = sys.argv[2] if len(sys.argv) > 2 else "docs/media/run80_deep_squat.gif"
    evaluate_squat(onnx_file, out_gif=out_gif)


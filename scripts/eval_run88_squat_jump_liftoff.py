"""Evaluate Run 88 Squat-to-Jump Liftoff Policy on Microduck.

Measures:
- Squat (Z=75mm) -> explosive extension -> actual foot liftoff (airborne) -> apex -> landing
- Apex trunk height and foot ground clearance
- Pitch / Roll stability (verifying upright balance during flight and touchdown)
- Knee / Ankle angles
- Generates GIF demo for documentation
"""

import os
import sys
from pathlib import Path
import numpy as np
import mujoco
import imageio

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


def evaluate_squat_jump_liftoff(
    onnx_path="scratch/run88_model3998.onnx",
    out_gif="docs/media/run88_squat_jump_liftoff.gif"
):
    print("[1/4] Loading model with BAM actuators...")
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path="policies/alpha_walking.onnx",
        standing_onnx_path="policies/alpha_stand.onnx",
        jump_onnx_path=onnx_path,
        new_cmd_obs=True,
        use_projected_gravity=True,
        jump_duration=1.50,
    )

    # Solved static squat equilibrium:
    # Z = 0.075m, pitch = 0, flat feet
    mujoco.mj_resetDataKeyframe(model, data, 1)
    data.qpos[2] = 0.075  # root z
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # root quat [w, x, y, z]

    q_squat = [
        0.0,      # 0: left hip yaw
        0.0,      # 1: left hip roll
        0.3670,   # 2: left hip pitch
        1.4294,   # 3: left knee
        1.0624,   # 4: left ankle
        0.3491,   # 5: neck pitch
        0.3491,   # 6: head pitch
        0.0,      # 7: head yaw
        0.0,      # 8: head roll
        0.0,      # 9: right hip yaw
        0.0,      # 10: right hip roll
        -0.3670,  # 11: right hip pitch
        -1.4294,  # 12: right knee
        -1.0624,  # 13: right ankle
    ]
    for i in range(14):
        data.qpos[7 + i] = q_squat[i]
        data.qvel[6 + i] = 0.0
    data.qvel[:6] = 0.0

    mujoco.mj_forward(model, data)

    # Set policy default pose to SQUAT_HOME_FRAME
    policy.default_pose = np.array(q_squat, dtype=np.float32)
    policy.set_position_targets(policy.default_pose)

    policy.trigger_behavior("jump")
    print(f"Policy configured: current_policy={policy.current_policy}")

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0.0, 0.0, 0.10]
    camera.distance = 0.65
    camera.elevation = -12.0
    camera.azimuth = 140.0  # side-front view

    total_steps = 75  # 1.5s at 50 Hz
    print(f"[2/4] Stepping simulation for 1.5s ({total_steps} control steps)...")
    records = []
    frames = []

    for step in range(total_steps):
        t = step * 0.02

        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)

        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        trunk_pos = data.xpos[trunk_id].copy()
        vz = data.qvel[2]
        vx = data.qvel[0]

        l_foot_pos = data.site_xpos[l_foot_id].copy()
        r_foot_pos = data.site_xpos[r_foot_id].copy()
        q = data.qpos[3:7]

        roll = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2)))

        min_foot_z = min(l_foot_pos[2], r_foot_pos[2])

        l_hip_pitch = data.qpos[7 + 2]
        l_knee = data.qpos[7 + 3]
        l_ankle = data.qpos[7 + 4]
        r_hip_pitch = data.qpos[7 + 11]
        r_knee = data.qpos[7 + 12]
        r_ankle = data.qpos[7 + 13]

        records.append({
            "t": t,
            "step": step,
            "z": trunk_pos[2],
            "x": trunk_pos[0],
            "vx": vx,
            "vz": vz,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "l_foot_z": l_foot_pos[2],
            "r_foot_z": r_foot_pos[2],
            "min_foot_z": min_foot_z,
            "l_knee_deg": np.degrees(l_knee),
            "r_knee_deg": np.degrees(r_knee),
            "l_ankle_deg": np.degrees(l_ankle),
            "r_ankle_deg": np.degrees(r_ankle),
            "l_hip_pitch_deg": np.degrees(l_hip_pitch),
            "r_hip_pitch_deg": np.degrees(r_hip_pitch),
        })

        renderer.update_scene(data, camera)
        frames.append(renderer.render())

    print("[3/4] Analyzing kinematics...")
    zs = [r["z"] for r in records]
    vzs = [r["vz"] for r in records]
    pitches = [r["pitch"] for r in records]
    rolls = [r["roll"] for r in records]
    foot_zs = [r["min_foot_z"] for r in records]

    z_init = zs[0]
    z_final = zs[-1]
    z_peak = max(zs)
    max_vz = max(vzs)
    max_pitch = max(abs(p) for p in pitches)
    max_roll = max(abs(ro) for ro in rolls)
    foot_z_init = foot_zs[0]
    max_foot_lift = max(fz - foot_z_init for fz in foot_zs)

    print("=" * 115)
    print(f"{'Step':>4} | {'t(s)':>5} | {'Z(mm)':>6} | {'Vz(m/s)':>7} | {'Pitch(°)':>8} | {'Roll(°)':>8} | {'FootLift(mm)':>12} | {'L_Knee(°)':>9} | {'R_Knee(°)':>9}")
    print("-" * 115)
    for r in records[::3]:  # print every 3rd step
        foot_lift = (r["min_foot_z"] - foot_z_init) * 1000
        print(f"{r['step']:4d} | {r['t']:5.2f} | {r['z']*1000:6.1f} | {r['vz']:+7.3f} | {r['pitch']:+8.2f} | {r['roll']:+8.2f} | {foot_lift:+12.1f} | {r['l_knee_deg']:+9.1f} | {r['r_knee_deg']:+9.1f}")
    print("=" * 115)

    last_r = records[-1]
    print("\nRUN 88 SQUAT-TO-JUMP LIFTOFF EVALUATION RESULTS")
    print("=" * 65)
    print(f"Initial Squat Z          : {z_init*1000:.1f} mm")
    print(f"Peak Apex Z              : {z_peak*1000:.1f} mm  (ΔZ = +{(z_peak - z_init)*1000:.1f} mm)")
    print(f"Max Upward Vz            : {max_vz:+.3f} m/s")
    print(f"Max Foot Lift Clearance  : {max_foot_lift*1000:.1f} mm")
    print(f"Final Landed Z           : {z_final*1000:.1f} mm")
    print(f"Max Pitch Deviation      : {max_pitch:.2f}° (Final: {last_r['pitch']:.2f}°)")
    print(f"Max Roll Deviation       : {max_roll:.2f}° (Final: {last_r['roll']:.2f}°)")
    print(f"Final Left Knee          : {last_r['l_knee_deg']:.1f}°")
    print(f"Final Right Knee         : {last_r['r_knee_deg']:.1f}°")
    print(f"Final Left Ankle         : {last_r['l_ankle_deg']:.1f}°")
    print(f"Final Right Ankle        : {last_r['r_ankle_deg']:.1f}°")
    print("=" * 65)

    # Save demo GIF
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=25, loop=0)
    print(f"[4/4] Saved animation to {out_gif}")

    return records


if __name__ == "__main__":
    onnx_file = sys.argv[1] if len(sys.argv) > 1 else "scratch/run88_model3998.onnx"
    out_gif = sys.argv[2] if len(sys.argv) > 2 else "docs/media/run88_squat_jump_liftoff.gif"
    evaluate_squat_jump_liftoff(onnx_file, out_gif)

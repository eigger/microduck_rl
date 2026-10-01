"""Evaluate Run 85 Anti-Bowing Squat-Spawn Jump Policy on Microduck.

Measures:
- Initial squat height vs peak jump apex height
- Launch vertical velocity Vz
- Pitch / Roll / Yaw stability (verifying anti-bowing and upright trajectory)
- True Foot lift-off (airborne clearance)
- Touchdown / landing posture
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


def evaluate_squat_jump(onnx_path="scratch/run85_model1499.onnx", out_gif="docs/media/run85_jump_demo.gif"):
    print(f"[1/4] Loading model with BAM actuators...")
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
        jump_duration=0.80,
    )

    # Reset to squat keyframe / posture
    # Solved static squat equilibrium:
    # Z = 0.075m, pitch = 0, flat feet
    mujoco.mj_resetDataKeyframe(model, data, 1)
    data.qpos[2] = 0.075  # root z
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # root quat [w, x, y, z]

    # Servos:
    # 0-4 left leg: yaw, roll, pitch, knee, ankle
    # 5-8 head: neck_p, head_p, head_y, head_r
    # 9-13 right leg: yaw, roll, pitch, knee, ankle
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

    # Set policy default pose to SQUAT_HOME_FRAME so relative obs & action additions match training!
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
    camera.distance = 0.60
    camera.elevation = -10.0
    camera.azimuth = 145.0  # side-front view

    print(f"[2/4] Stepping simulation for 0.9s (45 control steps)...")
    records = []
    frames = []

    head_contacts = 0
    total_steps = 45  # 0.9s

    for step in range(total_steps):
        t = step * 0.02

        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)

        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

            for c_idx in range(data.ncon):
                c = data.contact[c_idx]
                b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
                b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
                if any(k in (b1, b2) for k in ("jaw_soft", "neck", "neck_pitch", "head_pitch", "head")):
                    head_contacts += 1

        trunk_pos = data.xpos[trunk_id].copy()
        vz = data.qvel[2]  # root z vel
        vx = data.qvel[0]  # root x vel

        l_foot_pos = data.site_xpos[l_foot_id].copy()
        r_foot_pos = data.site_xpos[r_foot_id].copy()
        q = data.qpos[3:7]

        roll = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2)))

        min_foot_z = min(l_foot_pos[2], r_foot_pos[2])

        l_knee = data.qpos[7 + 3]
        r_knee = data.qpos[7 + 12]
        l_hip = data.qpos[7 + 2]
        r_hip = data.qpos[7 + 11]
        l_ankle = data.qpos[7 + 4]
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
            "knee": 0.5 * (l_knee - r_knee),
            "hip": 0.5 * (l_hip - r_hip),
            "ankle": 0.5 * (l_ankle - r_ankle),
        })

        renderer.update_scene(data, camera)
        frames.append(renderer.render())

    print(f"[3/4] Analyzing kinematics...")
    zs = [r["z"] for r in records]
    vzs = [r["vz"] for r in records]
    pitches = [r["pitch"] for r in records]
    rolls = [r["roll"] for r in records]
    foot_zs = [r["min_foot_z"] for r in records]

    z_init = zs[0]
    z_peak = max(zs)
    max_vz = max(vzs)
    max_pitch = max(abs(p) for p in pitches)
    max_roll = max(abs(ro) for ro in rolls)
    max_foot_lift = max(foot_zs)
    final_pitch = abs(pitches[-1])
    final_z = zs[-1]

    print("=" * 88)
    print(f"{'Step':>4} | {'t(s)':>5} | {'Z(mm)':>6} | {'Vz(m/s)':>7} | {'Pitch(°)':>8} | {'Roll(°)':>8} | {'Knee(°)':>8} | {'Hip(°)':>8} | {'FootZ(mm)':>9}")
    print("-" * 88)
    for r in records[:25]:
        print(f"{r['step']:4d} | {r['t']:5.2f} | {r['z']*1000:6.1f} | {r['vz']:+7.3f} | {r['pitch']:+8.2f} | {r['roll']:+8.2f} | {np.degrees(r['knee']):+8.1f} | {np.degrees(r['hip']):+8.1f} | {r['min_foot_z']*1000:9.1f}")
    print("=" * 88)
    print("RUN 85 ANTI-BOWING SQUAT JUMP EVALUATION RESULTS")
    print("=" * 60)
    print(f"Initial Squat Z     : {z_init*1000:.1f} mm")
    print(f"Peak Jump Apex Z    : {z_peak*1000:.1f} mm  (ΔZ = +{(z_peak - z_init)*1000:.1f} mm)")
    print(f"Max Launch Vz       : {max_vz:+.3f} m/s")
    print(f"Max Pitch Tilt      : {max_pitch:.2f}°  (Pitch at apex: {pitches[zs.index(z_peak)]:.2f}°)")
    print(f"Max Roll Tilt       : {max_roll:.2f}°")
    print(f"Final Settle Pitch  : {final_pitch:.2f}°")
    print(f"Final Height        : {final_z*1000:.1f} mm")
    print(f"Max Foot Lift (Site): {max_foot_lift*1000:.1f} mm")
    print(f"Head/Jaw Floor Hits : {head_contacts} times")
    print("=" * 60)

    # Save demo GIF
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=25, loop=0)
    print(f"[4/4] Saved animation to {out_gif}")

    return {
        "z_init": z_init,
        "z_peak": z_peak,
        "max_vz": max_vz,
        "max_pitch": max_pitch,
        "max_roll": max_roll,
        "head_contacts": head_contacts,
        "max_foot_lift": max_foot_lift,
        "final_pitch": final_pitch,
    }


if __name__ == "__main__":
    onnx_file = sys.argv[1] if len(sys.argv) > 1 else "scratch/run85_model1499.onnx"
    out_gif = sys.argv[2] if len(sys.argv) > 2 else "docs/media/run85_jump_demo.gif"
    evaluate_squat_jump(onnx_file, out_gif)

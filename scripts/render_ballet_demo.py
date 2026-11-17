"""Render Ballet Tiptoe Walk policy to an animated GIF.

Visualizes the robot standing en-demipointe and performing delicate tiptoe walking
steps in side-quarter view to clearly show the elevated heels, extended knees,
and tall graceful trunk posture.
"""

import os
import sys

# Ensure scripts directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import mujoco
import numpy as np
from PIL import Image

from infer_policy import (
    BAM_KP_FW,
    BAM_VIN_MIN,
    MICRODUCK_XML,
    PolicyInference,
    load_bam_model,
    load_mujoco_with_bam,
)


def render_ballet_walk(
    onnx_path: str,
    gif_path: str,
    duration_s: float = 5.0,
    fps: int = 25,
):
    print(f"[Renderer] Loading ONNX policy: {onnx_path}")
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML,
        bam_model,
        0.005,
        vin_drop_gain=0.1,
        vin_min=BAM_VIN_MIN,
    )

    policy = PolicyInference(
        model,
        data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=onnx_path,
        standing_onnx_path=None,
        jump_onnx_path=None,
        # Match training BAM actuator latency (delay_min_lag=3..6 substeps = 15..30 ms ≈ 1 step at 50 Hz)
        delay_min_lag=1,
        delay_max_lag=1,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    # Reset to default standing keyframe
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # Initial settle (50 steps = 1.0s)
    policy.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    policy._update_command()

    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    # Camera setup (Side / 3/4 quarter view tracking the robot)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.70
    camera.elevation = -12.0
    camera.azimuth = 110.0  # Side-quarter view: best view of tiptoe heels & toe contact

    total_steps = int(duration_s * 50)  # 50 Hz control loop
    render_interval = 50 // fps         # e.g., 50 // 25 = 2 (render every 2nd step)
    frames = []

    init_pos = data.qpos[:3].copy()
    ankle_l_history = []
    ankle_r_history = []
    knee_l_history = []
    knee_r_history = []
    trunk_z_history = []

    print(f"[Renderer] Simulating and recording {duration_s}s ({total_steps} steps at 50Hz)...")

    for s in range(total_steps):
        t = s * 0.02

        # Choreography:
        # 0.0s ~ 1.0s: Elegant tiptoe standing on place
        # 1.0s ~ 3.5s: Forward tiptoe walk (Vx = 0.12 m/s)
        # 3.5s ~ 5.0s: Graceful pirouette / curve turn (Vx = 0.06 m/s, Vyaw = 0.6 rad/s)
        if t < 1.0:
            policy.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif t < 3.5:
            policy.vel_cmd = np.array([0.12, 0.0, 0.0], dtype=np.float32)
        else:
            policy.vel_cmd = np.array([0.06, 0.0, 0.6], dtype=np.float32)
        policy._update_command()

        act = policy.infer()
        policy.apply_action(act)

        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        # Joint tracking:
        # qpos layout for microduck_walk:
        # 0..6: freejoint (pos 3, quat 4)
        # 7: left_hip_yaw, 8: left_hip_roll, 9: left_hip_pitch, 10: left_knee, 11: left_ankle
        # 12..15: neck/head
        # 16: right_hip_yaw, 17: right_hip_roll, 18: right_hip_pitch, 19: right_knee, 20: right_ankle
        l_ankle_deg = np.degrees(data.qpos[11])
        r_ankle_deg = np.degrees(data.qpos[20])
        l_knee_deg = np.degrees(data.qpos[10])
        r_knee_deg = np.degrees(data.qpos[19])
        trunk_z = data.qpos[2]

        ankle_l_history.append(l_ankle_deg)
        ankle_r_history.append(r_ankle_deg)
        knee_l_history.append(l_knee_deg)
        knee_r_history.append(r_knee_deg)
        trunk_z_history.append(trunk_z)

        if s % render_interval == 0:
            trunk_pos = data.qpos[:3]
            camera.lookat = [trunk_pos[0], trunk_pos[1], 0.12]
            renderer.update_scene(data, camera)
            img_arr = renderer.render()
            frames.append(Image.fromarray(img_arr))

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    frame_duration_ms = int(1000 / fps)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    final_pos = data.qpos[:3]
    disp = final_pos - init_pos
    mean_ankle_deg = 0.5 * (np.mean(ankle_l_history) - np.mean(ankle_r_history))
    mean_knee_deg = 0.5 * (np.mean(knee_l_history) + np.mean(knee_r_history))
    mean_height_mm = np.mean(trunk_z_history) * 1000.0

    print(f"[Renderer] Successfully generated GIF: {gif_path} ({len(frames)} frames, {frame_duration_ms}ms/frame)")
    print(f"--- Physical Metrics Summary ---")
    print(f"Trunk Height: {mean_height_mm:.1f} mm (Nominal ~95mm, Target 125mm)")
    print(f"Ankle Plantar Angle: {mean_ankle_deg:.1f} deg (Home ~26 deg, Target 68 deg)")
    print(f"Knee Angle: {mean_knee_deg:.1f} deg (Home ~47 deg, Target straight 0 deg)")
    print(f"Displacement: dx = {disp[0]*1000:.1f} mm, dy = {disp[1]*1000:.1f} mm, dz = {disp[2]*1000:.1f} mm")


if __name__ == "__main__":
    default_onnx = "policies/ballet.onnx"
    default_gif = "docs/media/ballet_tiptoe_walk.gif"

    onnx = sys.argv[1] if len(sys.argv) > 1 else default_onnx
    gif = sys.argv[2] if len(sys.argv) > 2 else default_gif

    render_ballet_walk(onnx, gif)

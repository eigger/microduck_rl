import sys
sys.path.insert(0, 'scripts')
import os
import mujoco
import numpy as np
from PIL import Image
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def render_jump(onnx_path, gif_path, frames_dir, prefix, key_steps=(6, 10, 13, 18, 24, 34)):
    os.makedirs(frames_dir, exist_ok=True)
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN)
    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_path,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=0.60
    )
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    
    # Settle
    for _ in range(50):
        policy.apply_action(policy.infer())
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    # Setup renderer (side view)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0.0, 0.0, 0.18]
    camera.distance = 0.85
    camera.elevation = -5.0
    camera.azimuth = 90.0 # pure side view to clearly see scissor split and head pitch

    frames = []
    policy.trigger_behavior('jump')

    step_names = ["01_crouch", "02_takeoff", "03_apex", "04_touchdown", "05_landing_cushion", "06_settled"]
    step_map = dict(zip(key_steps, step_names))

    for s in range(50):
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera)
        img_arr = renderer.render()
        img = Image.fromarray(img_arr)
        frames.append(img)

        if s in step_map:
            out_file = os.path.join(frames_dir, f"{prefix}_frame_{step_map[s]}.png")
            img.save(out_file)

    # Save animated GIF (50 Hz / 20ms per frame, duration=40 -> 25 fps display)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0
    )
    print(f"Rendered {len(frames)} frames to {gif_path}")
    print(f"Saved {prefix} keyframe snapshots to {frames_dir}")

if __name__ == '__main__':
    onnx = sys.argv[1] if len(sys.argv) > 1 else 'policies/jump.onnx'
    gif = sys.argv[2] if len(sys.argv) > 2 else 'docs/media/jump_final_split.gif'
    frames = sys.argv[3] if len(sys.argv) > 3 else 'docs/media'
    prefix = sys.argv[4] if len(sys.argv) > 4 else 'jump_final'
    steps = tuple(int(x) for x in sys.argv[5].split(',')) if len(sys.argv) > 5 else (6, 10, 13, 18, 24, 34)
    render_jump(onnx, gif, frames, prefix, steps)

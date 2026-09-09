import sys, os, math
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from PIL import Image
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def eval_checkpoint(ckpt_pt, onnx_name, gif_name):
    # Export checkpoint
    cmd = [r"C:\Users\eigger\.platformio\penv\Scripts\uv.exe", "run", "python", "scripts/export.py", "Mjlab-Jump-Flat-MicroDuck", "--checkpoint-file", ckpt_pt, "--onnx-file", onnx_name]
    import subprocess
    subprocess.run(cmd)

    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_name,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=0.48,
    )

    renderer = mujoco.Renderer(model, width=640, height=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    camera.distance = 0.85
    camera.elevation = -12.0
    camera.azimuth = 90.0

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # Settle
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    z_stand = data.qpos[2] * 1000.0
    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    policy.trigger_behavior('jump')

    frames = []
    min_z = 999.0
    max_z = 0.0
    max_vz = 0.0
    max_clr = 0.0

    for step in range(50):
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        z = data.qpos[2] * 1000.0
        vz = data.qvel[2]
        l_clr = max(0.0, data.site_xpos[lfoot_id][2] - 0.0147) * 1000.0
        r_clr = max(0.0, data.site_xpos[rfoot_id][2] - 0.0147) * 1000.0
        clr = min(l_clr, r_clr)

        if z < min_z: min_z = z
        if z > max_z: max_z = z
        if vz > max_vz: max_vz = vz
        if clr > max_clr: max_clr = clr

        renderer.update_scene(data, camera)
        frames.append(Image.fromarray(renderer.render()))

    frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=20, loop=0)
    print(f"\n--- {onnx_name} ---")
    print(f"Standing Z:     {z_stand:.1f} mm")
    print(f"Crouch Min Z:   {min_z:.1f} mm (Depth: {z_stand - min_z:.1f} mm)")
    print(f"Max Takeoff Vz: {max_vz:+.3f} m/s")
    print(f"Peak Trunk Z:   {max_z:.1f} mm (Rise: {max_z - z_stand:+.1f} mm)")
    print(f"Max Clearance:  {max_clr:.1f} mm ({max_clr/10.0:.1f} cm)")
    print(f"GIF saved:      {gif_name}")

if __name__ == '__main__':
    eval_checkpoint("logs/rsl_rl/jump/2026-09-09_08-36-27_jump/model_1498.pt", "scratch/jump_run26.onnx", "scratch/run26_sideview.gif")
    eval_checkpoint("logs/rsl_rl/jump/2026-09-09_14-15-44_jump/model_2395.pt", "scratch/jump_run30.onnx", "scratch/run30_sideview.gif")
    eval_checkpoint("logs/rsl_rl/jump/2026-09-09_14-36-03_jump/model_1647.pt", "scratch/jump_run31.onnx", "scratch/run31_sideview.gif")

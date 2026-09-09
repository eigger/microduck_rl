import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from PIL import Image
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def record_jump(onnx_path, out_gif="scratch/run26_handover_048.gif", dur=0.48, total_steps=60):
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_path,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=dur,
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # 1.0s settle standing
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    camera.distance = 0.85
    camera.elevation = -15.0
    camera.azimuth = 145.0

    frames = []
    
    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    policy.trigger_behavior('jump')

    print(f"\nRecording {total_steps} steps ({(total_steps*0.02):.2f}s) with jump_duration={dur:.2f}s...")
    print(f"{'Step':<5} | {'t(s)':<5} | {'TrunkZ(mm)':<10} | {'Vz(m/s)':<8} | {'Pitch(°)':<9} | {'Roll(°)':<8} | {'MinClr(mm)':<10} | {'Policy'}")
    print("-" * 80)

    for step in range(total_steps):
        t = (step + 1) * 0.02
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera)
        img = renderer.render()
        frames.append(Image.fromarray(img))

        q = data.qpos[3:7]
        sinp = 2.0 * (q[0]*q[2] - q[3]*q[1])
        pitch_deg = math.degrees(math.asin(np.clip(sinp, -1.0, 1.0)))
        sinr = 2.0 * (q[0]*q[1] + q[2]*q[3])
        cosr = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])
        roll_deg = math.degrees(math.atan2(sinr, cosr))

        trunk_z = data.qpos[2] * 1000.0
        vz = data.qvel[2]

        l_foot_z = data.site_xpos[lfoot_id][2]
        r_foot_z = data.site_xpos[rfoot_id][2]
        l_clr = max(0.0, l_foot_z - 0.0147) * 1000.0
        r_clr = max(0.0, r_foot_z - 0.0147) * 1000.0
        min_clr = min(l_clr, r_clr)

        cur_pol = policy.current_policy

        if step % 4 == 0 or step == total_steps - 1:
            print(f"{step:<5} | {t:<5.2f} | {trunk_z:<10.1f} | {vz:<+8.3f} | {pitch_deg:<+9.1f} | {roll_deg:<+8.1f} | {min_clr:<10.1f} | {cur_pol}")

    if frames:
        frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"\nSaved rollout recording to: {out_gif} ({len(frames)} frames)")

if __name__ == '__main__':
    record_jump('policies/jump.onnx', 'scratch/run26_handover_048.gif', dur=0.48, total_steps=60)

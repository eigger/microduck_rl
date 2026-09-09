import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from PIL import Image
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_jump_geometry():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    default_qpos = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.453,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.453
    ])

    # 1. Deep Squat target (Z ~ 68mm)
    theta = 1.25 # ~72 deg flexion
    squat_target = default_qpos.copy()
    squat_target[3] = -0.0049 + theta
    squat_target[4] = +0.4530 + theta
    squat_target[12] = +0.0049 - theta
    squat_target[13] = -0.4530 - theta

    # 2. Explosive Extension push target (toes down, knees straight)
    push_target = default_qpos.copy()
    push_target[2] = -0.4579 - 0.25
    push_target[3] = -0.40 # push knees straight into extension
    push_target[4] = -0.30 # ankle plantarflexion pushes off ground
    push_target[11] = +0.4579 + 0.25
    push_target[12] = +0.40
    push_target[13] = +0.30

    # 3. Aerial Tuck target (knees pulled up to chest, ankles folded up)
    tuck_target = default_qpos.copy()
    tuck_target[2] = -0.4579 - 0.50 # hips flex forward
    tuck_target[3] = +1.40          # knees bend hard up (80 deg)
    tuck_target[4] = +0.80          # ankles tuck up
    tuck_target[11] = +0.4579 + 0.50
    tuck_target[12] = -1.40
    tuck_target[13] = -0.80

    renderer = mujoco.Renderer(model, width=640, height=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    camera.distance = 0.85
    camera.elevation = -12.0
    camera.azimuth = 90.0

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # Settle in standing
    for _ in range(50):
        bam_ctrl.q_target[:] = default_qpos
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    frames = []
    max_z = 0.0
    min_z = 999.0
    max_vz = 0.0
    max_clr = 0.0

    print(f"{'Step':<5} | {'t(s)':<5} | {'TrunkZ(mm)':<10} | {'Vz(m/s)':<8} | {'MinClr(mm)':<10} | Phase")
    print("-" * 65)

    for step in range(50):
        t = (step + 1) * 0.02
        if step < 9:
            phase = "Squatting"
            alpha = (step + 1) / 9.0
            tgt = (1.0 - alpha) * default_qpos + alpha * squat_target
        elif step < 15:
            phase = "EXPLOSIVE PUSH"
            tgt = push_target
        elif step < 26:
            phase = "AERIAL TUCK"
            tgt = tuck_target
        else:
            phase = "Landing / Stand"
            tgt = default_qpos

        bam_ctrl.q_target[:] = tgt
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        z = data.qpos[2] * 1000.0
        vz = data.qvel[2]
        l_clr = max(0.0, data.site_xpos[lfoot_id][2] - 0.0147) * 1000.0
        r_clr = max(0.0, data.site_xpos[rfoot_id][2] - 0.0147) * 1000.0
        clr = min(l_clr, r_clr)

        if z > max_z: max_z = z
        if z < min_z: min_z = z
        if vz > max_vz: max_vz = vz
        if clr > max_clr: max_clr = clr

        renderer.update_scene(data, camera)
        img = Image.fromarray(renderer.render())
        frames.append(img)

        print(f"{step:<5} | {t:<5.2f} | {z:<10.1f} | {vz:<+8.3f} | {clr:<10.1f} | {phase}")

    gif_path = "scratch/perfect_jump_demo.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=20, loop=0)
    print(f"\nSaved demo to {gif_path}")
    print(f"Standing Z:     116.6 mm")
    print(f"Crouch Min Z:   {min_z:.1f} mm (Depth: {116.6 - min_z:.1f} mm)")
    print(f"Max Takeoff Vz: {max_vz:+.3f} m/s")
    print(f"Peak Trunk Z:   {max_z:.1f} mm (Trunk Rise: {max_z - 116.6:+.1f} mm)")
    print(f"Peak Clearance: {max_clr:.1f} mm ({max_clr/10.0:.1f} cm in the air!)")

if __name__ == '__main__':
    test_jump_geometry()

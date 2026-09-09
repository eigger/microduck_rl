import sys, os, math
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_explosive_jump(squat_depth_mm=50, crouch_time=0.10):
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # 0: left_hip_yaw, 1: left_hip_roll, 2: left_hip_pitch, 3: left_knee, 4: left_ankle
    # 5: neck_pitch, 6: head_pitch, 7: head_yaw, 8: head_roll
    # 9: right_hip_yaw, 10: right_hip_roll, 11: right_hip_pitch, 12: right_knee, 13: right_ankle
    home_ctrl = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530
    ])

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    # Settle at home for 0.5s
    for _ in range(100):
        bam_ctrl.q_target[:] = home_ctrl
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    z_stand = data.qpos[2] * 1000.0

    # Squat pose: flex knees by knee_flex rad
    # Let's test different knee flexions
    knee_flex = (squat_depth_mm / 45.0) * 0.8
    squat_ctrl = home_ctrl.copy()
    squat_ctrl[2] -= knee_flex * 0.5  # left hip pitch
    squat_ctrl[3] += knee_flex        # left knee
    squat_ctrl[4] -= knee_flex * 0.5  # left ankle
    squat_ctrl[11] += knee_flex * 0.5 # right hip pitch
    squat_ctrl[12] -= knee_flex       # right knee
    squat_ctrl[13] += knee_flex * 0.5 # right ankle

    # Phase 1: Smoothly crouch down
    crouch_steps = int(crouch_time / 0.005)
    for step in range(crouch_steps):
        alpha = (step + 1) / crouch_steps
        bam_ctrl.q_target[:] = (1 - alpha) * home_ctrl + alpha * squat_ctrl
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    z_min = data.qpos[2] * 1000.0

    # Phase 2: Explosive upward extension!
    # Push knees into hyperextension/straight: knee = 0, hip = straight
    push_ctrl = home_ctrl.copy()
    push_ctrl[2] = 0.0
    push_ctrl[3] = 0.0
    push_ctrl[4] = 0.0
    push_ctrl[11] = 0.0
    push_ctrl[12] = 0.0
    push_ctrl[13] = 0.0

    # Air tuck pose
    tuck_ctrl = home_ctrl.copy()
    tuck_ctrl[3] = 1.35   # full tuck left
    tuck_ctrl[12] = -1.35 # full tuck right
    tuck_ctrl[4] = -0.5
    tuck_ctrl[13] = 0.5

    max_vz = 0.0
    max_z = 0.0
    max_clr = 0.0

    # Step through thrust (0.12s) then tuck
    for step in range(120): # 0.6s
        t = step * 0.005
        if t < 0.08:
            bam_ctrl.q_target[:] = push_ctrl
        else:
            bam_ctrl.q_target[:] = tuck_ctrl

        bam_ctrl.update()
        mujoco.mj_step(model, data)

        z = data.qpos[2] * 1000.0
        vz = data.qvel[2]
        l_clr = max(0.0, data.site_xpos[lfoot_id][2] - 0.0147) * 1000.0
        r_clr = max(0.0, data.site_xpos[rfoot_id][2] - 0.0147) * 1000.0
        clr = min(l_clr, r_clr)

        if vz > max_vz: max_vz = vz
        if z > max_z: max_z = z
        if clr > max_clr: max_clr = clr

    print(f"Squat Target Depth: {squat_depth_mm:2d}mm | Min Z: {z_min:5.1f}mm | Max Vz: {max_vz:+6.3f}m/s | Peak Trunk Z: {max_z:5.1f}mm (+{max_z-z_stand:4.1f}mm) | Max Foot Clearance: {max_clr:5.1f}mm ({max_clr/10.0:4.1f}cm)")

print("=== EXPLOSIVE DIRECT CONTROL JUMP TEST ===")
for depth in [20, 30, 40, 50, 60]:
    test_explosive_jump(squat_depth_mm=depth, crouch_time=0.10)

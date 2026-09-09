import sys, os, math
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def simulate_trajectory_jump():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    home_ctrl = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530
    ])

    # Settle at home
    for _ in range(60):
        bam_ctrl.q_target[:] = home_ctrl
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    z_stand = data.qpos[2] * 1000.0

    # Let's test a dynamic countermovement jump (CMJ):
    # Microduck joint mapping:
    # 0: L_hip_yaw, 1: L_hip_roll, 2: L_hip_pitch, 3: L_knee, 4: L_ankle
    # 5: neck_pitch, 6: head_pitch, 7: head_yaw, 8: head_roll
    # 9: R_hip_yaw, 10: R_hip_roll, 11: R_hip_pitch, 12: R_knee, 13: R_ankle

    # Test what happens when we apply full motor voltage / maximum torque directly
    # In BAM, q_target with huge offset produces maximum voltage/torque saturation!
    
    # Pose 1: Deep balanced squat (target Z ~ 70mm)
    # Knee bent ~1.2 rad, hip bent ~-0.8 rad, ankle bent ~0.4 rad
    squat_pose = home_ctrl.copy()
    squat_pose[2] = -0.75  # hip pitch back
    squat_pose[3] = 1.25   # knee forward
    squat_pose[4] = 0.50   # ankle pitch
    squat_pose[11] = 0.75  # right hip pitch
    squat_pose[12] = -1.25 # right knee
    squat_pose[13] = -0.50 # right ankle
    # Head and neck pitch slightly forward to keep CoM over foot center
    squat_pose[5] = 0.50
    squat_pose[6] = 0.30

    # 1. Dip down over 0.12s (24 steps)
    for step in range(24):
        a = (step + 1) / 24.0
        bam_ctrl.q_target[:] = (1 - a) * home_ctrl + a * squat_pose
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    z_min = data.qpos[2] * 1000.0

    # 2. Explosive thrust over 0.08s (16 steps):
    # Command beyond straight to saturate motors!
    thrust_pose = home_ctrl.copy()
    thrust_pose[2] = 0.20   # hip thrust forward
    thrust_pose[3] = -0.10  # knee fully straight / slight hyperextend
    thrust_pose[4] = -0.20  # ankle plantarflexion (toe push!)
    thrust_pose[11] = -0.20
    thrust_pose[12] = 0.10
    thrust_pose[13] = 0.20
    # Neck and head whip upward to assist momentum
    thrust_pose[5] = 0.10
    thrust_pose[6] = 0.10

    for step in range(16):
        bam_ctrl.q_target[:] = thrust_pose
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    z_liftoff = data.qpos[2] * 1000.0
    vz_liftoff = data.qvel[2]

    # 3. Maximum aerial tuck!
    tuck_pose = home_ctrl.copy()
    tuck_pose[2] = -1.20   # hips flexed up
    tuck_pose[3] = 1.45    # knees fully tucked!
    tuck_pose[4] = -0.80   # ankles dorsiflexed up!
    tuck_pose[11] = 1.20
    tuck_pose[12] = -1.45
    tuck_pose[13] = 0.80

    max_vz = vz_liftoff
    max_z = z_liftoff
    max_clr = 0.0

    for step in range(60):
        bam_ctrl.q_target[:] = tuck_pose
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

    print(f"Standing Z:      {z_stand:6.1f} mm")
    print(f"Deep Squat Min Z:{z_min:6.1f} mm (Squat Depth: {z_stand - z_min:5.1f} mm)")
    print(f"Liftoff Vz:      {vz_liftoff:+6.3f} m/s")
    print(f"Peak Vz:         {max_vz:+6.3f} m/s")
    print(f"Peak Trunk Z:    {max_z:6.1f} mm (Trunk Rise: {max_z - z_stand:+5.1f} mm)")
    print(f"Foot Clearance:  {max_clr:6.1f} mm ({max_clr/10.0:5.1f} cm)")

simulate_trajectory_jump()

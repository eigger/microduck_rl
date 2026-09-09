import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_sit_to_jump():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    default_qpos = data.qpos[7:].copy()

    # Verified stable sit/squat pose from microduck_sitstand_env_cfg.py:
    squat_target = default_qpos.copy()
    squat_target[1] = 0.0
    squat_target[2] = -0.4079
    squat_target[3] = 1.20      # Deep knee bend (~69°)
    squat_target[4] = 0.05
    squat_target[10] = 0.0
    squat_target[11] = 0.4079
    squat_target[12] = -1.20
    squat_target[13] = -0.05

    # Full extension push target
    push_target = default_qpos.copy()
    push_target[2] = -0.15      # straight hip
    push_target[3] = 0.00       # fully straight knee
    push_target[4] = 0.15
    push_target[11] = 0.15
    push_target[12] = 0.00
    push_target[13] = -0.15

    # Flight tuck target (bring feet up in air)
    tuck_target = default_qpos.copy()
    tuck_target[2] = -0.50
    tuck_target[3] = 0.80
    tuck_target[4] = 0.40
    tuck_target[11] = 0.50
    tuck_target[12] = -0.80
    tuck_target[13] = -0.40

    # Landing target
    land_target = default_qpos.copy()

    # Step-by-step trajectory:
    # 0.00 - 0.20s: smooth crouch down to squat_target (10 steps)
    # 0.20 - 0.26s: hold bottom of squat & load motors (3 steps)
    # 0.26 - 0.40s: EXPLOSIVE THRUST to push_target (7 steps)
    # 0.40 - 0.60s: flight tuck
    # 0.60 - 0.90s: land and return to stand

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    print(f"\n{'Step':<5} | {'t(s)':<5} | {'TrunkZ(mm)':<10} | {'Vz(m/s)':<8} | {'L_Clr(mm)':<9} | {'R_Clr(mm)':<9} | {'Contacts':<8} | Phase")
    print("-" * 75)

    z_stand = data.qpos[2] * 1000.0
    max_vz = 0.0
    peak_z = z_stand
    min_z = z_stand
    max_min_clr = 0.0

    for step in range(45):
        t = (step + 1) * 0.02
        if step < 10:
            phase = "Squat Down"
            alpha = (step + 1) / 10.0
            target = (1.0 - alpha) * default_qpos + alpha * squat_target
        elif step < 13:
            phase = "Squat Bottom"
            target = squat_target
        elif step < 20:
            phase = "EXPLOSIVE THRUST"
            target = push_target
        elif step < 30:
            phase = "Aerial Flight"
            target = tuck_target
        else:
            phase = "Land & Stand"
            target = land_target

        bam_ctrl.q_target[:] = target
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        trunk_z = data.qpos[2] * 1000.0
        vz = data.qvel[2]

        l_foot_z = data.site_xpos[lfoot_id][2]
        r_foot_z = data.site_xpos[rfoot_id][2]
        l_clr = max(0.0, l_foot_z - 0.0147) * 1000.0
        r_clr = max(0.0, r_foot_z - 0.0147) * 1000.0
        min_clr = min(l_clr, r_clr)

        if trunk_z < min_z: min_z = trunk_z
        if trunk_z > peak_z: peak_z = trunk_z
        if vz > max_vz: max_vz = vz
        if min_clr > max_min_clr: max_min_clr = min_clr

        lf = rf = False
        for c in range(data.ncon):
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom1) or ''
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom2) or ''
            if 'left_foot' in g1 or 'left_foot' in g2: lf = True
            if 'right_foot' in g1 or 'right_foot' in g2: rf = True

        con_str = f"{'L' if lf else '.'}{'R' if rf else '.'}"

        if step % 2 == 0 or (12 <= step <= 25):
            print(f"{step:<5} | {t:<5.2f} | {trunk_z:<10.1f} | {vz:<+8.3f} | {l_clr:<9.1f} | {r_clr:<9.1f} | {con_str:<8} | {phase}")

    print("\n--- TRAJECTORY SUMMARY ---")
    print(f"Standing Z:            {z_stand:.1f} mm")
    print(f"Deep Squat Min Z:      {min_z:.1f} mm (Squat Depth: {z_stand - min_z:.1f} mm)")
    print(f"Max Takeoff Upward Vz: {max_vz:+.3f} m/s")
    print(f"Peak Trunk Height:     {peak_z:.1f} mm (Trunk Rise: {peak_z - z_stand:+.1f} mm)")
    print(f"BILATERAL CLEARANCE:   {max_min_clr:.1f} mm")

if __name__ == '__main__':
    test_sit_to_jump()

import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_perfect_jump():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    default_qpos = data.qpos[7:].copy()

    # In standing (HOME pose):
    # left_hip_pitch (2): -0.4579
    # left_knee (3):      -0.0049
    # left_ankle (4):     +0.4530
    #
    # When squatting (Countermovement):
    # To keep the trunk vertical and CoM directly over the center of the feet:
    # As the knee flexes forward by +theta, the shank tilts forward.
    # To keep the foot flat on the ground, the ankle must dorsiflex by +theta!
    # And to keep the trunk vertical, the hip must flex to match!
    # Let's test pure knee + ankle flexion:
    
    # Crouch depth (e.g. knee flex = 0.8 rad ~ 46°)
    theta = 0.70 # rad
    crouch_qpos = default_qpos.copy()
    crouch_qpos[2] = -0.4579 - theta * 0.15  # slight hip flexion to keep CoM centered
    crouch_qpos[3] = -0.0049 + theta         # knee flex forward
    crouch_qpos[4] = +0.4530 + theta * 0.85  # ankle flex to keep foot flat
    
    crouch_qpos[11] = +0.4579 + theta * 0.15
    crouch_qpos[12] = +0.0049 - theta
    crouch_qpos[13] = -0.4530 - theta * 0.85
    
    # Push target: explosive extension back to slightly above standing / fully straight
    push_qpos = default_qpos.copy()
    push_qpos[3] = 0.00
    push_qpos[12] = 0.00

    # Tuck target: in air, tuck legs to maximize foot clearance
    tuck_qpos = default_qpos.copy()
    tuck_qpos[3] = 0.70
    tuck_qpos[4] = 0.60
    tuck_qpos[12] = -0.70
    tuck_qpos[13] = -0.60

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    z_stand = data.qpos[2] * 1000.0
    print(f"Settled Standing Z: {z_stand:.1f} mm")

    print(f"\n{'Step':<5} | {'t(s)':<5} | {'TrunkZ(mm)':<10} | {'Vz(m/s)':<8} | {'L_Clr(mm)':<9} | {'R_Clr(mm)':<9} | {'Pitch(°)':<8} | {'Contacts':<8} | Phase")
    print("-" * 85)

    records = []
    
    for step in range(50): # 1.0s @ 50 Hz
        t = (step + 1) * 0.02
        
        if step < 8:
            phase = "Squat Dip"
            # Ramp down to crouch in 8 steps (0.16s)
            alpha = (step + 1) / 8.0
            target = (1.0 - alpha) * default_qpos + alpha * crouch_qpos
        elif step < 11:
            phase = "Hold Bottom"
            target = crouch_qpos
        elif step < 18:
            phase = "EXPLOSIVE PUSH"
            target = push_qpos
        elif step < 26:
            phase = "AERIAL TUCK"
            target = tuck_qpos
        else:
            phase = "LAND & STAND"
            target = default_qpos

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

        quat = data.qpos[3:7]
        w, x, y, z = quat
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))

        lf = rf = False
        for c in range(data.ncon):
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom1) or ''
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom2) or ''
            if 'left_foot' in g1 or 'left_foot' in g2: lf = True
            if 'right_foot' in g1 or 'right_foot' in g2: rf = True

        con_str = f"{'L' if lf else '.'}{'R' if rf else '.'}"

        records.append({
            'step': step, 't': t, 'trunk_z': trunk_z, 'vz': vz,
            'l_clr': l_clr, 'r_clr': r_clr, 'pitch': pitch,
            'lf': lf, 'rf': rf
        })

        if step <= 32:
            print(f"{step:<5} | {t:<5.2f} | {trunk_z:<10.1f} | {vz:<+8.3f} | {l_clr:<9.1f} | {r_clr:<9.1f} | {pitch:<+8.1f} | {con_str:<8} | {phase}")

    min_crouch = min(r['trunk_z'] for r in records[:15])
    max_vz = max(r['vz'] for r in records[:20])
    peak_z = max(r['trunk_z'] for r in records)
    peak_l = max(r['l_clr'] for r in records)
    peak_r = max(r['r_clr'] for r in records)
    peak_min = max(min(r['l_clr'], r['r_clr']) for r in records)

    print("\n--- RESULTS ---")
    print(f"Standing Z:            {z_stand:.1f} mm")
    print(f"Squat Bottom Z:        {min_crouch:.1f} mm (Dip: {z_stand - min_crouch:.1f} mm)")
    print(f"Max Takeoff Vz:        {max_vz:+.3f} m/s")
    print(f"Peak Trunk Height:     {peak_z:.1f} mm (Trunk Rise: {peak_z - z_stand:+.1f} mm)")
    print(f"BILATERAL CLEARANCE:   {peak_min:.1f} mm")
    print(f"Left Foot Peak:        {peak_l:.1f} mm, Right Foot Peak: {peak_r:.1f} mm")

if __name__ == '__main__':
    test_perfect_jump()

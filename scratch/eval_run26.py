import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def evaluate_run26(onnx_path, name="Run 26"):
    print(f"\n{'='*80}")
    print(f"EVALUATING {name}: {onnx_path}")
    print(f"{'='*80}")
    
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_path,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=1.0,
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    
    # Settle in standing for 50 steps (1.0s)
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    z_stand = data.qpos[2]
    print(f"Settled Standing Trunk Z: {z_stand*1000.0:.1f} mm")

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    policy.trigger_behavior('jump')

    print(f"\n{'Step':<5} | {'t(s)':<5} | {'TrunkZ(mm)':<10} | {'Vz(m/s)':<8} | {'L_Clr(mm)':<9} | {'R_Clr(mm)':<9} | {'MinClr(mm)':<10} | {'L_Knee(°)':<9} | {'R_Knee(°)':<9} | {'Contacts':<8}")
    print("-" * 95)

    records = []
    
    for step in range(50): # 1.0s @ 50 Hz
        t = (step + 1) * 0.02
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        trunk_z = data.qpos[2]
        vz = data.qvel[2]

        # Foot clearance above floor (sole site offset ~0.0147)
        l_foot_z = data.site_xpos[lfoot_id][2]
        r_foot_z = data.site_xpos[rfoot_id][2]
        l_clr = max(0.0, l_foot_z - 0.0147) * 1000.0
        r_clr = max(0.0, r_foot_z - 0.0147) * 1000.0
        min_clr = min(l_clr, r_clr)

        # Knee angles in degrees
        lk_rad = data.qpos[7+3]
        rk_rad = data.qpos[7+12]
        lk_deg = math.degrees(lk_rad)
        rk_deg = math.degrees(rk_rad)

        # Contacts
        lf_con = rf_con = False
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ''
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ''
            if 'left_foot' in g1 or 'left_foot' in g2: lf_con = True
            if 'right_foot' in g1 or 'right_foot' in g2: rf_con = True

        con_str = f"{'L' if lf_con else '.'}{'R' if rf_con else '.'}"

        records.append({
            'step': step,
            't': t,
            'trunk_z': trunk_z * 1000.0,
            'vz': vz,
            'l_clr': l_clr,
            'r_clr': r_clr,
            'min_clr': min_clr,
            'lk_deg': lk_deg,
            'rk_deg': rk_deg,
            'lf_con': lf_con,
            'rf_con': rf_con,
        })

        if step < 50:
            print(f"{step:<5} | {t:<5.2f} | {trunk_z*1000.0:<10.1f} | {vz:<+8.3f} | {l_clr:<9.1f} | {r_clr:<9.1f} | {min_clr:<10.1f} | {lk_deg:<+9.1f} | {rk_deg:<+9.1f} | {con_str:<8}")

    # Summary
    crouch_min_z = min(r['trunk_z'] for r in records[:15])
    max_vz = max(r['vz'] for r in records[8:25])
    peak_trunk_z = max(r['trunk_z'] for r in records)
    peak_min_clr = max(r['min_clr'] for r in records)
    peak_l_clr = max(r['l_clr'] for r in records)
    peak_r_clr = max(r['r_clr'] for r in records)
    
    air_steps = sum(1 for r in records if not r['lf_con'] and not r['rf_con'])

    print("\n--- KPI SUMMARY ---")
    print(f"Crouch Dip:          {z_stand*1000.0 - crouch_min_z:.1f} mm (Min Z: {crouch_min_z:.1f} mm)")
    print(f"Takeoff Max Vz:      {max_vz:+.3f} m/s")
    print(f"Peak Trunk Rise:     {peak_trunk_z - z_stand*1000.0:+.1f} mm (Peak Z: {peak_trunk_z:.1f} mm)")
    print(f"Left Foot Peak Clr:  {peak_l_clr:.1f} mm")
    print(f"Right Foot Peak Clr: {peak_r_clr:.1f} mm")
    print(f"BILATERAL CLEARANCE: {peak_min_clr:.1f} mm (Min of both feet off ground)")
    print(f"Both-Feet-Air Time:  {air_steps * 0.02:.3f} s ({air_steps} steps)")
    
    # Check hyperextension during air
    max_rk_hyp = max(r['rk_deg'] for r in records[10:35])
    print(f"Right Knee Hyperextension Peak: {max_rk_hyp:+.1f}° (target <= 2.3°)")
    print(f"Final Step Trunk Z:  {records[-1]['trunk_z']:.1f} mm")

if __name__ == '__main__':
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'policies/jump.onnx'
    name = sys.argv[2] if len(sys.argv) > 2 else 'Policy'
    evaluate_run26(onnx_path, name)

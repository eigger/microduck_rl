import sys, math
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def evaluate_policy(name, onnx_path, jump_dur=0.55):
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN)

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_path,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=jump_dur,
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    z_stand = data.qpos[2]
    policy.trigger_behavior('jump')

    min_z = z_stand
    peak_z = z_stand
    max_vz = 0.0
    air_steps = 0
    max_foot_fwd = -999.0
    jump_count = 0
    in_jump = False
    max_foot_clearance = 0.0
    max_lknee_fwd = 0.0
    max_rknee_fwd = 0.0
    max_rknee_hyperext = 0.0

    for step in range(60): # 1.2s
        t = (step + 1) * 0.02
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        
        z = data.qpos[2]
        vz = data.qvel[2]
        
        if z < min_z: min_z = z
        if z > peak_z: peak_z = z
        if vz > max_vz: max_vz = vz

        lf = rf = False
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ''
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ''
            if 'left_foot' in g1 or 'left_foot' in g2: lf = True
            if 'right_foot' in g1 or 'right_foot' in g2: rf = True
        
        is_air = (not lf and not rf)
        if step >= 6: # after crouch dip
            if is_air:
                air_steps += 1
                if not in_jump:
                    jump_count += 1
                    in_jump = True
                foot_z_pts = [data.geom_xpos[g][2] for g in range(model.ngeom) if 'foot' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
                if foot_z_pts:
                    min_f = min(foot_z_pts)
                    if min_f > max_foot_clearance:
                        max_foot_clearance = min_f
                lk_fwd = max(0.0, data.qpos[7+3])
                rk_fwd = max(0.0, -data.qpos[7+12])
                rk_hyp = max(0.0, data.qpos[7+12])
                if lk_fwd > max_lknee_fwd: max_lknee_fwd = lk_fwd
                if rk_fwd > max_rknee_fwd: max_rknee_fwd = rk_fwd
                if rk_hyp > max_rknee_hyperext: max_rknee_hyperext = rk_hyp
            else:
                if in_jump and step > 10:
                    in_jump = False

        # foot forward displacement
        trunk_x = data.xpos[model.body('trunk_base').id][0]
        lfoot_x = data.xpos[model.body('ankle_left').id][0]
        rfoot_x = data.xpos[model.body('ankle_right').id][0]
        foot_fwd = (0.5 * (lfoot_x + rfoot_x) - trunk_x) * 1000.0 # mm
        if is_air and foot_fwd > max_foot_fwd:
            max_foot_fwd = foot_fwd

    quat = data.qpos[3:7]
    w, x, y, z = quat
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))

    print(f"\n=== Results for {name} ({onnx_path}) ===")
    print(f"Standing Z: {z_stand:.4f} m")
    print(f"Crouch Min Z: {min_z:.4f} m (Dip: {(z_stand - min_z)*1000:.1f} mm)")
    print(f"Takeoff Max Vz: +{max_vz:.4f} m/s")
    print(f"Apex Peak Z: {peak_z:.4f} m (Rise: {(peak_z - z_stand)*1000:.1f} mm)")
    print(f"Foot Ground Clearance: {max_foot_clearance*1000:.1f} mm")
    print(f"Air Time: {air_steps * 0.02:.3f} s ({air_steps} steps)")
    print(f"Aerial Forward Knee Tuck: L={max_lknee_fwd:.2f} rad, R={max_rknee_fwd:.2f} rad (diff={abs(max_lknee_fwd-max_rknee_fwd):.2f})")
    print(f"Right Knee Hyperextension in Air: {max_rknee_hyperext:.2f} rad (target: 0.00)")
    print(f"Max Foot Forward in Air: {max_foot_fwd:.1f} mm")
    print(f"Real Jump Count: {jump_count}")
    print(f"Final Posture (t=1.2s): Z={data.qpos[2]:.4f}m, Pitch={pitch:.1f} deg, Roll={roll:.1f} deg")

if __name__ == '__main__':
    import os
    if os.path.exists('scratch/jump_run14.onnx'):
        evaluate_policy('Run 14 (Baseline)', 'scratch/jump_run14.onnx', jump_dur=0.55)
    if os.path.exists('scratch/jump_run18.onnx'):
        evaluate_policy('Run 18 (Hyperextension)', 'scratch/jump_run18.onnx', jump_dur=0.55)
    if os.path.exists('scratch/jump_run19.onnx'):
        evaluate_policy('Run 19 (100% Symmetric, Timid Push)', 'scratch/jump_run19.onnx', jump_dur=0.55)
    if os.path.exists('scratch/jump_run20.onnx'):
        evaluate_policy('Run 20 (Explosive Push + Symmetric Tuck)', 'scratch/jump_run20.onnx', jump_dur=0.55)

import sys, os, math
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def diagnose():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path='policies/jump.onnx',
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=0.50,
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # Settle in standing
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    policy.trigger_behavior('jump')

    print(f"{'Step':<5} | {'t(s)':<5} | {'TrunkZ':<7} | {'Vz':<7} | {'L_Knee_q':<9} | {'L_Knee_tgt':<10} | {'L_Knee_tau':<10} | {'L_Ank_tau':<9} | {'GRF(N)':<7}")
    print("-" * 85)

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")

    for step in range(25):
        t = (step + 1) * 0.02
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)

        # Record torques and forces across the 4 physics sub-steps
        sub_taus = []
        sub_grfs = []
        for _ in range(4):
            bam_ctrl.update()
            sub_taus.append(data.qfrc_actuator.copy())
            
            # Ground contact forces
            grf = 0.0
            for c in range(data.ncon):
                con = data.contact[c]
                c_f = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, c_f)
                grf += c_f[0] # normal force
            sub_grfs.append(grf)
            mujoco.mj_step(model, data)

        avg_tau = np.mean(sub_taus, axis=0)
        avg_grf = np.mean(sub_grfs)

        trunk_z = data.qpos[2] * 1000.0
        vz = data.qvel[2]

        lk_q = math.degrees(data.qpos[7+3])
        lk_tgt = math.degrees(bam_ctrl.q_target[3])
        lk_tau = avg_tau[6+3]
        la_tau = avg_tau[6+4]

        print(f"{step:<5} | {t:<5.2f} | {trunk_z:<7.1f} | {vz:<+7.3f} | {lk_q:<+9.1f} | {lk_tgt:<+10.1f} | {lk_tau:<+10.3f} | {la_tau:<+9.3f} | {avg_grf:<7.1f}")

if __name__ == '__main__':
    diagnose()

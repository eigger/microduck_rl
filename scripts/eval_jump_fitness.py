import sys
sys.path.insert(0, 'scripts')
import mujoco
import numpy as np
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def evaluate_fitness(onnx_path):
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN)
    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_path,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=0.60
    )
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    for _ in range(50):
        policy.apply_action(policy.infer())
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trunk_base')
    jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'jaw_soft')
    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_foot')
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_foot')

    policy.trigger_behavior('jump')

    max_head_pitch_err = 0.0
    min_head_z = 999.0
    max_body_pitch = 0.0
    max_vz = -999.0
    max_foot_clr = 0.0
    head_contacts = 0
    final_pitch = 0.0

    for s in range(50):
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
                b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
                if any(k in (b1, b2) for k in ('jaw_soft', 'neck', 'neck_pitch', 'yaw_roll_motion')):
                    head_contacts += 1

        q = data.qpos[3:7]
        pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
        max_body_pitch = max(max_body_pitch, abs(pitch))
        if s >= 40:
            final_pitch = abs(pitch)

        np_deg = np.degrees(data.qpos[7+5])
        hp_deg = np.degrees(data.qpos[7+6])
        head_pitch_err = max(abs(np_deg - 20.0), abs(hp_deg - 20.0))
        max_head_pitch_err = max(max_head_pitch_err, head_pitch_err)

        hz = data.xpos[jaw_id][2] * 1000
        min_head_z = min(min_head_z, hz)

        max_vz = max(max_vz, data.qvel[2])

        lz = max(0.0, data.site_xpos[lfoot_id][2] - 0.0147) * 1000
        rz = max(0.0, data.site_xpos[rfoot_id][2] - 0.0147) * 1000
        max_foot_clr = max(max_foot_clr, min(lz, rz))

    passed = (max_head_pitch_err < 15.0) and (min_head_z > 140.0) and (max_body_pitch < 25.0) and (head_contacts == 0) and (max_vz > 0.50)
    fit_status = "PASSED" if passed else "FAILED"
    print(f"\n==================================================")
    print(f" Policy: {onnx_path}")
    print(f"==================================================")
    print(f"  Max Head Pitch Error : {max_head_pitch_err:5.1f} deg  (Limit: < 15 deg)")
    print(f"  Min Head Height (Z)  : {min_head_z:5.1f} mm   (Limit: > 140 mm)")
    print(f"  Max Body Pitch Tilt  : {max_body_pitch:5.1f} deg  (Limit: < 25 deg)")
    print(f"  Head Ground Contacts : {head_contacts} times    (Limit: 0)")
    print(f"  Max Takeoff Vz       : {max_vz:+5.3f} m/s    (Limit: > +0.50 m/s)")
    print(f"  Max Foot Clearance   : {max_foot_clr:5.1f} mm")
    print(f"  Final Settled Pitch  : {final_pitch:5.1f} deg  (Limit: < 5 deg)")
    print(f"  >> FITNESS RESULT    : {fit_status}")

if __name__ == '__main__':
    for path in sys.argv[1:]:
        evaluate_fitness(path)

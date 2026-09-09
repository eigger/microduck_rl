import sys, os, math
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def analyze_trajectory(onnx_name):
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )
    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_name,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=0.48,
    )
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    policy.trigger_behavior('jump')
    print(f"\n{'Step':>4} | {'t(s)':>5} | {'Z(mm)':>6} | {'Vz(m/s)':>7} | {'Pitch(deg)':>10} | {'LClr(mm)':>8} | {'RClr(mm)':>8} | {'LKnee':>6} | {'RKnee':>6} | {'NeckP':>6} | {'HeadP':>6}")
    print("-" * 88)

    for step in range(45):
        t = step * 0.02
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        z = data.qpos[2] * 1000.0
        vz = data.qvel[2]
        
        # Euler pitch from quat [w, x, y, z]
        qw, qx, qy, qz = data.qpos[3:7]
        # pitch (around y axis): sin(pitch) = 2*(qw*qy - qz*qx)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

        l_clr = max(0.0, data.site_xpos[lfoot_id][2] - 0.0147) * 1000.0
        r_clr = max(0.0, data.site_xpos[rfoot_id][2] - 0.0147) * 1000.0

        # Servo joint positions:
        # left: hip_pitch=2, knee=3, ankle=4
        # right: hip_pitch=11, knee=12, ankle=13
        lhp = data.qpos[7 + 2]
        rhp = data.qpos[7 + 11]
        lknee = data.qpos[7 + 3]
        rknee = data.qpos[7 + 12]
        lankle = data.qpos[7 + 4]
        rankle = data.qpos[7 + 13]

        lak_act = act[4]
        rak_act = act[13]
        lkn_act = act[3]
        rkn_act = act[12]
        print(f"{step:2d}|{t:4.2f}|{z:5.1f}|{vz:+6.2f}|{l_clr:5.1f}|{r_clr:5.1f}|kn:{lknee:+5.2f},{rknee:+5.2f}|ak:{lankle:+5.2f},{rankle:+5.2f}")

print("=== RUN 26 TRAJECTORY ===")
analyze_trajectory("scratch/jump_run26.onnx")

print("\n=== RUN 38 (iter 2000 - Tuck Gate) TRAJECTORY ===")
analyze_trajectory("scratch/jump_run38_2000.onnx")

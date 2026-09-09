import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def main():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path='policies/jump.onnx',
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=1.0,
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    policy.trigger_behavior('jump')

    print(f"{'Step':<5} | {'t(s)':<5} | {'TrunkZ(mm)':<10} | {'Pitch(°)':<9} | {'Roll(°)':<8} | {'X_pos(m)':<9} | {'NeckPitch':<10} | {'HeadPitch':<10}")
    print("-" * 80)

    for step in range(60):
        t = (step + 1) * 0.02
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        q = data.qpos[3:7] # quat w, x, y, z
        # Pitch = rotation around Y axis
        # sin(pitch) = 2*(w*y - z*x)
        sinp = 2.0 * (q[0]*q[2] - q[3]*q[1])
        pitch_deg = math.degrees(math.asin(np.clip(sinp, -1.0, 1.0)))
        # Roll = rotation around X axis
        sinr = 2.0 * (q[0]*q[1] + q[2]*q[3])
        cosr = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])
        roll_deg = math.degrees(math.atan2(sinr, cosr))

        trunk_z = data.qpos[2] * 1000.0
        x_pos = data.qpos[0]
        neck_p = data.qpos[7+5] # neck pitch
        head_p = data.qpos[7+6] # head pitch

        print(f"{step:<5} | {t:<5.2f} | {trunk_z:<10.1f} | {pitch_deg:<+9.1f} | {roll_deg:<+8.1f} | {x_pos:<+9.3f} | {math.degrees(neck_p):<+10.1f} | {math.degrees(head_p):<+10.1f}")

if __name__ == '__main__':
    main()

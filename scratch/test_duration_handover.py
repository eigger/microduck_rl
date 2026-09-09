import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

for dur in [0.38, 0.42, 0.46, 0.50, 0.55]:
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )
    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path='policies/jump.onnx',
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=dur,
    )
    mujoco.mj_resetDataKeyframe(model, data, 1)
    for _ in range(50):
        policy.apply_action(policy.infer())
        for _ in range(4): bam_ctrl.update(); mujoco.mj_step(model, data)
    
    policy.trigger_behavior('jump')
    fallen = False
    for step in range(75):
        policy.update_behavior(0.02)
        policy.apply_action(policy.infer())
        for _ in range(4): bam_ctrl.update(); mujoco.mj_step(model, data)
        q = data.qpos[3:7]
        sinp = 2.0 * (q[0]*q[2] - q[3]*q[1])
        pitch = math.degrees(math.asin(np.clip(sinp, -1.0, 1.0)))
        if abs(pitch) > 35.0 or data.qpos[2] < 0.06:
            fallen = True
            break
    print(f"jump_duration={dur:.2f}s -> {'FALLEN at step ' + str(step) if fallen else 'SURVIVED upright!'}")

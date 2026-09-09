import sys
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
model, data, bam_ctrl, _ = load_mujoco_with_bam(MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN)

policy = PolicyInference(
    model, data, bam_ctrl=bam_ctrl,
    walking_onnx_path='policies/alpha_walking.onnx',
    standing_onnx_path='policies/alpha_stand.onnx',
    jump_onnx_path='policies/jump.onnx',
    new_cmd_obs=True, use_projected_gravity=True, jump_duration=0.55,
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

print("Step |  Time |  Z(mm) | Vz(m/s) | L_knee_act | R_knee_act | L_knee_qpos | L_knee_tau | Max_tau(Nm)")
print("-" * 85)

for step in range(30):
    t = (step + 1) * 0.02
    policy.update_behavior(0.02)
    act = policy.infer()
    policy.apply_action(act)
    
    # Check max torque during this step
    max_tau = 0.0
    for _ in range(4):
        bam_ctrl.update()
        mujoco.mj_step(model, data)
        tau_mag = np.max(np.abs(data.ctrl))
        if tau_mag > max_tau:
            max_tau = tau_mag
            
    z = data.qpos[2] * 1000.0
    vz = data.qvel[2]
    lk_act = act[3]
    rk_act = act[12]
    lk_qpos = data.qpos[7+3]
    lk_tau = data.ctrl[3]
    
    print(f"{step:4d} | {t:5.2f} | {z:6.1f} | {vz:7.3f} | {lk_act:10.3f} | {rk_act:10.3f} | {lk_qpos:11.3f} | {lk_tau:10.3f} | {max_tau:11.3f}")

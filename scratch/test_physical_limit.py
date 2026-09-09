import sys
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
model, data, bam_ctrl, _ = load_mujoco_with_bam(MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN)

# Keyframe 1: standing
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)

# Let's test a maximal physical jump trajectory:
# 1. Deep squat: hip pitch, knee, ankle
# 2. Maximum extension: push against the ground with full torque/velocity
# 3. Flight tuck: tuck knees

# Stand default
default_qpos = data.qpos[7:].copy()
print("Starting physical limits test...")

# Phase 1: Settle in deep squat (t=0 to 0.25s)
crouch_qpos = default_qpos.copy()
# Left leg: hip_yaw(0), hip_roll(1), hip_pitch(2), knee(3), ankle(4)
# Right leg: hip_yaw(9), hip_roll(10), hip_pitch(11), knee(12), ankle(13)
crouch_qpos[2] = -0.80   # hip pitch flex
crouch_qpos[3] = 0.90    # knee flex
crouch_qpos[4] = 0.70    # ankle flex
crouch_qpos[11] = 0.80   # right hip pitch
crouch_qpos[12] = -0.90  # right knee flex
crouch_qpos[13] = -0.70  # right ankle flex

# Phase 2: Full extension (push off)
push_qpos = default_qpos.copy()
push_qpos[2] = -0.10
push_qpos[3] = 0.00
push_qpos[4] = 0.10
push_qpos[11] = 0.10
push_qpos[12] = 0.00
push_qpos[13] = -0.10

# Phase 3: Aerial tuck
tuck_qpos = default_qpos.copy()
tuck_qpos[2] = -0.60
tuck_qpos[3] = 0.80
tuck_qpos[4] = 0.60
tuck_qpos[11] = 0.60
tuck_qpos[12] = -0.80
tuck_qpos[13] = -0.60

z_init = data.qpos[2]
max_z = z_init
max_vz = 0.0
max_clearance = 0.0

for step in range(150): # 0.75s total
    t = step * 0.005
    if t < 0.20:
        target = crouch_qpos
    elif t < 0.32:
        target = push_qpos
    elif t < 0.55:
        target = tuck_qpos
    else:
        target = default_qpos
    
    bam_ctrl.q_target[:] = target
    for _ in range(4):
        bam_ctrl.update()
        mujoco.mj_step(model, data)
    
    z = data.qpos[2]
    vz = data.qvel[2]
    if z > max_z: max_z = z
    if vz > max_vz: max_vz = vz
    
    foot_pts = [data.geom_xpos[g][2] for g in range(model.ngeom) if 'foot' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
    if foot_pts:
        clearance = min(foot_pts) - 0.0086
        if clearance > max_clearance:
            max_clearance = clearance

print(f"Maximal Physical Test Results:")
print(f"Initial Z: {z_init*1000:.1f} mm")
print(f"Max Takeoff Vz: {max_vz:.3f} m/s")
print(f"Peak Z: {max_z*1000:.1f} mm (Rise: {(max_z - z_init)*1000:.1f} mm)")
print(f"Max Actual Foot Clearance: {max_clearance*1000:.1f} mm")

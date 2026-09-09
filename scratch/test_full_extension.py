import sys
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
model, data, bam_ctrl, _ = load_mujoco_with_bam(MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN)

mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)

# Let's test deep crouch and full extension push:
# Default pose
default_qpos = data.qpos[7:].copy()

# Deep crouch pose: hip pitch, knee, ankle
crouch_qpos = default_qpos.copy()
crouch_qpos[2] = -0.75   # hip pitch
crouch_qpos[3] = 0.85    # knee flex
crouch_qpos[4] = 0.65    # ankle
crouch_qpos[11] = 0.75   # right hip pitch
crouch_qpos[12] = -0.85  # right knee flex
crouch_qpos[13] = -0.65  # right ankle

# Extension pose: straight legs
push_qpos = default_qpos.copy()
push_qpos[2] = -0.10
push_qpos[3] = -0.02   # full extension
push_qpos[4] = 0.05
push_qpos[11] = 0.10
push_qpos[12] = 0.02
push_qpos[13] = -0.05

# 1. Settle in crouch (50 steps = 0.25s)
for _ in range(50):
    bam_ctrl.q_target[:] = crouch_qpos
    for _ in range(4):
        bam_ctrl.update()
        mujoco.mj_step(model, data)

z_crouch = data.qpos[2]
print(f"Deep Crouch Z: {z_crouch*1000:.1f} mm")

# 2. Explosive push! (target = push_qpos)
max_vz = 0.0
peak_z = z_crouch
air_time = 0.0
max_clearance = 0.0

for step in range(60): # 0.30s
    t = step * 0.005 * 4
    if t < 0.14:
        bam_ctrl.q_target[:] = push_qpos
    else:
        # hold push pose in air
        bam_ctrl.q_target[:] = push_qpos
        
    for _ in range(4):
        bam_ctrl.update()
        mujoco.mj_step(model, data)
        
    z = data.qpos[2]
    vz = data.qvel[2]
    if z > peak_z: peak_z = z
    if vz > max_vz: max_vz = vz
    
    foot_pts = [data.geom_xpos[g][2] for g in range(model.ngeom) if 'foot' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
    if foot_pts:
        clr = min(foot_pts) - 0.0086
        if clr > max_clearance: max_clearance = clr
        if clr > 0.002: air_time += 0.005 * 4

print(f"Explosive Full Extension Results:")
print(f"Takeoff Max Vz: {max_vz:.3f} m/s")
print(f"Peak Z: {peak_z*1000:.1f} mm (Rise above stand: {(peak_z - 0.1166)*1000:.1f} mm)")
print(f"Peak Rise above crouch: {(peak_z - z_crouch)*1000:.1f} mm")
print(f"Max Actual Foot Clearance: {max_clearance*1000:.1f} mm")
print(f"Air Time: {air_time:.3f} s")

import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_motor_limit():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )
    
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    
    # Standing height
    z0 = data.qpos[2]
    print(f"Stand Z: {z0*1000:.1f} mm")
    
    # Try different thrust targets from settled crouch
    best_vz = 0.0
    best_z = 0.0
    best_config = None
    
    # Search thrust joint angles:
    # Hip pitch: [-0.4579 to +0.2]
    # Knee: [-0.2 to +0.2]
    # Ankle: [-0.2 to +0.8]
    # Head/Neck pitch: swing up [0.0 to -0.5]
    
    for hip_t in [-0.2, 0.0, +0.2]:
        for knee_t in [-0.15, -0.05, +0.05]:
            for ankle_t in [+0.1, +0.3, +0.5]:
                for head_t in [0.35, 0.0, -0.2]:
                    # Reset
                    mujoco.mj_resetDataKeyframe(model, data, 1)
                    mujoco.mj_forward(model, data)
                    
                    # Crouch 0.2s
                    crouch = np.array([
                        0.0, -0.0873, -0.4579, 1.15, 1.60, 0.35, 0.35, 0.0, 0.0,
                        0.0,  0.0873,  0.4579, -1.15, -1.60
                    ])
                    for _ in range(15):
                        bam_ctrl.q_target[:] = crouch
                        for _ in range(4): bam_ctrl.update(); mujoco.mj_step(model, data)
                        
                    # Thrust
                    thrust = np.array([
                        0.0, -0.0873, hip_t, knee_t, ankle_t, head_t, head_t, 0.0, 0.0,
                        0.0,  0.0873, -hip_t, -knee_t, -ankle_t
                    ])
                    bam_ctrl.q_target[:] = thrust
                    
                    max_vz = 0.0
                    max_z = 0.0
                    for _ in range(25):
                        for _ in range(4): bam_ctrl.update(); mujoco.mj_step(model, data)
                        if data.qvel[2] > max_vz: max_vz = data.qvel[2]
                        if data.qpos[2] > max_z: max_z = data.qpos[2]
                        
                    if max_vz > best_vz:
                        best_vz = max_vz
                        best_z = max_z
                        best_config = (hip_t, knee_t, ankle_t, head_t)
                        
    print(f"\nBest Thrust Configuration:")
    print(f"  Hip={best_config[0]:.2f}, Knee={best_config[1]:.2f}, Ankle={best_config[2]:.2f}, Head={best_config[3]:.2f}")
    print(f"  Peak Vz = {best_vz:+.3f} m/s")
    print(f"  Peak Z  = {best_z*1000:.1f} mm (Rise: +{(best_z - z0)*1000:.1f} mm)")

if __name__ == '__main__':
    test_motor_limit()

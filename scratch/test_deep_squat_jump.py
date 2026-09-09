import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_squat_jump():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    
    default_qpos = data.qpos[7:].copy()
    
    # Let's test a sequence of squats to find the deepest stable squat:
    # In standing:
    # left_hip_pitch: -0.4579, left_knee: -0.0049, left_ankle: +0.4530
    # To squat while keeping CoM over the foot center (no falling forward or backward):
    # If knee flexes +delta, hip pitch must flex -delta_hip, ankle must flex +delta_ankle.
    # The 4-bar leg linkage keeps trunk vertical if delta_knee approx delta_hip + delta_ankle!
    
    print("Testing stable squat depth...")
    for flex in [0.3, 0.5, 0.7, 0.85, 1.0]:
        mujoco.mj_resetDataKeyframe(model, data, 1)
        mujoco.mj_forward(model, data)
        
        target = default_qpos.copy()
        # Left leg
        target[2] = -0.4579 - flex * 0.5   # hip pitch
        target[3] = -0.0049 + flex * 1.0   # knee flex (+ for left)
        target[4] = +0.4530 + flex * 0.5   # ankle dorsiflex
        # Right leg (mirrored)
        target[11] = +0.4579 + flex * 0.5
        target[12] = +0.0049 - flex * 1.0  # knee flex (- for right)
        target[13] = -0.4530 - flex * 0.5
        
        # Hold squat for 0.5s (25 steps)
        fallen = False
        for _ in range(25):
            bam_ctrl.q_target[:] = target
            for _ in range(4):
                bam_ctrl.update()
                mujoco.mj_step(model, data)
            # check pitch
            quat = data.qpos[3:7]
            w, x, y, z = quat
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
            if abs(pitch) > 30:
                fallen = True
                break
                
        trunk_z = data.qpos[2] * 1000.0
        print(f"Flex scale {flex:.2f} -> Trunk Z: {trunk_z:.1f} mm, Pitch: {pitch:.1f}°, Fallen: {fallen}")

if __name__ == '__main__':
    test_squat_jump()

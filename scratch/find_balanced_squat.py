import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def find_balanced_squat():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    default_qpos = data.qpos[7:].copy()

    # Grid search for balanced squat:
    # We want knee flexion from 0.6 to 1.1 rad (35° to 65°)
    # and find hip_pitch and ankle offsets that result in min |pitch| after settling
    
    best_combo = None
    best_pitch_err = 999.0
    
    print("Searching for perfectly balanced deep squat...")
    
    for knee_deg in [40, 50, 60, 70]:
        knee_rad = math.radians(knee_deg)
        for hip_ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for ankle_ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
                mujoco.mj_resetDataKeyframe(model, data, 1)
                mujoco.mj_forward(model, data)
                
                target = default_qpos.copy()
                d_knee = knee_rad
                d_hip = knee_rad * hip_ratio
                d_ank = knee_rad * ankle_ratio
                
                # Left leg
                target[2] = -0.4579 - d_hip
                target[3] = -0.0049 + d_knee
                target[4] = +0.4530 + d_ank
                # Right leg
                target[11] = +0.4579 + d_hip
                target[12] = +0.0049 - d_knee
                target[13] = -0.4530 - d_ank
                
                # Settle for 20 steps (0.4s)
                for _ in range(20):
                    bam_ctrl.q_target[:] = target
                    for _ in range(4):
                        bam_ctrl.update()
                        mujoco.mj_step(model, data)
                
                quat = data.qpos[3:7]
                w, x, y, z = quat
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
                roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
                z_mm = data.qpos[2] * 1000.0
                
                # Check if feet are on floor
                lf_on = rf_on = False
                for c in range(data.ncon):
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom1) or ''
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom2) or ''
                    if 'left_foot' in g1 or 'left_foot' in g2: lf_on = True
                    if 'right_foot' in g1 or 'right_foot' in g2: rf_on = True
                    
                if knee_deg == 50 and hip_ratio == 0.5:
                    print(f"Sample Knee {knee_deg}° (hip_r={hip_ratio:.1f}, ank_r={ankle_ratio:.1f}): Z={z_mm:.1f}mm, Pitch={pitch:+.1f}°, LF={lf_on}, RF={rf_on}")
                if lf_on and rf_on and abs(pitch) < 15.0:
                    print(f"Knee {knee_deg}° (hip_r={hip_ratio:.1f}, ank_r={ankle_ratio:.1f}): Z={z_mm:.1f}mm, Pitch={pitch:+.1f}°, Roll={roll:+.1f}° [BALANCED STABLE!]")
                    if abs(pitch) < best_pitch_err:
                        best_pitch_err = abs(pitch)
                        best_combo = (knee_deg, hip_ratio, ankle_ratio, z_mm, pitch)

    print(f"\nBest balanced squat: {best_combo}")

if __name__ == '__main__':
    find_balanced_squat()

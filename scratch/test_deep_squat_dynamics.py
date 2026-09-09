import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_squat_depths():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    default_qpos = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.453,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.453
    ])

    print("Testing stable squat profiles from standing:")
    
    # We want to find a squat where:
    # 1. Trunk Z drops to 70-80mm (deep visible squat!)
    # 2. Torso pitch stays upright (abs(pitch) < 15°)
    # 3. Feet remain flat on ground
    for knee_bend_deg in [45, 60, 75, 90]:
        knee_rad = math.radians(knee_bend_deg)
        # To squat vertically without tilting:
        # knee flexes forward: +knee_rad (left), -knee_rad (right)
        # ankle flexes forward: +ankle_rad (left), -ankle_rad (right)
        # hip flexes: -hip_rad (left), +hip_rad (right)
        # The key to balance: the hip must shift slightly BACK or torso lean slightly FORWARD so CoM stays over foot!
        for hip_delta_deg in [0, 5, 10, 15, 20]:
            for ank_ratio in [0.7, 0.85, 1.0]:
                mujoco.mj_resetDataKeyframe(model, data, 1)
                mujoco.mj_forward(model, data)
                
                hip_rad = math.radians(hip_delta_deg)
                ank_rad = knee_rad * ank_ratio
                
                target = default_qpos.copy()
                target[2] = -0.4579 + hip_rad   # hip pitch (+ reduces forward lean or leans torso)
                target[3] = -0.0049 + knee_rad  # knee flex
                target[4] = +0.4530 + ank_rad   # ankle flex
                
                target[11] = +0.4579 - hip_rad
                target[12] = +0.0049 - knee_rad
                target[13] = -0.4530 - ank_rad
                
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
                
                # Check foot contacts
                lf_on = rf_on = False
                for c in range(data.ncon):
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom1) or ''
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[c].geom2) or ''
                    if 'left_foot' in g1 or 'left_foot' in g2: lf_on = True
                    if 'right_foot' in g1 or 'right_foot' in g2: rf_on = True
                    
                if lf_on and rf_on and abs(pitch) < 12.0 and abs(roll) < 5.0 and z_mm < 90.0:
                    print(f"-> SUCCESS: Knee {knee_bend_deg}°, Hip_d {hip_delta_deg}°, Ank_r {ank_ratio:.2f} => Z={z_mm:.1f}mm, Pitch={pitch:+.1f}°, Roll={roll:+.1f}°")

if __name__ == '__main__':
    test_squat_depths()

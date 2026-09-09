import sys, math, os
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from infer_policy import load_bam_model, load_mujoco_with_bam, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def test_explosive_thrust():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    default_qpos = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.453,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.453
    ])

    knee_rad = math.radians(80) # 80 deg bend
    ank_rad = knee_rad * 1.0

    squat_target = default_qpos.copy()
    squat_target[3] = -0.0049 + knee_rad
    squat_target[4] = +0.4530 + ank_rad
    squat_target[12] = +0.0049 - knee_rad
    squat_target[13] = -0.4530 - ank_rad

    # Max explosive extension:
    # We command knees straight and ankles pushing down hard (plantarflexion)
    # Target overshoot beyond physical limits pushes BAM torque to the max (0.963 Nm limit)!
    for push_ank in [0.0, -0.3, -0.6]:
        for push_knee in [0.0, -0.2, -0.5]:
            push_target = default_qpos.copy()
            push_target[2] = -0.4579 - 0.2
            push_target[3] = push_knee
            push_target[4] = push_ank
            push_target[11] = 0.4579 + 0.2
            push_target[12] = -push_knee
            push_target[13] = -push_ank

            mujoco.mj_resetDataKeyframe(model, data, 1)
            mujoco.mj_forward(model, data)

            # Squat for 10 steps (0.2s)
            for _ in range(10):
                bam_ctrl.q_target[:] = squat_target
                for _ in range(4):
                    bam_ctrl.update()
                    mujoco.mj_step(model, data)

            z_crouch = data.qpos[2] * 1000.0

            # Explosive push for 6 steps (0.12s)
            max_vz = 0.0
            max_z = 0.0
            max_clearance = 0.0

            lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
            rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

            for step in range(30):
                if step < 6:
                    bam_ctrl.q_target[:] = push_target
                else:
                    # Aerial tuck
                    tuck_target = default_qpos.copy()
                    tuck_target[3] = 1.0
                    tuck_target[4] = 0.5
                    tuck_target[12] = -1.0
                    tuck_target[13] = -0.5
                    bam_ctrl.q_target[:] = tuck_target

                for _ in range(4):
                    bam_ctrl.update()
                    mujoco.mj_step(model, data)

                vz = data.qvel[2]
                z = data.qpos[2] * 1000.0
                l_clr = max(0.0, data.site_xpos[lfoot_id][2] - 0.0147) * 1000.0
                r_clr = max(0.0, data.site_xpos[rfoot_id][2] - 0.0147) * 1000.0
                clr = min(l_clr, r_clr)

                if vz > max_vz: max_vz = vz
                if z > max_z: max_z = z
                if clr > max_clearance: max_clearance = clr

            print(f"Push Knee={push_knee:.1f}, Push Ank={push_ank:.1f} | Crouch Z={z_crouch:.1f}mm | Max Vz={max_vz:+.3f}m/s | Peak Trunk Z={max_z:.1f}mm | Clearance={max_clearance:.1f}mm")

if __name__ == '__main__':
    test_explosive_thrust()

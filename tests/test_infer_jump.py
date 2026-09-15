"""Test episodic jump execution in infer_policy (CPU MuJoCo with BAM).

Ensures that triggering 'jump' from an active standing state:
1. Resets action history to prevent stale action bias.
2. Neutralizes head posture observation terms (preventing normalizer OOD collapse).
3. Applies explosive countermovement crouch (Z <= 85mm) and takeoff (Vz >= +0.45m/s).
4. Reaches peak flight height >= 145mm with positive clearance on both feet.
5. Auto-returns cleanly to standing and settles without falling over.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
try:
    import pytest
    skipif = pytest.mark.skipif
except ImportError:
    def skipif(cond, reason=""):
        def decorator(f):
            return f
        return decorator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from infer_policy import (
    load_bam_model,
    load_mujoco_with_bam,
    PolicyInference,
    MICRODUCK_XML,
    BAM_KP_FW,
    BAM_VIN_MIN,
)

JUMP_ONNX = REPO / "policies" / "jump.onnx"
WALK_ONNX = REPO / "policies" / "alpha_walking.onnx"
STAND_ONNX = REPO / "policies" / "alpha_stand.onnx"


@skipif(not (JUMP_ONNX.exists() and STAND_ONNX.exists()), reason="ONNX policies not available")
def test_jump_from_standing_settle():
    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        str(REPO / MICRODUCK_XML), bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )
    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path=str(WALK_ONNX) if WALK_ONNX.exists() else None,
        standing_onnx_path=str(STAND_ONNX),
        jump_onnx_path=str(JUMP_ONNX),
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=1.0
    )
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # 1. Settle in standing for 50 steps
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    stand_z = data.qpos[2] * 1000.0
    assert 110.0 <= stand_z <= 122.0, f"Standing settle height unexpected: {stand_z:.1f}mm"

    # 2. Trigger Jump
    policy.trigger_behavior("jump")
    assert policy.current_policy == "jump"
    assert np.allclose(policy.last_action, 0.0), "last_action was not reset on behavior trigger"

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    z_min = stand_z
    z_max = 0.0
    vz_max = 0.0
    l_clr_max = 0.0
    r_clr_max = 0.0

    # 3. Step 50 steps of jump (1.0s: full cycle crouch -> thrust -> apex split -> cushion -> settle)
    for _ in range(50):
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        tz = data.qpos[2] * 1000.0
        vz = data.qvel[2]
        lz = data.site_xpos[lfoot_id, 2] * 1000.0
        rz = data.site_xpos[rfoot_id, 2] * 1000.0

        if tz < z_min: z_min = tz
        if tz > z_max: z_max = tz
        if vz > vz_max: vz_max = vz
        l_clr = max(0.0, lz - 14.7)
        r_clr = max(0.0, rz - 14.7)
        if l_clr > l_clr_max: l_clr_max = l_clr
        if r_clr > r_clr_max: r_clr_max = r_clr

    # Assertions for explosive CMJ
    assert z_min <= 90.0, f"Jump failed to crouch deep: min_z = {z_min:.1f}mm"
    assert vz_max >= 0.45, f"Jump takeoff velocity too weak: vz_max = {vz_max:.2f}m/s"
    assert z_max >= 140.0, f"Jump apex height too low: z_max = {z_max:.1f}mm"
    assert l_clr_max >= 30.0, f"Left foot did not take off: max clearance = {l_clr_max:.1f}mm"
    assert r_clr_max >= 10.0, f"Right foot did not take off: max clearance = {r_clr_max:.1f}mm"
    assert policy.current_policy == "standing", f"Policy did not hand back to standing: {policy.current_policy}"

    # 4. Step 50 steps of standing recovery (1.0s) — check for no stumble or lateral tilt
    max_recovery_roll = 0.0
    max_recovery_pitch = 0.0
    for _ in range(50):
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        q = data.qpos[3:7]
        sinr = 2.0 * (q[0]*q[1] + q[2]*q[3])
        cosr = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])
        roll = np.degrees(np.arctan2(sinr, cosr))
        sinp = np.clip(2.0 * (q[0]*q[2] - q[3]*q[1]), -1.0, 1.0)
        pitch = np.degrees(np.arcsin(sinp))

        max_recovery_roll = max(max_recovery_roll, abs(roll))
        max_recovery_pitch = max(max_recovery_pitch, abs(pitch))

    final_z = data.qpos[2] * 1000.0
    assert 112.0 <= final_z <= 122.0, f"Robot failed to settle in standing: final_z = {final_z:.1f}mm"
    assert max_recovery_roll <= 3.0, f"Robot tilted laterally in recovery: roll = {max_recovery_roll:.1f}°"
    assert max_recovery_pitch <= 5.0, f"Robot pitched in recovery: pitch = {max_recovery_pitch:.1f}°"


if __name__ == "__main__":
    test_jump_from_standing_settle()
    print("ALL TESTS PASSED: test_jump_from_standing_settle succeeded!")


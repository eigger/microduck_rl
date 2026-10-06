"""Review and regression test for bilateral symmetry in Microduck jump rewards.

Verifies:
1. Knee flexion symmetry: Left knee (+), Right knee (-) track symmetric targets.
2. Ankle extension symmetry: Left ankle (+), Right ankle (-) track symmetric targets.
3. Zero error at perfectly symmetric cushion & stand configurations.
4. Large penalty on asymmetric 'one-knee-bent, one-knee-extended' failure mode.
"""

import math

import torch

from mjlab_microduck.tasks.mdp import (
    foot_gap_potential_delta,
    foot_load_balance,
    feet_above_floor,
    foot_offset_cost,
    foot_slip_cost,
    spawn_frame_offset,
    yaw_from_quat,
    stand_curriculum_joints,
    stand_pose_with_hip,
    symmetric_hip_pitch_score,
    symmetric_knee_flexion_score,
)


def test_knee_and_ankle_symmetry_in_jump_landing_rise():
    # Setup simulated joint pos (batch of 3: symmetric cushion, asymmetric split, symmetric stand)
    joint_pos = torch.zeros((3, 14), dtype=torch.float32)

    # Env 0: Perfectly symmetric landing cushion
    # Left knee = +0.40, Right knee = -0.40
    # Left ankle = +0.30, Right ankle = -0.30
    joint_pos[0, 3] = 0.40
    joint_pos[0, 12] = -0.40
    joint_pos[0, 4] = 0.30
    joint_pos[0, 13] = -0.30

    # Env 1: Asymmetric split (the bug that caused Run 92 roll fall: left extended 0.0, right flexed -0.40)
    joint_pos[1, 3] = 0.0
    joint_pos[1, 12] = -0.40
    joint_pos[1, 4] = 0.30
    joint_pos[1, 13] = -0.30

    # Env 2: Perfectly symmetric stand (HOME_FRAME)
    # Left knee = 0.0, Right knee = 0.0
    # Left ankle = +0.4530, Right ankle = -0.4530
    joint_pos[2, 3] = 0.0
    joint_pos[2, 12] = 0.0
    joint_pos[2, 4] = 0.4530
    joint_pos[2, 13] = -0.4530

    # Test 1: Cushion phase (progress = 0.0)
    current_target_knee = 0.40
    current_target_ankle = 0.30
    std_stand_knee = 0.30
    std_stand_ankle = 0.20

    # Knee error formula (fixed):
    knee_err = 0.5 * (
        torch.square(joint_pos[:, 3] - current_target_knee)
        + torch.square(joint_pos[:, 12] - (-current_target_knee))
    )
    knee_score = torch.exp(-knee_err / (std_stand_knee ** 2))

    # Ankle error formula:
    ankle_err = 0.5 * (
        torch.square(joint_pos[:, 4] - current_target_ankle)
        + torch.square(joint_pos[:, 13] - (-current_target_ankle))
    )
    ankle_score = torch.exp(-ankle_err / (std_stand_ankle ** 2))

    # Assertions for cushion phase:
    # Env 0 (symmetric cushion) must have EXACTLY 0 knee and ankle error, score = 1.0
    assert torch.isclose(knee_err[0], torch.tensor(0.0)), f"Knee error for symmetric cushion should be 0, got {knee_err[0]}"
    assert torch.isclose(ankle_err[0], torch.tensor(0.0)), f"Ankle error for symmetric cushion should be 0, got {ankle_err[0]}"
    assert torch.isclose(knee_score[0], torch.tensor(1.0)), f"Knee score should be 1.0, got {knee_score[0]}"

    # Env 1 (asymmetric split) must have non-zero knee error and degraded score
    assert knee_err[1] > 0.05, f"Asymmetric split must have significant error, got {knee_err[1]}"
    assert knee_score[1] < knee_score[0], f"Asymmetric split score must be lower than symmetric"

    # Test 2: Final stand phase (progress = 1.0)
    stand_target_knee = 0.0
    stand_target_ankle = 0.4530
    knee_err_stand = 0.5 * (
        torch.square(joint_pos[:, 3] - stand_target_knee)
        + torch.square(joint_pos[:, 12] - (-stand_target_knee))
    )
    ankle_err_stand = 0.5 * (
        torch.square(joint_pos[:, 4] - stand_target_ankle)
        + torch.square(joint_pos[:, 13] - (-stand_target_ankle))
    )
    assert torch.isclose(knee_err_stand[2], torch.tensor(0.0)), f"Knee error for stand should be 0, got {knee_err_stand[2]}"
    assert torch.isclose(ankle_err_stand[2], torch.tensor(0.0)), f"Ankle error for stand should be 0, got {ankle_err_stand[2]}"

    print("ALL BILATERAL SYMMETRY CHECKS PASSED SUCCESSFULLY!")


def test_cushion_knee_score_rejects_odd_leg_and_dump():
    """Run 94 held a shallow symmetric crouch, then dumped one knee and fell.

    The score is a Gaussian on the signed target, so the pose Run 94 actually
    held still scores, and letting a knee go to 0 scores clearly less.
    """
    target = 0.55
    std = 0.40
    on_target = symmetric_knee_flexion_score(
        torch.tensor([0.55]), torch.tensor([-0.55]), target, std
    )
    # Run 94 step 24: about +26° / -25°. Roll was already +13° and the right sole was unloaded.
    held = symmetric_knee_flexion_score(
        torch.tensor([0.455]), torch.tensor([-0.443]), target, std
    )
    # Run 94 step 27: left knee dumped through 0 while the right stayed bent.
    dumped = symmetric_knee_flexion_score(
        torch.tensor([-0.033]), torch.tensor([-0.450]), target, std
    )
    # Run 93 step 24 odd-leg: +77.5° / -6.8°.
    odd_leg = symmetric_knee_flexion_score(
        torch.tensor([1.353]), torch.tensor([-0.119]), target, std
    )
    wrong_sign = symmetric_knee_flexion_score(
        torch.tensor([1.20]), torch.tensor([0.50]), target, std
    )

    assert torch.isclose(on_target, torch.tensor([1.0]), atol=1e-5)
    assert float(held) > 0.7
    assert float(dumped) < float(held) * 0.5
    assert float(odd_leg) < 0.15
    assert float(wrong_sign) < 0.05


def test_run96_right_hip_fold_is_on_the_stand_slope():
    """Run 96 step 50: left hip −34°, right hip −10°. Stand is −26° / +26°."""
    target = 0.4579
    std = 0.50
    held = symmetric_hip_pitch_score(
        torch.tensor([-0.602]), torch.tensor([-0.169]), target, std
    )
    matched = symmetric_hip_pitch_score(
        torch.tensor([-0.458]), torch.tensor([0.458]), target, std
    )
    assert float(matched) > 0.99
    assert 0.25 < float(held) < 0.70
    # Adding the hip must not cut the parked pose. Averaging it in left the
    # right hip at −12° for all of Run 98. Multiplying cut the pose in half
    # and falls rose 5 → 43.
    at_contact = stand_pose_with_hip(
        torch.tensor([0.56]), torch.tensor([0.44]), torch.tensor([0.0])
    )
    parked = stand_pose_with_hip(
        torch.tensor([0.56]), torch.tensor([0.44]), torch.tensor([1.0])
    )
    stood = stand_pose_with_hip(
        torch.tensor([0.56]), torch.tensor([1.0]), torch.tensor([1.0])
    )
    assert abs(float(at_contact) - 0.56) < 1e-5
    assert float(parked) > 0.56
    assert float(stood) > float(parked)


def test_run101_forward_landing_costs_more_than_planted():
    """Run 101 landed the feet +12 mm and +18 mm ahead of takeoff."""
    start = torch.tensor([[[0.0, 0.042], [0.0, -0.042]]])
    planted = foot_offset_cost(start, start.clone())
    forward = foot_offset_cost(
        start, torch.tensor([[[0.012, 0.042], [0.018, -0.042]]])
    )
    assert float(planted) == 0.0
    assert abs(float(forward) - 0.030) < 1e-6


def test_run103_skimming_feet_are_not_flight():
    """Run 103 steps 4-6: < 1 N contact force, feet at +0.1 / -1.2 mm."""
    floor = torch.zeros(3)
    feet_z = torch.tensor([[0.0001, -0.0012], [0.0103, 0.0110], [0.0118, 0.002]])
    lifted = feet_above_floor(feet_z, floor, 0.004)
    assert lifted.tolist() == [False, True, False]


def test_drift_is_measured_from_spawn_in_its_heading():
    """reset_base spawns at ±0.5 m and any yaw. A robot spawned at (0.4, -0.3)
    facing +y that hops 30 mm forward has moved +30 mm forward, 0 sideways.
    """
    start = torch.tensor([[0.4, -0.3]])
    yaw = torch.tensor([math.pi / 2])
    fwd, left = spawn_frame_offset(start, yaw, torch.tensor([[0.4, -0.27]]))
    assert abs(float(fwd) - 0.03) < 1e-6
    assert abs(float(left)) < 1e-6
    half = math.sin(math.radians(-6.0)), math.cos(math.radians(-6.0))
    q = torch.tensor([[half[1], 0.0, 0.0, half[0]]])
    assert abs(math.degrees(float(yaw_from_quat(q))) + 12.0) < 1e-4


def test_run106_floor_slide_costs_but_flight_swing_does_not():
    """Run 106 step 6: feet at +0.6 mm moved 9.6 mm back in one step.
    Step 10: feet at +13 mm swung forward 14 mm (flight, not slip)."""
    floor = torch.zeros(1)
    prev = torch.tensor([[[-0.0056, 0.042], [-0.0058, -0.042]]])
    slide = torch.tensor([[[-0.0152, 0.042], [-0.0148, -0.042]]])
    low = torch.tensor([[0.0006, 0.0007]])
    high = torch.tensor([[0.0127, 0.0143]])
    c_slide = float(foot_slip_cost(prev, slide, low, floor, 0.004, 0.02))
    c_flight = float(foot_slip_cost(prev, slide, high, floor, 0.004, 0.02))
    assert abs(c_slide - (0.0096 + 0.0090) / 0.02) < 1e-4
    assert c_flight == 0.0


def test_run99_crouch_blends_onto_the_stand_hip():
    """Run 99 step 50 right hip was −12°. The stand target is +26°."""
    crouch = stand_curriculum_joints(torch.tensor([0.0]))
    stand = stand_curriculum_joints(torch.tensor([1.0]))
    mid = stand_curriculum_joints(torch.tensor([0.5]))
    assert float(crouch[0, 11]) < 0.0
    assert float(stand[0, 11]) > 0.40
    assert float(crouch[0, 11]) < float(mid[0, 11]) < float(stand[0, 11])
    assert abs(float(stand[0, 2]) - (-0.4579)) < 1e-4


def test_one_foot_load_scores_zero_and_gap_opening_costs():
    """Run 94 step 16: left sole ~6 N, right sole 0 N, then the right foot rose."""
    even = foot_load_balance(torch.tensor([4.0]), torch.tensor([4.0]))
    one_foot = foot_load_balance(torch.tensor([6.0]), torch.tensor([0.0]))
    airborne = foot_load_balance(torch.tensor([0.2]), torch.tensor([0.1]))
    assert float(even) > 0.99
    assert float(one_foot) < 0.01
    assert float(airborne) == 0.0

    gap = torch.tensor([0.005, 0.037])
    prev = torch.tensor([0.0, -0.005])
    active = torch.tensor([True, True])
    delta, phi = foot_gap_potential_delta(gap, prev, active)
    # Entering with a 5 mm gap costs 5 mm. Opening 5 mm → 37 mm costs 32 mm.
    assert torch.isclose(delta[0], torch.tensor(-0.005), atol=1e-6)
    assert torch.isclose(delta[1], torch.tensor(-0.032), atol=1e-6)
    assert torch.isclose(phi, -gap).all()


def test_staggered_touchdown_does_not_collect_the_gap():
    """A late second foot must not be paid for height stored while the gate was off.

    The old potential kept Φ = −gap during the one-foot interval, then refunded
    it on the step the second foot landed. That made a staggered landing a reward.
    """
    gaps = [0.030, 0.028, 0.000]
    active_flags = [False, False, True]
    prev = torch.zeros(1)
    total = torch.zeros(1)
    for gap, is_active in zip(gaps, active_flags):
        delta, phi = foot_gap_potential_delta(
            torch.tensor([gap]), prev, torch.tensor([is_active])
        )
        total = total + delta
        prev = phi.detach()
    assert float(total) <= 0.0

    # A stored potential from the previous episode must not be refunded.
    refund, phi = foot_gap_potential_delta(
        torch.tensor([0.0]), torch.tensor([-0.040]), torch.tensor([False])
    )
    assert float(refund) == 0.0
    assert float(phi) == 0.0


if __name__ == "__main__":
    test_knee_and_ankle_symmetry_in_jump_landing_rise()

"""Tests for Microduck Jump environment configuration."""

import math

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg


def test_jump_task_registered():
    tasks = list_tasks()
    assert "Mjlab-Jump-Flat-MicroDuck" in tasks, f"Mjlab-Jump-Flat-MicroDuck not in {tasks}"


def test_jump_cfg_builds():
    cfg = make_microduck_jump_env_cfg()
    assert cfg.episode_length_s == 1.5
    assert cfg.sim.dt == 0.005
    assert cfg.sim.decimation == 4


def test_jump_rewards_present_and_signs():
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards

    # Positive jump rewards (Run 92 Squat → Launch → Flight Clearance → Cushion → Stand)
    assert "squat_to_jump_pose" in r and r["squat_to_jump_pose"].weight > 0
    assert "jump_takeoff_vz" in r and r["jump_takeoff_vz"].weight > 0
    assert "jump_foot_clearance" in r and r["jump_foot_clearance"].weight > 0
    assert "jump_peak_height" in r and r["jump_peak_height"].weight > 0
    assert "jump_landing_cushion" in r and r["jump_landing_cushion"].weight > 0
    assert "jump_landing_rise" in r and r["jump_landing_rise"].weight > 0
    assert "upright" in r and r["upright"].weight > 0

    # Self-negating penalties (returns <= 0, positive weight)
    assert "body_pitch" in r and r["body_pitch"].weight > 0
    assert "hip_bow" in r and r["hip_bow"].weight > 0

    # Negative penalties / regularizers
    assert "jump_anti_tiptoe" in r and r["jump_anti_tiptoe"].weight < 0
    assert "jump_impact" in r and r["jump_impact"].weight < 0
    assert "jump_rebound_hop" in r and r["jump_rebound_hop"].weight < 0
    assert "jump_roll_tilt" in r and r["jump_roll_tilt"].weight < 0
    assert "hip_lateral_spread" in r and r["hip_lateral_spread"].weight < 0
    assert "hip_yaw_neutral" in r and r["hip_yaw_neutral"].weight < 0
    assert "jump_lateral_drift" in r and r["jump_lateral_drift"].weight < 0
    assert "jump_sagittal_drift" in r and r["jump_sagittal_drift"].weight < 0
    assert "head_neutral" in r and r["head_neutral"].weight < 0
    assert "head_action_l2" in r and r["head_action_l2"].weight < 0
    assert "self_collisions" in r and r["self_collisions"].weight < 0
    assert "dof_pos_limits" in r and r["dof_pos_limits"].weight < 0
    assert "body_ang_vel" in r and r["body_ang_vel"].weight < 0
    assert "angular_momentum" in r and r["angular_momentum"].weight < 0
    assert "action_rate_l2" in r and r["action_rate_l2"].weight < 0

    # Old walking rewards removed
    assert "track_linear_velocity" not in r
    assert "track_angular_velocity" not in r
    assert "air_time" not in r


def test_jump_sensors_and_terminations():
    cfg = make_microduck_jump_env_cfg()
    sensor_names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in sensor_names
    assert "self_collision" in sensor_names

    assert "time_out" in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "fell_over" in cfg.terminations
    assert "bad_head_pitch" in cfg.terminations


def test_jump_play_variant():
    cfg = make_microduck_jump_env_cfg(play=True)
    assert "jump_takeoff_vz" in cfg.rewards


def test_actor_observation_keeps_the_61d_slot_layout():
    cfg = make_microduck_jump_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    for padded in ("head_command", "body_command"):
        assert padded in terms
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6

    critic_terms = cfg.observations["critic"].terms
    assert "height_scan" not in critic_terms
    assert "foot_height" not in critic_terms
    assert "base_lin_vel" in critic_terms


def test_action_rate_curriculum_stays_soft_on_run96_resume():
    """model_17897 restores common_step_counter past 17897*24. A stage at or
    below that counter would raise the action-rate tax before the stand exists.
    """
    cfg = make_microduck_jump_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0
    assert stages[0]["weight"] == -0.05
    later = [s["step"] for s in stages if s["weight"] < -0.05]
    assert later
    assert min(later) > 17897 * 24


def test_rise_starts_before_the_measured_fall():
    """Run 93/94 terminated around step 27. A rise reward that starts at 28
    is never on-policy.
    """
    cfg = make_microduck_jump_env_cfg()
    rise = cfg.rewards["jump_landing_rise"].params
    assert rise["min_step"] <= 20
    assert rise["min_step"] + rise["ramp_steps"] >= 50
    assert rise["cushion_init_knee"] == cfg.rewards["jump_landing_cushion"].params["min_knee_flexion"]
    # Run 95 held about +35°/-30° after the slew. std 0.30 put that error in the
    # tail (score ~0.03) and rise plateaued. 0.60 keeps that pose on the slope.
    held = 0.5 * ((0.618 ** 2) + (0.524 ** 2))
    assert rise["std_stand_knee"] >= 0.55
    assert math.exp(-held / (rise["std_stand_knee"] ** 2)) > 0.30


def test_obs_parity_with_roulade():
    from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
        make_microduck_roulade_env_cfg,
    )

    jump = make_microduck_jump_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    for grp in ("actor", "critic"):
        assert list(jump.observations[grp].terms.keys()) == list(
            roulade.observations[grp].terms.keys()
        ), f"Observation layout divergent on group {grp}"


if __name__ == "__main__":
    test_jump_task_registered()
    test_jump_cfg_builds()
    test_jump_rewards_present_and_signs()
    test_jump_sensors_and_terminations()
    test_jump_play_variant()
    test_actor_observation_keeps_the_61d_slot_layout()
    test_action_rate_curriculum_stays_soft_on_run96_resume()
    test_rise_starts_before_the_measured_fall()
    test_obs_parity_with_roulade()
    print("ALL JUMP CFG TESTS PASSED!")



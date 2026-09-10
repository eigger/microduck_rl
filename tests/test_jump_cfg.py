"""Tests for Microduck Jump environment configuration."""

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg


def test_jump_task_registered():
    tasks = list_tasks()
    assert "Mjlab-Jump-Flat-MicroDuck" in tasks, f"Mjlab-Jump-Flat-MicroDuck not in {tasks}"


def test_jump_cfg_builds():
    cfg = make_microduck_jump_env_cfg()
    assert cfg.episode_length_s == 1.2
    assert cfg.sim.dt == 0.005
    assert cfg.sim.decimation == 4


def test_jump_rewards_present_and_signs():
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards

    # Positive jump rewards
    assert "jump_crouch" in r and r["jump_crouch"].weight > 0
    assert "jump_flight" in r and r["jump_flight"].weight > 0
    assert "jump_takeoff_vz" in r and r["jump_takeoff_vz"].weight > 0
    assert "jump_landing_cushion" in r and r["jump_landing_cushion"].weight > 0
    assert "jump_landing_rest" in r and r["jump_landing_rest"].weight > 0
    assert "upright" in r and r["upright"].weight > 0

    # Self-negating penalty (returns <= 0, positive weight)
    assert "leg_symmetry" in r and r["leg_symmetry"].weight > 0
    assert "knee_hyperextension" in r and r["knee_hyperextension"].weight > 0
    assert "jump_sagittal_foot" in r and r["jump_sagittal_foot"].weight > 0


    # Negative regularizers
    assert "hip_lateral_spread" in r and r["hip_lateral_spread"].weight < 0
    assert "head_neutral" in r and r["head_neutral"].weight < 0
    assert "jump_drift" in r and r["jump_drift"].weight < 0
    assert "soft_landing" in r and r["soft_landing"].weight < 0
    assert "self_collisions" in r and r["self_collisions"].weight < 0
    assert "dof_pos_limits" in r and r["dof_pos_limits"].weight < 0
    assert "body_ang_vel" in r and r["body_ang_vel"].weight < 0
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


def test_jump_play_variant():
    cfg = make_microduck_jump_env_cfg(play=True)
    assert "jump_flight" in cfg.rewards


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


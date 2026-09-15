"""Tests for Microduck Squat environment configuration."""

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks
from mjlab_microduck.tasks.microduck_squat_env_cfg import make_microduck_squat_env_cfg


def test_squat_task_registered():
    tasks = list_tasks()
    assert "Mjlab-Squat-Flat-MicroDuck" in tasks, f"Mjlab-Squat-Flat-MicroDuck not in {tasks}"
    assert "Mjlab-Squat-Rough-MicroDuck" in tasks, f"Mjlab-Squat-Rough-MicroDuck not in {tasks}"
    assert "Mjlab-Squat-Flat-Backlash-MicroDuck" in tasks, f"Mjlab-Squat-Flat-Backlash-MicroDuck not in {tasks}"
    assert "Mjlab-Squat-Rough-Backlash-MicroDuck" in tasks, f"Mjlab-Squat-Rough-Backlash-MicroDuck not in {tasks}"


def test_squat_cfg_builds():
    cfg = make_microduck_squat_env_cfg()
    assert cfg.episode_length_s == 4.0
    assert cfg.sim.mujoco.timestep == 0.005
    assert cfg.decimation == 4


def test_squat_command_and_reset_distribution():
    cfg = make_microduck_squat_env_cfg()
    # SitStandCommand configured for 100% squat
    twist_cmd = cfg.commands["twist"]
    assert twist_cmd.sit_prob == 1.0

    # Resets: 100% standing, 0% sitting
    reset_params = cfg.events["set_ground_state"].params
    assert reset_params["standing_prob"] == 1.0
    assert reset_params["sitting_prob"] == 0.0


def test_squat_rewards_present_and_signs():
    cfg = make_microduck_squat_env_cfg()
    r = cfg.rewards

    # Positive task rewards
    assert "posture_pose_legs" in r and r["posture_pose_legs"].weight > 0
    assert "head_pose_tracking" in r and r["head_pose_tracking"].weight > 0
    assert "posture_pose_l1" in r and r["posture_pose_l1"].weight > 0
    assert "posture_height" in r and r["posture_height"].weight > 0
    assert "posture_height_sharp" in r and r["posture_height_sharp"].weight > 0
    assert "posture_height_l1" in r and r["posture_height_l1"].weight > 0
    assert "upright_linear" in r and r["upright_linear"].weight > 0
    assert "upright_while_tall" in r and r["upright_while_tall"].weight > 0
    assert "posture_stillness" in r and r["posture_stillness"].weight > 0
    assert "posture_composite" in r and r["posture_composite"].weight > 0

    # Self-negating penalties (returns <= 0, positive weight)
    assert "descent_speed" in r and r["descent_speed"].weight > 0
    assert "gentle_motion" in r and r["gentle_motion"].weight > 0

    # Rise terms must NOT be present
    assert "rise_bootstrap" not in r
    assert "rise_speed" not in r

    # Negative regularizers
    assert "action_rate_l2" in r and r["action_rate_l2"].weight < 0
    assert "body_ang_vel" in r and r["body_ang_vel"].weight < 0
    assert "angular_momentum" in r and r["angular_momentum"].weight < 0
    assert "self_collisions" in r and r["self_collisions"].weight < 0

    # Old walking rewards removed
    assert "track_linear_velocity" not in r
    assert "track_angular_velocity" not in r
    assert "air_time" not in r


def test_squat_sensors_and_terminations():
    cfg = make_microduck_squat_env_cfg()
    sensor_names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in sensor_names
    assert "self_collision" in sensor_names

    assert "time_out" in cfg.terminations
    assert "nan_state" in cfg.terminations
    # No fell_over termination (tips/wobbles play out)
    assert "fell_over" not in cfg.terminations


def test_squat_play_variant():
    cfg = make_microduck_squat_env_cfg(play=True)
    assert "posture_composite" in cfg.rewards
    assert cfg.episode_length_s == 4.0


def test_actor_observation_keeps_the_61d_slot_layout():
    cfg = make_microduck_squat_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    assert "head_command" in terms
    assert "body_command" in terms
    assert terms["body_command"].params["dim"] == 6

    critic_terms = cfg.observations["critic"].terms
    assert "height_scan" not in critic_terms
    assert "foot_height" not in critic_terms
    assert "base_lin_vel" in critic_terms


def test_obs_parity_with_sitstand():
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
        make_microduck_sitstand_env_cfg,
    )

    squat = make_microduck_squat_env_cfg()
    sitstand = make_microduck_sitstand_env_cfg()
    for grp in ("actor", "critic"):
        assert list(squat.observations[grp].terms.keys()) == list(
            sitstand.observations[grp].terms.keys()
        )

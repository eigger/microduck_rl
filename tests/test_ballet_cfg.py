"""Tests for Microduck Ballet (Tiptoe / En Pointe) environment configuration."""

import math

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks.microduck_ballet_env_cfg import make_microduck_ballet_env_cfg


def test_ballet_task_registered():
    tasks = list_tasks()
    assert "Mjlab-Ballet-Flat-MicroDuck" in tasks, f"Mjlab-Ballet-Flat-MicroDuck not in {tasks}"
    assert "Mjlab-Ballet-Flat-Backlash-MicroDuck" in tasks, f"Mjlab-Ballet-Flat-Backlash-MicroDuck not in {tasks}"


def test_ballet_cfg_builds():
    cfg = make_microduck_ballet_env_cfg()
    assert cfg.episode_length_s == 20.0
    assert cfg.decimation == 4


def test_ballet_rewards_present_and_signs():
    cfg = make_microduck_ballet_env_cfg()
    r = cfg.rewards

    # Positive ballet rewards
    assert "ballet_ankle_tiptoe" in r and r["ballet_ankle_tiptoe"].weight > 0
    assert "ballet_straight_knees" in r and r["ballet_straight_knees"].weight > 0
    assert "ballet_tall_trunk" in r and r["ballet_tall_trunk"].weight > 0
    assert "upright" in r and r["upright"].weight > 0

    # Penalties (costs >= 0, weight < 0)
    assert "ballet_ankle_asymmetry" in r and r["ballet_ankle_asymmetry"].weight < 0
    assert "ballet_flat_foot" in r and r["ballet_flat_foot"].weight < 0
    assert "ballet_graceful_torso" in r and r["ballet_graceful_torso"].weight < 0
    assert "foot_slip" in r and r["foot_slip"].weight < 0


def test_ballet_commands_and_obs():
    cfg = make_microduck_ballet_env_cfg()
    # Check twist command ranges are tuned for compact ballet steps
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (-0.15, 0.25)
    assert twist.ranges.lin_vel_y == (-0.15, 0.15)
    assert twist.ranges.ang_vel_z == (-0.8, 0.8)

    # Invariants: 61D actor observation structure
    # [twist(3), head_command(4), body_command(6)] in command block
    assert "head_command" in cfg.observations["actor"].terms
    assert "body_command" in cfg.observations["actor"].terms


def test_ballet_rl_cfg_loads():
    rl_cfg = load_rl_cfg("Mjlab-Ballet-Flat-MicroDuck")
    assert rl_cfg.experiment_name == "ballet"
    assert rl_cfg.run_name == "ballet_tiptoe"

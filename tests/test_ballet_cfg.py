"""Tests for Microduck Ballet (Tiptoe / En Pointe) environment configuration."""

import math
from types import SimpleNamespace

import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ballet_env_cfg import (
    BALLET_ANG_VEL_Z,
    BALLET_LIN_VEL_X,
    BALLET_LIN_VEL_Y,
    make_microduck_ballet_env_cfg,
)


def test_ballet_task_registered():
    tasks = list_tasks()
    assert "Mjlab-Ballet-Flat-MicroDuck" in tasks, f"Mjlab-Ballet-Flat-MicroDuck not in {tasks}"
    assert "Mjlab-Ballet-Flat-Backlash-MicroDuck" in tasks, f"Mjlab-Ballet-Flat-Backlash-MicroDuck not in {tasks}"


def test_ballet_cfg_builds():
    cfg = make_microduck_ballet_env_cfg()
    assert cfg.episode_length_s == 20.0
    assert cfg.decimation == 4
    # Actuator delay widened to cover lag=0..6 without mutating the shared constant
    assert cfg.scene.entities["robot"].articulation.actuators[0].delay_min_lag == 0
    assert MICRODUCK_WALK_ROBOT_CFG.articulation.actuators[0].delay_min_lag == 3


def test_ballet_rewards_present_and_signs():
    cfg = make_microduck_ballet_env_cfg()
    r = cfg.rewards

    # Positive ballet rewards
    assert "ballet_ankle_tiptoe" in r and r["ballet_ankle_tiptoe"].weight > 0
    assert "ballet_straight_knees" in r and r["ballet_straight_knees"].weight > 0
    assert "ballet_tall_trunk" in r and r["ballet_tall_trunk"].weight > 0
    assert "ballet_hip_centering" in r and r["ballet_hip_centering"].weight > 0
    assert "ballet_graceful_stride" in r and r["ballet_graceful_stride"].weight > 0
    assert "upright" in r and r["upright"].weight > 0
    assert "track_yaw_fine" in r and r["track_yaw_fine"].weight > 0

    # Penalties (costs >= 0, weight < 0)
    assert "ballet_ankle_asymmetry" in r and r["ballet_ankle_asymmetry"].weight < 0
    assert "ballet_flat_foot" in r and r["ballet_flat_foot"].weight < 0
    assert "ballet_graceful_torso" in r and r["ballet_graceful_torso"].weight < 0
    assert "ballet_hip_stagger" in r and r["ballet_hip_stagger"].weight < 0
    assert "ballet_gait_symmetry" not in r
    assert "ballet_head_poise" in r and r["ballet_head_poise"].weight < 0
    assert "ballet_short_step" in r and r["ballet_short_step"].weight < 0
    assert "ballet_no_stepping" in r and r["ballet_no_stepping"].weight < 0
    assert "foot_slip" in r and r["foot_slip"].weight < 0

    # En-pointe foot clearance target height (site_z_floor = 19.3mm -> 33mm target)
    assert r["foot_clearance"].params["target_height"] == 0.033
    assert r["foot_swing_height"].params["target_height"] == 0.033


def test_ballet_hip_stagger_ema_and_standing():
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros((2, 14)),
            joint_vel=torch.zeros((2, 14)),
        ),
        find_joints=lambda _pat: (list(range(14)), []),
    )
    # Staggered hips on both rows: L flexed forward -0.50 rad, R extended -0.10 rad
    asset.data.joint_pos[:, 2] = -0.50
    asset.data.joint_pos[:, 11] = -0.10
    # Row 0 walking, Row 1 standing
    cmd = torch.tensor([[0.12, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        step_dt=0.02,
        episode_length_buf=torch.tensor([50, 50]),
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command=lambda _name: cmd),
    )
    cost = microduck_mdp.ballet_hip_stagger_penalty(
        env, asset_cfg=SceneEntityCfg("robot")
    )
    # Row 0 (walking) builds up slowly via 1s EMA; Row 1 (standing) pays full instantaneous penalty
    assert 0.0 < cost[0].item() < 0.05
    assert cost[1].item() > 0.5


def test_ballet_stance_knee_gate_and_phase_tiptoe():
    # Mock single-support walking envs: left grounded, right swinging with -0.5 rad knee
    asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=torch.zeros((3, 14))),
        find_joints=lambda _pat: (list(range(14)), []),
    )
    # Row 0: stance L at 64 deg / straight knee, swing R at 68 deg (air=0.12s -> full ramp) / flexed knee
    # Row 1: early liftoff R (air=0.05s -> halfway target 66 deg), stance L at 64 deg / straight knee
    # Row 2: double-knee crouch (L knee +0.75, R knee +0.75 rad)
    asset.data.joint_pos[0, 4] = math.radians(64.0)
    asset.data.joint_pos[0, 13] = -math.radians(68.0)
    asset.data.joint_pos[0, 3] = 0.0
    asset.data.joint_pos[0, 12] = -0.5
    asset.data.joint_pos[1, 4] = math.radians(64.0)
    asset.data.joint_pos[1, 13] = -math.radians(66.0)
    asset.data.joint_pos[1, 3] = 0.0
    asset.data.joint_pos[1, 12] = -0.3
    asset.data.joint_pos[2, 4] = math.radians(64.0)
    asset.data.joint_pos[2, 13] = -math.radians(64.0)
    asset.data.joint_pos[2, 3] = 0.75
    asset.data.joint_pos[2, 12] = 0.75

    sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_contact_time=torch.tensor([[0.2, 0.0], [0.2, 0.0], [0.2, 0.2]]),
            current_air_time=torch.tensor([[0.0, 0.12], [0.0, 0.05], [0.0, 0.0]]),
        )
    )
    cmd = torch.tensor([[0.12, 0.0, 0.0], [0.12, 0.0, 0.0], [0.12, 0.0, 0.0]])

    class _Scene(dict):
        def __init__(self, robot, contact):
            super().__init__(robot=robot, feet_ground_contact=contact)
            self.sensors = {"feet_ground_contact": contact}

    env = SimpleNamespace(
        scene=_Scene(asset, sensor),
        command_manager=SimpleNamespace(get_command=lambda _name: cmd),
    )

    k2 = microduck_mdp._ballet_stance_knee_sq(env, asset)
    assert k2[0].item() == 0.0
    assert k2[1].item() == 0.0
    assert k2[2].item() > 1.0

    r = microduck_mdp.ballet_ankle_tiptoe_reward(
        env,
        asset_cfg=SceneEntityCfg("robot"),
    )
    assert r[0].item() > 0.99
    assert r[1].item() > 0.99
    assert r[2].item() < 0.02


def test_ballet_head_poise_locks_4dof_and_vel():
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros((2, 14)),
            joint_vel=torch.zeros((2, 14)),
            default_joint_pos=torch.zeros((2, 14)),
        ),
        find_joints=lambda _pat: (list(range(14)), []),
    )
    # Row 0: calm head on command; Row 1: head_roll whip + yaw error
    asset.data.joint_pos[1, 7] = 0.25  # head_yaw error
    asset.data.joint_vel[1, 8] = 2.0   # head_roll velocity
    head_cmd = torch.zeros((2, 4))
    env = SimpleNamespace(
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command=lambda _name: head_cmd),
    )
    cost = microduck_mdp.ballet_head_poise_penalty(
        env, asset_cfg=SceneEntityCfg("robot")
    )
    assert cost[0].item() == 0.0
    assert cost[1].item() > 0.4


def test_ballet_commands_and_obs():
    cfg = make_microduck_ballet_env_cfg()
    # Check twist command ranges are tuned for poised ballet steps
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == BALLET_LIN_VEL_X
    assert twist.ranges.lin_vel_y == BALLET_LIN_VEL_Y
    assert twist.ranges.ang_vel_z == BALLET_ANG_VEL_Z

    # Invariants: 61D actor observation structure
    # [twist(3), head_command(4), body_command(6)] in command block
    assert "head_command" in cfg.observations["actor"].terms
    assert "body_command" in cfg.observations["actor"].terms


def test_ballet_rl_cfg_loads():
    rl_cfg = load_rl_cfg("Mjlab-Ballet-Flat-MicroDuck")
    assert rl_cfg.experiment_name == "ballet"
    assert rl_cfg.run_name == "ballet_tiptoe"
    assert rl_cfg.algorithm.symmetry_cfg is not None
    assert rl_cfg.algorithm.symmetry_cfg["use_mirror_loss"] is True

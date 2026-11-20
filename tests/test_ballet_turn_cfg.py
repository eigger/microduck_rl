"""Tests for Microduck Ballet Turn (One-Legged Arabesque Turn) environment configuration."""

import mujoco

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ballet_turn_env_cfg import (
    ENABLE_SYMMETRY,
    NECK_PATTERN_NO_YAW,
    make_microduck_ballet_turn_env_cfg,
)


def test_ballet_turn_task_registered():
    tasks = list_tasks()
    assert "Mjlab-BalletTurn-Flat-MicroDuck" in tasks


def test_ballet_turn_rewards_and_signs():
    cfg = make_microduck_ballet_turn_env_cfg()
    assert cfg.episode_length_s == 8.0
    assert cfg.scene.entities["robot"].init_state == HOME_FRAME

    r = cfg.rewards
    assert "angular_momentum" not in r, "angular_momentum directly fights Z spin"

    # Positive task rewards
    for name in (
        "ballet_turn_arabesque_pose",
        "ballet_turn_spin_rate",
        "ballet_turn_stance_pose",
        "ballet_turn_stand_rest",
        "upright",
    ):
        assert name in r and r[name].weight > 0

    # Self-negating penalty (returns <= 0, positive weight)
    assert "knee_hyperextension" in r
    assert r["knee_hyperextension"].func is microduck_mdp.knee_hyperextension_penalty
    assert r["knee_hyperextension"].weight > 0
    assert r["knee_hyperextension"].params["limit_rad"] == 1.02

    # Standard cost penalties (return >= 0, negative weight)
    for name in (
        "ballet_turn_right_foot_down",
        "ballet_turn_stay_in_place",
        "ballet_turn_stance_hop",
        "ballet_turn_heel_rock",
        "body_ang_vel",
        "dof_pos_limits",
        "action_rate_l2",
        "self_collisions",
        "soft_landing",
        "joint_torques_l2",
        "neck_joint_pos_l2",
        "head_yaw_stand_l2",
    ):
        assert name in r and r[name].weight < 0

    # Head yaw is free for spotting during spin (only locked during stand rest)
    neck_pattern = r["neck_joint_pos_l2"].params["pattern"]
    assert neck_pattern == NECK_PATTERN_NO_YAW
    assert "head_yaw" not in neck_pattern

    # Run 02 sustained single-leg + strong stand-rest invariants
    assert r["ballet_turn_stance_hop"].params["max_free_z"] == 0.025
    assert r["ballet_turn_right_foot_down"].weight <= -5.0
    assert r["ballet_turn_stand_rest"].weight >= 12.0


def test_ballet_turn_condim4_and_events_and_terminations():
    cfg = make_microduck_ballet_turn_env_cfg()
    robot_cfg = cfg.scene.entities["robot"]
    spec = robot_cfg.spec_fn()
    for col in robot_cfg.collisions:
        col.edit_spec(spec)
    model = spec.compile()
    for geom_name in ("left_foot_collision", "right_foot_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert gid >= 0
        assert model.geom_condim[gid] == 4
        assert model.geom_friction[gid][1] > 0.0

    for ev_name in ("expand_bam_friction_fields", "foot_friction", "foot_torsional_friction", "ballet_tiptoe_spawn"):
        assert ev_name in cfg.events
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"
    assert cfg.events["foot_torsional_friction"].params["axes"] == [1]

    for term in ("time_out", "nan_state", "fell_over", "fell_down"):
        assert term in cfg.terminations
    assert cfg.terminations["nan_state"].params["sensor_names"] == ("feet_ground_contact",)


def test_ballet_turn_symmetry_disabled():
    """One-legged CCW Arabesque turn is asymmetric."""
    assert ENABLE_SYMMETRY is False
    rl_cfg = load_rl_cfg("Mjlab-BalletTurn-Flat-MicroDuck")
    assert getattr(rl_cfg.algorithm, "symmetry_cfg", None) is None


def test_ballet_turn_61d_obs_parity():
    cfg = make_microduck_ballet_turn_env_cfg()
    actor_terms = cfg.observations["actor"].terms
    assert list(actor_terms.keys()) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]
    assert actor_terms["head_command"].params["dim"] == 4
    assert actor_terms["body_command"].params["dim"] == 6
    assert list(cfg.observations["critic"].terms.keys()) == [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "foot_air_time",
        "foot_contact",
        "foot_contact_forces",
        "head_command",
        "body_command",
    ]

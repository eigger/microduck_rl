"""Tests for Microduck Ballet Jump (5-phase CMJ Grand Jeté) environment configuration."""

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ballet_jump_env_cfg import (
    make_microduck_ballet_jump_env_cfg,
)


def test_ballet_jump_task_registered():
    tasks = list_tasks()
    assert "Mjlab-BalletJump-Flat-MicroDuck" in tasks


def test_ballet_jump_cmj_rewards_and_signs():
    cfg = make_microduck_ballet_jump_env_cfg()
    assert cfg.episode_length_s == 1.0
    assert cfg.scene.entities["robot"].init_state == HOME_FRAME

    r = cfg.rewards
    # Positive 5-phase CMJ rewards
    for name in (
        "jump_crouch",
        "jump_takeoff_vz",
        "jump_flight",
        "jump_ballet_grand_jete",
        "jump_landing_cushion",
        "jump_landing_rest",
        "upright",
    ):
        assert name in r and r[name].weight > 0

    assert r["jump_landing_cushion"].func is microduck_mdp.jump_cmj_landing_cushion_reward
    assert r["jump_ballet_grand_jete"].func is microduck_mdp.jump_ballet_grand_jete_reward

    rest_p = r["jump_landing_rest"].params
    assert rest_p["leg_only"] is True
    assert rest_p["std_foot_align"] > 0
    assert rest_p["std_sole_level"] > 0

    # Self-negating penalties (return <= 0, positive weight)
    for name in ("foot_height_symmetry", "knee_hyperextension"):
        assert name in r and r[name].weight > 0

    assert r["foot_height_symmetry"].params["max_step"] <= 11
    assert r["knee_hyperextension"].params["limit_rad"] > 0.9

    # Negative penalties
    for name in (
        "jump_roll_tilt",
        "jump_yaw",
        "jump_lateral_drift",
        "jump_sagittal_drift",
        "feet_lateral_rail",
        "hip_lateral_spread",
        "head_neutral",
        "head_action_l2",
        "jump_drift",
        "jump_grounded_sole_level",
        "jump_post_landing_feet_align",
        "soft_landing",
        "self_collisions",
    ):
        assert name in r and r[name].weight < 0

    # Phase timing alignment: crouch -> takeoff -> apex split -> cushion -> rest
    assert r["jump_crouch"].params["max_step"] < r["jump_flight"].params["min_step"]
    split = r["jump_ballet_grand_jete"].params
    assert 10 <= split["min_step"] < split["max_step"] <= 20
    assert r["jump_landing_cushion"].params["max_step"] <= r["jump_landing_rest"].params["min_step"]


def test_ballet_jump_spawns_and_terminations():
    cfg = make_microduck_ballet_jump_env_cfg()
    assert "ballet_tiptoe_spawn" in cfg.events
    assert 0.0 < cfg.events["ballet_tiptoe_spawn"].params["fraction"] < 1.0
    event_names = list(cfg.events.keys())
    assert event_names.index("reset_base") < event_names.index("ballet_tiptoe_spawn")
    assert event_names.index("reset_robot_joints") < event_names.index("ballet_tiptoe_spawn")

    for term in ("time_out", "nan_state", "fell_over", "bad_head_pitch", "fell_down"):
        assert term in cfg.terminations


def test_ballet_jump_symmetry_disabled():
    """Grand Jeté is asymmetric (left forward, right backward)."""
    rl_cfg = load_rl_cfg("Mjlab-BalletJump-Flat-MicroDuck")
    assert getattr(rl_cfg.algorithm, "symmetry_cfg", None) is None


def test_ballet_jump_61d_obs_parity():
    cfg = make_microduck_ballet_jump_env_cfg()
    terms = cfg.observations["actor"].terms
    assert list(terms.keys()) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6
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


def test_ballet_jump_action_rate_curriculum():
    cfg = make_microduck_ballet_jump_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0
    assert stages[0]["weight"] == -0.0005

"""Microduck Ballet Turn (One-Legged Arabesque Turn) environment configuration.

Trains a command-gated one-legged Arabesque turn on Microduck:
  - When ``twist.ang_vel_z == 0`` (``rel_standing_envs = 0.20``): holds a still,
    aligned 2-foot HOME standing/braking pose (feet flat, zero yaw spin).
  - When ``twist.ang_vel_z in [1.5, 3.0]`` rad/s: shifts weight over the left
    stance foot, extends the right leg straight back with pointed toe
    (Arabesque), and spins smoothly CCW in place using the 280 g head as a
    spotting flywheel while keeping torsional foot friction (``condim=4``)
    active for sim2real fidelity.

Built on top of ``make_velocity_env_cfg`` with unified 61D actor observations
and BAM M6 actuators + domain randomization.
"""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUND_PICK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES

NUM_STEPS_PER_ENV = 24
EPISODE_LENGTH_S = 8.0

# Symmetry mirror-loss is disabled: left-foot stance + right-leg Arabesque + CCW spin.
ENABLE_SYMMETRY = False

# Domain randomization
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 5.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)

# Tuned Ballet Turn constants
BALLET_TIPTOE_SPAWN_FRACTION = 0.35
NECK_PATTERN_NO_YAW = r"^(neck_pitch|head_pitch|head_roll)$"
TURN_ANG_VEL_Z = (1.5, 3.0)
REL_STANDING_ENVS = 0.25
FOOT_TORSIONAL_FRICTION_RANGE = (0.004, 0.022)


def make_microduck_ballet_turn_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck one-legged Arabesque ballet turn environment configuration."""
    del rough  # Flat plane task; signature matches registry convention.

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Enable condim=4 torsional foot friction so single-foot spinning cannot
    # exploit zero-spin-friction sliding, and widen BAM lag range for deploy.
    robot_cfg = deepcopy(MICRODUCK_GROUND_PICK_ROBOT_CFG)
    robot_cfg.collisions = (
        CollisionCfg(
            geom_names_expr=[".*_collision"],
            condim={r"^(left|right)_foot_collision$": 4, ".*_collision": 1},
            priority={r"^(left|right)_foot_collision$": 1},
            friction={r"^(left|right)_foot_collision$": (1.0, 0.015, 0.0001)},
        ),
    )
    for act_cfg in robot_cfg.articulation.actuators:
        if hasattr(act_cfg, "delay_min_lag"):
            act_cfg.delay_min_lag = 0

    cfg.scene.entities = {"robot": robot_cfg}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Simulation & Control: 50 Hz policy (0.005s step * 4 decimation)
    cfg.sim.decimation = 4
    cfg.sim.dt = 0.005
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards ───────────────────────────────────────────────────────────────
    # angular_momentum is intentionally removed: it penalizes 3D angular
    # momentum and directly fights the commanded yaw spin.
    for term in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
        "angular_momentum",
    ):
        cfg.rewards.pop(term, None)

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].params["std"] = math.radians(22.0)
    cfg.rewards["upright"].weight = 2.5

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["action_rate_l2"].weight = -0.08

    cfg.rewards["ballet_turn_arabesque_pose"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_arabesque_pose_reward,
        weight=5.0,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["ballet_turn_spin_rate"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_spin_rate_reward,
        weight=10.0,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["ballet_turn_stance_pose"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_stance_pose_reward,
        weight=1.5,
    )
    cfg.rewards["ballet_turn_stand_rest"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_stand_rest_reward,
        weight=14.0,
        params={"sensor_name": feet_ground_cfg.name},
    )

    cfg.rewards["ballet_turn_right_foot_down"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_right_foot_down_penalty,
        weight=-8.0,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["ballet_turn_stay_in_place"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_stay_in_place_penalty,
        weight=-12.0,
    )
    cfg.rewards["ballet_turn_stance_hop"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_stance_hop_penalty,
        weight=-4.0,
        params={"max_free_z": 0.025},
    )
    cfg.rewards["ballet_turn_heel_rock"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_heel_rock_penalty,
        weight=-2.0,
    )
    # Self-negating penalty (returns <= 0), so positive weight gives negative reward contribution.
    cfg.rewards["knee_hyperextension"] = RewardTermCfg(
        func=microduck_mdp.knee_hyperextension_penalty,
        weight=5.0,
        params={"limit_rad": 1.02},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-2.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=mdp.soft_landing,
        weight=-0.002,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2,
        weight=-5e-4,
    )
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2,
        weight=-2.5,
        params={"pattern": NECK_PATTERN_NO_YAW},
    )
    cfg.rewards["head_yaw_stand_l2"] = RewardTermCfg(
        func=microduck_mdp.ballet_turn_head_yaw_stand_penalty,
        weight=-1.0,
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["time_out"].time_out = True
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        time_out=False,
        params={"limit_angle": math.radians(50.0)},
    )
    cfg.terminations["fell_down"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        time_out=False,
        params={"min_height": 0.055},
    )

    # ── Observations (61D actor / critic parity) ──────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    cfg.observations["actor"].terms["projected_gravity"] = deepcopy(
        cfg.observations["actor"].terms["projected_gravity"]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["joint_pos"] = deepcopy(
        cfg.observations["actor"].terms["joint_pos"]
    )
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 0 if play else 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 0 if play else 64
    cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 0
    cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 0 if play else 1
    cfg.observations["actor"].terms["projected_gravity"].delay_update_period = 0 if play else 64
    cfg.observations["actor"].terms["joint_pos"].delay_min_lag = 0
    cfg.observations["actor"].terms["joint_pos"].delay_max_lag = 0
    cfg.observations["actor"].terms["joint_pos"].delay_update_period = 0
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 0 if play else 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 0 if play else 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms["projected_gravity"]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: twist.ang_vel_z triggers Arabesque turn vs 2-foot stand ─────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0 if play else REL_STANDING_ENVS
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.rel_forward_envs = 0.0
    command.rel_world_envs = 0.0
    command.init_velocity_prob = 0.0
    command.resampling_time_range = (3.0, 4.5)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = TURN_ANG_VEL_Z
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Events (Spawns + Domain Randomization) ────────────────────────────────
    cfg.events.pop("push_robot", None)

    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.6, 1.3)
    cfg.events["foot_torsional_friction"] = EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=foot_frictions_geom_names),
            "operation": "abs",
            "ranges": FOOT_TORSIONAL_FRICTION_RANGE,
            "axes": [1],
            "shared_random": True,
        },
    )

    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.01, 0.01),
        "y": (-0.01, 0.01),
        "z": (0.115, 0.118),
        "roll": (-0.03, 0.03),
        "pitch": (-0.03, 0.03),
        "yaw": (-math.pi, math.pi),
    }
    cfg.events["reset_base"].params["velocity_range"] = {}

    cfg.events["reset_robot_joints"].params["position_range"] = (-0.03, 0.03)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

    cfg.events["ballet_tiptoe_spawn"] = EventTermCfg(
        func=microduck_mdp.reset_ballet_tiptoe_jump_spawn,
        mode="reset",
        params={
            "fraction": 0.0 if play else BALLET_TIPTOE_SPAWN_FRACTION,
            "ankle_range": (0.60, 1.18),
            "knee_range": (-0.02, 0.15),
            "hip_pitch_range": (-0.45, -0.25),
            "stride_offset_range": (-0.12, 0.12),
            "z_range": (0.120, 0.129),
            "pitch_range": (-0.06, 0.06),
            "roll_range": (-0.04, 0.04),
            "lin_vel_x_range": (-0.02, 0.06),
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curricula ─────────────────────────────────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.08},
                {"step": 300 * NUM_STEPS_PER_ENV, "weight": -0.15},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": -0.22},
            ],
        },
    )

    return cfg


MicroduckBalletTurnRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="ballet_turn",
    run_name="run02_arabesque_sustained_turn",
    save_interval=100,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=800,
)

"""Microduck Ballet Jump (Grand Jeté / 가위 점프) task configuration.

5-phase Counter-Movement Jump (CMJ) architecture matching Run 59's proven
``HOME_FRAME`` (`MICRODUCK_GROUND_PICK_ROBOT_CFG`, ``EPISODE_LENGTH_S = 1.0``)
recipe, plus ballet-tiptoe handover spawns so the same policy launches cleanly
from BOTH resting standing (HOME_FRAME) AND live ``ballet.onnx`` tiptoe walking:

  - Phase 1 (`jump_crouch`, steps 0–8): fast 0.12 s Plié dip below 88 mm
  - Phase 2 (`jump_takeoff_vz` + `foot_height_symmetry`, steps 6–14): synchronous push-off
  - Phase 3 (`jump_flight` + `jump_scissor_split`, steps 10–24 / 12–18): aerial Grand Jeté
  - Phase 4 (`jump_cmj_landing_cushion`, steps 18–26): compliant knee touchdown
  - Phase 5 (`jump_landing_rest`, steps 26–50): upright recovery to HOME_FRAME
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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUND_PICK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES

NUM_STEPS_PER_ENV = 24
EPISODE_LENGTH_S = 1.0  # 50 steps at 50 Hz (matching Run 59 CMJ cycle)

# Domain randomization constants (matching Run 59 / microduck_jump_env_cfg)
ENABLE_SYMMETRY = False  # Asymmetric left-forward / right-backward Grand Jeté
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
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 4.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)

# Tuned CMJ + Grand Jeté constants
BALLET_TIPTOE_SPAWN_FRACTION = 0.50
GRAND_JETE_WEIGHT = 600.0


def make_microduck_ballet_jump_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck 5-phase CMJ Grand Jeté ballet jump environment configuration."""
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

    cfg.scene.entities = {"robot": deepcopy(MICRODUCK_GROUND_PICK_ROBOT_CFG)}
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

    # ── Rewards (5-Phase CMJ + Grand Jeté Scissor Split) ──────────────────────
    for term in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
    ):
        cfg.rewards.pop(term, None)

    # Regularizers & physics guards
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].params["std"] = math.radians(10.0)
    cfg.rewards["upright"].weight = 15.0

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.01
    cfg.rewards["dof_pos_limits"].weight = -0.5
    cfg.rewards["action_rate_l2"].weight = -0.0005

    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=mdp.soft_landing,
        weight=-0.001,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Phase 1: Fast Plié / Deep Crouch Dip (steps 0–9)
    cfg.rewards["jump_crouch"] = RewardTermCfg(
        func=microduck_mdp.jump_crouch_reward,
        weight=80.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "target_z": 0.076,
            "std_z": 0.015,
            "max_step": 9,
            "min_knee_flexion": 0.65,
        },
    )

    # Phase 2: Symmetrical explosive upward thrust (steps 6–15, crouch-gated)
    cfg.rewards["jump_takeoff_vz"] = RewardTermCfg(
        func=microduck_mdp.jump_takeoff_velocity_reward,
        weight=150.0,
        params={
            "target_vz": 0.75,
            "min_step": 6,
            "max_step": 15,
            "min_crouch_z": 0.096,
            "full_crouch_z": 0.078,
        },
    )
    cfg.rewards["foot_height_symmetry"] = RewardTermCfg(
        func=microduck_mdp.foot_height_symmetry_penalty,
        weight=50.0,
        params={"max_step": 11},
    )

    # Phase 3: High aerial flight + Straight-leg Pointed-toe Grand Jeté
    cfg.rewards["jump_flight"] = RewardTermCfg(
        func=microduck_mdp.jump_flight_reward,
        weight=160.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "stand_z": 0.117,
            "target_height": 0.160,
            "min_clearance": 0.010,
            "target_clearance": 0.028,
            "max_clearance_score": 1.0,
            "min_step": 10,
            "max_step": 24,
            "min_crouch_z": 0.096,
            "full_crouch_z": 0.078,
        },
    )
    cfg.rewards["jump_ballet_grand_jete"] = RewardTermCfg(
        func=microduck_mdp.jump_ballet_grand_jete_reward,
        weight=GRAND_JETE_WEIGHT,
        params={
            "sensor_name": feet_ground_cfg.name,
            "min_step": 12,
            "max_step": 19,
            "min_crouch_z": 0.096,
            "full_crouch_z": 0.078,
            "target_l_hip": -1.43,
            "target_r_hip": 0.14,
            "target_l_knee": -0.73,
            "target_r_knee": 0.73,
            "target_l_ankle": 1.40,
            "target_r_ankle": -1.40,
            "std_hip": 0.55,
            "std_knee": 0.55,
            "std_ankle": 0.60,
            "target_split": 0.105,
        },
    )

    # Phase 4: Compliant touchdown & spring cushion (steps 19–26, crouch+air gated)
    cfg.rewards["jump_landing_cushion"] = RewardTermCfg(
        func=microduck_mdp.jump_cmj_landing_cushion_reward,
        weight=35.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "min_air_time": 0.03,
            "min_knee_flexion": 0.20,
            "min_step": 19,
            "max_step": 26,
        },
    )

    # Phase 5: Return to upright HOME_FRAME standing rest with neat aligned flat feet (steps 26–50)
    cfg.rewards["jump_landing_rest"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_rest_reward,
        weight=120.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "min_air_time": 0.04,
            "target_z": 0.117,
            "std_z": 0.015,
            "std_pose": 0.20,
            "min_step": 26,
            "leg_only": True,
            "std_foot_align": 0.025,
            "std_sole_level": 0.18,
        },
    )

    # Verticality, posture, and leg alignment penalties (Run 59 weights + ballet guards)
    cfg.rewards["jump_roll_tilt"] = RewardTermCfg(
        func=microduck_mdp.jump_roll_tilt_penalty,
        weight=-10.0,
    )
    cfg.rewards["jump_yaw"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_penalty,
        weight=-2.5,
    )
    cfg.rewards["jump_lateral_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_drift_penalty,
        weight=-4.0,
    )
    cfg.rewards["jump_sagittal_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_sagittal_drift_penalty,
        weight=-12.0,
    )
    cfg.rewards["feet_lateral_rail"] = RewardTermCfg(
        func=microduck_mdp.feet_lateral_rail_penalty,
        weight=-4.0,
    )
    cfg.rewards["hip_lateral_spread"] = RewardTermCfg(
        func=microduck_mdp.hip_lateral_abduction_penalty,
        weight=-5.0,
    )
    cfg.rewards["head_neutral"] = RewardTermCfg(
        func=microduck_mdp.head_neutral_penalty,
        weight=-50.0,
    )
    cfg.rewards["head_action_l2"] = RewardTermCfg(
        func=microduck_mdp.head_action_l2,
        weight=-30.0,
    )
    cfg.rewards["jump_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_drift_penalty,
        weight=-10.0,
    )
    cfg.rewards["jump_grounded_sole_level"] = RewardTermCfg(
        func=microduck_mdp.jump_grounded_sole_level_penalty,
        weight=-15.0,
        params={"crouch_max_step": 9, "land_min_step": 26},
    )
    cfg.rewards["jump_post_landing_feet_align"] = RewardTermCfg(
        func=microduck_mdp.jump_post_landing_feet_align_penalty,
        weight=-40.0,
        params={"min_step": 24},
    )
    cfg.rewards["knee_hyperextension"] = RewardTermCfg(
        func=microduck_mdp.knee_hyperextension_penalty,
        weight=15.0,
        params={"limit_rad": 1.02},
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
        params={"limit_angle": math.radians(35.0)},
    )
    cfg.terminations["bad_head_pitch"] = TerminationTermCfg(
        func=microduck_mdp.head_orientation_exceeded,
        time_out=False,
        params={"max_pitch_deg": 35.0, "min_pitch_deg": 5.0},
    )
    cfg.terminations["fell_down"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        time_out=False,
        params={"min_height": 0.045},
    )

    # ── Observations (61D actor / 74D critic matching Run 59) ─────────────────
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

    # Match Run 59 delays during training; zero delays in play
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

    # ── Command: tiny noise around zero (kept for 61D obs-shape parity) ───────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
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
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.01, 0.01),
        "y": (-0.01, 0.01),
        "z": (0.115, 0.118),
        "roll": (-0.02, 0.02),
        "pitch": (-0.02, 0.02),
        "yaw": (-0.03, 0.03),
    }
    cfg.events["reset_base"].params["velocity_range"] = {}

    cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

    # Run after reset_base and reset_robot_joints so ballet tiptoe state wins cleanly
    cfg.events["ballet_tiptoe_spawn"] = EventTermCfg(
        func=microduck_mdp.reset_ballet_tiptoe_jump_spawn,
        mode="reset",
        params={
            "fraction": BALLET_TIPTOE_SPAWN_FRACTION,
            "ankle_range": (0.60, 1.18),
            "knee_range": (-0.02, 0.15),
            "hip_pitch_range": (-0.45, -0.25),
            "stride_offset_range": (-0.12, 0.12),
            "z_range": (0.120, 0.129),
            "pitch_range": (-0.06, 0.06),
            "roll_range": (-0.04, 0.04),
            "lin_vel_x_range": (-0.02, 0.10),
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
                {"step": 0, "weight": -0.0005},
            ],
        },
    )

    return cfg


MicroduckBalletJumpRlCfg = RslRlOnPolicyRunnerCfg(
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
        learning_rate=1.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="ballet_jump",
    run_name="run08_ballet_straight_pointe",
    save_interval=50,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=600,
)

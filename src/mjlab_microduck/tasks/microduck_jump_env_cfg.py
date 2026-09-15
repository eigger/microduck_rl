"""Microduck Jump (점프) task configuration.

Episodic policy: Robot starts in a standing equilibrium pose, crouches,
launches upward explosively so both feet leave the ground simultaneously,
reaches peak apex height, and lands stably back on its feet returning to
the standard standing pose.
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
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUND_PICK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

NUM_STEPS_PER_ENV = 24
EPISODE_LENGTH_S = 1.2  # 60 steps (Two-Phase: 0-18 deep squat, 17-25 explosive thrust, 22-35 straight flight, 32-60 land & rest)

ENABLE_SYMMETRY = True
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


def make_microduck_jump_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck jump environment configuration."""

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

    cfg.scene.entities = {"robot": MICRODUCK_GROUND_PICK_ROBOT_CFG}
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
    # Clear velocity locomotion terms
    for term in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
    ]:
        if term in cfg.rewards:
            del cfg.rewards[term]

    # Two-Phase Vertical Jump Reward Stack (Run 78: Zero-Pitch Triple Extension Architecture):
    # Phase 1: Slow, smooth, controlled deep squat down (steps 0-18, t in 0.00-0.36s)
    # Direct joint pose matching (multiplicative composite): Knee +50.0°, Hip +36.1°, Ankle +21.8° guarantees 0° pitch!
    cfg.rewards["jump_slewed_crouch"] = RewardTermCfg(
        func=microduck_mdp.jump_slewed_crouch_reward,
        weight=10.0,
        params={
            "sensor_name": "feet_ground_contact",
            "ramp_steps": 14,
            "max_step": 18,
        },
    )

    # Phase 1b: L1 bootstrap on squat pose: provides constant gradient toward deep squat
    cfg.rewards["jump_slewed_crouch_l1"] = RewardTermCfg(
        func=microduck_mdp.jump_slewed_crouch_l1_reward,
        weight=3.0,
        params={
            "ramp_steps": 14,
            "max_step": 18,
        },
    )

    # Phase 1c: Anti-Early-Launch Penalty (steps 0-15, t in 0.00-0.30s)
    # Self-negating penalty (<= 0): strictly forbids hopping/launching before squat is complete!
    cfg.rewards["jump_early_launch"] = RewardTermCfg(
        func=microduck_mdp.jump_early_launch_penalty,
        weight=5.0,
        params={"max_step": 15, "threshold_vz": 0.03},
    )

    # Phase 2: Symmetrical explosive upward thrust & Triple Extension (steps 16-27, t in 0.32-0.54s)
    # Rewards upward launch velocity + Knee Extension + Ankle Push + Hip Posterior Extension for 0-Pitch CoM alignment!
    cfg.rewards["jump_vertical_thrust"] = RewardTermCfg(
        func=microduck_mdp.jump_vertical_thrust_reward,
        weight=15.0,
        params={
            "target_vz": 0.60,
            "min_step": 16,
            "max_step": 27,
        },
    )

    # Phase 2b: Symmetrical Takeoff Synchrony (steps 0-27, t in 0.00-0.54s)
    # Self-negating penalty (<= 0): forces BOTH feet to push and lift off simultaneously!
    cfg.rewards["foot_height_symmetry"] = RewardTermCfg(
        func=microduck_mdp.foot_height_symmetry_penalty,
        weight=4.0,
        params={"max_step": 27},
    )

    # Phase 3: Straight-leg airborne flight (steps 22-38, t in 0.44-0.76s)
    # CRITICAL: Rewards True Foot Lift-off with legs extended straight and parallel (no scissor split)!
    cfg.rewards["jump_straight_flight"] = RewardTermCfg(
        func=microduck_mdp.jump_straight_flight_reward,
        weight=20.0,
        params={
            "sensor_name": "feet_ground_contact",
            "stand_z": 0.1175,
            "target_height": 0.140,
            "min_step": 22,
            "max_step": 38,
        },
    )

    # Phase 4: Compliant touchdown & spring cushion (steps 32-42, t in 0.64-0.84s)
    cfg.rewards["jump_landing_cushion"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_cushion_reward,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "min_air_time": 0.03,
            "min_knee_flexion": 0.15,
            "min_step": 32,
            "max_step": 42,
        },
    )

    # Phase 5: Return to proud upright standing pose (steps 36-60, t in 0.72-1.20s)
    # Restores standard symmetrical standing pose without drift or splay after touchdown!
    cfg.rewards["jump_landing_rest"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_rest_reward,
        weight=5.0,
        params={
            "sensor_name": "feet_ground_contact",
            "min_air_time": 0.03,
            "target_z": 0.117,
            "std_z": 0.015,
            "std_pose": 0.25,
            "min_step": 36,
        },
    )

    # Upright orientation maintenance (std 10° keeps body upright during entire sequence)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].params["std"] = math.radians(10.0)
    cfg.rewards["upright"].weight = 4.0

    # Strict roll tilt penalty: eliminates sideways lean during takeoff & flight
    cfg.rewards["jump_roll_tilt"] = RewardTermCfg(
        func=microduck_mdp.jump_roll_tilt_penalty,
        weight=-4.0,
    )

    # Normalized yaw penalty: balanced gradient (-1.0) that guides heading without tax-bombing jump
    cfg.rewards["jump_yaw"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_penalty,
        weight=-1.0,
    )

    # Lateral drift penalty: prevents sideways velocity & displacement
    cfg.rewards["jump_lateral_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_drift_penalty,
        weight=-2.0,
    )

    # Sagittal (forward) drift penalty: prevents leaping forward, enforces in-place vertical jump
    cfg.rewards["jump_sagittal_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_sagittal_drift_penalty,
        weight=-2.0,
    )

    # Lateral foot rail penalty: forces feet to stay on the +/- 42mm nominal rails
    cfg.rewards["feet_lateral_rail"] = RewardTermCfg(
        func=microduck_mdp.feet_lateral_rail_penalty,
        weight=-2.0,
    )

    # Anti-split sideways: prevents lateral leg spreading (hip_roll)
    cfg.rewards["hip_lateral_spread"] = RewardTermCfg(
        func=microduck_mdp.hip_lateral_abduction_penalty,
        weight=-2.0,
    )

    # Hip yaw neutral: strictly locks left/right hip_yaw to 0 to prevent torque generation around Z
    cfg.rewards["hip_yaw_neutral"] = RewardTermCfg(
        func=microduck_mdp.hip_yaw_neutral_penalty,
        weight=-2.0,
    )
    # Head neutral: guides head posture without crushing dynamic jump
    cfg.rewards["head_neutral"] = RewardTermCfg(
        func=microduck_mdp.head_neutral_penalty,
        weight=-5.0,
    )
    # Head action freeze: keeps head actions calm
    cfg.rewards["head_action_l2"] = RewardTermCfg(
        func=microduck_mdp.head_action_l2,
        weight=-5.0,
    )
    # Drift penalty: keeps jump vertical
    cfg.rewards["jump_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_drift_penalty,
        weight=-3.0,
    )
    # Anti-hyperextension: gentle barrier against knees bending backward
    cfg.rewards["knee_hyperextension"] = RewardTermCfg(
        func=microduck_mdp.knee_hyperextension_penalty,
        weight=5.0,
    )

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
    cfg.rewards["dof_pos_limits"].weight = -0.5
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.01

    # Action smoothness: Run 12 level action rate penalty
    cfg.rewards["action_rate_l2"].weight = -0.3

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["time_out"].time_out = True
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    # Orientation termination (45°): allows exploration noise during deep crouch/launch
    # while terminating if robot completely loses balance.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        time_out=False,
        params={"limit_angle": math.radians(45.0)},
    )
    # Realistic head pitch limits (-35° to +75°): allows natural head motion during deep squat
    # without triggering premature 0.08s resets, while still preventing head collision cheating!
    cfg.terminations["bad_head_pitch"] = TerminationTermCfg(
        func=microduck_mdp.head_orientation_exceeded,
        time_out=False,
        params={"max_pitch_deg": 75.0, "min_pitch_deg": -35.0},
    )
    cfg.terminations["fell_down"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        time_out=False,
        params={"min_height": 0.038},
    )

    # ── Observations (identical 61D layout to walking / trick policies) ──────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # Remove terrain height sensor terms (flat floor, no ray scanner sensor)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Contact terms in critic with NaN-safe wrappers
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # Delays and noise
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

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

    # Command obs slots: zero padding for BOTH head (4) and body (6)
    # The 61D obs layout parity [twist(3), head(4), body(6)] is kept so runtime works unchanged.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
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

    # ── Events (Domain Randomization) ──────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.115, 0.118)

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

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
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    if "command_vel" in cfg.curriculum:
        del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.008},
            ],
        },
    )

    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
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
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="jump",
    run_name="run78_zero_pitch_triple_extension_jump",
    save_interval=15,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=1200,
)


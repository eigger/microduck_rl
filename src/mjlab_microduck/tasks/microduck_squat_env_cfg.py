"""Microduck *squat* task (mjlab 1.3.0) — stand to deep squat (lowest stable posture), GENTLY.

Derived from microduck_sitstand_env_cfg. Focuses purely on the descent and
settling into the deep squat (knees bent, trunk low and upright, feet planted).
Eliminates rise transitions to master stable crouching before jumping.

Command:
    cmd (twist slot) = [1.0, 0, 0]   always commanded to SIT/SQUAT.
Initial state:
    Always starts standing (standing_prob=1.0, sitting_prob=0.0).
Episode duration:
    4.0 seconds: 2.0 s ramp down + 2.0 s hold at deep squat rest.

Key constraint vs. Run 79:
    foot_xy_drift_penalty locks foot XY positions to their spawn coordinates.
    Without this, the policy slides feet forward to cheaply lower trunk height
    instead of doing a proper squat with planted feet.
"""

import math
from copy import deepcopy

# Symmetry
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity/sitstand envs) ──────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges ───────────────────────────────────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode length: 4.0 s is plenty for 2.0 s slewed descent + 2.0 s deep squat hold.
EPISODE_LENGTH_S = 4.0
POSTURE_DWELL_S  = (4.0, 4.0)
SIT_PROB         = 1.0  # Always SIT/SQUAT

# ── SQUAT keyframe: physically verified static equilibrium squat ─────────────
# Solved via inverse kinematics and CoM balancing in MuJoCo physics:
# - Both feet 100% flat on the ground (Delta ankle == Delta knee, cancels foot pitch)
# - Knee bent forward (+1.4294 rad)
# - Hip pitched forward (+0.3670 rad) to keep CoM_x = 0.0mm directly above the feet
# - Trunk Z settles at 0.075m (deep stable squat, 43.5mm drop from standing 0.1185m)
SITTING_TARGET_OVERRIDES = {
    1:   0.0,      # left  hip_roll   (HOME -0.0873)
    2:   0.3670,   # left  hip_pitch  (HOME -0.4579)
    3:   1.4294,   # left  knee       (HOME -0.0049)
    4:   1.0624,   # left  ankle      (HOME +0.4530)
    # neck/head steered by the head_pose command.
    10:  0.0,      # right hip_roll   (HOME +0.0873)
    11: -0.3670,   # right hip_pitch  (HOME +0.4579)
    12: -1.4294,   # right knee       (HOME +0.0049)
    13: -1.0624,   # right ankle      (HOME -0.4530)
}

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m)
STAND_Z = 0.115
SIT_Z   = 0.075

STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z   = 0.075

POSTURE_RAMP_S = 2.0
MAX_DESCENT_SPEED = 0.05
# Max foot XY slide allowed from spawn position (m). ~2 cm tolerance for DR jitter.
FOOT_XY_DRIFT_LIMIT = 0.02

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
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_squat_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck squat environment configuration."""

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

    # Standup robot variant: full collision meshes.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Foot site cfg for drift penalty (left_foot, right_foot sites) ──────────
    foot_site_cfg = SceneEntityCfg("robot", site_names=["left_foot", "right_foot"])

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    # foot_xy_drift_penalty (below) enforces planted feet via position locking.
    # foot_slip from the base velocity env is removed: it reads foot geom velocities
    # against the sensor slot layout which differs in the squat scene and crashes.
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: posture-conditioned squat stack ──────────────────────────────
    # Pose target — legs only (head is command-steered).
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=8.0,
        params={
            "command_name":  "twist",
            "std":           0.5,
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Head pose tracking (active commandable head).
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # L1 bootstrap toward the squat pose.
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1,
        weight=3.0,
        params={
            "command_name":  "twist",
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # Trunk height — two-layer Gaussian + L1 transition driver.
    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
            "std":          0.04,
        },
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
            "std":          0.015,
        },
    )
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=15.0,
        params={
            "command_name": "twist",
            "sit_z":        SIT_Z,
            "stand_z":      STAND_Z,
        },
    )

    # ── Gentleness: descent-speed cap and shock penalty ───────────────────────
    # Self-negating functions (returning <= 0) → POSITIVE weight.
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={
            "max_down_vel": MAX_DESCENT_SPEED,
            "asset_cfg":    SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Two-layer upright pressure
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_while_tall"] = RewardTermCfg(
        func=microduck_mdp.upright_while_tall,
        weight=0.2,
        params={
            "height_low":  SIT_UPRIGHT_Z,
            "height_high": STAND_UPRIGHT_Z,
            "asset_cfg":   SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Stillness at the commanded posture (rest quietly, upright)
    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name":  "twist",
            "sit_z":         SIT_Z,
            "stand_z":       STAND_Z,
            "band_full":     0.012,
            "band_zero":     0.03,
            "vel_std":       0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )

    # Multiplicative goal composite
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name":  "twist",
            "sit_overrides": SITTING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z":         SIT_Z,
            "stand_z":       STAND_Z,
            "height_std":    0.03,
            "upright_std":   0.40,
            "pose_std":      0.40,
            "head_std":      0.40,
        },
    )

    # ── Sim2real regularisers ────────────────────────────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Foot-plant enforcement ─────────────────────────────────────────────────
    # Self-negating (returns <=0) → POSITIVE weight.
    # Penalises cumulative XY drift of the foot sites from their spawn position.
    # Softened to 6.0 so the policy is free to explore leg flexion without being
    # paralyzed by foot placement taxes at early iterations.
    cfg.rewards["foot_xy_drift"] = RewardTermCfg(
        func=microduck_mdp.foot_xy_drift_penalty,
        weight=6.0,
        params={
            "asset_cfg": foot_site_cfg,
            "max_drift": FOOT_XY_DRIFT_LIMIT,
        },
    )

    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (61D shared layout) ─────────────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

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

    # ── Head pose command ────────────────────────────────────────────────────
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.07, 0.07),
            (-0.015, 0.015),
        ),
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: ALWAYS SIT/SQUAT ─────────────────────────────────────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(
        **{
            **vars(command),
            "sit_prob": SIT_PROB,
            "ramp_s":   POSTURE_RAMP_S,
            "sit_z":    SIT_Z,
            "stand_z":  STAND_Z,
        }
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
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

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.11, 0.12)

    # Always start standing (100% standing, 0% sitting) to train descent
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":          0.0,
            "face_up_prob":            0.0,
            "sitting_prob":            0.0,
            "standing_prob":           1.0,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.10,
            "sitting_tilt_max":        math.radians(8),
            "sitting_z_min":           0.06,
            "sitting_z_max":           0.075,
            "standing_z_min":          0.11,
            "standing_z_max":          0.12,
        },
    )

    # MuJoCo solver settings for seated contacts (nconmax=60 fits 4096 envs within 8GB VRAM)
    cfg.sim.nconmax = 60
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
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

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
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

    # ── Terrain ───────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ── Curriculum ────────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1000 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 1500 * 24, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2000 * 24, "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}},
                    {"step": 2500 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 500 * 24,   "weight": -0.2},
                {"step": 750 * 24,   "weight": -0.4},
                {"step": 1000 * 24,  "weight": -0.6},
                {"step": 1250 * 24,  "weight": -0.8},
                {"step": 1500 * 24,  "weight": -1.0},
            ],
        },
    )

    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 500 * 24,   "weight": -5e-4},
                {"step": 1000 * 24,  "weight": -1e-3},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckSquatRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_squat",
    run_name="microduck_squat",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3000,
)

"""Microduck Ballet (Tiptoe / En Pointe) environment.

Learns a graceful tiptoe walking gait (demipointe / pointe-like) where the robot:
  - Extends both ankles in plantar flexion (+left, -right) so only the front toes touch the ground
  - Keeps knees relatively straight and maintains a tall, elegant posture (higher trunk z)
  - Moves with delicate steps according to commanded twist velocities (vx, vy, vyaw)
  - Holds an upright, graceful torso with minimal tilt

Built on top of make_microduck_velocity_env_cfg to inherit 61D obs, BAM actuator DR,
and robustness stack.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

import mjlab_microduck.tasks.mdp as microduck_mdp
from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .symmetry import SYMMETRY_CFG

# === Tuned Constants for Ballet Tiptoe ===
# Phase-aware en pointe posture (no liftoff discontinuity):
# - Standing: 66 deg en pointe
# - Walking stance foot: 64 deg en pointe
# - Walking swing foot: ramps smoothly over 0.10s of air time to 68 deg full pointe
BALLET_TARGET_PLANTAR_DEG = 66.0
BALLET_SWING_PLANTAR_DEG = 68.0
BALLET_STANCE_PLANTAR_DEG = 64.0
BALLET_ANKLE_STD_RAD = 0.28
BALLET_TARGET_TRUNK_Z = 0.129      # Equilibrium trunk height (m) on 66 deg tiptoes
BALLET_TRUNK_Z_STD = 0.010         # Gaussian std for trunk height (m)
BALLET_KNEE_STD_RAD = 0.20         # Extended stance knees tolerance (rad)

# Command ranges: poised, unhurried ballet steps
BALLET_LIN_VEL_X = (-0.12, 0.18)
BALLET_LIN_VEL_Y = (-0.12, 0.12)
BALLET_ANG_VEL_Z = (-0.7, 0.7)


def make_microduck_ballet_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck ballet tiptoe walking environment configuration."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # Widen BAM actuator delay randomization to 0..6 substeps (0..30 ms) so
    # deployment works reliably at both 0-step and 1-step lag in infer_policy.
    robot_cfg = deepcopy(cfg.scene.entities["robot"])
    for act_cfg in robot_cfg.articulation.actuators:
        if hasattr(act_cfg, "delay_min_lag"):
            act_cfg.delay_min_lag = 0
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}

    # 1. Adjust velocity command ranges for delicate ballet steps
    twist_cmd = cfg.commands["twist"]
    twist_cmd.ranges.lin_vel_x = BALLET_LIN_VEL_X
    twist_cmd.ranges.lin_vel_y = BALLET_LIN_VEL_Y
    twist_cmd.ranges.ang_vel_z = BALLET_ANG_VEL_Z
    twist_cmd.turn_in_place_min_frac = 0.2
    twist_cmd.rel_standing_envs = 0.25

    # 2. Exclude ankle, knee, and hip_pitch from base nominal pose tracking
    # and relax hip_roll during walking so weight can shift smoothly over stance foot
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].params["asset_cfg"].joint_names = (
            r"^(?!passive_|.*neck.*|.*head.*|.*ankle.*|.*knee.*|.*hip_pitch.*).*",
        )
        for std_key in ("std_standing", "std_walking", "std_running"):
            if std_key in cfg.rewards["pose"].params:
                d = dict(cfg.rewards["pose"].params[std_key])
                d.pop(".*knee.*", None)
                d.pop(".*ankle.*", None)
                d.pop(".*hip_pitch.*", None)
                if std_key in ("std_walking", "std_running") and r".*hip_roll.*" in d:
                    d[r".*hip_roll.*"] = 0.15
                cfg.rewards["pose"].params[std_key] = d

    # 3. Add Ballet Tiptoe specific rewards & penalties
    # Phase-aware ankle posture: 64 deg stance -> smooth 0.10s ramp to 68 deg swing, 66 deg stand
    cfg.rewards["ballet_ankle_tiptoe"] = RewardTermCfg(
        func=microduck_mdp.ballet_ankle_tiptoe_reward,
        weight=4.0,
        params={
            "target_plantar_deg": BALLET_TARGET_PLANTAR_DEG,
            "swing_plantar_deg": BALLET_SWING_PLANTAR_DEG,
            "stance_plantar_deg": BALLET_STANCE_PLANTAR_DEG,
            "std_rad": BALLET_ANKLE_STD_RAD,
        },
    )

    # Ankle asymmetry penalty (active during double support; free during single-support swing)
    cfg.rewards["ballet_ankle_asymmetry"] = RewardTermCfg(
        func=microduck_mdp.ballet_ankle_asymmetry_penalty,
        weight=-1.5,
    )

    # Flat foot ground contact penalty: enforce true en-pointe ground contacts (blocking flat soles)
    cfg.rewards["ballet_flat_foot"] = RewardTermCfg(
        func=microduck_mdp.ballet_flat_foot_contact_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "min_tiptoe_alignment": 0.55,
        },
    )

    # Extended knees reward (gated on tiptoe stance to prevent flat-foot crouching)
    cfg.rewards["ballet_straight_knees"] = RewardTermCfg(
        func=microduck_mdp.ballet_straight_knees_reward,
        weight=2.5,
        params={
            "std_rad": BALLET_KNEE_STD_RAD,
            "min_ankle_deg": 26.0,
            "target_ankle_deg": 55.0,
        },
    )

    # Elevated trunk height reward (gated on en-pointe stance to prevent flat-foot crouching)
    cfg.rewards["ballet_tall_trunk"] = RewardTermCfg(
        func=microduck_mdp.ballet_tall_trunk_height_reward,
        weight=2.5,
        params={
            "target_height": BALLET_TARGET_TRUNK_Z,
            "std_m": BALLET_TRUNK_Z_STD,
            "min_ankle_deg": 26.0,
            "target_ankle_deg": 55.0,
        },
    )

    # Graceful torso penalty: minimize pitch and roll tilt
    cfg.rewards["ballet_graceful_torso"] = RewardTermCfg(
        func=microduck_mdp.ballet_graceful_torso_penalty,
        weight=-2.0,
    )

    # Center stride midpoint at en-pointe CoM equilibrium (hip_pitch ~19.2 deg)
    cfg.rewards["ballet_hip_centering"] = RewardTermCfg(
        func=microduck_mdp.ballet_hip_centering_reward,
        weight=1.5,
        params={
            "target_hip_deg": 19.2,
            "std_rad": 0.30,
        },
    )

    # Penalize standing stagger/velocity and permanent one-leg-lead limping during walking
    cfg.rewards["ballet_hip_stagger"] = RewardTermCfg(
        func=microduck_mdp.ballet_hip_stagger_penalty,
        weight=-4.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.01,
            "tau_s": 1.0,
            "target_hip_deg": 19.2,
        },
    )

    # Slow, graceful stride cadence & high-clearance alternating crossover shaping
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].params["target_height"] = 0.033
    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].params["target_height"] = 0.033
    if "air_time" in cfg.rewards:
        cfg.rewards["air_time"].weight = 4.5
        cfg.rewards["air_time"].params["threshold_min"] = 0.16
        cfg.rewards["air_time"].params["threshold_max"] = 0.50

    cfg.rewards["ballet_graceful_stride"] = RewardTermCfg(
        func=microduck_mdp.ballet_graceful_stride_reward,
        weight=6.0,
    )
    cfg.rewards["ballet_short_step"] = RewardTermCfg(
        func=microduck_mdp.ballet_short_step_penalty,
        weight=-6.0,
    )
    cfg.rewards["ballet_no_stepping"] = RewardTermCfg(
        func=microduck_mdp.no_stepping_penalty,
        weight=-8.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )

    # Pure yaw-rate tracking reward (free of roll/pitch oscillation attenuation)
    cfg.rewards["track_yaw_fine"] = RewardTermCfg(
        func=microduck_mdp.track_yaw_rate_fine,
        weight=3.5,
        params={
            "std": 0.25,
            "command_name": "twist",
        },
    )

    # Strong linear velocity tracking so graceful strides actively propel at command speed
    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 7.5
        cfg.rewards["track_linear_velocity"].params["std"] = 0.10

    # Lock all 4 head/neck joints (pose + velocity) to eliminate counterweight whipping
    cfg.rewards["ballet_head_poise"] = RewardTermCfg(
        func=microduck_mdp.ballet_head_poise_penalty,
        weight=-6.0,
        params={"command_name": "head_pose"},
    )
    if "head_pose_tracking" in cfg.rewards:
        cfg.rewards["head_pose_tracking"].weight = 2.5
        cfg.rewards["head_pose_tracking"].params["std"] = 0.40
        cfg.rewards["head_pose_tracking"].params["fine_std"] = 0.10
        cfg.rewards["head_pose_tracking"].params["fine_weight"] = 0.5
    if "head_pose_bias" in cfg.rewards:
        cfg.rewards["head_pose_bias"].weight = 2.5

    # Strengthen upright posture & allow crisp tiptoe pivot steps
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = 3.0
    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].weight = -0.15
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.20

    # Curricula tuned for en-pointe slow passe gait over 150-350 iter horizon
    if "head_pose_bias_weight" in cfg.curriculum:
        cfg.curriculum["head_pose_bias_weight"].params["weight_stages"] = [
            {"step": 0, "weight": 2.5},
            {"step": 500 * 24, "weight": 3.0},
        ]
    if "head_pose_range" in cfg.curriculum:
        cfg.curriculum["head_pose_range"].params["range_stages"] = [
            {"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
            {"step": 500 * 24, "ranges": ((-0.15, 0.15), (-0.15, 0.15), (-0.30, 0.30), (-0.08, 0.08))},
        ]
    if "standing_envs" in cfg.curriculum:
        cfg.curriculum["standing_envs"].params["standing_stages"] = [
            {"step": 0, "rel_standing_envs": 0.25},
            {"step": 400 * 24, "rel_standing_envs": 0.25},
        ]
    if "action_rate_weight" in cfg.curriculum:
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0, "weight": -0.20},
            {"step": 150 * 24, "weight": -0.22},
            {"step": 250 * 24, "weight": -0.22},
        ]

    # Reset spawn height adjusted slightly higher for tiptoe stance
    if "reset_base" in cfg.events:
        cfg.events["reset_base"].params["pose_range"]["z"] = (0.128, 0.138)

    return cfg


# Runner configuration
MicroduckBalletRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRlCfg)
MicroduckBalletRlCfg.experiment_name = "ballet"
MicroduckBalletRlCfg.run_name = "ballet_tiptoe"
MicroduckBalletRlCfg.save_interval = 50
MicroduckBalletRlCfg.algorithm.symmetry_cfg = deepcopy(SYMMETRY_CFG)
MicroduckBalletRlCfg.algorithm.symmetry_cfg["mirror_loss_coeff"] = 1.0

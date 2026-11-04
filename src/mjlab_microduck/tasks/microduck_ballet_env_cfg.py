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
from mjlab.rl import RslRlOnPolicyRunnerCfg

import mjlab_microduck.tasks.mdp as microduck_mdp
from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)

# === Tuned Constants for Ballet Tiptoe ===
# 68 deg (~1.187 rad): foot nearly perpendicular to floor, heel lifted >40mm
BALLET_TARGET_PLANTAR_DEG = 68.0
# std_rad=0.30 (~17.2 deg): provides clear gradient (exp(-3.0) ≈ 0.05) from 26 deg seed policy
BALLET_ANKLE_STD_RAD = 0.30
BALLET_TARGET_TRUNK_Z = 0.125      # Elevated trunk height (m) on tiptoes (vs ~0.095m nominal)
BALLET_TRUNK_Z_STD = 0.015         # Gaussian std for trunk height (m)
BALLET_KNEE_STD_RAD = 0.20         # Extended knees tolerance (rad)

# Command ranges: delicate, compact steps
BALLET_LIN_VEL_X = (-0.15, 0.25)
BALLET_LIN_VEL_Y = (-0.15, 0.15)
BALLET_ANG_VEL_Z = (-0.8, 0.8)


def make_microduck_ballet_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck ballet tiptoe walking environment configuration."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # 1. Adjust velocity command ranges for delicate ballet steps
    twist_cmd = cfg.commands["twist"]
    twist_cmd.ranges.lin_vel_x = BALLET_LIN_VEL_X
    twist_cmd.ranges.lin_vel_y = BALLET_LIN_VEL_Y
    twist_cmd.ranges.ang_vel_z = BALLET_ANG_VEL_Z

    # 2. Exclude ankle and knee from base nominal pose tracking so it doesn't fight tiptoe/straight knees
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].params["asset_cfg"].joint_names = (
            r"^(?!passive_|.*neck.*|.*head.*|.*ankle.*|.*knee.*).*",
        )
        for std_key in ("std_standing", "std_walking", "std_running"):
            if std_key in cfg.rewards["pose"].params:
                d = dict(cfg.rewards["pose"].params[std_key])
                d.pop(".*knee.*", None)
                d.pop(".*ankle.*", None)
                cfg.rewards["pose"].params[std_key] = d

    # 3. Add Ballet Tiptoe specific rewards & penalties
    # Tiptoe ankle posture: reward maintaining ~68 deg plantar-flexion
    cfg.rewards["ballet_ankle_tiptoe"] = RewardTermCfg(
        func=microduck_mdp.ballet_ankle_tiptoe_reward,
        weight=4.0,
        params={
            "target_plantar_deg": BALLET_TARGET_PLANTAR_DEG,
            "std_rad": BALLET_ANKLE_STD_RAD,
        },
    )

    # Ankle asymmetry penalty: enforce symmetrical tiptoeing
    cfg.rewards["ballet_ankle_asymmetry"] = RewardTermCfg(
        func=microduck_mdp.ballet_ankle_asymmetry_penalty,
        weight=-1.5,
    )

    # Flat foot ground contact penalty: heavily penalize ground contacts if foot is flat
    cfg.rewards["ballet_flat_foot"] = RewardTermCfg(
        func=microduck_mdp.ballet_flat_foot_contact_penalty,
        weight=-2.5,
        params={
            "min_tiptoe_alignment": 0.50,
        },
    )

    # Extended knees reward (gated on tiptoe stance to prevent flat-foot cheating)
    cfg.rewards["ballet_straight_knees"] = RewardTermCfg(
        func=microduck_mdp.ballet_straight_knees_reward,
        weight=1.5,
        params={
            "std_rad": BALLET_KNEE_STD_RAD,
            "min_ankle_deg": 26.0,
            "target_ankle_deg": 55.0,
        },
    )

    # Elevated trunk height reward (gated on tiptoe stance to prevent flat-foot cheating)
    cfg.rewards["ballet_tall_trunk"] = RewardTermCfg(
        func=microduck_mdp.ballet_tall_trunk_height_reward,
        weight=2.0,
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

    # Strongly incentivize walking when velocity is commanded (prevent idle camping)
    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 3.5

    # Strengthen upright posture & foot slip prevention for crisp tiptoe steps
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = 3.0
    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].weight = -0.3

    # Reset spawn height adjusted slightly higher for tiptoe stance
    if "reset_base" in cfg.events:
        cfg.events["reset_base"].params["pose_range"]["z"] = (0.128, 0.138)

    return cfg


# Runner configuration
MicroduckBalletRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRlCfg)
MicroduckBalletRlCfg.experiment_name = "ballet"
MicroduckBalletRlCfg.run_name = "ballet_tiptoe"
MicroduckBalletRlCfg.save_interval = 50

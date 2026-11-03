"""Microduck Jump (점프) task configuration.

Run 89: Squat Jump with Enhanced Thrust, Landing Cushion & Standing Hold.
Run 88 verified actual liftoff (+18.4mm clearance, 140.6mm apex, Vz=+0.502m/s, Roll<3°).
Run 89 addresses the two user requirements:
  1. Stronger thrust off the ground (jump_takeoff_vz boosted 15→25, Vz scale 4.0).
  2. Compliant landing cushion: on touchdown (0.32-0.60s), knees flex and trunk dips
     to absorb shock (jump_landing_cushion weight +10.0), then smoothly rises to
     standing equilibrium (jump_landing_rise weight +12.0, Z=0.118m).
  3. Rebound hopping eliminated: takeoff rewards gated to step<=18, and post-touchdown
     liftoff strictly penalized (jump_rebound_hop weight -15.0).
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

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GROUND_PICK_ROBOT_CFG,
    MICRODUCK_GROUND_PICK_SQUAT_ROBOT_CFG,
    JUMP_TARGET_POSE,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

NUM_STEPS_PER_ENV = 24
EPISODE_LENGTH_S = 1.5  # 75 steps (0-9 extend, 9-20 liftoff, 20-40 flight, 40-75 land & hold)

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

JUMP_MIN_FOOT_LIFT = 0.004  # m above floor for flight rewards; foot sites rest at 0 mm


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

    cfg.scene.entities = {"robot": MICRODUCK_GROUND_PICK_SQUAT_ROBOT_CFG}
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
    # Clear ALL velocity locomotion terms
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

    # ── Run 92: Aerial Foot Clearance, Knee Tuck, and Clean Stand ──────────────
    # Phase 1: Squat-to-pose launch extension (step 0-14, t <= 0.28s)
    # Phase 2: High apex flight + Aerial foot clearance (step 6-18, t = 0.12-0.36s)
    # Phase 3: Landing cushion (step 16-30, symmetric knee Gaussian at 0.55 rad, Z ~ 0.100m)
    # Phase 4: Slewed rise from touchdown (step 16-50), not from step 28 where episodes already died
    # Anti-rebound: step >= 22 forbids second liftoff (weight -15.0)

    # FOUNDATION: Joint pose tracking for launch phase extension (restricted to step <= 14)
    cfg.rewards["squat_to_jump_pose"] = RewardTermCfg(
        func=microduck_mdp.squat_to_jump_pose_reward,
        weight=8.0,
        params={
            "target_joint_pos": JUMP_TARGET_POSE,
            "std_hip": 0.15,
            "std_knee": 0.10,
            "std_ankle": 0.10,
            "max_step": 14,
        },
    )

    # TAKEOFF VZ: positive upward velocity when BOTH feet are fully airborne (step <= 14).
    # Summed over the air steps this is the trunk's ballistic rise. At weight 30
    # Run 101 earned ~2.2 per episode against ~18.5 for the stand, so it tucked
    # its feet 15 mm instead of lifting the body. Foot sites rest at 0 mm;
    # below 4 mm the feet are still skimming the floor.
    cfg.rewards["jump_takeoff_vz"] = RewardTermCfg(
        func=microduck_mdp.jump_takeoff_vz_reward,
        weight=60.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "vz_scale": 4.0,
            "max_step": 14,
            "min_foot_lift": JUMP_MIN_FOOT_LIFT,
        },
    )

    # TRUE FLIGHT HEIGHT: body rise AND minimum bilateral foot clearance.
    # Run 108's independent terms paid only 0.69/episode and allowed a 9 mm
    # foot tuck while the 19.4-point stand term dominated. The product makes
    # squat extension without flight and leg tucking without body lift score 0.
    # At the current 141 mm trunk / ~9 mm feet it remains visibly on-gradient
    # (~0.10 per active step); the 100 weight gives a target flight comparable
    # mass to the landing phase without creating a persistent goal jackpot.
    cfg.rewards["jump_flight_height"] = RewardTermCfg(
        func=microduck_mdp.jump_flight_height_reward,
        weight=100.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "stand_z": 0.118,
            "target_z": 0.155,
            "min_foot_lift": JUMP_MIN_FOOT_LIFT,
            "target_foot_lift": 0.035,
            "min_step": 6,
            "max_step": 20,
        },
    )

    # ANTI-TIPTOE: forbid unilateral tiptoe cheat (ankle asymmetry and excessive plantar flexion)
    cfg.rewards["jump_anti_tiptoe"] = RewardTermCfg(
        func=microduck_mdp.jump_anti_tiptoe_penalty,
        weight=-6.0,
        params={"max_plantar_flexion_deg": 35.0},
    )

    # LANDING CUSHION: spring-like knee flexion & trunk dip on touchdown (step 16-38)
    cfg.rewards["jump_landing_cushion"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_cushion_reward,
        weight=10.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "min_step": 16,
            "max_step": 30,
            "target_cushion_z": 0.100,
            "std_cushion_z": 0.025,
            "min_knee_flexion": 0.55,
            "knee_sym_std": 0.40,
        },
    )

    # One-foot landing: Run 94 held symmetric knees while the right foot rose to 44 mm
    # and the left sole took all the load, before roll left 0°. Potential on the
    # foot-height gap, so closing pays once and oscillating does not.
    cfg.rewards["jump_foot_gap"] = RewardTermCfg(
        func=microduck_mdp.jump_foot_gap_shaping,
        weight=100.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "min_step": 14,
            "max_step": 30,
        },
    )

    # LANDING RISE & STAND: slewed rise from touchdown (step 16) to standing Z=0.118m
    cfg.rewards["jump_landing_rise"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_rise_reward,
        # Run 109 height discovery used 2.0 and raised the 256-env median apex
        # 140.9 -> 145.7 mm, but final stand fell to 109.5 mm. A 64-env pilot
        # at 5.0 collapsed the apex to 120 mm in four updates. Stage 2 therefore
        # uses 3.0: enough added landing slope without making no-jump optimal.
        weight=3.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "min_step": 16,
            "ramp_steps": 34,
            "target_stand_z": 0.118,
            "std_stand_z": 0.025,
            "target_stand_knee": 0.0,
            "std_stand_knee": 0.60,
            "cushion_init_z": 0.100,
            "cushion_init_knee": 0.55,
            "target_stand_ankle": 0.4530,
            "std_stand_ankle": 0.20,
            "cushion_init_ankle": 0.30,
            "target_stand_hip": 0.4579,
            "std_stand_hip": 0.50,
        },
    )

    # ANTI-REBOUND HOP: strictly penalize any feet liftoff after touchdown (step >= 24)
    cfg.rewards["jump_rebound_hop"] = RewardTermCfg(
        func=microduck_mdp.jump_rebound_hop_penalty,
        weight=-15.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "min_step": 24,
        },
    )

    # FOOT LANDING OFFSET: feet must come down where they took off. Start at the
    # flight/landing boundary so legitimate launch exploration is not taxed at
    # every pre-takeoff step; final displacement and stepping still cost.
    cfg.rewards["jump_foot_landing_offset"] = RewardTermCfg(
        func=microduck_mdp.jump_foot_landing_offset_penalty,
        weight=-90.0,
        params={"min_step": 14},
    )

    # GROUND FOOT SLIP (cost ≥ 0): horizontal speed only within 4 mm of floor.
    # Run 106 hopped ~7 mm at steps 4-6, swept the feet 22 mm back, re-landed
    # on its toes (site above 4 mm while touching), then swung them forward in
    # flight. Run 108 used max_height=1 m, so normal airborne foot repositioning
    # was charged throughout the episode and height regressed. Keeping the full
    # episode window still catches pre-launch sliding and post-landing steps.
    cfg.rewards["jump_foot_slip"] = RewardTermCfg(
        func=microduck_mdp.jump_foot_slip_penalty,
        weight=-30.0,
        params={"max_step": 75, "max_height": JUMP_MIN_FOOT_LIFT},
    )

    # IMPACT: Penalize hard landings — descending velocity at foot contact
    cfg.rewards["jump_impact"] = RewardTermCfg(
        func=microduck_mdp.jump_impact_penalty,
        weight=-3.0,
        params={"sensor_name": feet_ground_cfg.name},
    )

    # UPRIGHT: Direct upright reward (std 6° — same tight balance as Run 87)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].params["std"] = math.radians(6.0)
    cfg.rewards["upright"].weight = 8.0

    # PITCH penalty (≤ 0, positive weight): blocks torso pitch > 8° during liftoff
    cfg.rewards["body_pitch"] = RewardTermCfg(
        func=microduck_mdp.body_pitch_penalty,
        weight=8.0,
    )

    # ROLL tilt penalty (≤ 0, negative weight): the Run-86 killer kept at -8
    cfg.rewards["jump_roll_tilt"] = RewardTermCfg(
        func=microduck_mdp.jump_roll_tilt_penalty,
        weight=-8.0,
    )

    # HIP bow (≤ 0, positive weight): forbids trunk collapsing forward at hips
    cfg.rewards["hip_bow"] = RewardTermCfg(
        func=microduck_mdp.hip_bow_penalty,
        weight=6.0,
    )

    # HIP lateral spread (≤ 0): prevents legs splaying sideways
    cfg.rewards["hip_lateral_spread"] = RewardTermCfg(
        func=microduck_mdp.hip_lateral_abduction_penalty,
        weight=-4.0,
    )

    # HIP yaw neutral (≤ 0): locks hip_yaw=0, prevents torque around Z
    cfg.rewards["hip_yaw_neutral"] = RewardTermCfg(
        func=microduck_mdp.hip_yaw_neutral_penalty,
        weight=-3.0,
    )

    # LATERAL drift (≤ 0): penalizes sideways velocity and displacement
    cfg.rewards["jump_lateral_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_drift_penalty,
        weight=-3.0,
    )

    # SAGITTAL drift (≤ 0): penalizes forward leaping. Both drift terms measure
    # from the spawn spot in the spawn heading frame; up to Run 105 they used
    # env_origins in world axes, which reset_base offsets by ±0.5 m at any yaw.
    cfg.rewards["jump_sagittal_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_sagittal_drift_penalty,
        # Run 110 model_33950 balances a 146.1 mm apex, +13.5 mm forward drift,
        # +1.2° yaw and 117.9 mm final stand over 1024 DR rollouts. Increasing
        # this to -8 after yaw refinement moved the cheat back into +14° yaw.
        weight=-4.0,
    )

    # YAW drift (cost ≥ 0): Run 105 turned 12° in flight, landing L +29 / R +8 mm.
    # At -10 Run 107 still ended at median +6° (p90 +16°).
    cfg.rewards["jump_yaw_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_drift_penalty,
        # Run 110 model_33950 ends at +1.2° yaw. Increasing this to -40 fixed
        # one checkpoint's yaw but displaced the policy +20.3 mm forward; the
        # following sagittal refinement then regressed to +14.4° yaw.
        weight=-20.0,
    )

    # HEAD neutral + action freeze: keeps head calm
    cfg.rewards["head_neutral"] = RewardTermCfg(
        func=microduck_mdp.head_neutral_penalty,
        weight=-3.0,
    )
    cfg.rewards["head_action_l2"] = RewardTermCfg(
        func=microduck_mdp.head_action_l2,
        weight=-3.0,
    )

    # SELF COLLISION sensor
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    cfg.rewards["dof_pos_limits"].weight = -0.5
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.01

    # ACTION smoothness (start near-zero, curriculum ramp)
    cfg.rewards["action_rate_l2"].weight = -0.01

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["time_out"].time_out = True
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    # Strict 20° fell_over: any lateral lean ≥20° immediately terminates.
    # This is the primary fix for Run-86's sideways roll bias — rapid episode
    # termination under lateral tilt drives the policy to avoid it entirely.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        time_out=False,
        params={"limit_angle": math.radians(20.0)},
    )
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
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.073, 0.077)
    # After the squat reset: some episodes start in the measured crouch→stand
    # blend so the rise has on-policy data. Must stay after reset_robot_joints.
    cfg.events["reset_stand_curriculum"] = EventTermCfg(
        func=microduck_mdp.reset_stand_curriculum,
        mode="reset",
        # Stage 2 restores recovery data after height discovery reduced the
        # final stand from 114.5 mm to 109.6 mm.
        params={"fraction": 0.25},
    )

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
                # Run 110 resumes model_33450 for 1000 iterations. Do not raise
                # the attempt tax while preserving height and restoring landing;
                # 35500 is beyond the planned end at iter 34450.
                {"step": 0, "weight": -0.05},
                {"step": 35500 * 24, "weight": -0.15},
                {"step": 37500 * 24, "weight": -0.30},
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
    run_name="run110_preserve_height_restore_landing",
    save_interval=25,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=12000,
)


"""Arm v5: learn arm + gripper on a FREE-STANDING Spot whose legs run a frozen walking policy."""
import copy
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .env_cfg import SPOT_ARM_FIXED_CFG, SpotArmCubeLiftEnvCfg_V4

WALK_POLICY = "@SPOT_DEMO_ROOT@/policies/spot_arm_flat_v2/exported/policy.pt"
LEG_JOINTS = "[fh][lr]_.*"
ARM_VEL_LIMIT = 3.0  # rad/s per arm joint. [Guessing] realistic cap; USD default was 999


@configclass
class FrozenWalkActionCfg(JointPositionActionCfg):
    # referenced by name, resolved only after the simulator is running (Isaac Lab 3.0 convention)
    class_type: type | str = "{DIR}.frozen_walk_action:FrozenWalkAction"
    policy_path: str = MISSING
    obs_group: str = "walk"


@configclass
class WalkObsCfg(ObsGroup):
    # exactly the walking policy's 48-dim input (order matters)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[LEG_JOINTS])})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[LEG_JOINTS])})
    actions = ObsTerm(func=mdp.last_action, params={"action_name": "legs"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


SPOT_V5_CFG = copy.deepcopy(SPOT_ARM_FIXED_CFG)
SPOT_V5_CFG.spawn.articulation_props.fix_root_link = False
SPOT_V5_CFG.init_state.pos = (0.0, 0.0, 0.6)
SPOT_V5_CFG.actuators = {
    **SPOT_V5_CFG.actuators,
    "spot_arm": ImplicitActuatorCfg(
        joint_names_expr=["arm0_.*"], stiffness=400.0, damping=40.0, velocity_limit_sim=ARM_VEL_LIMIT
    ),
}


@configclass
class SpotArmStandLiftEnvCfg_V5(SpotArmCubeLiftEnvCfg_V4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPOT_V5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
            spawn=GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)),
        )
        self.events.reset_robot_base = None          # body is free now; no pinned-pose randomization

        # learned action = arm (6) + gripper (1); legs come from the frozen walking policy (0 dims)
        self.actions.leg_hold = None
        self.actions.legs = FrozenWalkActionCfg(
            asset_name="robot", joint_names=[LEG_JOINTS], scale=0.2, use_default_offset=True,
            policy_path=WALK_POLICY, obs_group="walk",
        )

        # walking policy needs its own observation group and a zero velocity command
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(100.0, 100.0),
            rel_standing_envs=1.0,
            heading_command=False,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        )
        self.observations.walk = WalkObsCfg()

        # Spot fell: tilted past ~45 deg or body below 0.30 m
        self.terminations.base_tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
        self.terminations.base_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.30})


@configclass
class SpotArmStandLiftEnvCfg_V5_PLAY(SpotArmStandLiftEnvCfg_V5):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True


@configclass
class SpotArmStandLiftEnvCfg_V5_CONT(SpotArmStandLiftEnvCfg_V5):
    """For RESUMED runs: penalties fixed at their final weights. The curriculum step counter restarts on resume,
    which would otherwise switch the penalties off again and then snap them back on."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate.weight = -0.1
        self.rewards.joint_vel.weight = -0.1
        self.rewards.gripper_toggle.weight = -0.2
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None
        self.curriculum.gripper_toggle = None

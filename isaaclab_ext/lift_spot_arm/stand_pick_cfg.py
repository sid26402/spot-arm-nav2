import copy
import os
_DZ = float(os.environ.get("STAND_DZ", "0.0"))   # lower bench + cube by this many metres

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot_arm.flat_env_cfg import (
    LEG_JOINTS, SPOT_ARM_CFG, SpotArmFlatEnvCfg,
)

from .env_cfg import ARM_JOINTS, BENCH_H, BENCH_X, CUBE_Z, READY

# Free-standing Spot (NOT pinned), arm starting in the same READY pose the arm policy was trained from
SPOT_STAND_CFG = copy.deepcopy(SPOT_ARM_CFG)
SPOT_STAND_CFG.spawn.usd_path = "@SPOT_ASSETS@/spot_with_arm_v3.usd"
SPOT_STAND_CFG.init_state.pos = (0.0, 0.0, 0.6)
SPOT_STAND_CFG.init_state.joint_pos = {**SPOT_ARM_CFG.init_state.joint_pos, **READY}


@configclass
class StandPickActionsCfg:
    # order matters for the runner: legs (12) from walking policy, then arm (6) + gripper (1) from arm policy
    legs = loco_mdp.JointPositionActionCfg(asset_name="robot", joint_names=[LEG_JOINTS], scale=0.2, use_default_offset=True)
    arm_action = lift_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True)
    gripper_action = lift_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm0_f1x"],
        open_command_expr={"arm0_f1x": -1.0},
        close_command_expr={"arm0_f1x": 0.0},
    )


@configclass
class ArmObsCfg(ObsGroup):
    # identical order to the arm training env: 7 + 7 + 3 + 7 + 6 + 1 = 31
    joint_pos = ObsTerm(func=lift_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm0_.*"])})
    joint_vel = ObsTerm(func=lift_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm0_.*"])})
    object_position = ObsTerm(func=lift_mdp.object_position_in_robot_root_frame)
    target_object_position = ObsTerm(func=lift_mdp.generated_commands, params={"command_name": "object_pose"})
    actions = ObsTerm(func=lift_mdp.last_action, params={"action_name": "arm_action"})
    gripper_last = ObsTerm(func=lift_mdp.last_action, params={"action_name": "gripper_action"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SpotArmStandPickEnvCfg(SpotArmFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.episode_length_s = 8.0

        self.scene.robot = SPOT_STAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            debug_vis=False,
        )
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.arm_contact = None          # the arm is supposed to touch the cube now

        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(BENCH_X, 0.0, (BENCH_H - _DZ) / 2)),
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 1.2, BENCH_H - _DZ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.30, 0.25)),
            ),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[BENCH_X, 0.0, CUBE_Z - _DZ], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.actions = StandPickActionsCfg()

        # walking: always "stand still"; arm: one goal for the whole episode
        self.commands.base_velocity.ranges = loco_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )
        self.commands.base_velocity.debug_vis = False
        self.commands.object_pose = lift_mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="arm0_link_wr1",
            resampling_time_range=(100.0, 100.0),
            debug_vis=True,
            ranges=lift_mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.7, 1.0), pos_y=(-0.25, 0.25), pos_z=(0.05, 0.30), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            ),
        )

        self.observations.policy.enable_corruption = False
        self.observations.arm = ArmObsCfg()

        # clean test: no random spawn pose, kicks, pushes or mass changes; cube position randomized as in training
        ev = self.events
        ev.reset_base.params = {**ev.reset_base.params, "pose_range": {}, "velocity_range": {}}
        ev.reset_robot_joints.params = {**ev.reset_robot_joints.params, "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
        ev.push_robot = None
        ev.add_base_mass = None
        ev.reset_object = EventTerm(
            func=lift_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

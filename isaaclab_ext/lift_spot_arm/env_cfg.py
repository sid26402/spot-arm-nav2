import copy
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot_arm.flat_env_cfg import SPOT_ARM_CFG

USD_PATH = "@SPOT_ASSETS@/spot_with_arm_v3.usd"
BASE_Z = 0.55
BENCH_X, BENCH_H = 0.85, 0.40
CUBE_Z = BENCH_H + 0.055
LIFTED_Z = BENCH_H + 0.06
ARM_JOINTS = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"]
READY = {"arm0_sh0": 0.0, "arm0_sh1": -0.9, "arm0_el0": 1.8, "arm0_el1": 0.0,
         "arm0_wr0": -0.9, "arm0_wr1": 0.0, "arm0_f1x": -1.0}

SPOT_ARM_FIXED_CFG = copy.deepcopy(SPOT_ARM_CFG)
SPOT_ARM_FIXED_CFG.spawn.usd_path = USD_PATH
SPOT_ARM_FIXED_CFG.spawn.articulation_props.fix_root_link = True
SPOT_ARM_FIXED_CFG.init_state.pos = (0.0, 0.0, BASE_Z)
SPOT_ARM_FIXED_CFG.init_state.joint_pos = {**SPOT_ARM_CFG.init_state.joint_pos, **READY}


def _t(x):
    if torch.is_tensor(x):
        return x
    import warp as wp
    return wp.to_torch(x)


def arm_action_rate(env) -> torch.Tensor:
    """Action-rate penalty on the 6 arm joints only (gripper and leg-hold excluded)."""
    am = env.action_manager
    return torch.sum(torch.square(am.action[:, :6] - am.prev_action[:, :6]), dim=1)


def gripper_toggle(env) -> torch.Tensor:
    """1.0 on every step the gripper command flips between open and closed."""
    am = env.action_manager
    return ((am.action[:, 6] < 0) != (am.prev_action[:, 6] < 0)).float()


def object_grasped(env, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """1.0 when the fingertip frame is at the cube AND the finger is stopped partway (pinching)."""
    ee = _t(env.scene["ee_frame"].data.target_pos_w)[..., 0, :]
    obj = _t(env.scene["object"].data.root_pos_w)
    near = torch.norm(obj - ee, dim=1) < threshold
    f1x = _t(env.scene["robot"].data.joint_pos)[:, asset_cfg.joint_ids].reshape(-1)
    pinching = (f1x > -0.9) & (f1x < -0.2)
    return (near & pinching).float()


def object_raised_while_grasped(env, rest_z: float, span: float, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """0..1 for raising the cube 0..span metres above its resting height while pinching it."""
    pinch = object_grasped(env, threshold, asset_cfg)
    z = _t(env.scene["object"].data.root_pos_w)[:, 2]
    return pinch * torch.clamp((z - rest_z) / span, 0.0, 1.0)


def gripper_closed_empty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """1.0 when the gripper is (almost) fully closed with nothing between the jaws."""
    f1x = _t(env.scene["robot"].data.joint_pos)[:, asset_cfg.joint_ids].reshape(-1)
    return (f1x > -0.15).float()


def ee_below_bench(env, bench_top: float, margin: float) -> torch.Tensor:
    """0..1 as the fingertip frame drops below bench top + margin (arm scraping the bench)."""
    ee_z = _t(env.scene["ee_frame"].data.target_pos_w)[..., 0, 2]
    return torch.clamp((bench_top + margin - ee_z) / 0.02, 0.0, 1.0)


def cube_pushed(env, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """0..1 cube horizontal speed while it is NOT being pinched (knocking / shoving it)."""
    v = _t(env.scene["object"].data.root_lin_vel_w)[:, :2]
    pinch = object_grasped(env, threshold, asset_cfg)
    return (1.0 - pinch) * torch.clamp(torch.norm(v, dim=1) / 0.2, 0.0, 1.0)


def arm_cfg():
    return SceneEntityCfg("robot", joint_names=["arm0_.*"])


@configclass
class SpotArmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SPOT_ARM_FIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
            spawn=GroundPlaneCfg(),
        )
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(BENCH_X, 0.0, BENCH_H / 2)),
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 1.2, BENCH_H),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.30, 0.25)),
            ),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[BENCH_X, 0.0, CUBE_Z], rot=[0, 0, 0, 1]),
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

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/arm0_link_wr1",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.19, 0.0, 0.0]),
                ),
            ],
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm0_f1x"],
            open_command_expr={"arm0_f1x": -1.0},
            close_command_expr={"arm0_f1x": 0.0},
        )
        self.actions.leg_hold = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["[fh][lr]_.*"], scale=0.0, use_default_offset=True
        )

        self.commands.object_pose.body_name = "arm0_link_wr1"
        self.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.7, 1.0), pos_y=(-0.25, 0.25), pos_z=(0.05, 0.30), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )

        pol = self.observations.policy
        pol.joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": arm_cfg()})
        pol.joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": arm_cfg()})
        pol.actions = ObsTerm(func=mdp.last_action, params={"action_name": "arm_action"})
        pol.gripper_last = ObsTerm(func=mdp.last_action, params={"action_name": "gripper_action"})

        r = self.rewards
        r.action_rate.func = arm_action_rate
        r.joint_vel.params = {"asset_cfg": arm_cfg()}
        r.lifting_object.params = {**r.lifting_object.params, "minimal_height": LIFTED_Z}
        r.object_goal_tracking.params = {**r.object_goal_tracking.params, "minimal_height": LIFTED_Z}
        r.object_goal_tracking_fine_grained.params = {
            **r.object_goal_tracking_fine_grained.params, "minimal_height": LIFTED_Z
        }
        r.grasping_object = RewTerm(
            func=object_grasped,
            params={"threshold": 0.04, "asset_cfg": SceneEntityCfg("robot", joint_names=["arm0_f1x"])},
            weight=5.0,
        )

        self.terminations.object_dropping.params = {
            **self.terminations.object_dropping.params, "minimum_height": BENCH_H - 0.10
        }


@configclass
class SpotArmCubeLiftEnvCfg_PLAY(SpotArmCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True


@configclass
class SpotArmCubeLiftEnvCfg_V4(SpotArmCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # same physics rate as the walking env: 500 Hz physics, 50 Hz policy
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        # pinned body pose randomized to cover a real standing Spot: height 0.48-0.56 m, tilt about +-4.6 deg
        self.events.reset_robot_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"z": (-0.07, 0.01), "roll": (-0.08, 0.08), "pitch": (-0.08, 0.08)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # anti-chatter: gripper flip penalty, off until grasping is learned (~iteration 1000), then -0.2
        self.rewards.gripper_toggle = RewTerm(func=gripper_toggle, weight=0.0)
        self.curriculum.gripper_toggle = CurrTerm(
            func=mdp.modify_reward_weight, params={"term_name": "gripper_toggle", "weight": -0.2, "num_steps": 30000}
        )
        # smoothness penalties: v4 learns ~2x slower at 500 Hz, so delay them to ~iteration 1500 (v2 used ~417)
        self.curriculum.action_rate.params = {**self.curriculum.action_rate.params, "num_steps": 36000}
        self.curriculum.joint_vel.params = {**self.curriculum.joint_vel.params, "num_steps": 36000}
        # v6 rewards: react to the cube instead of a fixed routine
        f1x = lambda: SceneEntityCfg("robot", joint_names=["arm0_f1x"])
        self.rewards.grasping_object.weight = 2.0
        self.rewards.lift_shaping = RewTerm(
            func=object_raised_while_grasped,
            params={"rest_z": BENCH_H + 0.024, "span": 0.05, "threshold": 0.04, "asset_cfg": f1x()}, weight=8.0)
        self.rewards.closed_empty = RewTerm(func=gripper_closed_empty, params={"asset_cfg": f1x()}, weight=-2.0)
        self.rewards.scrape_bench = RewTerm(func=ee_below_bench, params={"bench_top": BENCH_H, "margin": 0.015}, weight=-3.0)
        self.rewards.cube_pushed = RewTerm(func=cube_pushed, params={"threshold": 0.04, "asset_cfg": f1x()}, weight=-2.0)


@configclass
class SpotArmCubeLiftEnvCfg_V4_PLAY(SpotArmCubeLiftEnvCfg_V4):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True

"""v7a: generalist TOP-DOWN grasp-and-lift on a free-standing Spot.
Surface height random 0.0-0.65 m (kinematic platform), stiff arm (kp 4000), object orientation observed.
Built on v5 (frozen walking legs, 500 Hz, v6 rewards, delayed penalties)."""
import copy
import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass

from .env_cfg import _t, object_grasped
from .stand_train_cfg import SPOT_V5_CFG, SpotArmStandLiftEnvCfg_V5

PLAT_X, PLAT_H = 0.85, 1.2            # platform centre in front of the body, platform box height
TOP_MIN, TOP_MAX = 0.0, 0.65          # surface height range: floor .. just above Spot's back
CUBE_HALF = 0.024
ARM_KP, ARM_KD, ARM_VEL = 4000.0, 40.0, 3.0
# arm raised high, gripper pointing DOWN, above the highest surface. [Guessing] angles: verify visually
READY_HIGH = {"arm0_sh0": 0.0, "arm0_sh1": -1.40, "arm0_el0": 1.39, "arm0_el1": 0.0,
              "arm0_wr0": 1.58, "arm0_wr1": 0.0, "arm0_f1x": -1.0}

SPOT_V7_CFG = copy.deepcopy(SPOT_V5_CFG)
SPOT_V7_CFG.init_state.joint_pos = {**SPOT_V5_CFG.init_state.joint_pos, **READY_HIGH}
SPOT_V7_CFG.actuators = {
    **SPOT_V5_CFG.actuators,
    "spot_arm": ImplicitActuatorCfg(joint_names_expr=["arm0_.*"], stiffness=ARM_KP, damping=ARM_KD, velocity_limit_sim=ARM_VEL),
}


def _top(env):
    return _t(env.scene["table"].data.root_pos_w)[:, 2] + PLAT_H / 2


def reset_platform_and_object(env, env_ids, top_range, xy_range, yaw_range):
    """Random surface height per episode; cube placed ON that surface (one function = guaranteed order)."""
    plat, obj = env.scene["table"], env.scene["object"]
    dev = obj.device
    ids = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=dev)
    n = len(ids)
    origins = env.scene.env_origins[ids]
    top = math_utils.sample_uniform(top_range[0], top_range[1], (n,), device=dev)
    pp = plat.data.default_root_pose.torch[ids].clone()
    pp[:, 0:3] += origins
    pp[:, 2] = top - PLAT_H / 2
    plat.write_root_pose_to_sim_index(root_pose=pp, env_ids=ids)
    op = obj.data.default_root_pose.torch[ids].clone()
    op[:, 0:3] += origins
    op[:, 0] += math_utils.sample_uniform(xy_range[0][0], xy_range[0][1], (n,), device=dev)
    op[:, 1] += math_utils.sample_uniform(xy_range[1][0], xy_range[1][1], (n,), device=dev)
    op[:, 2] = top + CUBE_HALF + 0.005
    yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (n,), device=dev)
    zero = torch.zeros_like(yaw)
    op[:, 3:7] = math_utils.quat_mul(op[:, 3:7], math_utils.quat_from_euler_xyz(zero, zero, yaw))
    obj.write_root_pose_to_sim_index(root_pose=op, env_ids=ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)


def object_quat_in_robot_frame(env):
    rq = _t(env.scene["robot"].data.root_quat_w)
    oq = _t(env.scene["object"].data.root_quat_w)
    return math_utils.quat_mul(math_utils.quat_inv(rq), oq)


def object_lifted_rel(env, height):
    return ((_t(env.scene["object"].data.root_pos_w)[:, 2] - _top(env)) > height).float()


def hold_height(env, target, std, min_lift):
    dz = _t(env.scene["object"].data.root_pos_w)[:, 2] - _top(env)
    return (dz > min_lift).float() * (1.0 - torch.tanh(torch.abs(dz - target) / std))


def raised_while_grasped_rel(env, span, threshold, asset_cfg):
    pinch = object_grasped(env, threshold, asset_cfg)
    dz = _t(env.scene["object"].data.root_pos_w)[:, 2] - _top(env) - CUBE_HALF
    return pinch * torch.clamp(dz / span, 0.0, 1.0)


def ee_below_surface_near(env, margin, near):
    ee = _t(env.scene["ee_frame"].data.target_pos_w)[..., 0, :]
    ob = _t(env.scene["object"].data.root_pos_w)
    close = (torch.norm(ee[:, :2] - ob[:, :2], dim=1) < near).float()
    return close * torch.clamp((_top(env) + margin - ee[:, 2]) / 0.02, 0.0, 1.0)


def gripper_points_down(env, near):
    """0..1: gripper axis pointing down, weighted up as the fingertip gets close to the object."""
    q = _t(env.scene["ee_frame"].data.target_quat_w)[..., 0, :]
    fwd = math_utils.quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3))
    ee = _t(env.scene["ee_frame"].data.target_pos_w)[..., 0, :]
    ob = _t(env.scene["object"].data.root_pos_w)
    w = torch.clamp(1.0 - torch.norm(ee - ob, dim=1) / near, 0.0, 1.0)
    return w * torch.clamp(-fwd[:, 2], 0.0, 1.0)


def object_below_surface(env, margin):
    return _t(env.scene["object"].data.root_pos_w)[:, 2] < _top(env) - margin


@configclass
class SpotArmV7aEnvCfg(SpotArmStandLiftEnvCfg_V5):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPOT_V7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(PLAT_X, 0.0, 0.4 - PLAT_H / 2)),
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 1.2, PLAT_H),
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.30, 0.25)),
            ),
        )
        # one event moves the platform to a random height and puts the cube on it
        self.events.reset_object_position = None
        self.events.reset_platform_object = EventTerm(
            func=reset_platform_and_object, mode="reset",
            params={"top_range": (TOP_MIN, TOP_MAX), "xy_range": ((-0.10, 0.10), (-0.20, 0.20)), "yaw_range": (-math.pi, math.pi)},
        )
        # observations: no goal point; add object orientation (body frame)
        pol = self.observations.policy
        pol.target_object_position = None
        pol.object_orientation = ObsTerm(func=object_quat_in_robot_frame)
        # rewards: lift task relative to the surface + top-down shaping (v6 terms kept)
        r = self.rewards
        f1x = lambda: SceneEntityCfg("robot", joint_names=["arm0_f1x"])
        r.lifting_object = None
        r.object_goal_tracking = None
        r.object_goal_tracking_fine_grained = None
        r.lift_shaping = None
        r.scrape_bench = None
        r.lifted_rel = RewTerm(func=object_lifted_rel, params={"height": 0.06}, weight=15.0)
        r.hold_height = RewTerm(func=hold_height, params={"target": 0.15, "std": 0.05, "min_lift": 0.04}, weight=16.0)
        r.hold_height_fine = RewTerm(func=hold_height, params={"target": 0.15, "std": 0.02, "min_lift": 0.04}, weight=5.0)
        r.lift_shaping_rel = RewTerm(func=raised_while_grasped_rel, params={"span": 0.05, "threshold": 0.04, "asset_cfg": f1x()}, weight=8.0)
        r.scrape_surface = RewTerm(func=ee_below_surface_near, params={"margin": 0.015, "near": 0.2}, weight=-3.0)
        r.top_down = RewTerm(func=gripper_points_down, params={"near": 0.4}, weight=3.0)
        # cube falling off the surface ends the episode (relative to the surface)
        self.terminations.object_dropping = DoneTerm(func=object_below_surface, params={"margin": 0.10})


@configclass
class SpotArmV7aEnvCfg_CONT(SpotArmV7aEnvCfg):
    """Resumed runs: penalties fixed at final weights (curricula restart on resume)."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate.weight = -0.1
        self.rewards.joint_vel.weight = -0.1
        self.rewards.gripper_toggle.weight = -0.2
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None
        self.curriculum.gripper_toggle = None


@configclass
class SpotArmV7aEnvCfg_PLAY(SpotArmV7aEnvCfg_CONT):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True

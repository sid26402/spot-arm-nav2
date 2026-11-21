"""v8: multi-object top-down picking (cube, foam brick, marker), then + random bench height.
Built on v7e (grasp counts only if commanded closed AND finger stalled). Observations unchanged (31 -> 7)
so every stage resumes from the previous one. All object origins are at their geometric centre (object_centres.py)."""
import copy

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms

from .env_cfg import _t
from .v7_cfg import PLAT_X, PLAT_H, _top, reset_platform_and_object, ee_below_surface_near, object_below_surface
from .v7d_cfg import SpotArmV7eEnvCfg, SpotArmV7eEnvCfg_PLAY, cube_held

PROPS = "@SPOT_ASSETS@/props"
EXTRA_OBJECTS = [f"{PROPS}/061_foam_brick_physics.usd", f"{PROPS}/040_large_marker_physics.usd"]
TOP_RANGE = (0.15, 0.50)      # bench top height range for v8b (start pose READY_HIGH clears 0.65)
REST_OFFSET = 0.024           # 'resting' reference above the surface (cube half-height)
LIFT_MIN = 0.06               # lifted = more than 6 cm above the surface


def _apply_multi_object(cfg):
    cube = copy.deepcopy(cfg.scene.object.spawn)
    rp = getattr(cube, "rigid_props", None)
    assets = [cube] + [sim_utils.UsdFileCfg(usd_path=p, rigid_props=copy.deepcopy(rp)) for p in EXTRA_OBJECTS]
    cfg.scene.object.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets, random_choice=True)
    cfg.scene.replicate_physics = False
    # 4096 separately-built scenes overflow Isaac Lab's lift defaults (aggregate pairs 16384 -> PhysX needed ~16400):
    # contacts were dropped and ~20% of robots sank through the floor (v8a). Enlarge the GPU buffers generously.
    px = cfg.sim.physics          # Isaac Lab 3.0: physics backend settings (was cfg.sim.physx)
    for name, value in (("gpu_total_aggregate_pairs_capacity", 2 ** 21), ("gpu_found_lost_aggregate_pairs_capacity", 2 ** 26),
                        ("gpu_found_lost_pairs_capacity", 2 ** 23), ("gpu_max_rigid_contact_count", 2 ** 24),
                        ("gpu_max_rigid_patch_count", 2 ** 21)):
        if px is not None and hasattr(px, name):
            setattr(px, name, max(int(getattr(px, name) or 0), value))


def lifted_rel(env, minimal_height):
    return ((_t(env.scene["object"].data.root_pos_w)[:, 2] - _top(env)) > minimal_height).float()


def raised_while_held_rel(env, rest_z, span, threshold, asset_cfg):
    """v7e lift bridge, but 'resting height' is measured from the current surface (rest_z ignored)."""
    z = _t(env.scene["object"].data.root_pos_w)[:, 2]
    return cube_held(env, asset_cfg) * torch.clamp((z - _top(env) - REST_OFFSET) / span, 0.0, 1.0)


def goal_distance_rel(env, std, minimal_height, command_name):
    robot, obj = env.scene["robot"], env.scene["object"]
    cmd = _t(env.command_manager.get_command(command_name))
    goal_w, _ = combine_frame_transforms(_t(robot.data.root_pos_w)[:, :3], _t(robot.data.root_quat_w), cmd[:, :3])
    pos = _t(obj.data.root_pos_w)[:, :3]
    lifted = (pos[:, 2] - _top(env)) > minimal_height
    return lifted.float() * (1.0 - torch.tanh(torch.norm(goal_w - pos, dim=1) / std))


def _apply_heights(cfg):
    cfg.scene.table = RigidObjectCfg(
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
    cfg.events.reset_object_position = None
    cfg.events.reset_platform_object = EventTerm(
        func=reset_platform_and_object, mode="reset",
        params={"top_range": TOP_RANGE, "xy_range": ((-0.10, 0.10), (-0.25, 0.25)), "yaw_range": (0.0, 0.0)},
    )
    r = cfg.rewards
    std = r.object_goal_tracking.params.get("std", 0.3)
    std_fine = r.object_goal_tracking_fine_grained.params.get("std", 0.05)
    r.lifting_object = RewTerm(func=lifted_rel, params={"minimal_height": LIFT_MIN}, weight=r.lifting_object.weight)
    r.object_goal_tracking = RewTerm(func=goal_distance_rel, params={"std": std, "minimal_height": LIFT_MIN, "command_name": "object_pose"},
                                     weight=r.object_goal_tracking.weight)
    r.object_goal_tracking_fine_grained = RewTerm(func=goal_distance_rel, params={"std": std_fine, "minimal_height": LIFT_MIN, "command_name": "object_pose"},
                                                  weight=r.object_goal_tracking_fine_grained.weight)
    r.lift_shaping.func = raised_while_held_rel
    r.scrape_bench = RewTerm(func=ee_below_surface_near, params={"margin": 0.015, "near": 0.2}, weight=-3.0)
    cfg.terminations.object_dropping = DoneTerm(func=object_below_surface, params={"margin": 0.10})


@configclass
class SpotArmV8aEnvCfg(SpotArmV7eEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_multi_object(self)


@configclass
class SpotArmV8aEnvCfg_PLAY(SpotArmV7eEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_multi_object(self)


@configclass
class SpotArmV8bEnvCfg(SpotArmV8aEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_heights(self)


@configclass
class SpotArmV8bEnvCfg_PLAY(SpotArmV8aEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_heights(self)


# ===== v9a (2026-11-02): CUBE ONLY + bench height CURRICULUM (v8b2 collapsed when thrown into the full range at once) =====
from isaaclab.managers import CurriculumTermCfg as CurrTerm

H_START, H_END = (0.37, 0.43), TOP_RANGE           # start near the trained 0.40 m bench, widen to 0.15-0.50 m
H_STEPS = 4000 * 24                                 # ~4000 iterations (24 env steps / iteration); restarts on resume


def widen_platform_range(env, env_ids, start_range, end_range, num_steps):
    frac = min(1.0, float(env.common_step_counter) / float(num_steps))
    lo = start_range[0] + (end_range[0] - start_range[0]) * frac
    hi = start_range[1] + (end_range[1] - start_range[1]) * frac
    tc = env.event_manager.get_term_cfg("reset_platform_object")
    tc.params["top_range"] = (lo, hi)
    env.event_manager.set_term_cfg("reset_platform_object", tc)
    return hi - lo                                    # logged as Curriculum/platform_range (width in m)


def _apply_v9a(cfg):
    _apply_heights(cfg)
    cfg.events.reset_platform_object.params["top_range"] = H_START
    cfg.curriculum.platform_range = CurrTerm(func=widen_platform_range,
                                             params={"start_range": H_START, "end_range": H_END, "num_steps": H_STEPS})
    px = cfg.sim.physics
    for name, value in (("gpu_total_aggregate_pairs_capacity", 2 ** 21), ("gpu_found_lost_aggregate_pairs_capacity", 2 ** 26)):
        if px is not None and hasattr(px, name):
            setattr(px, name, max(int(getattr(px, name) or 0), value))


@configclass
class SpotArmV9aEnvCfg(SpotArmV7eEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9a(self)


@configclass
class SpotArmV9aEnvCfg_PLAY(SpotArmV7eEnvCfg_PLAY):
    """Evaluation: the FULL height range, no curriculum."""
    def __post_init__(self):
        super().__post_init__()
        _apply_v9a(self)
        self.events.reset_platform_object.params["top_range"] = H_END
        self.curriculum.platform_range = None


# ===== v8c / v9b (2026-11-02): PRECISION for top-down =====
# be centred over the cube in the JAW frame before descending; jaws square to the cube faces.
import isaaclab.utils.math as math_utils

JAW_CENTRE = (-0.019, -0.004)   # cube centre in gripper (y, z) during real grasps (measure_jaws.py: y -0.029..-0.008, z -0.009..0.002)


def _cube_in_gripper(env):
    ee = env.scene["ee_frame"].data
    p = _t(ee.target_pos_w)[..., 0, :]
    q = _t(ee.target_quat_w)[..., 0, :]
    return math_utils.quat_apply(math_utils.quat_inv(q), _t(env.scene["object"].data.root_pos_w)[:, :3] - p), q


def pregrasp_align(env, std, max_depth):
    """0..1: cube centred sideways between the jaws while it is still ahead of (below) the gripper."""
    loc, _ = _cube_in_gripper(env)
    lat = torch.sqrt((loc[:, 1] - JAW_CENTRE[0]) ** 2 + (loc[:, 2] - JAW_CENTRE[1]) ** 2)
    ahead = (loc[:, 0] > -0.02) & (loc[:, 0] < max_depth)
    return ahead.float() * (1.0 - torch.tanh(lat / std))


def jaw_square(env, near):
    """0..1: closing axis of the jaws aligned with a cube face (1) rather than a diagonal (0), near the cube."""
    loc, q = _cube_in_gripper(env)
    yw = math_utils.quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(q.shape[0], 3))[:, :2]
    a = yw.abs().max(dim=1).values / torch.clamp(torch.norm(yw, dim=1), min=1e-6)
    close = (torch.norm(loc, dim=1) < near).float()
    return close * torch.clamp((a - 0.7071) / (1.0 - 0.7071), 0.0, 1.0)


def _apply_align(cfg):
    cfg.rewards.pregrasp_align = RewTerm(func=pregrasp_align, params={"std": 0.015, "max_depth": 0.20}, weight=2.0)
    cfg.rewards.jaw_square = RewTerm(func=jaw_square, params={"near": 0.15}, weight=0.5)


@configclass
class SpotArmV8cEnvCfg(SpotArmV7eEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_align(self)


@configclass
class SpotArmV8cEnvCfg_PLAY(SpotArmV7eEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_align(self)


@configclass
class SpotArmV9bEnvCfg(SpotArmV9aEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_align(self)


@configclass
class SpotArmV9bEnvCfg_PLAY(SpotArmV9aEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_align(self)


# ===== v9c (2026-11-03): v9b continued on the FULL height range (a resume would restart the curriculum at 0.37-0.43) =====
@configclass
class SpotArmV9cEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_platform_object.params["top_range"] = H_END
        self.curriculum.platform_range = None


# ===== v9d (2026-11-04): no wrist roll + reachable heights only =====
# diagnose_failures.py on v9b: rotated >75 deg -> 64% vs 97% success (fixed bench); left-side cubes 2-55%;
# benches below 0.25 m 0% (reach limit while standing).
from isaaclab.managers import SceneEntityCfg

H_D = (0.28, 0.50)


def wrist_roll_dev(env, asset_cfg):
    """|wrist roll - start pose|: the cube never spawns rotated, so the grasp needs no wrist roll."""
    rd = env.scene["robot"].data
    q = _t(rd.joint_pos)[:, asset_cfg.joint_ids].reshape(-1)
    q0 = _t(rd.default_joint_pos)[:, asset_cfg.joint_ids].reshape(-1)
    return torch.abs(q - q0)


def _apply_v9d(cfg):
    cfg.events.reset_platform_object.params["top_range"] = H_D
    cfg.curriculum.platform_range = None
    cfg.rewards.wrist_roll = RewTerm(func=wrist_roll_dev, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm0_wr1"])}, weight=-0.3)


@configclass
class SpotArmV9dEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9d(self)


@configclass
class SpotArmV9dEnvCfg_PLAY(SpotArmV9bEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9d(self)


# ===== v9e (2026-11-05): stop the far-left falls + keep the gripper out of the body =====
# diagnose (v9d 49k, fixed bench): falls 13%, ALL far-left (y>+0.15: ok 4%, fell 65%), all after 4 s, never while holding,
# wrist >1.8 rad -> fell 48%. Demo zone |y|<0.10: ok 92%, fell 1%.
# check_body_intrusion: gripper inside body v9b 0%, v9d 2% (front-lower corner, ~2 cm deep, failed grasps only).
# self-collision stays OFF (would also change leg-body contacts under the frozen walking policy).
from isaaclab.envs import mdp as _base_mdp

BODY_BOX = ((-0.45, 0.50), (-0.13, 0.13), (-0.12, 0.10))   # base frame (m); x front edge has +5 cm margin


def gripper_in_body(env, box=BODY_BOX):
    """1 + depth/2 cm while the gripper (ee_frame, base frame) is inside the body box, else 0."""
    ee = _t(env.scene["ee_frame"].data.target_pos_source)[:, 0, :]
    lo = torch.tensor([b[0] for b in box], device=ee.device)
    hi = torch.tensor([b[1] for b in box], device=ee.device)
    inside = ((ee > lo) & (ee < hi)).all(dim=1)
    depth = torch.minimum(ee - lo, hi - ee).min(dim=1).values.clamp(min=0.0)
    return inside.float() * (1.0 + depth / 0.02)


def _apply_v9e(cfg):
    _apply_v9d(cfg)
    cfg.rewards.base_tilt = RewTerm(func=_base_mdp.flat_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot")}, weight=-10.0)
    cfg.rewards.fall_penalty.weight = -1500.0
    cfg.rewards.gripper_in_body = RewTerm(func=gripper_in_body, weight=-2.0)


@configclass
class SpotArmV9eEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9e(self)


@configclass
class SpotArmV9eEnvCfg_PLAY(SpotArmV9bEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9e(self)


# ===== v9n (2026-11-06): camera-error tolerant pick (for perception) =====
# Policy sees the cube position with a per-episode bias (camera calibration error) + per-step jitter.
# Rewards still use the TRUE cube position. Built on v9e (tilt + fall + gripper-in-body).
from isaaclab.utils.noise import UniformNoiseCfg as _Unoise
from isaaclab.managers import EventTermCfg as _EvTerm

CAM_BIAS = (0.010, 0.010, 0.005)     # +- m, constant per episode (x, y, z in robot frame)
CAM_JITTER = 0.004                   # +- m, new every step
_OBJ_POS_FUNC = None                 # original object_position obs function, captured in _apply_v9n


def sample_cam_bias(env, env_ids, scale=CAM_BIAS):
    n = env.num_envs
    if getattr(env, "_cam_bias", None) is None or env._cam_bias.shape[0] != n:
        env._cam_bias = torch.zeros(n, 3, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(n, device=env.device)
    s = torch.tensor(scale, device=env.device)
    env._cam_bias[env_ids] = (torch.rand(len(env_ids), 3, device=env.device) * 2 - 1) * s


def object_position_biased(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                           object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    p = _OBJ_POS_FUNC(env, robot_cfg=robot_cfg, object_cfg=object_cfg)
    b = getattr(env, "_cam_bias", None)
    return p if b is None else p + b


def _apply_v9n(cfg):
    global _OBJ_POS_FUNC
    _apply_v9e(cfg)
    term = cfg.observations.policy.object_position
    if term.func is not object_position_biased:
        _OBJ_POS_FUNC = term.func
    term.func = object_position_biased
    term.noise = _Unoise(n_min=-CAM_JITTER, n_max=CAM_JITTER)
    cfg.observations.policy.enable_corruption = True
    cfg.events.cam_bias = _EvTerm(func=sample_cam_bias, mode="reset")


@configclass
class SpotArmV9nEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9n(self)


@configclass
class SpotArmV9nEnvCfg_PLAY(SpotArmV9bEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9n(self)


# ===== v9m (2026-11-10): v9g skills on 3 objects: cube, foam brick, marker (pen-like) =====
# Same as v9n (camera error, tilt/fall/body penalties, heights 0.28-0.50) + _apply_multi_object (v8a; enlarged PhysX buffers).
# Observation unchanged (31) so it resumes from v9g. Object yaw stays fixed (reset yaw_range 0) -> no orientation input needed.
# Asset order from _apply_multi_object: 0 = cube, 1 = foam brick, 2 = marker.


def _apply_v9m(cfg):
    _apply_v9n(cfg)
    _apply_multi_object(cfg)


def _single_object(cfg, k):
    """Evaluation only: every robot gets object k."""
    cfg.scene.object.spawn = cfg.scene.object.spawn.assets_cfg[k]
    cfg.scene.replicate_physics = True


@configclass
class SpotArmV9mEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9m(self)


@configclass
class SpotArmV9mEnvCfg_PLAY(SpotArmV9bEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9m(self)


@configclass
class SpotArmV9mEnvCfg_PLAY_Cube(SpotArmV9mEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _single_object(self, 0)


@configclass
class SpotArmV9mEnvCfg_PLAY_Brick(SpotArmV9mEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _single_object(self, 1)


@configclass
class SpotArmV9mEnvCfg_PLAY_Marker(SpotArmV9mEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _single_object(self, 2)


# ===== v9w (2026-11-17): fine-tune v9g for the demo's perception observation model ("wanc") =====
# Demo (perception chat): cube position = ONE camera look, frozen in the WORLD frame (so body motion is accounted for),
# then, once the finger has stalled on the cube for N steps, ANCHORED TO THE WRIST (offset fixed in the gripper frame).
# Measured with v9g: wanc 83-86% held vs 94% live; demo first grasps 50-80%. Train on that observation directly.
# Per episode: 30% live (keeps the old skill), 70% wanc with N = 2..8 stall steps. Camera bias/jitter from v9n kept.
# A small random base push at reset imitates the ~4 cm shove UNFOLD gives the body after the look.
from isaaclab.utils.math import subtract_frame_transforms, quat_apply

W_P_LIVE = 0.3
W_STALL = (2, 8)
W_F1X_HOLD = (-0.9, -0.2)     # finger angle range = "closed on something"
W_F1X_VEL = 0.5               # rad/s: finger counts as stopped below this
W_PUSH = 0.15                 # m/s random base velocity at reset (x, y); 0.25 gave 7 cm drift, demo shove ~4 cm


def _wanc_state(env):
    s = getattr(env, "_wanc", None)
    if s is None or s["need"].shape[0] != env.num_envs:
        n, d = env.num_envs, env.device
        s = {"need": torch.ones(n, dtype=torch.bool, device=d), "look_w": torch.zeros(n, 3, device=d),
             "att": torch.zeros(n, dtype=torch.bool, device=d), "off": torch.zeros(n, 3, device=d),
             "stall": torch.zeros(n, dtype=torch.long, device=d), "nstall": torch.full((n,), 2, dtype=torch.long, device=d),
             "live": torch.zeros(n, dtype=torch.bool, device=d), "f1x": None}
        env._wanc = s
    return s


def reset_wanc(env, env_ids, p_live=W_P_LIVE, stall_range=W_STALL):
    s = _wanc_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    k = len(env_ids)
    s["need"][env_ids] = True
    s["att"][env_ids] = False
    s["stall"][env_ids] = 0
    s["nstall"][env_ids] = torch.randint(stall_range[0], stall_range[1] + 1, (k,), device=env.device)
    s["live"][env_ids] = torch.rand(k, device=env.device) < p_live


def object_position_wanc(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    s = _wanc_state(env)
    robot, obj = env.scene[robot_cfg.name], env.scene[object_cfg.name]
    rp, rq = _t(robot.data.root_pos_w)[:, :3], _t(robot.data.root_quat_w)
    cube_w = _t(obj.data.root_pos_w)[:, :3]
    bias_b = getattr(env, "_cam_bias", None)
    bias_w = quat_apply(rq, bias_b) if bias_b is not None else torch.zeros_like(cube_w)
    if s["need"].any():                                   # the camera look: once per episode, world frame
        s["look_w"] = torch.where(s["need"][:, None], cube_w + bias_w, s["look_w"])
        s["need"] = torch.zeros_like(s["need"])
    if s["f1x"] is None:
        s["f1x"] = robot.find_joints("arm0_f1x")[0][0]
    q, qd = _t(robot.data.joint_pos)[:, s["f1x"]], _t(robot.data.joint_vel)[:, s["f1x"]]
    ft = env.scene["ee_frame"]
    ep, eq = _t(ft.data.target_pos_w)[:, 0, :], _t(ft.data.target_quat_w)[:, 0, :]
    holding = (q > W_F1X_HOLD[0]) & (q < W_F1X_HOLD[1]) & (qd.abs() < W_F1X_VEL)
    s["stall"] = torch.where(holding, s["stall"] + 1, torch.zeros_like(s["stall"]))
    new = (s["stall"] >= s["nstall"]) & ~s["att"]
    if new.any():                                         # grasp detected: anchor the (estimated) cube to the wrist
        off, _ = subtract_frame_transforms(ep, eq, s["look_w"])
        s["off"] = torch.where(new[:, None], off, s["off"])
    released = (q < W_F1X_HOLD[0] - 0.1) | (q > -0.1)     # finger opened again, or closed fully on nothing
    s["att"] = (s["att"] | new) & ~released
    att_w, _ = combine_frame_transforms(ep, eq, s["off"])
    seen_w = torch.where(s["att"][:, None], att_w, s["look_w"])
    seen_w = torch.where(s["live"][:, None], cube_w + bias_w, seen_w)
    seen_b, _ = subtract_frame_transforms(rp, rq, seen_w)
    return seen_b


def _apply_v9w(cfg, p_live=W_P_LIVE):
    _apply_v9n(cfg)
    cfg.observations.policy.object_position.func = object_position_wanc
    cfg.events.wanc_reset = _EvTerm(func=reset_wanc, mode="reset", params={"p_live": p_live})
    cfg.events.pre_pick_shove = _EvTerm(func=_base_mdp.push_by_setting_velocity, mode="reset",
                                        params={"velocity_range": {"x": (-W_PUSH, W_PUSH), "y": (-W_PUSH, W_PUSH)}})


@configclass
class SpotArmV9wEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9w(self)


@configclass
class SpotArmV9wEnvCfg_PLAY(SpotArmV9bEnvCfg_PLAY):
    """Evaluation: wanc for every robot (no live episodes)."""
    def __post_init__(self):
        super().__post_init__()
        _apply_v9w(self, p_live=0.0)


# ===== v9y (2026-11-18): turned cubes (yaw +-45 deg) + arm start-pose spread, from v9g =====
# Perception chat: demo input is correct (<= 0.3 cm) but ~40% of first grasps miss, closing to the -y side and turning the
# cube 10-40 deg; a turned cube is then unpickable (v9g only ever saw square cubes). v9w (wanc obs + base push) failed
# (worse everywhere, falls back), so no obs change and no push here. The arm start pose is spread +-0.08 rad to cover the
# residual error UNFOLD leaves before PICK. jaw_square now measures alignment against the CUBE's faces, not the world axes.
# No yaw input: with a square cube +-45 deg covers every orientation; the policy has to learn a grasp that works anyway.
Y_YAW = 0.785          # rad, cube yaw range at reset
Y_ARM = 0.08           # rad, arm joint offset range at reset (sh0..wr1, not the finger)
Y_ARM_JOINTS = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"]


def jaw_square_rel(env, near):
    """0..1: jaw closing axis aligned with a face of the (possibly turned) cube, near the cube."""
    loc, q = _cube_in_gripper(env)
    oq = _t(env.scene["object"].data.root_quat_w)
    yw = math_utils.quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(q.shape[0], 3))
    yc = math_utils.quat_apply(math_utils.quat_inv(oq), yw)[:, :2]
    a = yc.abs().max(dim=1).values / torch.clamp(torch.norm(yc, dim=1), min=1e-6)
    close = (torch.norm(loc, dim=1) < near).float()
    return close * torch.clamp((a - 0.7071) / (1.0 - 0.7071), 0.0, 1.0)


def _apply_v9y(cfg):
    _apply_v9n(cfg)
    cfg.events.reset_platform_object.params["yaw_range"] = (-Y_YAW, Y_YAW)
    cfg.rewards.jaw_square = RewTerm(func=jaw_square_rel, params={"near": 0.15}, weight=0.5)
    cfg.events.arm_start_spread = _EvTerm(func=_base_mdp.reset_joints_by_offset, mode="reset",
                                          params={"position_range": (-Y_ARM, Y_ARM), "velocity_range": (0.0, 0.0),
                                                  "asset_cfg": SceneEntityCfg("robot", joint_names=Y_ARM_JOINTS)})


@configclass
class SpotArmV9yEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9y(self)


@configclass
class SpotArmV9yEnvCfg_PLAY(SpotArmV9bEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9y(self)


# ===== v9y (2026-11-18): turned cubes (yaw +-45 deg) + arm start-pose spread, from v9g =====
# Perception chat: demo input is correct (<= 0.3 cm) but ~40% of first grasps miss, closing to the -y side and turning the
# cube 10-40 deg; a turned cube is then unpickable (v9g only ever saw square cubes). v9w (wanc obs + base push) failed
# (worse everywhere, falls back), so no obs change and no push here. The arm start pose is spread +-0.08 rad to cover the
# residual error UNFOLD leaves before PICK. jaw_square now measures alignment against the CUBE's faces, not the world axes.
# No yaw input: with a square cube +-45 deg covers every orientation; the policy has to learn a grasp that works anyway.
Y_YAW = 0.785          # rad, cube yaw range at reset
Y_ARM = 0.08           # rad, arm joint offset range at reset (sh0..wr1, not the finger)
Y_ARM_JOINTS = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"]


def jaw_square_rel(env, near):
    """0..1: jaw closing axis aligned with a face of the (possibly turned) cube, near the cube."""
    loc, q = _cube_in_gripper(env)
    oq = _t(env.scene["object"].data.root_quat_w)
    yw = math_utils.quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(q.shape[0], 3))
    yc = math_utils.quat_apply(math_utils.quat_inv(oq), yw)[:, :2]
    a = yc.abs().max(dim=1).values / torch.clamp(torch.norm(yc, dim=1), min=1e-6)
    close = (torch.norm(loc, dim=1) < near).float()
    return close * torch.clamp((a - 0.7071) / (1.0 - 0.7071), 0.0, 1.0)


def _apply_v9y(cfg):
    _apply_v9n(cfg)
    cfg.events.reset_platform_object.params["yaw_range"] = (-Y_YAW, Y_YAW)
    cfg.rewards.jaw_square = RewTerm(func=jaw_square_rel, params={"near": 0.15}, weight=0.5)
    cfg.events.arm_start_spread = _EvTerm(func=_base_mdp.reset_joints_by_offset, mode="reset",
                                          params={"position_range": (-Y_ARM, Y_ARM), "velocity_range": (0.0, 0.0),
                                                  "asset_cfg": SceneEntityCfg("robot", joint_names=Y_ARM_JOINTS)})


@configclass
class SpotArmV9yEnvCfg(SpotArmV9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9y(self)


@configclass
class SpotArmV9yEnvCfg_PLAY(SpotArmV9bEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v9y(self)

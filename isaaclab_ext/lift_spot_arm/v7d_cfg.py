"""v7d: TOP-DOWN pick on v5's proven setup. Changes vs v5 (4, all needed for top-down):
1) start pose READY_HIGH (gripper down, clear above the bench; checked visually in v7a)
2) grasping_object -> 'cube between the jaws' check in the GRIPPER frame (zone measured from v5's real grasps)
3) small top-down reward (+1, near the cube only)
4) fall penalty -500 (~ -10 per fall after Isaac Lab's dt scaling)
Arm motors unchanged from v5 (400/40) - v7b's 2000 setting blew up the simulation."""
import copy

import isaaclab.envs.mdp as base_mdp
import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from .env_cfg import _t
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from .stand_train_cfg import SPOT_V5_CFG, SpotArmStandLiftEnvCfg_V5, SpotArmStandLiftEnvCfg_V5_CONT
from .v7_cfg import READY_HIGH, gripper_points_down

# cube-centre zone in the GRIPPER frame (from measure_jaws.py on v5: x -0.005..0.044, y -0.029..-0.008, z -0.009..0.002), with margin
JAW_BOX = ((-0.015, 0.055), (-0.040, 0.002), (-0.020, 0.012))
CLOSED_ON = (-0.9, -0.2)          # gripper partly closed = stopped by something

SPOT_V7D_CFG = copy.deepcopy(SPOT_V5_CFG)
SPOT_V7D_CFG.init_state.joint_pos = {**SPOT_V5_CFG.init_state.joint_pos, **READY_HIGH}


def cube_in_jaw_box(env):
    """Bool per env: cube centre inside the measured between-the-jaws zone (gripper frame)."""
    ee = env.scene["ee_frame"].data
    p = _t(ee.target_pos_w)[..., 0, :]
    q = _t(ee.target_quat_w)[..., 0, :]
    loc = math_utils.quat_apply(math_utils.quat_inv(q), _t(env.scene["object"].data.root_pos_w)[:, :3] - p)
    inside = torch.ones(loc.shape[0], dtype=torch.bool, device=loc.device)
    for i, (lo, hi) in enumerate(JAW_BOX):
        inside &= (loc[:, i] > lo) & (loc[:, i] < hi)
    return inside


def cube_between_jaws(env, asset_cfg):
    f1x = _t(env.scene["robot"].data.joint_pos)[:, asset_cfg.joint_ids].reshape(-1)
    closed = (f1x > CLOSED_ON[0]) & (f1x < CLOSED_ON[1])
    return (closed & cube_in_jaw_box(env)).float()


def unstable_state(env, joint_vel_max, obj_vel_max):
    """End the episode of any robot whose physics is going unstable, BEFORE invalid numbers reach training."""
    jv = _t(env.scene["robot"].data.joint_vel)
    ov = _t(env.scene["object"].data.root_lin_vel_w)
    bad = ~torch.isfinite(jv).all(dim=1) | ~torch.isfinite(ov).all(dim=1)
    bad |= torch.nan_to_num(jv, nan=0.0, posinf=1e6, neginf=-1e6).abs().max(dim=1).values > joint_vel_max
    bad |= torch.nan_to_num(ov, nan=0.0, posinf=1e6, neginf=-1e6).norm(dim=1) > obj_vel_max
    return bad


def _apply_v7d(cfg):
    cfg.scene.robot = SPOT_V7D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.terminations.unstable = DoneTerm(func=unstable_state, params={"joint_vel_max": 50.0, "obj_vel_max": 10.0})
    rp = getattr(cfg.scene.object.spawn, "rigid_props", None)
    if rp is not None:
        rp.max_depenetration_velocity = 1.0      # a jammed cube is pushed apart gently, not shot out
    cfg.rewards.grasping_object = RewTerm(
        func=cube_between_jaws, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm0_f1x"])}, weight=2.0)
    cfg.rewards.top_down = RewTerm(func=gripper_points_down, params={"near": 0.4}, weight=1.0)
    cfg.rewards.reaching_far = RewTerm(func=lift_mdp.object_ee_distance, params={"std": 0.3}, weight=1.0)   # signal from the raised start pose
    cfg.rewards.fall_penalty = RewTerm(
        func=base_mdp.is_terminated_term, params={"term_keys": ["base_tilted", "base_low"]}, weight=-500.0)


@configclass
class SpotArmV7dEnvCfg(SpotArmStandLiftEnvCfg_V5):
    def __post_init__(self):
        super().__post_init__()
        _apply_v7d(self)


@configclass
class SpotArmV7dEnvCfg_CONT(SpotArmStandLiftEnvCfg_V5_CONT):
    def __post_init__(self):
        super().__post_init__()
        _apply_v7d(self)


@configclass
class SpotArmV7dEnvCfg_PLAY(SpotArmV7dEnvCfg_CONT):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True


# ===== v7dR (2026-10-28): lift/push rewards use the SAME jaw check as grasping + penalty ramp + NaN-safe =====
from isaaclab.managers import CurriculumTermCfg as CurrTerm


def raised_while_between_jaws(env, rest_z, span, threshold, asset_cfg):
    """Same formula as v6 object_raised_while_grasped, but gated by the between-the-jaws check."""
    pinch = cube_between_jaws(env, asset_cfg)
    z = _t(env.scene["object"].data.root_pos_w)[:, 2]
    return pinch * torch.clamp((z - rest_z) / span, 0.0, 1.0)


def cube_pushed_jaws(env, threshold, asset_cfg):
    """Same formula as v6 cube_pushed, but 'held' means the between-the-jaws check."""
    v = _t(env.scene["object"].data.root_lin_vel_w)[:, :2]
    pinch = cube_between_jaws(env, asset_cfg)
    return (1.0 - pinch) * torch.clamp(torch.norm(v, dim=1) / 0.2, 0.0, 1.0)


def ramp_reward_weight(env, env_ids, term_name, start, end, num_steps):
    """Linearly move a reward weight from start to end over num_steps env steps (restarts on resume)."""
    frac = min(1.0, float(env.common_step_counter) / float(num_steps))
    w = start + (end - start) * frac
    tc = env.reward_manager.get_term_cfg(term_name)
    tc.weight = w
    env.reward_manager.set_term_cfg(term_name, tc)
    return w


RAMP_STEPS = 400 * 24   # ~400 iterations (24 env steps per iteration)


def _apply_v7dr(cfg):
    cfg.rewards.lift_shaping.func = raised_while_between_jaws
    cfg.rewards.cube_pushed.func = cube_pushed_jaws
    cfg.terminations.unstable.time_out = True
    cfg.rewards.action_rate.weight = -0.02
    cfg.rewards.joint_vel.weight = -0.02
    cfg.rewards.gripper_toggle.weight = -1.0     # chatter was 34 flips/s; -0.2 cost only ~10% of grasp reward
    cfg.curriculum.action_rate = CurrTerm(func=ramp_reward_weight, params={"term_name": "action_rate", "start": -0.02, "end": -0.05, "num_steps": RAMP_STEPS})
    cfg.curriculum.joint_vel = CurrTerm(func=ramp_reward_weight, params={"term_name": "joint_vel", "start": -0.02, "end": -0.05, "num_steps": RAMP_STEPS})
    cfg.curriculum.gripper_toggle = None


@configclass
class SpotArmV7dREnvCfg(SpotArmStandLiftEnvCfg_V5):
    def __post_init__(self):
        super().__post_init__()
        _apply_v7d(self)
        _apply_v7dr(self)


@configclass
class SpotArmV7dREnvCfg_PLAY(SpotArmV7dREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True
        self.rewards.action_rate.weight = -0.1
        self.rewards.joint_vel.weight = -0.1
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


# ===== v7e (2026-10-29): a grasp counts ONLY if the gripper is COMMANDED closed AND the finger has STALLED =====
# check_grasp_quality.py on v7dr it 10807: only 37% of 'grasping' steps were real; 46% commanded open, 50% finger moving.
FINGER_STALL_VEL = 0.5   # rad/s


def cube_held(env, asset_cfg):
    rd = env.scene["robot"].data
    f1x = _t(rd.joint_pos)[:, asset_cfg.joint_ids].reshape(-1)
    f1v = _t(rd.joint_vel)[:, asset_cfg.joint_ids].reshape(-1).abs()
    cmd = _t(env.action_manager.get_term("gripper_action").raw_actions).reshape(-1)
    closed_on = (f1x > CLOSED_ON[0]) & (f1x < CLOSED_ON[1])
    return (closed_on & (cmd < 0) & (f1v < FINGER_STALL_VEL) & cube_in_jaw_box(env)).float()


def grasp_held(env, asset_cfg):
    return cube_held(env, asset_cfg)


def raised_while_held(env, rest_z, span, threshold, asset_cfg):
    z = _t(env.scene["object"].data.root_pos_w)[:, 2]
    return cube_held(env, asset_cfg) * torch.clamp((z - rest_z) / span, 0.0, 1.0)


def cube_pushed_unheld(env, threshold, asset_cfg):
    v = _t(env.scene["object"].data.root_lin_vel_w)[:, :2]
    return (1.0 - cube_held(env, asset_cfg)) * torch.clamp(torch.norm(v, dim=1) / 0.2, 0.0, 1.0)


def _apply_v7e(cfg):
    cfg.rewards.grasping_object.func = grasp_held
    cfg.rewards.lift_shaping.func = raised_while_held
    cfg.rewards.cube_pushed.func = cube_pushed_unheld


@configclass
class SpotArmV7eEnvCfg(SpotArmV7dREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v7e(self)


@configclass
class SpotArmV7eEnvCfg_PLAY(SpotArmV7dREnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v7e(self)

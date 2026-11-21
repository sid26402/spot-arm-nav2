"""v7b: EXACTLY v5's task (bench 0.40 m, side grasp, goal point, v6 rewards, frozen walking legs, 500 Hz)
with ONLY the arm changed: kp 2000 (was 400), shoulder/elbow damping 90, wrist/gripper damping 20,
plus a fall penalty (v7a showed a stiffer arm can tip Spot, and negative rewards made falling cheap)."""
import copy

import isaaclab.envs.mdp as base_mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from .stand_train_cfg import SPOT_V5_CFG, SpotArmStandLiftEnvCfg_V5, SpotArmStandLiftEnvCfg_V5_CONT

ARM_KP = 2000.0
BIG_KD, SMALL_KD = 90.0, 20.0     # shoulder/elbow ~critically damped; light wrist/gripper joints need less

SPOT_V7B_CFG = copy.deepcopy(SPOT_V5_CFG)
SPOT_V7B_CFG.actuators = {k: v for k, v in SPOT_V5_CFG.actuators.items()
                          if not any("arm0" in e for e in v.joint_names_expr)}
SPOT_V7B_CFG.actuators["arm_big"] = ImplicitActuatorCfg(
    joint_names_expr=["arm0_sh0", "arm0_sh1", "arm0_el0"], stiffness=ARM_KP, damping=BIG_KD, velocity_limit_sim=3.0)
SPOT_V7B_CFG.actuators["arm_small"] = ImplicitActuatorCfg(
    joint_names_expr=["arm0_el1", "arm0_wr0", "arm0_wr1", "arm0_f1x"], stiffness=ARM_KP, damping=SMALL_KD, velocity_limit_sim=3.0)


def _apply_v7b(cfg):
    cfg.scene.robot = SPOT_V7B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.rewards.fall_penalty = RewTerm(
        func=base_mdp.is_terminated_term, params={"term_keys": ["base_tilted", "base_low"]}, weight=-50.0)


@configclass
class SpotArmV7bEnvCfg(SpotArmStandLiftEnvCfg_V5):
    def __post_init__(self):
        super().__post_init__()
        _apply_v7b(self)


@configclass
class SpotArmV7bEnvCfg_CONT(SpotArmStandLiftEnvCfg_V5_CONT):
    def __post_init__(self):
        super().__post_init__()
        _apply_v7b(self)


@configclass
class SpotArmV7bEnvCfg_PLAY(SpotArmV7bEnvCfg_CONT):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.scene.ee_frame.debug_vis = True

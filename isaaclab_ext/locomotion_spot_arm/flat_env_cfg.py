import copy
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotFlatEnvCfg,
    SpotPhysxEventCfg,
)

USD_PATH = "@SPOT_ASSETS@/spot_with_arm_rootfix.usd"
LEG_JOINTS = "[fh][lr]_.*"
ARM_JOINTS = "arm0_.*"

STOW = {
    "arm0_sh0": 0.0, "arm0_sh1": -3.1, "arm0_el0": 3.1, "arm0_el1": 0.0,
    "arm0_wr0": 0.0, "arm0_wr1": 0.0, "arm0_f1x": -1.5,
}

SPOT_ARM_CFG = copy.deepcopy(SPOT_CFG)
SPOT_ARM_CFG.spawn.usd_path = USD_PATH
SPOT_ARM_CFG.spawn.articulation_props.enabled_self_collisions = False
SPOT_ARM_CFG.init_state.joint_pos = {**SPOT_CFG.init_state.joint_pos, **STOW}
SPOT_ARM_CFG.actuators = {
    **SPOT_CFG.actuators,
    "spot_arm": ImplicitActuatorCfg(joint_names_expr=[ARM_JOINTS], stiffness=400.0, damping=40.0),
}


def leg_action_smoothness(env) -> torch.Tensor:
    am = env.action_manager
    return torch.linalg.norm(am.action[:, :12] - am.prev_action[:, :12], dim=1)


@configclass
class SpotArmActionsCfg:
    legs = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[LEG_JOINTS], scale=0.2, use_default_offset=True)
    arm_hold = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[ARM_JOINTS], scale=0.0, use_default_offset=True)


@configclass
class SpotArmEventCfg(SpotPhysxEventCfg):
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class SpotArmFlatEnvCfg(SpotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        legs = lambda: SceneEntityCfg("robot", joint_names=LEG_JOINTS)

        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions = SpotArmActionsCfg()
        self.events = SpotArmEventCfg()

        pol = self.observations.policy
        pol.joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": legs()}, noise=Unoise(n_min=-0.05, n_max=0.05))
        pol.joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": legs()}, noise=Unoise(n_min=-0.5, n_max=0.5))
        pol.actions = ObsTerm(func=mdp.last_action, params={"action_name": "legs"})

        r = self.rewards
        r.action_smoothness.func = leg_action_smoothness
        r.joint_pos.params = {**r.joint_pos.params, "asset_cfg": legs()}
        r.joint_torques.params = {**r.joint_torques.params, "asset_cfg": legs()}
        r.air_time.params = {**r.air_time.params, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lleg")}
        r.air_time_variance.params = {
            **r.air_time_variance.params,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lleg"),
        }
        r.gait.params = {
            **r.gait.params,
            "synced_feet_pair_names": (("fl_lleg", "hr_lleg"), ("fr_lleg", "hl_lleg")),
        }
        r.foot_clearance = None
        r.foot_slip = None

        self.terminations.body_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_uleg"]), "threshold": 1.0},
        )
        self.terminations.arm_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["arm0_link.*"]), "threshold": 1.0},
        )


class SpotArmFlatEnvCfg_PLAY(SpotArmFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False

        import os
        if os.environ.get("SPOTARM_DEBUG") == "1":
            zero6 = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
            self.events.reset_base.params = {**self.events.reset_base.params, "pose_range": {"yaw": (0.0, 0.0)}, "velocity_range": zero6}
            self.events.reset_robot_joints.params = {**self.events.reset_robot_joints.params, "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
            self.events.push_robot = None
            self.events.add_base_mass = None

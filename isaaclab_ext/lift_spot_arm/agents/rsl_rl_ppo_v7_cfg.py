from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import SpotArmLiftPPORunnerCfg


def _algo(entropy):
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=entropy,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-4, schedule="adaptive",
        gamma=0.98, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class SpotArmV7PPORunnerCfg_E01(SpotArmLiftPPORunnerCfg):
    experiment_name = "spot_arm_lift_v7"
    algorithm = _algo(0.01)


@configclass
class SpotArmV7PPORunnerCfg_E002(SpotArmLiftPPORunnerCfg):
    experiment_name = "spot_arm_lift_v7"
    algorithm = _algo(0.002)


@configclass
class SpotArmV7bPPORunnerCfg_E01(SpotArmLiftPPORunnerCfg):
    experiment_name = "spot_arm_lift_v7b"
    algorithm = _algo(0.01)


@configclass
class SpotArmV7bPPORunnerCfg_E002(SpotArmLiftPPORunnerCfg):
    experiment_name = "spot_arm_lift_v7b"
    algorithm = _algo(0.002)


@configclass
class SpotArmV7dPPORunnerCfg_E01(SpotArmLiftPPORunnerCfg):
    experiment_name = "spot_arm_lift_v7d"
    algorithm = _algo(0.01)


@configclass
class SpotArmV7dPPORunnerCfg_E002(SpotArmLiftPPORunnerCfg):
    experiment_name = "spot_arm_lift_v7d"
    algorithm = _algo(0.002)

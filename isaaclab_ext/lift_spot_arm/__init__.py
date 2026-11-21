import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Cube-SpotArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotArmCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-SpotArm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotArmCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SpotArm-StandPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.stand_pick_cfg:SpotArmStandPickEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-SpotArm-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotArmCubeLiftEnvCfg_V4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-SpotArm-v4-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpotArmCubeLiftEnvCfg_V4_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-SpotArm-Stand-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_train_cfg:SpotArmStandLiftEnvCfg_V5",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-SpotArm-Stand-v5-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_train_cfg:SpotArmStandLiftEnvCfg_V5_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-SpotArm-Stand-v5-Cont",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_train_cfg:SpotArmStandLiftEnvCfg_V5_CONT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

for _id, _cfg, _agent in [
    ("Isaac-SpotArm-V7a-S1", "SpotArmV7aEnvCfg", "SpotArmV7PPORunnerCfg_E01"),
    ("Isaac-SpotArm-V7a-S2", "SpotArmV7aEnvCfg", "SpotArmV7PPORunnerCfg_E002"),
    ("Isaac-SpotArm-V7a-S3", "SpotArmV7aEnvCfg_CONT", "SpotArmV7PPORunnerCfg_E002"),
    ("Isaac-SpotArm-V7a-Play", "SpotArmV7aEnvCfg_PLAY", "SpotArmV7PPORunnerCfg_E002"),
]:
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v7_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:{_agent}",
        },
        disable_env_checker=True,
    )

for _id, _cfg, _agent in [
    ("Isaac-SpotArm-V7b-S1", "SpotArmV7bEnvCfg", "SpotArmV7bPPORunnerCfg_E01"),
    ("Isaac-SpotArm-V7b-S2", "SpotArmV7bEnvCfg", "SpotArmV7bPPORunnerCfg_E002"),
    ("Isaac-SpotArm-V7b-S3", "SpotArmV7bEnvCfg_CONT", "SpotArmV7bPPORunnerCfg_E002"),
    ("Isaac-SpotArm-V7b-Play", "SpotArmV7bEnvCfg_PLAY", "SpotArmV7bPPORunnerCfg_E002"),
]:
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v7b_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:{_agent}",
        },
        disable_env_checker=True,
    )

for _id, _cfg, _agent in [
    ("Isaac-SpotArm-V7d-S1", "SpotArmV7dEnvCfg", "SpotArmV7dPPORunnerCfg_E01"),
    ("Isaac-SpotArm-V7d-S2", "SpotArmV7dEnvCfg", "SpotArmV7dPPORunnerCfg_E002"),
    ("Isaac-SpotArm-V7d-S3", "SpotArmV7dEnvCfg_CONT", "SpotArmV7dPPORunnerCfg_E002"),
    ("Isaac-SpotArm-V7d-Play", "SpotArmV7dEnvCfg_PLAY", "SpotArmV7dPPORunnerCfg_E002"),
]:
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v7d_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:{_agent}",
        },
        disable_env_checker=True,
    )

for _id, _cfg in [("Isaac-SpotArm-V7dR", "SpotArmV7dREnvCfg"), ("Isaac-SpotArm-V7dR-Play", "SpotArmV7dREnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v7d_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )

for _id, _cfg in [("Isaac-SpotArm-V7e", "SpotArmV7eEnvCfg"), ("Isaac-SpotArm-V7e-Play", "SpotArmV7eEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v7d_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )

for _id, _cfg in [("Isaac-SpotArm-V8a", "SpotArmV8aEnvCfg"), ("Isaac-SpotArm-V8a-Play", "SpotArmV8aEnvCfg_PLAY"),
                  ("Isaac-SpotArm-V8b", "SpotArmV8bEnvCfg"), ("Isaac-SpotArm-V8b-Play", "SpotArmV8bEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )

for _id, _cfg in [("Isaac-SpotArm-V9a", "SpotArmV9aEnvCfg"), ("Isaac-SpotArm-V9a-Play", "SpotArmV9aEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )

for _id, _cfg in [("Isaac-SpotArm-V8c", "SpotArmV8cEnvCfg"), ("Isaac-SpotArm-V8c-Play", "SpotArmV8cEnvCfg_PLAY"),
                  ("Isaac-SpotArm-V9b", "SpotArmV9bEnvCfg"), ("Isaac-SpotArm-V9b-Play", "SpotArmV9bEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )

gym.register(
    id="Isaac-SpotArm-V9c",
    entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v8_cfg:SpotArmV9cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
    },
    disable_env_checker=True,
)

for _id, _cfg in [("Isaac-SpotArm-V9d", "SpotArmV9dEnvCfg"), ("Isaac-SpotArm-V9d-Play", "SpotArmV9dEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )


for _id, _cfg in [("Isaac-SpotArm-V9e", "SpotArmV9eEnvCfg"), ("Isaac-SpotArm-V9e-Play", "SpotArmV9eEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )


for _id, _cfg in [("Isaac-SpotArm-V9n", "SpotArmV9nEnvCfg"), ("Isaac-SpotArm-V9n-Play", "SpotArmV9nEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )


for _id, _cfg in [("Isaac-SpotArm-V9m", "SpotArmV9mEnvCfg"), ("Isaac-SpotArm-V9m-Play", "SpotArmV9mEnvCfg_PLAY"),
                  ("Isaac-SpotArm-V9m-Play-Cube", "SpotArmV9mEnvCfg_PLAY_Cube"), ("Isaac-SpotArm-V9m-Play-Brick", "SpotArmV9mEnvCfg_PLAY_Brick"),
                  ("Isaac-SpotArm-V9m-Play-Marker", "SpotArmV9mEnvCfg_PLAY_Marker")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )


for _id, _cfg in [("Isaac-SpotArm-V9w", "SpotArmV9wEnvCfg"), ("Isaac-SpotArm-V9w-Play", "SpotArmV9wEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )


for _id, _cfg in [("Isaac-SpotArm-V9y", "SpotArmV9yEnvCfg"), ("Isaac-SpotArm-V9y-Play", "SpotArmV9yEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )


for _id, _cfg in [("Isaac-SpotArm-V9y", "SpotArmV9yEnvCfg"), ("Isaac-SpotArm-V9y-Play", "SpotArmV9yEnvCfg_PLAY")]:
    gym.register(
        id=_id,
        entry_point=f"{__name__}.safe_env:SafeManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v8_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_v7_cfg:SpotArmV7dPPORunnerCfg_E002",
        },
        disable_env_checker=True,
    )

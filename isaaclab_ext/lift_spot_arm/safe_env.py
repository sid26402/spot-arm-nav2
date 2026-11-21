"""ManagerBasedRLEnv that zeroes non-finite rewards/observations, so one broken env cannot poison a PPO update."""
import torch

from isaaclab.envs import ManagerBasedRLEnv


def _clean(x):
    if torch.is_tensor(x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x


class SafeManagerBasedRLEnv(ManagerBasedRLEnv):
    def step(self, action):
        obs, rew, terminated, truncated, extras = super().step(action)
        return _clean(obs), _clean(rew), terminated, truncated, extras

    def reset(self, *args, **kwargs):
        obs, extras = super().reset(*args, **kwargs)
        return _clean(obs), extras

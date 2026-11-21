"""Leg action term driven by a frozen, pre-trained walking policy (loaded lazily, after the simulator starts)."""
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class FrozenWalkAction(JointPositionAction):
    """Leg joint targets from a frozen walking policy. Contributes 0 dims to the learned action."""

    def __init__(self, cfg, env):
        self._ready = False
        super().__init__(cfg, env)          # builds 12-dim buffers while action_dim still reports 12
        self._ready = True
        self._walk = torch.jit.load(cfg.policy_path, map_location=self.device).eval()
        self.cmd = torch.zeros(self.num_envs, 3, device=self.device)  # leg velocity command, 0 = stand (training default)

    @property
    def action_dim(self) -> int:
        return 0 if getattr(self, "_ready", False) else self._num_joints

    def process_actions(self, actions: torch.Tensor):
        obs_buf = getattr(self._env, "obs_buf", None)
        obs = obs_buf.get(self.cfg.obs_group) if isinstance(obs_buf, dict) else None
        if obs is None:                      # very first step before any observation exists
            walk = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        else:
            o = obs.clone()
            o[:, 9:12] = self.cmd            # velocity command slots (zero during training = stand still)
            with torch.no_grad():
                walk = self._walk(o)[:, : self._num_joints]
        super().process_actions(walk)       # stored as raw_actions -> walking last_action obs stays consistent

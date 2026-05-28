"""
Thin adapter that bridges SumoEnv (our SUMO intersection env) to the
interface expected by disc-gym's A2C / DQN / LCPO trainers.

SumoEnv interface (non-gymnasium):
    obs              = env.reset()             -> np.ndarray [24]
    obs, r, done, info = env.step(action: int) -> ...

Adapter adds:
    env.obs_dim   = 24
    env.action_dim = 1
    env.n_bins    = 2
    env.reset()   -> (obs, {})     (gymnasium-style tuple)
    env.step(act) -> (obs, r, done, False, info)  (adds truncated=False)
    env.is_different(new_obs, recent_obs) -> bool[N]   (for LCPO OOD buffer)
"""

from __future__ import annotations

import numpy as np


class SumoEnvAdapter:
    """
    Wraps a SumoEnv (or any env with reset()->obs, step(int)->(obs,r,done,info))
    into the disc-gym trainer interface.

    Parameters
    ----------
    env         : SumoEnv instance
    ood_sigma   : threshold multiplier for is_different()  (default 2.0)
    """

    def __init__(self, env, ood_sigma: float = 2.0):
        self._env       = env
        self.ood_sigma  = ood_sigma

        # disc-gym interface attributes
        self.obs_dim    = env.observation_space_shape[0]   # 24
        self.action_dim = 1                                 # single discrete action
        self.n_bins     = env.action_space_n               # 2

    # ------------------------------------------------------------------
    # Core env methods
    # ------------------------------------------------------------------

    def reset(self):
        obs = self._env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, act: np.ndarray):
        """act: np.ndarray shape [1] from FCNPolicy.sample_action"""
        obs, rew, done, info = self._env.step(int(act[0]))
        return np.array(obs, dtype=np.float32), float(rew), done, False, info

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # LCPO: OOD detection in raw observation space
    # Used by vanilla LCPO only; GRU-LCPO overrides this in latent space
    # ------------------------------------------------------------------

    def is_different(self, new_obs: np.ndarray, recent_obs: np.ndarray) -> np.ndarray:
        """
        Return bool[N] — True if new_obs[i] is OOD relative to recent_obs.

        Uses mean + sigma * std distance in raw observation space.
        """
        mu        = recent_obs.mean(axis=0)
        var       = recent_obs.var(axis=0).mean() + 1e-6
        dists     = np.linalg.norm(new_obs - mu, axis=-1)
        threshold = self.ood_sigma * np.sqrt(var * self.obs_dim)
        return dists > threshold

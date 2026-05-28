"""
GRU-LCPO Agent
==============
LCPO with a GRU-based latent context encoder instead of manually-labeled z_t.

Key changes vs vanilla LCPO
----------------------------
1. GRUEncoder takes the last k observations -> h_t  (context vector)
2. Policy and Value nets receive  aug_obs = concat(s_t, h_t)  as input
3. OOD detection happens in h_t latent space instead of raw obs space
4. No manual context labels required at any point

Training flow
-------------
    collect  : aug_obs = [s_t, GRUEncoder(history)]  stored in replay buffer
    update   : train_lcpo(aug_obs, ...)    -- TRPO / LCPO gradient step
    OOD buf  : stores aug_obs; is_different() checks h_t portion only
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from agent.core_alg.core_lcpo import train_lcpo
from buffer.buffer import TransitionBuffer
from buffer.buffer_ood import OutOfDSampler
from neural_net.gru_encoder import GRUEncoder, ObsHistoryEncoder
from neural_net.nn import FCNPolicy, FullyConnectNN
from utils.rms import RunningMeanStd


# ---------------------------------------------------------------------------
# OOD detector in latent context space
# ---------------------------------------------------------------------------

def _make_ctx_ood_detector(obs_dim: int, context_dim: int, sigma: float = 2.0):
    """
    Return an is_different(new_aug_obs, recent_aug_obs) -> bool[N] function.

    A stored sample is considered OOD if its h_t (context part of aug_obs)
    deviates more than `sigma` normalised standard deviations from the cluster
    of recent h_t vectors.
    """
    def _is_different(new_aug: np.ndarray, recent_aug: np.ndarray) -> np.ndarray:
        new_h    = new_aug[:, obs_dim:]     # [N, context_dim]
        recent_h = recent_aug[:, obs_dim:]  # [M, context_dim]

        mu  = recent_h.mean(axis=0)                          # [context_dim]
        var = recent_h.var(axis=0).mean() + 1e-6             # scalar

        dists = np.linalg.norm(new_h - mu, axis=-1)         # [N]
        threshold = sigma * np.sqrt(var * context_dim)
        return dists > threshold

    return _is_different


# ---------------------------------------------------------------------------
# GRU-LCPO Agent
# ---------------------------------------------------------------------------

class GRULCPOAgent:
    """
    GRU-LCPO agent compatible with the SUMO intersection environment.

    Parameters
    ----------
    obs_dim       : raw observation size  (e.g. 24)
    act_bins      : number of discrete actions  (e.g. 2)
    nn_hids       : hidden layer sizes for policy / value nets
    gru_hidden    : GRU hidden state size
    context_dim   : dimension of h_t (replaces z_t)
    k             : history window length fed to GRU
    trpo_kl_in    : KL constraint for in-distribution states (LCPO local)
    trpo_kl_out   : KL constraint for OOD states (LCPO global)
    trpo_dual     : use dual-step TRPO (recommended)
    ood_mini_len  : recent-window size for OOD sampler
    ood_len       : reservoir capacity for OOD sampler
    """

    def __init__(
        self,
        obs_dim:      int,
        act_bins:     int,
        nn_hids:      List[int]    = (128, 128),
        gru_hidden:   int          = 64,
        context_dim:  int          = 32,
        k:            int          = 10,
        batch_size:   int          = 2048,
        lr_rate:      float        = 3e-4,
        val_lr_rate:  float        = 1e-3,
        gamma:        float        = 0.99,
        lam:          float        = 0.95,
        trpo_kl_in:   float        = 0.01,
        trpo_kl_out:  float        = 0.05,
        trpo_damping: float        = 0.1,
        trpo_dual:    bool         = True,
        ood_mini_len: int          = 50,
        ood_len:      int          = 5000,
        entropy_max:  float        = 0.5,
        entropy_min:  float        = 0.01,
        entropy_decay:float        = 1e-5,
        reward_scale: float        = 1.0,
        output_folder:str          = "results/gru_lcpo",
        device:       str          = "cpu",
        seed:         int          = 0,
        monitor:      Optional[SummaryWriter] = None,
    ):
        self.obs_dim     = obs_dim
        self.context_dim = context_dim
        self.aug_dim     = obs_dim + context_dim   # input to policy / value
        self.act_bins    = act_bins
        self.k           = k
        self.gamma       = gamma
        self.lamb        = lam
        self.trpo_kl_in  = trpo_kl_in
        self.trpo_kl_out = trpo_kl_out
        self.trpo_damping= trpo_damping
        self.trpo_dual   = trpo_dual
        self.output_folder = output_folder
        self.monitor     = monitor
        self.it          = 0

        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # -- GRU encoder --------------------------------------------------
        self.gru_encoder  = GRUEncoder(obs_dim, gru_hidden, context_dim).to(self.device)
        self.obs_history  = ObsHistoryEncoder(self.gru_encoder, k, obs_dim, self.device)

        # -- Policy net: input = aug_obs = [s_t, h_t] ---------------------
        # act_len=1 (single action dim), act_bins=2 (keep / switch)
        self.policy_net = FCNPolicy(
            self.aug_dim, list(nn_hids), act_bins, 1,
            act=nn.ReLU, final_layer_act=False,
        )
        self.policy_net = torch.jit.script(self.policy_net).to(self.device)

        # -- Value net: same input ----------------------------------------
        self.value_net = FullyConnectNN(
            self.aug_dim, list(nn_hids), 1, 1,
            act=nn.ReLU, final_layer_act=False,
        )
        self.value_net = torch.jit.script(self.value_net).to(self.device)

        self.net_opt_p = torch.optim.Adam(
            self.policy_net.parameters(), lr=lr_rate, weight_decay=1e-4, eps=1e-5
        )
        self.net_opt_v = torch.optim.Adam(
            self.value_net.parameters(), lr=val_lr_rate, weight_decay=1e-4, eps=1e-5
        )
        self.net_loss = nn.MSELoss(reduction="mean")

        # -- Replay buffer (stores aug_obs) --------------------------------
        self.buff = TransitionBuffer(self.aug_dim, 1, batch_size, reward_scale)
        self.batch_rng = np.random.RandomState(seed=seed)

        # -- OOD buffer: stores aug_obs, detects via h_t portion ----------
        self.ood_buf = OutOfDSampler(
            self.aug_dim, ood_mini_len, ood_len,
            _make_ctx_ood_detector(obs_dim, context_dim),
        )

        # -- Entropy schedule ---------------------------------------------
        self.entropy_factor = entropy_max
        self.entropy_min    = entropy_min
        self.entropy_decay  = entropy_decay

        self.ret_rms = RunningMeanStd(shape=())

        os.makedirs(output_folder + "/models", exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Call at every episode boundary."""
        self.obs_history.reset()

    def augment(self, obs: np.ndarray) -> np.ndarray:
        """Return aug_obs = concat(s_t, h_t).  Updates internal GRU history."""
        h_t = self.obs_history.encode(obs)
        return np.concatenate([obs, h_t])

    def sample_action(self, aug_obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(aug_obs, dtype=torch.float, device=self.device).unsqueeze(0)
        return self.policy_net.sample_action(t).cpu().numpy()

    def run_training(self, env, num_epochs: int, save_interval: int = 100) -> None:
        """
        Main training loop.

        env must expose the SumoEnv interface:
            obs          = env.reset()          -> np.ndarray
            obs, r, done, info = env.step(int)
        """
        obs = env.reset()
        self.reset_history()
        ret = 0.0
        ret_list: List[float] = []

        for epoch in range(num_epochs):
            self.buff.reset_head()
            ret_list.clear()

            # -- Collect one batch of experience --------------------------
            while not self.buff.buffer_full():
                aug_obs  = self.augment(obs)
                act      = self.sample_action(aug_obs)

                next_obs, rew, done, info = env.step(int(act[0]))
                aug_next = self.augment(next_obs)

                self.buff.add_exp(aug_obs, act, rew, aug_next, done, False)
                ret = ret * self.gamma + rew
                ret_list.append(ret)

                obs = next_obs
                if done:
                    obs = env.reset()
                    self.reset_history()
                    ret = 0.0

            (aug_obs_np, aug_next_np,
             acts_np, rews_np,
             term_np, trunc_np) = self.buff.get()

            # -- Normalise rewards ----------------------------------------
            self.ret_rms.update(np.array(ret_list))
            norm_rews = rews_np / np.sqrt(self.ret_rms.var + 1)

            # -- LCPO update ----------------------------------------------
            pg_loss, v_loss, entropy = self._update(
                aug_obs_np, aug_next_np, acts_np, norm_rews, term_np, trunc_np
            )

            # -- Logging --------------------------------------------------
            if self.monitor is not None:
                self.monitor.add_scalar("train/pg_loss",     pg_loss,            epoch)
                self.monitor.add_scalar("train/v_loss",      v_loss,             epoch)
                self.monitor.add_scalar("train/entropy",     entropy,            epoch)
                self.monitor.add_scalar("train/mean_reward", np.mean(rews_np),   epoch)
                ctx_info = self._context_stats(aug_obs_np)
                self.monitor.add_scalar("ctx/h_mean_norm",  ctx_info[0],         epoch)
                self.monitor.add_scalar("ctx/h_std",        ctx_info[1],         epoch)

            if epoch % save_interval == 0:
                self._save(epoch)
                print(
                    f"[GRU-LCPO] epoch={epoch:5d} | "
                    f"pg={pg_loss:7.3f} | v={v_loss:7.3f} | "
                    f"ent={entropy:.3f} | r̄={np.mean(rews_np):8.1f}"
                )

        self._save(num_epochs)

    def save(self, path: str) -> None:
        torch.save({
            "policy_net":  self.policy_net.state_dict(),
            "value_net":   self.value_net.state_dict(),
            "gru_encoder": self.gru_encoder.state_dict(),
            "net_opt_p":   self.net_opt_p.state_dict(),
            "net_opt_v":   self.net_opt_v.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.gru_encoder.load_state_dict(ckpt["gru_encoder"])
        if "net_opt_p" in ckpt:
            self.net_opt_p.load_state_dict(ckpt["net_opt_p"])
            self.net_opt_v.load_state_dict(ckpt["net_opt_v"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update(
        self,
        aug_obs_np:  np.ndarray,
        aug_next_np: np.ndarray,
        acts_np:     np.ndarray,
        rews_np:     np.ndarray,
        term_np:     np.ndarray,
        trunc_np:    np.ndarray,
    ) -> Tuple[float, float, float]:

        # Feed aug_obs into OOD sampler; get_dist detects via h_t portion
        self.ood_buf.add_many_exp(aug_obs_np, self.batch_rng)
        ood_raw = self.ood_buf.get(self.batch_rng, len(aug_obs_np))
        ood_np  = np.array(ood_raw) if ood_raw else np.zeros((0, self.aug_dim), dtype=np.float32)

        monitor = self.monitor if self.monitor is not None else _DummyMonitor()

        pg_loss, v_loss, entropy, *_ = train_lcpo(
            self.value_net, self.policy_net,
            self.net_opt_p, self.net_opt_v, self.net_loss,
            self.device,
            acts_np, aug_next_np, rews_np, aug_obs_np, term_np, trunc_np,
            self.gamma, self.lamb,
            self.trpo_kl_in, self.trpo_kl_out, self.trpo_damping, self.trpo_dual,
            self.entropy_factor,
            ood_obs_np=ood_np,
            monitor=monitor,
            it=self.it,
        )

        self.it += 1
        self.entropy_factor = max(self.entropy_factor - self.entropy_decay, self.entropy_min)
        return float(pg_loss), float(v_loss), float(entropy)

    def _save(self, epoch: int) -> None:
        self.save(f"{self.output_folder}/models/model_{epoch}")

    def _context_stats(self, aug_obs_np: np.ndarray) -> Tuple[float, float]:
        h = aug_obs_np[:, self.obs_dim:]        # [B, context_dim]
        norms = np.linalg.norm(h, axis=-1)
        return float(norms.mean()), float(h.std())


class _DummyMonitor:
    """Drop-in for SummaryWriter when no tensorboard is configured."""
    def add_scalar(self, *args, **kwargs): pass

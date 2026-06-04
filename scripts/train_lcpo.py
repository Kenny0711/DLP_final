"""
Person 2 ── LCPO (Locally-Constrained Policy Optimisation)
===========================================================
Initialises from the A2C warm-up checkpoint, then trains with the LCPO
TRPO-style constrained update that prevents catastrophic forgetting.

從 A2C warm-up 權重出發，改用 LCPO 的 TRPO 約束更新。
LCPO 會偵測 OOD 狀態（過去 context 的觀測值），加入 KL 約束保護舊知識。

Prerequisites / 前置條件:
  Run collect_warmup.py first!

Usage:
  python train_lcpo.py

Output:
  results/lcpo/models/model_*.pt
  results/lcpo/log.csv
"""

import os, sys, csv
import numpy as np
import torch
import torch.nn as nn

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SUMO_DIR)

from neural_net.nn import FCNPolicy, FullyConnectNN
from agent.core_alg.core_lcpo import train_lcpo
from buffer.buffer import TransitionBuffer
from buffer.buffer_ood import OutOfDSampler
from utils.rms import RunningMeanStd
from utils.sumo_path import ensure_sumo_in_path  # noqa
from sumo_env import SumoIntersectionEnv
import traci as _traci

# ── Config ───────────────────────────────────────────────────────────────────
FULL_CFG   = os.path.join(HERE, "nets", "intersection.sumocfg")
WARMUP_PT  = os.path.join(HERE, "results", "warmup", "checkpoint.pt")
RESULT_DIR = os.path.join(HERE, "results", "lcpo")
LOG_PATH   = os.path.join(RESULT_DIR, "log.csv")

OBS_DIM    = 24
ACT_BINS   = 2
NN_HIDS    = [128, 128]
N_EPOCHS   = 500
BATCH_SIZE = 1080
LR_P       = 3e-4
LR_V       = 1e-3
GAMMA      = 0.99
LAM        = 0.95
ENTROPY    = 0.05
# LCPO-specific / LCPO 專用參數
KL_IN      = 0.01   # local KL constraint (recent states) / 近期狀態 KL 上限
KL_OUT     = 0.05   # global KL constraint (OOD states)  / OOD 狀態 KL 上限
DAMPING    = 0.1    # conjugate gradient damping
DUAL       = True   # use dual-step TRPO
OOD_WIN    = 100    # recent-window size for OOD sampler
OOD_CAP    = 8000   # reservoir capacity for OOD sampler
SAVE_EVERY = 50
DEVICE     = "cpu"
SEED       = 42


def get_context(step: int) -> str:
    if step < 360:  return "morning"
    if step < 720:  return "off"
    return "evening"


def make_ood_detector(obs_dim: int):
    """
    OOD detector in raw obs space.
    A stored obs is OOD if its L2 distance from recent cluster > 2σ.
    OOD 偵測：若舊 obs 與近期 obs 的距離超過 2σ，視為 OOD。
    """
    def _is_different(new_obs: np.ndarray, recent_obs: np.ndarray) -> np.ndarray:
        mu  = recent_obs.mean(axis=0)
        var = recent_obs.var(axis=0).mean() + 1e-6
        return np.linalg.norm(new_obs - mu, axis=-1) > 2.0 * np.sqrt(var * obs_dim)
    return _is_different


class _NoMonitor:
    def add_scalar(self, *a, **k): pass


def main():
    os.makedirs(RESULT_DIR + "/models", exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    assert os.path.exists(WARMUP_PT), \
        f"找不到 {WARMUP_PT}，請先執行 collect_warmup.py"

    env = SumoIntersectionEnv(FULL_CFG, gui=False)

    # ── Networks (same architecture as A2C warm-up) ──────────────────────────
    policy_net = torch.jit.script(
        FCNPolicy(OBS_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    )
    value_net = torch.jit.script(
        FullyConnectNN(OBS_DIM, NN_HIDS, 1, 1, act=nn.ReLU, final_layer_act=False)
    )
    opt_p   = torch.optim.Adam(policy_net.parameters(), lr=LR_P, weight_decay=1e-4, eps=1e-5)
    opt_v   = torch.optim.Adam(value_net.parameters(),  lr=LR_V, weight_decay=1e-4, eps=1e-5)
    loss_fn = nn.MSELoss()

    # ── Load A2C warm-up weights / 載入 A2C 權重直接繼承 ────────────────────
    ckpt = torch.load(WARMUP_PT, map_location=DEVICE)
    policy_net.load_state_dict(ckpt["policy_net"])
    value_net.load_state_dict(ckpt["value_net"])
    print(f"[LCPO] Loaded A2C warm-up weights from {WARMUP_PT}")

    # ── OOD buffer / OOD 記憶緩衝區 ─────────────────────────────────────────
    batch_rng = np.random.RandomState(SEED)
    ood_buf   = OutOfDSampler(OBS_DIM, OOD_WIN, OOD_CAP, make_ood_detector(OBS_DIM))

    monitor = _NoMonitor()
    ret_rms = RunningMeanStd(shape=())
    buff    = TransitionBuffer(OBS_DIM, 1, BATCH_SIZE)
    entropy_factor = ENTROPY

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "mean_reward", "avg_wait",
            "r_morning", "r_off", "r_evening",
            "pg_loss", "v_loss", "ood_size",
        ])

    obs, _ = env.reset()

    for epoch in range(N_EPOCHS):
        buff.reset_head()
        step_rewards = {c: [] for c in ("morning", "off", "evening")}
        step_waits = []
        step_idx = 0

        while not buff.buffer_full():
            obs_t = torch.as_tensor(obs, dtype=torch.float).unsqueeze(0)
            act   = policy_net.sample_action(obs_t).numpy()

            next_obs, rew, done, trunc, info = env.step(int(act[0]))
            buff.add_exp(obs, act, rew, next_obs, done, trunc)
            step_rewards[get_context(step_idx)].append(rew)
            try:
                _w = sum(_traci.edge.getWaitingTime(e) for e in ["N2C","S2C","E2C","W2C","C2N","C2S","C2E","C2W"]) / 8.0
            except Exception:
                _w = 0.0
            step_waits.append(_w)
            step_idx += 1

            obs = next_obs
            if done or trunc:
                obs, _ = env.reset()
                step_idx = 0

        obs_b, next_b, act_b, rew_b, term_b, trunc_b = buff.get()
        ret_rms.update(rew_b)
        norm_rew = rew_b / np.sqrt(ret_rms.var + 1)

        # ── OOD sampling / 從緩衝區取 OOD 樣本 ──────────────────────────────
        ood_buf.add_many_exp(obs_b, batch_rng)
        ood_raw = ood_buf.get(batch_rng, BATCH_SIZE)
        ood_np  = np.array(ood_raw) if ood_raw else np.zeros((0, OBS_DIM), dtype=np.float32)

        # ── LCPO update / LCPO TRPO 約束更新 ────────────────────────────────
        pg_loss, v_loss, ent, *_ = train_lcpo(
            value_net, policy_net, opt_p, opt_v, loss_fn,
            torch.device(DEVICE),
            act_b, next_b, norm_rew, obs_b, term_b, trunc_b,
            GAMMA, LAM, KL_IN, KL_OUT, DAMPING, DUAL,
            entropy_factor, ood_obs_np=ood_np,
            monitor=monitor, it=epoch,
        )
        entropy_factor = max(entropy_factor * 0.999, 0.01)

        r_m = np.mean(step_rewards["morning"])  if step_rewards["morning"]  else 0
        r_o = np.mean(step_rewards["off"])      if step_rewards["off"]      else 0
        r_e = np.mean(step_rewards["evening"])  if step_rewards["evening"]  else 0
        mean_r = float(np.mean(rew_b))

        print(f"[LCPO] epoch={epoch:4d} | "
              f"r̄={mean_r:7.2f} | "
              f"morning={r_m:6.1f} off={r_o:6.1f} evening={r_e:6.1f} | wait={avg_wait:5.1f}s | "
              f"ood={len(ood_raw):4d} | pg={pg_loss:.4f}")

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, mean_r, r_m, r_o, r_e, pg_loss, v_loss, len(ood_raw)]
            )

        if epoch % SAVE_EVERY == 0:
            torch.save({"policy_net": policy_net.state_dict(),
                        "value_net":  value_net.state_dict()},
                       f"{RESULT_DIR}/models/model_{epoch}.pt")

    torch.save({"policy_net": policy_net.state_dict(),
                "value_net":  value_net.state_dict()},
               f"{RESULT_DIR}/models/model_final.pt")
    env.close()
    print(f"\n[LCPO] Done. Log → {LOG_PATH}")


if __name__ == "__main__":
    main()

"""
Person 1 ── A2C (Advantage Actor-Critic)
=========================================
Loads the A2C warm-up checkpoint and continues training on the full
3-context scenario (morning → off-peak → evening).

載入 warm-up 的 A2C checkpoint，在三段 context 的完整場景上繼續訓練。
Baseline: standard A2C without any catastrophic-forgetting mitigation.

Prerequisites / 前置條件:
  Run collect_warmup.py first!  先跑 collect_warmup.py！

Usage / 使用方式:
  python train_a2c.py

Output / 輸出:
  results/a2c/models/model_*.pt
  results/a2c/log.csv
"""

import os, sys, csv
import numpy as np
import torch
import torch.nn as nn

# ── Path setup ───────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, SUMO_DIR)

from neural_net.nn import FCNPolicy, FullyConnectNN
from agent.core_alg.core_pg import train_actor_critic
from buffer.buffer import TransitionBuffer
from utils.rms import RunningMeanStd
from utils.sumo_path import ensure_sumo_in_path  # noqa
from sumo_env_revised import SumoEnv

# ── Config ───────────────────────────────────────────────────────────────────
FULL_CFG   = os.path.join(HERE, "nets", "intersection.sumocfg")
WARMUP_PT  = os.path.join(HERE, "results", "warmup", "checkpoint.pt")
RESULT_DIR = os.path.join(HERE, "results", "a2c")
LOG_PATH   = os.path.join(RESULT_DIR, "log.csv")

OBS_DIM    = 24
ACT_BINS   = 2
NN_HIDS    = [128, 128]
N_EPOCHS   = 500       # 完整訓練輪數
BATCH_SIZE = 1080      # 1 episode = 10800s / 10s = 1080 steps
LR_P       = 3e-4
LR_V       = 1e-3
GAMMA      = 0.99
LAM        = 0.95
ENTROPY    = 0.05
SAVE_EVERY = 50
DEVICE     = "cpu"
SEED       = 42


def get_context(step: int) -> str:
    """Map env step → context name (within one 1080-step episode)"""
    if step < 360:  return "morning"    # SUMO 0–3600s
    if step < 720:  return "off"        # SUMO 3600–7200s
    return "evening"                    # SUMO 7200–10800s


class _NoMonitor:
    def add_scalar(self, *a, **k): pass


def main():
    os.makedirs(RESULT_DIR + "/models", exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    assert os.path.exists(WARMUP_PT), \
        f"找不到 {WARMUP_PT}，請先執行 collect_warmup.py"

    # ── Environment ──────────────────────────────────────────────────────────
    env = SumoEnv(use_gui = False)   # 1080 steps per episode

    # ── Networks ─────────────────────────────────────────────────────────────
    policy_net = torch.jit.script(
        FCNPolicy(OBS_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    )
    value_net = torch.jit.script(
        FullyConnectNN(OBS_DIM, NN_HIDS, 1, 1, act=nn.ReLU, final_layer_act=False)
    )
    opt_p   = torch.optim.Adam(policy_net.parameters(), lr=LR_P, weight_decay=1e-4, eps=1e-5)
    opt_v   = torch.optim.Adam(value_net.parameters(),  lr=LR_V, weight_decay=1e-4, eps=1e-5)
    loss_fn = nn.MSELoss()

    # ── Load warm-up weights / 載入 warm-up 模型 ────────────────────────────
    ckpt = torch.load(WARMUP_PT, map_location=DEVICE)
    policy_net.load_state_dict(ckpt["policy_net"])
    value_net.load_state_dict(ckpt["value_net"])
    print(f"[A2C] Loaded warm-up checkpoint from {WARMUP_PT}")

    monitor = _NoMonitor()
    ret_rms = RunningMeanStd(shape=())
    buff    = TransitionBuffer(OBS_DIM, 1, BATCH_SIZE)
    entropy_factor = ENTROPY

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "mean_reward", "avg_wait",
            "r_morning", "r_off", "r_evening",
            "pg_loss", "v_loss",
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

        pg_loss, v_loss, *_ = train_actor_critic(
            value_net, policy_net, opt_p, opt_v, loss_fn,
            torch.device(DEVICE),
            act_b, next_b, norm_rew, obs_b, term_b, trunc_b,
            GAMMA, LAM, entropy_factor, monitor=monitor, it=epoch,
        )
        entropy_factor = max(entropy_factor * 0.999, 0.01)

        r_m = np.mean(step_rewards["morning"])  if step_rewards["morning"]  else 0
        r_o = np.mean(step_rewards["off"])      if step_rewards["off"]      else 0
        r_e = np.mean(step_rewards["evening"])  if step_rewards["evening"]  else 0
        mean_r = float(np.mean(rew_b))
        avg_wait = float(np.mean(step_waits)) if step_waits else 0.0

        print(f"[A2C] epoch={epoch:4d} | "
              f"r̄={mean_r:7.2f} | "
              f"morning={r_m:6.1f} off={r_o:6.1f} evening={r_e:6.1f} | wait={avg_wait:5.1f}s | "
              f"pg={pg_loss:.4f} v={v_loss:.4f}")

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_r, avg_wait, r_m, r_o, r_e, pg_loss, v_loss])

        if epoch % SAVE_EVERY == 0:
            path = f"{RESULT_DIR}/models/model_{epoch}.pt"
            torch.save({"policy_net": policy_net.state_dict(),
                        "value_net":  value_net.state_dict()}, path)

    # Final save
    torch.save({"policy_net": policy_net.state_dict(),
                "value_net":  value_net.state_dict()},
               f"{RESULT_DIR}/models/model_final.pt")
    env.close()
    print(f"\n[A2C] Done. Log → {LOG_PATH}")


if __name__ == "__main__":
    main()

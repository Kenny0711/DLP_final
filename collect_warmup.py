"""
Step 0 ── A2C Warm-up + Trajectory Collection  (所有人先跑這個)
==============================================================
Run this FIRST. It trains A2C on morning_peak only (warmup.sumocfg),
saves a checkpoint and the collected trajectory for other algorithms.

先跑這個腳本！用 A2C 在早峰場景做 warm-up，
把模型和 trajectory 存起來，給其他 4 個算法用。

Outputs / 輸出:
  results/warmup/checkpoint.pt      A2C model weights
  results/warmup/trajectory.pkl     transitions collected by A2C
  results/warmup/log.csv            per-epoch training log

Usage / 使用方式:
  python collect_warmup.py
"""

import os, sys, pickle, csv
import numpy as np
import torch
import torch.nn as nn

# ── Path setup / 路徑設定 ────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SUMO_DIR)

from neural_net.nn import FCNPolicy, FullyConnectNN
from agent.core_alg.core_pg import train_actor_critic
from buffer.buffer import TransitionBuffer
from utils.rms import RunningMeanStd
from sumo_env import SumoIntersectionEnv

# ── Config / 超參數設定 ──────────────────────────────────────────────────────
WARMUP_CFG    = os.path.join(HERE, "nets", "warmup.sumocfg")
RESULT_DIR    = os.path.join(HERE, "results", "warmup")
CKPT_PATH     = os.path.join(RESULT_DIR, "checkpoint.pt")
TRAJ_PATH     = os.path.join(RESULT_DIR, "trajectory.pkl")
LOG_PATH      = os.path.join(RESULT_DIR, "log.csv")

OBS_DIM       = 24      # observation dimension / 觀測維度
ACT_BINS      = 2       # keep=0 / switch=1
NN_HIDS       = [128, 128]
WARMUP_EPOCHS = 200     # warm-up training epochs / warm-up 訓練輪數
BATCH_SIZE    = 360     # 1 episode = 3600s / 10s = 360 steps
LR_P          = 3e-4    # policy learning rate
LR_V          = 1e-3    # value learning rate
GAMMA         = 0.99
LAM           = 0.95    # GAE lambda
ENTROPY       = 0.1     # entropy bonus coefficient
DEVICE        = "cpu"
SEED          = 42


# ── Helpers / 輔助工具 ───────────────────────────────────────────────────────
class _NoMonitor:
    def add_scalar(self, *a, **k): pass


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Environment / 環境 ──────────────────────────────────────────────────
    env = SumoIntersectionEnv(WARMUP_CFG, gui=False)
    env.max_steps = 360   # warmup is 3600s / 10s per step = 360 steps

    # ── Networks / 網路 ─────────────────────────────────────────────────────
    # FCNPolicy: in=[24] → hidden → out=[2] (Q for keep/switch)
    policy_net = torch.jit.script(
        FCNPolicy(OBS_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    )
    value_net = torch.jit.script(
        FullyConnectNN(OBS_DIM, NN_HIDS, 1, 1, act=nn.ReLU, final_layer_act=False)
    )
    opt_p   = torch.optim.Adam(policy_net.parameters(), lr=LR_P, weight_decay=1e-4, eps=1e-5)
    opt_v   = torch.optim.Adam(value_net.parameters(),  lr=LR_V, weight_decay=1e-4, eps=1e-5)
    loss_fn = nn.MSELoss()
    monitor = _NoMonitor()
    ret_rms = RunningMeanStd(shape=())
    buff    = TransitionBuffer(OBS_DIM, 1, BATCH_SIZE)

    # ── Trajectory storage / trajectory 儲存容器 ────────────────────────────
    traj_obs, traj_act, traj_rew, traj_next, traj_done = [], [], [], [], []

    entropy_factor = ENTROPY

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "mean_reward", "pg_loss", "v_loss"])

    obs, _ = env.reset()

    for epoch in range(WARMUP_EPOCHS):
        buff.reset_head()
        ep_rews = []

        # ── Collect one episode / 收集一個 episode ───────────────────────
        while not buff.buffer_full():
            obs_t = torch.as_tensor(obs, dtype=torch.float).unsqueeze(0)
            act   = policy_net.sample_action(obs_t).numpy()        # shape [1]

            next_obs, rew, done, trunc, info = env.step(int(act[0]))
            buff.add_exp(obs, act, rew, next_obs, done, trunc)
            ep_rews.append(rew)

            # Save transition for trajectory file
            # 把每個 transition 存下來給其他算法用
            traj_obs.append(obs.copy())
            traj_act.append(int(act[0]))
            traj_rew.append(float(rew))
            traj_next.append(next_obs.copy())
            traj_done.append(bool(done or trunc))

            obs = next_obs
            if done or trunc:
                obs, _ = env.reset()

        # ── A2C update / A2C 更新 ────────────────────────────────────────
        obs_b, next_b, act_b, rew_b, term_b, trunc_b = buff.get()
        ret_rms.update(rew_b)
        norm_rew = rew_b / np.sqrt(ret_rms.var + 1)   # reward normalisation

        pg_loss, v_loss, ent, *_ = train_actor_critic(
            value_net, policy_net, opt_p, opt_v, loss_fn,
            torch.device(DEVICE),
            act_b, next_b, norm_rew, obs_b, term_b, trunc_b,
            GAMMA, LAM, entropy_factor, monitor=monitor, it=epoch,
        )
        entropy_factor = max(entropy_factor * 0.999, 0.01)

        mean_r = float(np.mean(ep_rews))
        print(f"[warmup] epoch={epoch:4d} | r̄={mean_r:8.2f} | "
              f"pg={pg_loss:.4f} | v={v_loss:.4f}")

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_r, pg_loss, v_loss])

    # ── Save checkpoint / 儲存 A2C 模型 ─────────────────────────────────────
    torch.save({
        "policy_net": policy_net.state_dict(),
        "value_net":  value_net.state_dict(),
        "opt_p":      opt_p.state_dict(),
        "opt_v":      opt_v.state_dict(),
    }, CKPT_PATH)
    print(f"\n[warmup] Checkpoint → {CKPT_PATH}")

    # ── Save trajectory / 儲存 trajectory ───────────────────────────────────
    traj = {
        "obs":      np.array(traj_obs,  dtype=np.float32),   # [T, 24]
        "actions":  np.array(traj_act,  dtype=np.int64),     # [T]
        "rewards":  np.array(traj_rew,  dtype=np.float32),   # [T]
        "next_obs": np.array(traj_next, dtype=np.float32),   # [T, 24]
        "dones":    np.array(traj_done, dtype=bool),          # [T]
    }
    with open(TRAJ_PATH, "wb") as f:
        pickle.dump(traj, f)
    print(f"[warmup] Trajectory → {TRAJ_PATH}  ({len(traj_obs)} transitions)")

    env.close()
    print("\n[warmup] Done! 接下來各人跑自己的訓練腳本。")
    print("  Person 1: python train_a2c.py")
    print("  Person 2: python train_lcpo.py")
    print("  Person 3: python train_ddqn.py")
    print("  Person 4: python train_gru_lcpo.py")


if __name__ == "__main__":
    main()

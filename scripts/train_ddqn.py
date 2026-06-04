"""
Person 3 ── DDQN (Double Deep Q-Network)
==========================================
Phase 1 (offline): Pre-fills replay buffer with A2C trajectory data,
                   then pre-trains the Q-network (behaviour cloning style).
Phase 2 (online):  Continues training with ε-greedy online exploration
                   on the full 3-context scenario.

第一階段（offline）：把 A2C 的 trajectory 載入 replay buffer，先做離線預訓練。
第二階段（online）： 接著在完整三段場景做 ε-greedy 線上訓練。

Cannot directly reuse A2C policy weights (different architecture).
不能直接用 A2C 的 policy 權重（網路架構不同，DDQN 用 Q-value 輸出）。

Prerequisites: collect_warmup.py

Usage:
  python train_ddqn.py

Output:
  results/ddqn/models/model_*.pt
  results/ddqn/log.csv
"""

import os, sys, csv, pickle
from collections import deque
import numpy as np
import torch
import torch.nn as nn

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SUMO_DIR)

from neural_net.nn import FCNPolicy
from agent.core_alg.core_dqn import train_dqn, soft_copy
from utils.sumo_path import ensure_sumo_in_path  # noqa
from sumo_env import SumoIntersectionEnv
import traci as _traci

# ── Config ───────────────────────────────────────────────────────────────────
FULL_CFG    = os.path.join(HERE, "nets", "intersection.sumocfg")
TRAJ_PATH   = os.path.join(HERE, "results", "warmup", "trajectory.pkl")
RESULT_DIR  = os.path.join(HERE, "results", "ddqn")
LOG_PATH    = os.path.join(RESULT_DIR, "log.csv")

OBS_DIM     = 24
ACT_BINS    = 2
NN_HIDS     = [128, 128]

# Offline pre-train / 離線預訓練
OFFLINE_STEPS  = 5_000    # SGD steps on A2C trajectory data
OFFLINE_BATCH  = 128

# Online training / 線上訓練
N_ONLINE_EPOCHS = 500
ONLINE_BATCH    = 1080    # steps per epoch = 1 full episode
REPLAY_CAP      = 50_000  # replay buffer capacity
LR              = 1e-3
GAMMA           = 0.99
TAU             = 0.005   # soft target update rate
EPS_START       = 0.3     # exploration start (lower because pre-trained)
EPS_END         = 0.05
EPS_DECAY       = 0.995   # per-epoch decay
SAVE_EVERY      = 50
DEVICE          = "cpu"
SEED            = 42


def get_context(step: int) -> str:
    if step < 360:  return "morning"
    if step < 720:  return "off"
    return "evening"


# ── Simple replay buffer ─────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
        self.rng = np.random.default_rng(SEED)

    def push(self, obs, act, rew, next_obs, done):
        self.buf.append((obs, int(act), float(rew), next_obs, bool(done)))

    def sample(self, batch_size: int):
        idx  = self.rng.integers(0, len(self.buf), batch_size)
        data = [self.buf[i] for i in idx]
        obs      = np.array([d[0] for d in data], dtype=np.float32)
        actions  = np.array([[d[1]] for d in data], dtype=np.int64)  # [B, 1]
        rewards  = np.array([d[2] for d in data], dtype=np.float32)
        next_obs = np.array([d[3] for d in data], dtype=np.float32)
        dones    = np.array([d[4] for d in data], dtype=bool)
        return obs, actions, rewards, next_obs, dones

    def __len__(self): return len(self.buf)


def make_nets():
    q_net = torch.jit.script(
        FCNPolicy(OBS_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    )
    target = torch.jit.script(
        FCNPolicy(OBS_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    )
    soft_copy(target, q_net, tau=1.0)  # hard copy
    return q_net, target


def main():
    os.makedirs(RESULT_DIR + "/models", exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    assert os.path.exists(TRAJ_PATH), \
        f"找不到 {TRAJ_PATH}，請先執行 collect_warmup.py"

    # ── Load A2C trajectory / 載入 A2C trajectory ────────────────────────────
    with open(TRAJ_PATH, "rb") as f:
        traj = pickle.load(f)
    print(f"[DDQN] Loaded {len(traj['obs'])} transitions from trajectory")

    replay = ReplayBuffer(REPLAY_CAP)

    # Pre-fill replay buffer with A2C transitions
    # 先把 A2C 的 transition 填進 replay buffer
    for i in range(len(traj["obs"])):
        replay.push(traj["obs"][i], traj["actions"][i], traj["rewards"][i],
                    traj["next_obs"][i], traj["dones"][i])
    print(f"[DDQN] Replay buffer pre-filled: {len(replay)} transitions")

    q_net, target = make_nets()
    device = torch.device(DEVICE)
    opt    = torch.optim.Adam(q_net.parameters(), lr=LR, weight_decay=1e-4, eps=1e-5)
    loss_fn = nn.MSELoss()

    # ── Phase 1: Offline pre-train / 離線預訓練 ──────────────────────────────
    print(f"[DDQN] Phase 1: Offline pre-train ({OFFLINE_STEPS} steps)…")
    for step in range(OFFLINE_STEPS):
        obs_b, act_b, rew_b, next_b, done_b = replay.sample(OFFLINE_BATCH)
        q_loss, *_ = train_dqn(
            act_b, next_b, rew_b, obs_b, done_b, np.zeros_like(done_b),
            q_net, target, opt, loss_fn, device, GAMMA, TAU,
        )
        if step % 500 == 0:
            print(f"  offline step={step:5d} | Q_loss={q_loss:.4f}")

    # ── Phase 2: Online training / 線上訓練 ──────────────────────────────────
    print(f"\n[DDQN] Phase 2: Online training ({N_ONLINE_EPOCHS} epochs)…")
    env = SumoIntersectionEnv(FULL_CFG, gui=False)
    eps = EPS_START
    act_rng = np.random.default_rng(SEED + 1)

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "mean_reward", "avg_wait",
            "r_morning", "r_off", "r_evening",
            "q_loss", "epsilon",
        ])

    obs, _ = env.reset()

    for epoch in range(N_ONLINE_EPOCHS):
        step_rewards = {c: [] for c in ("morning", "off", "evening")}
        ep_transitions = []
        step_waits = []
        step_idx = 0
        steps_this_epoch = 0

        # collect ONLINE_BATCH steps
        while steps_this_epoch < ONLINE_BATCH:
            # ε-greedy action selection
            if act_rng.random() < eps:
                action = int(act_rng.integers(0, ACT_BINS))
            else:
                obs_t  = torch.as_tensor(obs, dtype=torch.float).unsqueeze(0)
                _, act_idx = q_net.max(obs_t)       # greedy: argmax Q
                action = int(act_idx[0, 0].item())

            next_obs, rew, done, trunc, info = env.step(action)
            replay.push(obs, action, rew, next_obs, done or trunc)
            ep_transitions.append(rew)
            step_rewards[get_context(step_idx)].append(rew)
            try:
                _w = sum(_traci.edge.getWaitingTime(e) for e in ["N2C","S2C","E2C","W2C","C2N","C2S","C2E","C2W"]) / 8.0
            except Exception:
                _w = 0.0
            step_waits.append(_w)
            step_idx += 1
            steps_this_epoch += 1

            obs = next_obs
            if done or trunc:
                obs, _ = env.reset()
                step_idx = 0

        # ── Update Q-network / 更新 Q-network ────────────────────────────
        if len(replay) >= OFFLINE_BATCH:
            obs_b, act_b, rew_b, next_b, done_b = replay.sample(OFFLINE_BATCH)
            q_loss, *_ = train_dqn(
                act_b, next_b, rew_b, obs_b, done_b, np.zeros_like(done_b),
                q_net, target, opt, loss_fn, device, GAMMA, TAU,
            )
        else:
            q_loss = 0.0

        eps = max(eps * EPS_DECAY, EPS_END)

        r_m = np.mean(step_rewards["morning"])  if step_rewards["morning"]  else 0
        r_o = np.mean(step_rewards["off"])      if step_rewards["off"]      else 0
        r_e = np.mean(step_rewards["evening"])  if step_rewards["evening"]  else 0
        mean_r   = float(np.mean(ep_transitions))
        avg_wait = float(np.mean(step_waits)) if step_waits else 0.0

        print(f"[DDQN] epoch={epoch:4d} | "
              f"r̄={mean_r:7.2f} | "
              f"morning={r_m:6.1f} off={r_o:6.1f} evening={r_e:6.1f} | "
              f"wait={avg_wait:5.1f}s | ε={eps:.3f} q={q_loss:.4f}")

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_r, avg_wait, r_m, r_o, r_e, q_loss, eps])

        if epoch % SAVE_EVERY == 0:
            torch.save({"q_net": q_net.state_dict(),
                        "target": target.state_dict()},
                       f"{RESULT_DIR}/models/model_{epoch}.pt")

    torch.save({"q_net": q_net.state_dict(), "target": target.state_dict()},
               f"{RESULT_DIR}/models/model_final.pt")
    env.close()
    print(f"\n[DDQN] Done. Log → {LOG_PATH}")


if __name__ == "__main__":
    main()

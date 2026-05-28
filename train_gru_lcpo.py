"""
Person 4 ── GRU-LCPO (proposed method)
=======================================
Phase 1 – Behaviour Cloning warm-up:
  Uses A2C trajectory sequences to pre-train GRU encoder + policy network
  via supervised (cross-entropy) imitation of A2C actions.
  把 A2C 的 trajectory 重建成 sequence，用 BC loss 訓練 GRU encoder + policy。

Phase 2 – Online GRU-LCPO:
  Runs the full 3-context scenario with LCPO constrained update.
  OOD detection is performed in the GRU latent context (h_t) space,
  removing the need for manually labeled context.
  在完整三段場景做 LCPO 線上訓練，OOD 偵測在 h_t 潛在空間，不需要 context 標籤。

Prerequisites: collect_warmup.py

Usage:
  python train_gru_lcpo.py

Output:
  results/gru_lcpo/models/model_*.pt
  results/gru_lcpo/log.csv
"""

import os, sys, csv, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SUMO_DIR)

from neural_net.nn import FCNPolicy, FullyConnectNN
from neural_net.gru_encoder import GRUEncoder, ObsHistoryEncoder
from agent.core_alg.core_lcpo import train_lcpo
from buffer.buffer import TransitionBuffer
from buffer.buffer_ood import OutOfDSampler
from utils.rms import RunningMeanStd
from sumo_env import SumoIntersectionEnv

# ── Config ───────────────────────────────────────────────────────────────────
FULL_CFG    = os.path.join(HERE, "nets", "intersection.sumocfg")
TRAJ_PATH   = os.path.join(HERE, "results", "warmup", "trajectory.pkl")
RESULT_DIR  = os.path.join(HERE, "results", "gru_lcpo")
LOG_PATH    = os.path.join(RESULT_DIR, "log.csv")

OBS_DIM     = 24
ACT_BINS    = 2
NN_HIDS     = [128, 128]

# GRU encoder / GRU 編碼器
GRU_HIDDEN  = 64    # GRU hidden state size
CONTEXT_DIM = 32    # output h_t dimension
K           = 10    # history window length / 歷史窗口長度

AUG_DIM     = OBS_DIM + CONTEXT_DIM   # 24 + 32 = 56

# BC warm-up / 行為克隆 warm-up
BC_EPOCHS      = 50     # number of BC training epochs
BC_BATCH       = 256    # batch size for BC
BC_LR          = 3e-4

# Online / 線上訓練
N_EPOCHS    = 500
BATCH_SIZE  = 1080
LR_P        = 3e-4
LR_V        = 1e-3
GAMMA       = 0.99
LAM         = 0.95
ENTROPY     = 0.05
KL_IN       = 0.01
KL_OUT      = 0.05
DAMPING     = 0.1
DUAL        = True
OOD_WIN     = 100
OOD_CAP     = 8000
SAVE_EVERY  = 50
DEVICE      = "cpu"
SEED        = 42


def get_context(step: int) -> str:
    if step < 360:  return "morning"
    if step < 720:  return "off"
    return "evening"


class _NoMonitor:
    def add_scalar(self, *a, **k): pass


# ── OOD detector in h_t latent space / 在 h_t 潛在空間做 OOD 偵測 ──────────
def make_ctx_ood_detector(obs_dim: int, context_dim: int, sigma: float = 2.0):
    def _is_different(new_aug: np.ndarray, recent_aug: np.ndarray) -> np.ndarray:
        new_h    = new_aug[:, obs_dim:]
        recent_h = recent_aug[:, obs_dim:]
        mu  = recent_h.mean(axis=0)
        var = recent_h.var(axis=0).mean() + 1e-6
        dists = np.linalg.norm(new_h - mu, axis=-1)
        return dists > sigma * np.sqrt(var * context_dim)
    return _is_different


# ── Build obs sequence window from trajectory ────────────────────────────────
def build_sequences(obs_array: np.ndarray, k: int) -> np.ndarray:
    """
    For each timestep t, build a [k, obs_dim] window ending at t.
    Left-pad with zeros when t < k.
    為每個時間步 t 建立長度 k 的觀測序列（不足時左補 0）。
    """
    T, D = obs_array.shape
    seqs = np.zeros((T, k, D), dtype=np.float32)
    for t in range(T):
        start = max(0, t - k + 1)
        seqs[t, k - (t - start + 1):] = obs_array[start: t + 1]
    return seqs  # [T, k, D]


def main():
    os.makedirs(RESULT_DIR + "/models", exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)

    assert os.path.exists(TRAJ_PATH), \
        f"找不到 {TRAJ_PATH}，請先執行 collect_warmup.py"

    # ── Load trajectory / 載入 trajectory ────────────────────────────────────
    with open(TRAJ_PATH, "rb") as f:
        traj = pickle.load(f)
    T = len(traj["obs"])
    print(f"[GRU-LCPO] Loaded {T} transitions from trajectory")

    # Build sequences for GRU / 建立 GRU 用的序列資料
    seqs    = build_sequences(traj["obs"], K)     # [T, k, 24]
    actions = traj["actions"]                      # [T]  int64

    # ── Networks / 網路 ──────────────────────────────────────────────────────
    gru_enc   = GRUEncoder(OBS_DIM, GRU_HIDDEN, CONTEXT_DIM).to(device)
    policy_net = torch.jit.script(
        FCNPolicy(AUG_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    ).to(device)
    value_net  = torch.jit.script(
        FullyConnectNN(AUG_DIM, NN_HIDS, 1, 1, act=nn.ReLU, final_layer_act=False)
    ).to(device)

    opt_gru = torch.optim.Adam(gru_enc.parameters(),   lr=BC_LR, weight_decay=1e-4)
    opt_p   = torch.optim.Adam(policy_net.parameters(), lr=LR_P,  weight_decay=1e-4, eps=1e-5)
    opt_v   = torch.optim.Adam(value_net.parameters(),  lr=LR_V,  weight_decay=1e-4, eps=1e-5)
    loss_fn = nn.MSELoss()

    rng = np.random.default_rng(SEED)

    # ────────────────────────────────────────────────────────────────────────
    # Phase 1: Behaviour Cloning warm-up
    # GRU encoder + policy 模仿 A2C 的動作（監督式學習）
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[GRU-LCPO] Phase 1: BC warm-up ({BC_EPOCHS} epochs)…")
    ce_loss = nn.CrossEntropyLoss()

    for bc_epoch in range(BC_EPOCHS):
        idx = rng.permutation(T)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, T, BC_BATCH):
            batch_idx = idx[start: start + BC_BATCH]
            if len(batch_idx) < 4:
                continue

            seq_t  = torch.as_tensor(seqs[batch_idx], dtype=torch.float, device=device)  # [B, k, 24]
            obs_t  = torch.as_tensor(traj["obs"][batch_idx], dtype=torch.float, device=device)  # [B, 24]
            act_t  = torch.as_tensor(actions[batch_idx], dtype=torch.long, device=device)       # [B]

            # Encode context / 編碼 context
            h_t    = gru_enc(seq_t)                                  # [B, 32]
            aug_t  = torch.cat([obs_t, h_t], dim=-1)                 # [B, 56]

            # Policy logits shape: [B, grp_size=1, bins=2]
            logits = policy_net.forward(aug_t).squeeze(1)            # [B, 2]
            loss   = ce_loss(logits, act_t)

            opt_gru.zero_grad()
            opt_p.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(gru_enc.parameters()) + list(policy_net.parameters()), 0.5
            )
            opt_gru.step()
            opt_p.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  BC epoch={bc_epoch:3d} | CE loss={avg_loss:.4f}")

    print("[GRU-LCPO] BC warm-up done.\n")

    # ────────────────────────────────────────────────────────────────────────
    # Phase 2: Online GRU-LCPO
    # 線上 GRU-LCPO 訓練（三段 context，OOD 偵測在 h_t 空間）
    # ────────────────────────────────────────────────────────────────────────
    env = SumoIntersectionEnv(FULL_CFG, gui=False)

    obs_history = ObsHistoryEncoder(gru_enc, K, OBS_DIM, device)
    batch_rng   = np.random.RandomState(SEED)
    ood_buf     = OutOfDSampler(
        AUG_DIM, OOD_WIN, OOD_CAP,
        make_ctx_ood_detector(OBS_DIM, CONTEXT_DIM),
    )
    ret_rms = RunningMeanStd(shape=())
    buff    = TransitionBuffer(AUG_DIM, 1, BATCH_SIZE)
    monitor = _NoMonitor()
    entropy_factor = ENTROPY

    def augment(raw_obs: np.ndarray) -> np.ndarray:
        """Append GRU context to raw obs → aug_obs [56]"""
        h_t = obs_history.encode(raw_obs)
        return np.concatenate([raw_obs, h_t])

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "mean_reward",
            "r_morning", "r_off", "r_evening",
            "pg_loss", "v_loss", "ood_size",
        ])

    obs, _ = env.reset()
    obs_history.reset()

    for epoch in range(N_EPOCHS):
        buff.reset_head()
        step_rewards = {c: [] for c in ("morning", "off", "evening")}
        step_idx = 0

        while not buff.buffer_full():
            aug_obs = augment(obs)
            aug_t   = torch.as_tensor(aug_obs, dtype=torch.float, device=device).unsqueeze(0)
            act     = policy_net.sample_action(aug_t).numpy()   # [1]

            next_obs, rew, done, trunc, info = env.step(int(act[0]))
            aug_next = augment(next_obs)

            buff.add_exp(aug_obs, act, rew, aug_next, done, trunc)
            step_rewards[get_context(step_idx)].append(rew)
            step_idx += 1

            obs = next_obs
            if done or trunc:
                obs, _ = env.reset()
                obs_history.reset()
                step_idx = 0

        aug_b, aug_next_b, act_b, rew_b, term_b, trunc_b = buff.get()
        ret_rms.update(rew_b)
        norm_rew = rew_b / np.sqrt(ret_rms.var + 1)

        # OOD sampling in h_t space / 在 h_t 潛在空間取 OOD 樣本
        ood_buf.add_many_exp(aug_b, batch_rng)
        ood_raw = ood_buf.get(batch_rng, BATCH_SIZE)
        ood_np  = np.array(ood_raw) if ood_raw else np.zeros((0, AUG_DIM), dtype=np.float32)

        # LCPO update on aug_obs / 用 aug_obs 做 LCPO 更新
        pg_loss, v_loss, ent, *_ = train_lcpo(
            value_net, policy_net, opt_p, opt_v, loss_fn,
            device,
            act_b, aug_next_b, norm_rew, aug_b, term_b, trunc_b,
            GAMMA, LAM, KL_IN, KL_OUT, DAMPING, DUAL,
            entropy_factor, ood_obs_np=ood_np,
            monitor=monitor, it=epoch,
        )
        entropy_factor = max(entropy_factor * 0.999, 0.01)

        r_m = np.mean(step_rewards["morning"])  if step_rewards["morning"]  else 0
        r_o = np.mean(step_rewards["off"])      if step_rewards["off"]      else 0
        r_e = np.mean(step_rewards["evening"])  if step_rewards["evening"]  else 0
        mean_r = float(np.mean(rew_b))

        print(f"[GRU-LCPO] epoch={epoch:4d} | "
              f"r̄={mean_r:7.2f} | "
              f"morning={r_m:6.1f} off={r_o:6.1f} evening={r_e:6.1f} | "
              f"ood={len(ood_raw):4d} | pg={pg_loss:.4f}")

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, mean_r, r_m, r_o, r_e, pg_loss, v_loss, len(ood_raw)]
            )

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "policy_net":  policy_net.state_dict(),
                "value_net":   value_net.state_dict(),
                "gru_encoder": gru_enc.state_dict(),
            }, f"{RESULT_DIR}/models/model_{epoch}.pt")

    torch.save({
        "policy_net":  policy_net.state_dict(),
        "value_net":   value_net.state_dict(),
        "gru_encoder": gru_enc.state_dict(),
    }, f"{RESULT_DIR}/models/model_final.pt")
    env.close()
    print(f"\n[GRU-LCPO] Done. Log → {LOG_PATH}")


if __name__ == "__main__":
    main()

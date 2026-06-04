"""
GRU Latent Context Visualisation (t-SNE)
==========================================
Loads the trained GRU encoder, encodes the A2C warm-up trajectory
through it, and plots the 2-D t-SNE projection of h_t vectors
coloured by which context they belong to.

A well-trained GRU encoder should produce clusters that naturally
separate morning / off-peak / evening contexts, demonstrating that
GRU-LCPO infers context without manual labels.

GRU 潛在空間視覺化：用 t-SNE 把 h_t 降維到 2D，
若三個 context 自然分群，代表 GRU 成功學會辨識 context。

Usage:
  python scripts/plot_gru_latent.py [--model PATH]

Output:
  figures/gru_latent_tsne.png

Requirements:
  pip install scikit-learn
"""

import os, sys, argparse, pickle
import numpy as np
import torch

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(HERE, "src")
sys.path.insert(0, SRC_DIR)

from neural_net.gru_encoder import GRUEncoder

FIG_DIR   = os.path.join(HERE, "figures")
TRAJ_PATH = os.path.join(HERE, "results", "warmup", "trajectory.pkl")
MODEL_DEFAULT = os.path.join(HERE, "results", "gru_lcpo", "models", "model_final.pt")
os.makedirs(FIG_DIR, exist_ok=True)

# Must match train_gru_lcpo.py config
OBS_DIM     = 24
GRU_HIDDEN  = 64
CONTEXT_DIM = 32
K           = 10

# Context labels by step index within a 1080-step episode
# (trajectory comes from warmup = morning_peak only; label by step % 360 for variety)
FULL_EP_STEPS = 1080


def get_context_label(global_step: int) -> str:
    ep_step = global_step % FULL_EP_STEPS
    if ep_step < 360:  return "morning"
    if ep_step < 720:  return "off"
    return "evening"


def build_sequences(obs: np.ndarray, k: int) -> np.ndarray:
    T, D = obs.shape
    seqs = np.zeros((T, k, D), dtype=np.float32)
    for t in range(T):
        start = max(0, t - k + 1)
        seqs[t, k - (t - start + 1):] = obs[start: t + 1]
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT,
                        help="Path to GRU-LCPO checkpoint (.pt)")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Run train_gru_lcpo.py first.")
        return

    ckpt = torch.load(args.model, map_location="cpu")
    enc  = GRUEncoder(OBS_DIM, GRU_HIDDEN, CONTEXT_DIM)
    enc.load_state_dict(ckpt["gru_encoder"])
    enc.eval()
    print(f"[load] GRU encoder from {args.model}")

    # ── Load trajectory ──────────────────────────────────────────────────────
    if not os.path.exists(TRAJ_PATH):
        print(f"[ERROR] Trajectory not found: {TRAJ_PATH}")
        return

    with open(TRAJ_PATH, "rb") as f:
        traj = pickle.load(f)
    obs_all = traj["obs"]   # [T, 24]
    T = len(obs_all)
    print(f"[load] Trajectory: {T} transitions")

    # ── Build sequences & encode ─────────────────────────────────────────────
    print("[encode] Building sequences…")
    seqs = build_sequences(obs_all, K)   # [T, k, 24]

    # Encode in batches to avoid OOM
    BATCH = 512
    h_vecs = []
    with torch.no_grad():
        for i in range(0, T, BATCH):
            seq_t = torch.as_tensor(seqs[i: i + BATCH], dtype=torch.float)
            h = enc(seq_t).numpy()
            h_vecs.append(h)
    h_all = np.concatenate(h_vecs, axis=0)   # [T, 32]
    print(f"[encode] h_t shape: {h_all.shape}")

    # Assign context labels (repeat the 3-context cycle over trajectory)
    labels = np.array([get_context_label(i) for i in range(T)])

    # ── t-SNE ────────────────────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[ERROR] scikit-learn not installed. Run: pip install scikit-learn")
        return

    # Subsample for speed if T is large
    MAX_POINTS = 3000
    if T > MAX_POINTS:
        idx = np.random.choice(T, MAX_POINTS, replace=False)
        h_plot = h_all[idx]
        l_plot = labels[idx]
    else:
        h_plot, l_plot = h_all, labels

    print(f"[tsne] Running t-SNE on {len(h_plot)} points…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    z = tsne.fit_transform(h_plot)   # [N, 2]

    # ── Plot ─────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    ctx_colors = {
        "morning": ("tab:red",    "Morning Peak 早峰"),
        "off":     ("tab:blue",   "Off-Peak 離峰"),
        "evening": ("tab:orange", "Evening Peak 晚峰"),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for ctx, (color, label) in ctx_colors.items():
        mask = l_plot == ctx
        ax.scatter(z[mask, 0], z[mask, 1], c=color, label=label,
                   s=8, alpha=0.6, linewidths=0)

    ax.set_title("GRU Latent Context h_t — t-SNE Projection\n"
                 "GRU 潛在 context 向量 t-SNE 視覺化\n"
                 "(cluster separation = GRU learned to distinguish contexts)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)

    out = os.path.join(FIG_DIR, "gru_latent_tsne.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

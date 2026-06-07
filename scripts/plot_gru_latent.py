"""
GRU Latent Context Visualisation (t-SNE)  — corrected version
==============================================================
Runs the trained GRU-LCPO policy in the FULL 3-context intersection scenario
and records h_t at every step with its REAL context label (based on actual
step position within the episode, not a modulo cycle over warmup data).

Previous version: loaded warmup-only trajectory (morning_peak only) and
  labelled steps with step%1080, so all data was the same context → no clusters.

This version: actually executes SumoIntersectionEnv for N episodes,
  labels step 0–359 as "morning", 360–719 as "off", 720–1079 as "evening".

Usage:
  python scripts/plot_gru_latent.py [--model PATH] [--episodes N]

Output:
  figures/gru_latent_tsne.png
"""

import os, sys, argparse, shutil
import numpy as np
import torch
import torch.nn as nn

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SUMO_DIR)

# ── Make sure the sumo binary is on PATH ─────────────────────────────────────
_SUMO_CANDIDATES = [
    os.path.join(os.path.dirname(HERE), "LCPO", "venv", "bin"),
    os.path.join(HERE, "venv", "bin"),
]
if not shutil.which("sumo"):
    for _d in _SUMO_CANDIDATES:
        if os.path.isfile(os.path.join(_d, "sumo")):
            os.environ["PATH"] = _d + ":" + os.environ.get("PATH", "")
            print(f"[sumo] Added to PATH: {_d}")
            break

from neural_net.nn import FCNPolicy, FullyConnectNN
from neural_net.gru_encoder import GRUEncoder, ObsHistoryEncoder
from sumo_env import SumoIntersectionEnv

FIG_DIR       = os.path.join(HERE, "figures")
FULL_CFG      = os.path.join(HERE, "nets", "intersection.sumocfg")
MODEL_DEFAULT = os.path.join(HERE, "results", "gru_lcpo", "models", "model_final.pt")
os.makedirs(FIG_DIR, exist_ok=True)

OBS_DIM     = 28
GRU_HIDDEN  = 64
CONTEXT_DIM = 32
AUG_DIM     = OBS_DIM + CONTEXT_DIM  # 28 + 32 = 60
NN_HIDS     = [128, 128]
ACT_BINS    = 2
K           = 10


def get_context_label(step: int) -> str:
    """Real context label based on step index within the 1080-step episode."""
    if step < 360:  return "morning"
    if step < 720:  return "off"
    return "evening"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT,
                        help="Path to GRU-LCPO checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of full episodes to collect h_t from (default: 3)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Run scripts/train_gru_lcpo.py first.")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)

    enc = GRUEncoder(OBS_DIM, GRU_HIDDEN, CONTEXT_DIM)
    enc.load_state_dict(ckpt["gru_encoder"])
    enc.eval()

    policy_net = torch.jit.script(
        FCNPolicy(AUG_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    )
    policy_net.load_state_dict(ckpt["policy_net"])
    policy_net.eval()
    print(f"[load] GRU encoder + policy from {args.model}")

    obs_history = ObsHistoryEncoder(enc, K, OBS_DIM, torch.device("cpu"))

    # ── Run full 3-context episodes and collect h_t (with disk cache) ───────────
    CACHE_H   = os.path.join(FIG_DIR, "_h_vecs_cache.npy")
    CACHE_L   = os.path.join(FIG_DIR, "_labels_cache.npy")

    if os.path.exists(CACHE_H) and os.path.exists(CACHE_L) and not getattr(args, "nocache", False):
        h_all  = np.load(CACHE_H)
        labels = np.load(CACHE_L, allow_pickle=True)
        print(f"[cache] Loaded {len(h_all)} h_t vectors from disk")
    else:
        env = SumoIntersectionEnv(FULL_CFG, gui=False)
        h_vecs: list[np.ndarray] = []
        ctx_labels: list[str] = []

        for ep in range(args.episodes):
            obs, _ = env.reset()
            obs_history.reset()
            step = 0
            done = False

            while not done:
                h_t = obs_history.encode(obs)
                h_vecs.append(h_t.copy())
                ctx_labels.append(get_context_label(step))

                aug = np.concatenate([obs, h_t])
                aug_t = torch.as_tensor(aug, dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    action = int(policy_net.sample_action(aug_t).numpy()[0])

                obs, _, done, trunc, _ = env.step(action)
                step += 1
                done = done or trunc

            print(f"[episode {ep + 1}/{args.episodes}] "
                  f"steps={step}  total collected={len(h_vecs)}")

        env.close()
        h_all  = np.array(h_vecs, dtype=np.float32)
        labels = np.array(ctx_labels)
        np.save(CACHE_H, h_all)
        np.save(CACHE_L, labels)
        print(f"[cache] Saved {len(h_all)} h_t vectors to disk")
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"[collected] {len(h_all)} h_t vectors | "
          + " | ".join(f"{u}={c}" for u, c in zip(uniq, cnt)))

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[ERROR] scikit-learn not installed. Run: pip install scikit-learn")
        return

    # Balanced sampling: equal points per context so minority classes are visible
    rng = np.random.default_rng(42)
    ctx_names = ["morning", "off", "evening"]
    per_ctx   = min(700, min((labels == c).sum() for c in ctx_names))
    idx_bal   = np.concatenate([
        rng.choice(np.where(labels == c)[0], per_ctx, replace=False)
        for c in ctx_names
    ])
    rng.shuffle(idx_bal)
    h_plot = h_all[idx_bal]
    l_plot = labels[idx_bal]
    print(f"[balance] {per_ctx} points per context = {len(h_plot)} total")

    print(f"[tsne] Running t-SNE on {len(h_plot)} points…")
    tsne = TSNE(n_components=2, perplexity=20, random_state=42,
                max_iter=2000, init="pca", learning_rate="auto")
    z = tsne.fit_transform(h_plot)   # [N, 2]

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA

    plt.rcParams["axes.unicode_minus"] = False

    ctx_styles = {
        "morning": ("#E63946", "o", "Morning Peak (WB=489 vph)"),
        "off":     ("#457B9D", "s", "Off-Peak    (WB=418 vph)"),
        "evening": ("#F4A261", "^", "Evening Peak (WB=657 vph)"),
    }

    # PCA for the left panel
    pca = PCA(n_components=2, random_state=42)
    z_pca = pca.fit_transform(h_plot)

    def draw_confidence_ellipse(ax, x, y, color, n_std=1.8):
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * n_std * np.sqrt(vals)
        ell = Ellipse(xy=(x.mean(), y.mean()), width=w, height=h, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.12, linewidth=1.5)
        ax.add_patch(ell)
        Ellipse_border = Ellipse(xy=(x.mean(), y.mean()), width=w, height=h, angle=angle,
                                  edgecolor=color, facecolor="none", linewidth=1.5, linestyle="--")
        ax.add_patch(Ellipse_border)

    n_ep = args.episodes
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"GRU Latent Context $h_t$ — PCA & t-SNE  "
        f"({n_ep} episodes, {len(h_plot)} points)",
        fontsize=12
    )

    titles  = ["PCA Projection (PC1 vs PC2)", "t-SNE Projection (perplexity=20)"]
    z_list  = [z_pca, z]
    for ax, proj, title in zip(axes, z_list, titles):
        for ctx, (color, marker, label) in ctx_styles.items():
            mask = l_plot == ctx
            ax.scatter(proj[mask, 0], proj[mask, 1], c=color, marker=marker,
                       label=label, s=10, alpha=0.55, linewidths=0)
            if mask.sum() > 10:
                draw_confidence_ellipse(ax, proj[mask, 0], proj[mask, 1], color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(markerscale=2.5, fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "gru_latent_tsne.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

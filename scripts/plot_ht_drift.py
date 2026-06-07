"""
Within-episode h_t OOD drift (analog of LCPO paper Fig 2).

Runs ONE full 1080-step episode with the trained GRU-LCPO policy.
Records at every step:
  - h_t  (32-dim latent context)
  - OOD distance = ||h_t - mu_recent|| / sqrt(var_recent * 32)
    (normalized so threshold = 2.0)

Outputs: figures/poster_fig_ht_drift.png
"""

import os, sys, shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["axes.unicode_minus"] = False

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SUMO_DIR)

_SUMO_CANDIDATES = [
    os.path.join(os.path.dirname(HERE), "LCPO", "venv", "bin"),
    os.path.join(HERE, "venv", "bin"),
]
if not shutil.which("sumo"):
    for _d in _SUMO_CANDIDATES:
        if os.path.isfile(os.path.join(_d, "sumo")):
            os.environ["PATH"] = _d + ":" + os.environ.get("PATH", "")
            break

from neural_net.nn import FCNPolicy, FullyConnectNN
from neural_net.gru_encoder import GRUEncoder, ObsHistoryEncoder
from sumo_env import SumoIntersectionEnv

FIG_DIR   = os.path.join(HERE, "figures")
FULL_CFG  = os.path.join(HERE, "nets", "intersection.sumocfg")
MODEL     = os.path.join(HERE, "results", "gru_lcpo", "models", "model_final.pt")
os.makedirs(FIG_DIR, exist_ok=True)

OBS_DIM     = 28
GRU_HIDDEN  = 64
CONTEXT_DIM = 32
AUG_DIM     = OBS_DIM + CONTEXT_DIM
NN_HIDS     = [128, 128]
ACT_BINS    = 2
K           = 10
OOD_WIN     = 100   # same as training


def ood_distance(h_t: np.ndarray, recent: list[np.ndarray]) -> float:
    """Normalized OOD distance; threshold = 2.0 in training."""
    if len(recent) < 5:
        return 0.0
    mat = np.array(recent)           # [N, 32]
    mu  = mat.mean(axis=0)
    var = mat.var(axis=0).mean() + 1e-6
    return float(np.linalg.norm(h_t - mu) / np.sqrt(var * CONTEXT_DIM))


def main():
    assert os.path.exists(MODEL), f"Model not found: {MODEL}"
    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)

    enc = GRUEncoder(OBS_DIM, GRU_HIDDEN, CONTEXT_DIM)
    enc.load_state_dict(ckpt["gru_encoder"])
    enc.eval()

    policy_net = torch.jit.script(
        FCNPolicy(AUG_DIM, NN_HIDS, ACT_BINS, 1, act=nn.ReLU, final_layer_act=False)
    )
    policy_net.load_state_dict(ckpt["policy_net"])
    policy_net.eval()

    obs_hist = ObsHistoryEncoder(enc, K, OBS_DIM, torch.device("cpu"))

    env = SumoIntersectionEnv(FULL_CFG, gui=False)
    obs, _ = env.reset()
    obs_hist.reset()

    steps, ood_dists, h_norms = [], [], []
    recent_h: list[np.ndarray] = []

    step = 0
    while True:
        h_t = obs_hist.encode(obs)
        d   = ood_distance(h_t, recent_h)
        ood_dists.append(d)
        h_norms.append(float(np.linalg.norm(h_t)))
        steps.append(step)

        recent_h.append(h_t.copy())
        if len(recent_h) > OOD_WIN:
            recent_h.pop(0)

        aug   = np.concatenate([obs, h_t])
        aug_t = torch.as_tensor(aug, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            action = int(policy_net.sample_action(aug_t).numpy()[0])

        obs, _, done, trunc, _ = env.step(action)
        step += 1
        if done or trunc:
            break

    env.close()
    print(f"[run] Episode length: {step} steps")

    steps      = np.array(steps)
    ood_dists  = np.array(ood_dists)
    h_norms    = np.array(h_norms)

    # ── Plot ──────────────────────────────────────────────────────────────────
    ctx_colors = {
        "Morning\nPeak\n(WB=489)":  (0,   360, "#FFDDC1"),
        "Off-Peak\n(WB=418)":       (360, 720, "#C1E1FF"),
        "Evening\nPeak\n(WB=657)":  (720, min(step, 1080), "#FFD6D6"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"hspace": 0.08})

    for ax in axes:
        for label, (s, e, col) in ctx_colors.items():
            ax.axvspan(s, e, color=col, alpha=0.35, zorder=0)
        ax.axvline(360, color="gray", linewidth=0.8, linestyle="--", zorder=1)
        ax.axvline(720, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    # Top: OOD distance (normalized)
    ax0 = axes[0]
    ax0.plot(steps, ood_dists, color="#AA3377", linewidth=0.8, alpha=0.85)
    ax0.axhline(2.0, color="red", linewidth=1.2, linestyle="--", label="OOD threshold (σ=2)")
    # Shade triggered region
    triggered = ood_dists >= 2.0
    ax0.fill_between(steps, 0, ood_dists, where=triggered,
                     color="red", alpha=0.25, label="OOD triggered")
    ax0.set_ylabel("Normalized OOD Dist.\n" r"$\|h_t - \mu_\mathrm{recent}\| / \sqrt{\hat\sigma^2 \cdot d}$",
                   fontsize=8)
    ax0.legend(fontsize=8, loc="upper left")
    ax0.set_ylim(bottom=0)
    ax0.set_title("Within-Episode $h_t$ Drift — GRU-LCPO  (one full episode)",
                  fontsize=11)

    # Bottom: h_t L2 norm
    ax1 = axes[1]
    ax1.plot(steps, h_norms, color="#4477AA", linewidth=0.8, alpha=0.85)
    ax1.set_ylabel(r"$\|h_t\|_2$  (context vector norm)", fontsize=8)
    ax1.set_xlabel("Step within Episode", fontsize=9)

    # Context boundary labels (top axis)
    for (label, (s, e, _)) in ctx_colors.items():
        mid = (s + min(e, step)) / 2
        axes[0].text(mid, ax0.get_ylim()[1] * 0.92,
                     label.replace("\n", " "),
                     ha="center", va="top", fontsize=7.5,
                     color="black", style="italic")

    # Annotate boundaries
    for x in [360, 720]:
        if x < step:
            axes[1].annotate("context\nswitch",
                             xy=(x, h_norms[min(x, len(h_norms)-1)]),
                             xytext=(x + 30, h_norms[min(x, len(h_norms)-1)] * 1.05),
                             fontsize=7, color="gray",
                             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    out = os.path.join(FIG_DIR, "poster_fig_ht_drift.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

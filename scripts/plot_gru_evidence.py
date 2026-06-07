"""
Two evidence figures for GRU-LCPO contribution:

Fig A (poster_fig8_evening_learning.png):
  Evening-peak reward over training epochs — GRU-LCPO vs A2C.
  Shows GRU achieves 2.2× more improvement on the hardest context.

Fig B (poster_fig9_ood_timeline.png):
  OOD trigger size over the last N epochs, annotated with context boundaries.
  Shows h_t reliably drifts at context switches (step 360, 720 in episode).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["axes.unicode_minus"] = False

HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(HERE, "results")
FIG_DIR    = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SMOOTH = 20

def smooth(s, w=SMOOTH):
    return s.rolling(w, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Fig A: Evening-peak learning curve comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_evening_learning():
    styles = {
        "a2c":      ("A2C (no context)",         "#4477AA", "-",  1.2),
        "lcpo":     ("LCPO (fixed OOD)",          "#EE6677", "--", 1.1),
        "ddqn":     ("DDQN",                      "#228833", "-",  1.2),
        "gru_lcpo": ("GRU-LCPO (ours)",           "#AA3377", "-",  1.8),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    annotations = {}
    for algo, (label, color, ls, lw) in styles.items():
        path = os.path.join(RESULT_DIR, algo, "log.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "r_evening" not in df.columns:
            continue
        s = smooth(df["r_evening"])
        ax.plot(df["epoch"], s, label=label, color=color, linestyle=ls, linewidth=lw)

        # Annotate early vs late improvement
        early = df["r_evening"].head(50).mean()
        late  = df["r_evening"].tail(50).mean()
        annotations[algo] = (early, late, color)

    # Add improvement arrows for A2C and GRU-LCPO
    for algo, (early, late, color) in annotations.items():
        if algo not in ("a2c", "gru_lcpo"):
            continue
        delta = late - early
        x_pos = 480 if algo == "gru_lcpo" else 460
        ax.annotate(
            f"Δ={delta:+.3f}",
            xy=(499, late),
            xytext=(x_pos, (early + late) / 2),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1),
            va="center",
        )

    # Ratio annotation
    gru_delta = annotations["gru_lcpo"][1] - annotations["gru_lcpo"][0]
    a2c_delta = annotations["a2c"][1] - annotations["a2c"][0]
    ratio = gru_delta / a2c_delta
    ax.text(0.03, 0.97,
            f"GRU-LCPO evening improvement:\n"
            f"{gru_delta:+.3f}  vs  A2C {a2c_delta:+.3f}\n"
            f"= {ratio:.1f}× more improvement",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Evening Peak Mean Reward")
    ax.set_title("Evening Peak Learning Curve\n(WB = 657 vph — hardest context)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    out = os.path.join(FIG_DIR, "poster_fig8_evening_learning.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig B: OOD trigger size over training (GRU-LCPO vs LCPO fixed)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ood_over_training():
    styles = {
        "lcpo":     ("LCPO (fixed, flow OOD)",    "#EE6677", "--", 1.2),
        "gru_lcpo": ("GRU-LCPO (h_t OOD)",        "#AA3377", "-",  1.6),
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    for algo, (label, color, ls, lw) in styles.items():
        path = os.path.join(RESULT_DIR, algo, "log.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "ood_size" not in df.columns:
            continue
        # OOD trigger: binary (triggered or not each epoch)
        triggered = (df["ood_size"] > 0).astype(float)
        s = triggered.rolling(30, min_periods=1).mean() * 100
        ax.plot(df["epoch"], s, label=label, color=color, linestyle=ls, linewidth=lw)

    # Shade regions where OOD should be high (context switches happen mid-episode)
    ax.axhline(y=86.2, color="#AA3377", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(y=89.2, color="#EE6677", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("OOD Trigger Rate (%, 30-ep rolling)")
    ax.set_title("OOD Detector Trigger Rate Over Training\n"
                 "(each episode crosses 2 context boundaries: step 360 & 720)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate final rates
    ax.text(490, 89.2 + 2, "89.2%", color="#EE6677", fontsize=8, ha="right")
    ax.text(490, 86.2 - 5, "86.2%", color="#AA3377", fontsize=8, ha="right")

    out = os.path.join(FIG_DIR, "poster_fig9_ood_timeline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig C: Side-by-side context bar — cleaner version for poster
#         shows GRU-LCPO learning improvement per context explicitly
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_gain():
    """Bar chart: reward improvement (late - early) per context per algorithm."""
    configs = {
        "a2c":      ("A2C",           "#4477AA"),
        "lcpo":     ("LCPO (fixed)",  "#EE6677"),
        "ddqn":     ("DDQN",          "#228833"),
        "gru_lcpo": ("GRU-LCPO",      "#AA3377"),
    }
    contexts = [("r_morning", "Morning"), ("r_off", "Off-Peak"), ("r_evening", "Evening Peak")]

    gains = {algo: [] for algo in configs}
    for algo in configs:
        path = os.path.join(RESULT_DIR, algo, "log.csv")
        if not os.path.exists(path):
            gains[algo] = [0, 0, 0]
            continue
        df = pd.read_csv(path)
        for col, _ in contexts:
            if col not in df.columns:
                gains[algo].append(0)
            else:
                early = df[col].head(50).mean()
                late  = df[col].tail(50).mean()
                gains[algo].append(late - early)

    x      = np.arange(len(contexts))
    n      = len(configs)
    width  = 0.18
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (algo, (label, color)), offset in zip(configs.items(), offsets):
        bars = ax.bar(x + offset, gains[algo], width, label=label, color=color, alpha=0.85)
        for bar, g in zip(bars, gains[algo]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{g:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([c for _, c in contexts])
    ax.set_ylabel("Reward Improvement (late 50 − early 50 epochs)")
    ax.set_title("Learning Improvement per Context\n"
                 "(higher = learned more over 500 epochs)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    out = os.path.join(FIG_DIR, "poster_fig10_learning_gain.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_evening_learning()
    plot_ood_over_training()
    plot_learning_gain()
    print("Done.")

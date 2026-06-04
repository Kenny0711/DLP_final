"""
Adaptation Speed Plot
=====================
Visualises how quickly each algorithm recovers performance after a
context transition (morning→off at epoch step 360, off→evening at 720).

Uses per-context reward columns (r_morning, r_off, r_evening) from each
algorithm's log.csv as a proxy for adaptation speed:
  - r_off in the epochs right after training begins shows how fast
    the policy adapts to the off-peak context
  - Same for r_evening

Two plots:
  1. Rolling reward per context over training epochs (line chart)
  2. "Adaptation gap" = |r_context_early − r_context_late| per algorithm (bar)

Output / 輸出:
  figures/adaptation_speed.png
  figures/context_reward_curves.png

Usage:
  python scripts/plot_adaptation.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(HERE, "results")
FIG_DIR    = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALGORITHMS = {
    "a2c":        ("A2C (baseline)",      "tab:blue"),
    "lcpo":       ("LCPO",                "tab:orange"),
    "ddqn":       ("DDQN",                "tab:green"),
    "gru_lcpo":   ("GRU-LCPO (proposed)", "tab:red"),
    "fixed_time": ("Fixed-Time",          "tab:gray"),
}
SMOOTH = 20


def load_log(algo: str) -> pd.DataFrame | None:
    path = os.path.join(RESULT_DIR, algo, "log.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def smooth(s, w=SMOOTH):
    return s.rolling(w, min_periods=1).mean()


# ── Plot 1: Per-context reward curves over training ──────────────────────────
def plot_context_curves(logs: dict):
    contexts = [
        ("r_morning", "Morning Peak 早峰",  0),
        ("r_off",     "Off-Peak 離峰",       1),
        ("r_evening", "Evening Peak 晚峰",   2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Per-Context Reward Over Training\n各時段 Reward 訓練曲線", fontsize=13)

    for ax, (col, title, _) in zip(axes, contexts):
        for algo, (label, color) in ALGORITHMS.items():
            df = logs.get(algo)
            if df is None or col not in df.columns:
                continue
            s = smooth(df[col])
            ax.plot(df["epoch"], s, label=label, color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "context_reward_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")
    plt.close(fig)


# ── Plot 2: Adaptation speed — reward drop right after context switch ─────────
# Proxy: compare r_context in first 25 epochs (before agent adapts) vs
#        last 50 epochs (fully adapted).  Smaller gap = faster adaptation.
def plot_adaptation_gap(logs: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Adaptation Speed After Context Switch\n切換後適應速度（gap 越小代表越快適應）", fontsize=13)

    contexts = [
        ("r_off",     "morning→off-peak shift",    axes[0]),
        ("r_evening", "off-peak→evening shift",     axes[1]),
    ]

    for col, title, ax in contexts:
        labels, gaps, colors = [], [], []
        for algo, (label, color) in ALGORITHMS.items():
            df = logs.get(algo)
            if df is None or col not in df.columns:
                continue
            n = len(df)
            early = df[col].iloc[:max(n//8, 5)].mean()
            late  = df[col].iloc[-50:].mean()
            gap   = abs(early - late)
            labels.append(label)
            gaps.append(gap)
            colors.append(color)

        bars = ax.bar(labels, gaps, color=colors, alpha=0.85)
        for bar, g in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{g:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(title)
        ax.set_ylabel("|early reward − late reward|  (↓ better)")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "adaptation_speed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")
    plt.close(fig)


def main():
    logs = {}
    for algo in ALGORITHMS:
        df = load_log(algo)
        if df is not None:
            logs[algo] = df
            print(f"[load] {algo}: {len(df)} epochs")
        else:
            print(f"[skip] {algo}: log not found")

    if not logs:
        print("No result CSVs found.")
        return

    plot_context_curves(logs)
    plot_adaptation_gap(logs)
    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

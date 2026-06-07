"""
Aggregate & plot results from all 4 algorithms.

Usage:
  python scripts/plot_comparison.py

Output:
  figures/reward_comparison.png
  figures/context_comparison.png
  figures/forgetting.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# line style: solid for RL, dashed for LCPO (A2C and LCPO overlap — use different dash)
ALGORITHMS = {
    "a2c":      ("A2C (baseline)",         "tab:blue",   "-",   1.2),
    "lcpo":     ("LCPO",                   "tab:orange", "--",  1.1),
    "ddqn":     ("DDQN",                   "tab:green",  "-",   1.2),
    "gru_lcpo": ("GRU-LCPO (proposed)",    "tab:red",    "-",   1.5),
}

SMOOTH = 20


def load_log(algo: str) -> pd.DataFrame | None:
    path = os.path.join(RESULT_DIR, algo, "log.csv")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"  [loaded] {algo}: {len(df)} epochs")
    return df


def smooth(series, w: int = SMOOTH) -> pd.Series:
    return series.rolling(w, min_periods=1).mean()


def plot_reward_comparison(logs: dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    for algo, (label, color, ls, lw) in ALGORITHMS.items():
        df = logs.get(algo)
        if df is None or "mean_reward" not in df.columns:
            continue
        s = smooth(df["mean_reward"])
        ax.plot(df["epoch"], s, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Reward (smoothed, w=20)")
    ax.set_title("Algorithm Comparison — Overall Mean Reward\n(A2C and LCPO overlap: LCPO OOD triggered 0/500 epochs)")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(FIG_DIR, "reward_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved -> {path}")
    plt.close(fig)


def plot_context_comparison(logs: dict):
    contexts = [("r_morning", "Morning Peak"),
                ("r_off",     "Off-Peak"),
                ("r_evening", "Evening Peak")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Per-Context Reward Comparison")

    for ax, (col, title) in zip(axes, contexts):
        for algo, (label, color, ls, lw) in ALGORITHMS.items():
            df = logs.get(algo)
            if df is None or col not in df.columns:
                continue
            s = smooth(df[col])
            ax.plot(df["epoch"], s, label=label, color=color,
                    linestyle=ls, linewidth=lw)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Mean Reward (smoothed)")
    axes[-1].legend(loc="lower right")

    path = os.path.join(FIG_DIR, "context_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved -> {path}")
    plt.close(fig)


def plot_forgetting(logs: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo, (label, color, ls, lw) in ALGORITHMS.items():
        df = logs.get(algo)
        if df is None or "r_morning" not in df.columns:
            continue
        n = len(df)
        early  = df["r_morning"].iloc[:n // 4].mean()
        late   = df["r_morning"].iloc[-n // 4:].mean()
        forget = early - late

        ax.bar(label, forget, color=color, alpha=0.8)
        ax.text(label, forget + 0.5, f"{forget:.1f}", ha="center", fontsize=9)

    ax.set_ylabel("Forgetting Score (lower = less forgetting)\n= early morning reward - late morning reward")
    ax.set_title("Catastrophic Forgetting Metric\n(morning reward degradation after context shift)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIG_DIR, "forgetting.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved -> {path}")
    plt.close(fig)


def main():
    print("Loading logs...")
    logs = {}
    for algo in ALGORITHMS:
        df = load_log(algo)
        if df is not None:
            logs[algo] = df

    if not logs:
        print("No result CSVs found.")
        return

    plot_reward_comparison(logs)
    plot_context_comparison(logs)
    plot_forgetting(logs)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

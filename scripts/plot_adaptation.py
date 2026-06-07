"""
Adaptation Speed Plot

Usage:
  python scripts/plot_adaptation.py

Output:
  figures/adaptation_speed.png
  figures/context_reward_curves.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(HERE, "results")
FIG_DIR    = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALGORITHMS = {
    "a2c":        ("A2C (baseline)",      "tab:blue",   "-",   1.2),
    "lcpo":       ("LCPO",                "tab:orange", "--",  1.1),
    "ddqn":       ("DDQN",                "tab:green",  "-",   1.2),
    "gru_lcpo":   ("GRU-LCPO (proposed)", "tab:red",    "-",   1.5),
    "fixed_time": ("Fixed-Time",          "tab:gray",   ":",   1.0),
}
SMOOTH = 20


def load_log(algo: str) -> pd.DataFrame | None:
    path = os.path.join(RESULT_DIR, algo, "log.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def smooth(s, w=SMOOTH):
    return s.rolling(w, min_periods=1).mean()


def plot_context_curves(logs: dict):
    contexts = [
        ("r_morning", "Morning Peak"),
        ("r_off",     "Off-Peak"),
        ("r_evening", "Evening Peak"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Per-Context Reward Over Training", fontsize=13)

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
        ax.set_ylabel("Mean Reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "context_reward_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")
    plt.close(fig)


def plot_adaptation_gap(logs: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Adaptation Speed After Context Switch\n(smaller gap = faster adaptation)", fontsize=13)

    contexts = [
        ("r_off",     "morning -> off-peak shift",    axes[0]),
        ("r_evening", "off-peak -> evening shift",     axes[1]),
    ]

    for col, title, ax in contexts:
        labels, gaps, colors = [], [], []
        for algo, (label, color, ls, lw) in ALGORITHMS.items():
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
        ax.set_ylabel("|early reward - late reward| (lower = better)")
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

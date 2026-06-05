"""
Aggregate & plot results from all 4 algorithms.
彙整四個算法的結果並畫圖比較。

Usage / 使用方式:
  把 4 台電腦的 results/ 資料夾合併到同一台，再執行：
  python plot_comparison.py

Output / 輸出:
  figures/reward_comparison.png    整體 reward 趨勢比較
  figures/context_comparison.png   各 context 的 reward 比較
  figures/forgetting.png           Catastrophic Forgetting 指標
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALGORITHMS = {
    "a2c":      ("A2C (baseline)",         "tab:blue"),
    "lcpo":     ("LCPO",                   "tab:orange"),
    "ddqn":     ("DDQN",                   "tab:green"),
    "gru_lcpo": ("GRU-LCPO (proposed)",    "tab:red"),
}

SMOOTH = 20    # rolling-average window


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


# ── Plot 1: Overall mean reward comparison ───────────────────────────────────
def plot_reward_comparison(logs: dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    for algo, (label, color) in ALGORITHMS.items():
        df = logs.get(algo)
        if df is None or "mean_reward" not in df.columns:
            continue
        s = smooth(df["mean_reward"])
        ax.plot(df["epoch"], s, label=label, color=color)

    ax.set_xlabel("Epoch 訓練輪數")
    ax.set_ylabel("Mean Reward (smoothed)")
    ax.set_title("Algorithm Comparison — Overall Reward\n各算法整體 Reward 比較")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(FIG_DIR, "reward_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.close(fig)


# ── Plot 2: Per-context reward ───────────────────────────────────────────────
def plot_context_comparison(logs: dict):
    contexts = [("r_morning", "Morning Peak 早峰"),
                ("r_off",     "Off-Peak 離峰"),
                ("r_evening", "Evening Peak 晚峰")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Per-Context Reward / 各時段 Reward 比較")

    for ax, (col, title) in zip(axes, contexts):
        for algo, (label, color) in ALGORITHMS.items():
            df = logs.get(algo)
            if df is None or col not in df.columns:
                continue
            s = smooth(df[col])
            ax.plot(df["epoch"], s, label=label, color=color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Mean Reward (smoothed)")
    axes[-1].legend(loc="lower right")

    path = os.path.join(FIG_DIR, "context_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.close(fig)


# ── Plot 3: Catastrophic Forgetting metric ───────────────────────────────────
# Measure: how much does morning_peak performance degrade after the context
#          shifts to off_peak and evening_peak?
# 指標：訓練後期（off/evening context）時，morning_peak 的 reward 是否下降？
def plot_forgetting(logs: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo, (label, color) in ALGORITHMS.items():
        df = logs.get(algo)
        if df is None or "r_morning" not in df.columns:
            continue

        # Compare morning reward in first 25% vs last 25% of training
        n = len(df)
        early  = df["r_morning"].iloc[:n // 4].mean()
        late   = df["r_morning"].iloc[-n // 4:].mean()
        forget = early - late   # positive = forgetting occurred

        ax.bar(label, forget, color=color, alpha=0.8)
        ax.text(label, forget + 0.5, f"{forget:.1f}", ha="center", fontsize=9)

    ax.set_ylabel("Forgetting Score (↓ better)\n= early morning reward − late morning reward")
    ax.set_title("Catastrophic Forgetting Metric\n早峰 reward 退化量（越小代表越不 forget）")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(FIG_DIR, "forgetting.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.close(fig)


def main():
    print("Loading logs…")
    logs = {}
    for algo in ALGORITHMS:
        df = load_log(algo)
        if df is not None:
            logs[algo] = df

    if not logs:
        print("No result CSVs found. Run the training scripts first.")
        return

    plot_reward_comparison(logs)
    plot_context_comparison(logs)
    plot_forgetting(logs)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

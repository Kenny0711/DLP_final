"""
Poster figures for DLP Final — one file per figure, English only.

Output (all in figures/):
  poster_fig1_learning_curves.png   -- overall mean reward vs epoch (all algorithms)
  poster_fig2_context_bar.png       -- grouped bar: final reward per algorithm x context
  poster_fig3_ood_rate.png          -- OOD trigger rate comparison (LCPO vs GRU-LCPO)
  poster_fig4_evening_peak.png      -- evening peak horizontal bar ranking
  poster_fig5_episode_structure.png -- episode timeline with context segments + vph
  poster_fig6_cost_breakdown.png    -- stacked stopping cost per algorithm per context
  poster_fig7_context_curves.png    -- per-context reward curves (3 separate lines)

Usage:
  python scripts/plot_poster.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(HERE, "results")
FIG_DIR    = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALGOS = {
    "fixed_time": ("Fixed-Time",       "#888888", ":",  1.0),
    "a2c":        ("A2C",              "#4477AA", "-",  1.2),
    "lcpo":       ("LCPO",             "#EE6677", "--", 1.1),
    "ddqn":       ("DDQN",             "#228833", "-",  1.2),
    "gru_lcpo":   ("GRU-LCPO (ours)",  "#AA3377", "-",  1.6),
}

def compute_ood_data(logs: dict) -> dict:
    """Auto-compute OOD trigger counts from log CSVs (ood_size > 0 = triggered)."""
    result = {}
    for algo in ("lcpo", "gru_lcpo"):
        df = logs.get(algo)
        if df is None or "ood_size" not in df.columns:
            result[algo] = {"triggered": 0, "total": 0}
        else:
            total     = len(df)
            triggered = int((df["ood_size"] > 0).sum())
            result[algo] = {"triggered": triggered, "total": total}
    return result

SMOOTH = 20
N_LAST = 50  # epochs averaged for final performance


def load_log(algo: str) -> pd.DataFrame | None:
    path = os.path.join(RESULT_DIR, algo, "log.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def smooth(s: pd.Series, w: int = SMOOTH) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def last_n(df: pd.DataFrame | None, col: str, n: int = N_LAST) -> float:
    if df is None or col not in df.columns:
        return 0.0
    return float(df[col].iloc[-n:].mean())


def save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.close(fig)


# ── Fig 1: Overall learning curves ───────────────────────────────────────────
def fig1_learning_curves(logs: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo, (label, color, ls, lw) in ALGOS.items():
        df = logs.get(algo)
        if df is None or "mean_reward" not in df.columns:
            continue
        ax.plot(df["epoch"], smooth(df["mean_reward"]),
                label=label, color=color, linestyle=ls, linewidth=lw)

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Mean Reward per Step (smoothed)", fontsize=12)
    ax.set_title("Overall Learning Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.text(0.02, 0.03,
            "LCPO (fixed): OOD on flow dims only (4-dim z_t proxy, 89.2% trigger rate)",
            transform=ax.transAxes, fontsize=8.5, color="#888")
    fig.tight_layout()
    save(fig, "poster_fig1_learning_curves.png")


# ── Fig 2: Per-context grouped bar chart ─────────────────────────────────────
def fig2_context_bar(logs: dict):
    contexts = [
        ("r_morning", "Morning Peak\n(WB 489 vph)", "#E8A87C"),
        ("r_off",     "Off-Peak\n(WB 418 vph)",     "#82C0CC"),
        ("r_evening", "Evening Peak\n(WB 657 vph)", "#9B59B6"),
    ]
    algo_keys = list(ALGOS.keys())
    n_ctx = len(contexts)
    x     = np.arange(len(algo_keys))
    w     = 0.22
    offs  = np.linspace(-(n_ctx-1)/2*w, (n_ctx-1)/2*w, n_ctx)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (col, label, color) in enumerate(contexts):
        vals = [last_n(logs.get(a), col) for a in algo_keys]
        bars = ax.bar(x + offs[i], vals, w, label=label,
                      color=color, alpha=0.9, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() - 0.08,
                    f"{v:.2f}", ha="center", va="top",
                    fontsize=7.5, color="white", fontweight="bold")

    ax.set_ylabel("Mean Reward (last 50 epochs avg, higher = better)", fontsize=11)
    ax.set_title("Per-Context Final Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGOS[a][0] for a in algo_keys], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(bottom=-8.5, top=0.5)

    gru_idx = algo_keys.index("gru_lcpo")
    ev_val  = last_n(logs.get("gru_lcpo"), "r_evening")
    ax.annotate("Best in\nevening peak",
                xy=(gru_idx + offs[2], ev_val),
                xytext=(gru_idx + offs[2] + 0.15, -0.6),
                fontsize=9, color="#AA3377", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#AA3377", lw=1.2))

    fig.tight_layout()
    save(fig, "poster_fig2_context_bar.png")


# ── Fig 3: OOD trigger rate ───────────────────────────────────────────────────
def fig3_ood_rate(logs: dict, ood_data: dict = None):
    if ood_data is None:
        ood_data = compute_ood_data(logs)

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = [
        f"LCPO\n(flow-only OOD, {ood_data['lcpo']['total']} epochs)",
        f"GRU-LCPO\n(h_t latent, {ood_data['gru_lcpo']['total']} epochs)",
    ]
    rates  = [ood_data[k]["triggered"] / max(ood_data[k]["total"], 1) * 100
              for k in ("lcpo", "gru_lcpo")]
    colors = ["#EE6677", "#AA3377"]

    bars = ax.bar(labels, rates, color=colors, width=0.45, alpha=0.9, edgecolor="white")
    for bar, r, k in zip(bars, rates, ("lcpo", "gru_lcpo")):
        d = ood_data[k]
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{r:.1f}%\n({d['triggered']}/{d['total']})",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("OOD Trigger Rate (%)", fontsize=12)
    ax.set_title("OOD Detector Activation Rate", fontsize=14, fontweight="bold")
    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.annotate("h_t latent space:\ncontext-separable",
                xy=(1, rates[1] - 2), xytext=(1.35, rates[1] - 25),
                fontsize=9, color="#AA3377",
                arrowprops=dict(arrowstyle="->", color="#AA3377", lw=1.0))
    ax.annotate("raw obs noise\nswamps signal",
                xy=(0, rates[0] + 2), xytext=(-0.55, 25),
                fontsize=9, color="#EE6677",
                arrowprops=dict(arrowstyle="->", color="#EE6677", lw=1.0))

    fig.tight_layout()
    save(fig, "poster_fig3_ood_rate.png")


# ── Fig 4: Evening peak horizontal bar ranking ────────────────────────────────
def fig4_evening_peak(logs: dict):
    fig, ax = plt.subplots(figsize=(7, 5))

    algo_order  = ["fixed_time", "a2c", "lcpo", "ddqn", "gru_lcpo"]
    ev_vals     = [last_n(logs.get(a), "r_evening") for a in algo_order]
    ev_colors   = [ALGOS[a][1] for a in algo_order]
    ev_labels   = [ALGOS[a][0] for a in algo_order]

    bars = ax.barh(ev_labels[::-1], ev_vals[::-1],
                   color=ev_colors[::-1], alpha=0.9, edgecolor="white")
    for bar, v in zip(bars, ev_vals[::-1]):
        xpos = v - 0.08 if v < -0.5 else v + 0.05
        ha   = "right" if v < -0.5 else "left"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", ha=ha, fontsize=10)

    ax.set_xlabel("Mean Reward — last 50 epochs (higher = better)", fontsize=11)
    ax.set_title("Evening Peak Performance\n(WB = 657 vph, highest traffic load)", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(left=min(ev_vals) * 1.12)

    fig.tight_layout()
    save(fig, "poster_fig4_evening_peak.png")


# ── Fig 5: Episode structure timeline ────────────────────────────────────────
def fig5_episode_structure():
    fig, ax = plt.subplots(figsize=(10, 4))

    ctx_info = [
        {"name": "Morning Peak", "start": 0,   "steps": 360, "WB": 489, "EB": 319, "color": "#E8A87C"},
        {"name": "Off-Peak",     "start": 360,  "steps": 360, "WB": 418, "EB": 331, "color": "#82C0CC"},
        {"name": "Evening Peak", "start": 720,  "steps": 360, "WB": 657, "EB": 349, "color": "#9B59B6"},
    ]

    row_wb = 0.8
    row_eb = 0.2

    for info in ctx_info:
        s, w = info["start"], info["steps"]
        # WB bar
        ax.barh([row_wb], [w], left=s, height=0.28,
                color=info["color"], alpha=0.9, edgecolor="white", linewidth=1.5)
        # EB bar (lighter)
        ax.barh([row_eb], [w], left=s, height=0.28,
                color=info["color"], alpha=0.5, edgecolor="white", linewidth=1.5)
        # vph annotations
        ax.text(s + w/2, row_wb + 0.21, f"WB: {info['WB']} vph",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=info["color"])
        ax.text(s + w/2, row_eb - 0.21, f"EB: {info['EB']} vph",
                ha="center", va="top", fontsize=10, color=info["color"])
        # context name inside bar
        ax.text(s + w/2, row_wb, info["name"],
                ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")

    # context switch lines
    for vx in [360, 720]:
        ax.axvline(vx, color="#333", linewidth=2, linestyle="--", alpha=0.7)
        ax.text(vx, 1.18, f"Context switch\n(step {vx})", ha="center",
                fontsize=9, color="#444",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#ccc", alpha=0.9))

    # step bracket annotations
    for info in ctx_info:
        s, w = info["start"], info["steps"]
        ax.annotate("", xy=(s + w, -0.12), xytext=(s, -0.12),
                    arrowprops=dict(arrowstyle="<->", color="#666", lw=1.2))
        ax.text(s + w/2, -0.18, f"{w} steps\n(= 60 min sim)",
                ha="center", va="top", fontsize=8.5, color="#666")

    # row labels
    ax.text(-35, row_wb, "WB\n(E2C)", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(-35, row_eb, "EB\n(W2C)", ha="right", va="center", fontsize=10)

    ax.set_xlim(-60, 1140)
    ax.set_ylim(-0.55, 1.45)
    ax.set_xlabel("Episode Step  (1 step = 10 s simulation)", fontsize=11)
    ax.set_title("Episode Structure: 3 Equal Context Segments (1080 steps total)", fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.set_axis_on()
    ax.get_yaxis().set_visible(False)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False)

    # x-axis ticks
    ax.set_xticks([0, 180, 360, 540, 720, 900, 1080])
    ax.set_xticklabels(["0", "180", "360", "540", "720", "900", "1080"])

    fig.tight_layout()
    save(fig, "poster_fig5_episode_structure.png")


# ── Fig 6: Stacked stopping cost per context ─────────────────────────────────
def fig6_cost_breakdown(logs: dict):
    fig, ax = plt.subplots(figsize=(9, 6))

    algo_keys = list(ALGOS.keys())
    ctx_spec  = [
        ("r_morning", "#E8A87C", "Morning Peak"),
        ("r_off",     "#82C0CC", "Off-Peak"),
        ("r_evening", "#9B59B6", "Evening Peak"),
    ]
    x       = np.arange(len(algo_keys))
    bottoms = np.zeros(len(algo_keys))

    for col, color, name in ctx_spec:
        vals = np.array([abs(last_n(logs.get(a), col)) for a in algo_keys])
        bars = ax.bar(x, vals, bottom=bottoms, color=color,
                      alpha=0.9, edgecolor="white", linewidth=0.8, label=name)
        for idx, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0.15:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bottoms[idx] + v/2,
                        f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottoms += vals

    for i, b in enumerate(bottoms):
        ax.text(i, b + 0.04, f"{b:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ALGOS[a][0] for a in algo_keys], fontsize=11)
    ax.set_ylabel("Cumulative Stopping Cost  (|reward| per step, lower = better)", fontsize=11)
    ax.set_title("Context-wise Stopping Cost Breakdown", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    save(fig, "poster_fig6_cost_breakdown.png")


# ── Fig 7: Per-context learning curves (3 panels in one row) ─────────────────
def fig7_context_curves(logs: dict):
    contexts = [
        ("r_morning", "Morning Peak (WB 489 vph)"),
        ("r_off",     "Off-Peak (WB 418 vph)"),
        ("r_evening", "Evening Peak (WB 657 vph)"),
    ]

    for col, title in contexts:
        fname_part = col.replace("r_", "")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algo, (label, color, ls, lw) in ALGOS.items():
            df = logs.get(algo)
            if df is None or col not in df.columns:
                continue
            ax.plot(df["epoch"], smooth(df[col]),
                    label=label, color=color, linestyle=ls, linewidth=lw)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean Reward (smoothed)", fontsize=12)
        ax.set_title(f"Per-Context Reward: {title}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        save(fig, f"poster_fig7_{fname_part}_curve.png")


def main():
    print("Loading logs...")
    logs = {}
    for algo in ALGOS:
        df = load_log(algo)
        if df is not None:
            logs[algo] = df
            print(f"  [ok] {algo}: {len(df)} epochs")
        else:
            print(f"  [skip] {algo}: not found")

    print("\nGenerating poster figures...")
    ood_data = compute_ood_data(logs)
    print(f"  OOD stats: LCPO={ood_data['lcpo']['triggered']}/{ood_data['lcpo']['total']}  "
          f"GRU-LCPO={ood_data['gru_lcpo']['triggered']}/{ood_data['gru_lcpo']['total']}")

    fig1_learning_curves(logs)
    fig2_context_bar(logs)
    fig3_ood_rate(logs, ood_data)
    fig4_evening_peak(logs)
    fig5_episode_structure()
    fig6_cost_breakdown(logs)
    fig7_context_curves(logs)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

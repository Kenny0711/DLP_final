"""
Fixed-Time Control Baseline
============================
Simulates a standard fixed-time traffic signal controller:
  - Phase 0 (NS+WE green): 60 seconds
  - Phase 2 (SN+EW green): 60 seconds
  - Yellow: 3 seconds between phases (handled by env)

No learning, no gradient updates.
Runs the full 3-context + sudden-burst scenario and records
reward and avg_wait per epoch for comparison with RL methods.

固定時制基準線：每 60 秒切換一次綠燈，不做任何學習，
只記錄 reward / avg_wait 供與 RL 方法比較。

Usage:
  python scripts/train_fixed_time.py [--sudden]

Output:
  results/fixed_time/log.csv
"""

import os, sys, csv, argparse
import numpy as np

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(HERE, "src")
SUMO_DIR = os.path.join(os.path.dirname(HERE), "LCPO", "sumo_intersection")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SUMO_DIR)

from utils.sumo_path import ensure_sumo_in_path  # noqa
from sumo_env import SumoIntersectionEnv

# ── Config ───────────────────────────────────────────────────────────────────
FULL_CFG    = os.path.join(HERE, "nets", "intersection.sumocfg")
SUDDEN_CFG  = os.path.join(HERE, "nets", "sudden.sumocfg")
RESULT_DIR  = os.path.join(HERE, "results", "fixed_time")
LOG_PATH    = os.path.join(RESULT_DIR, "log.csv")

N_EPOCHS        = 500
GREEN_DURATION  = 60    # seconds per green phase
STEP_LENGTH     = 10    # env step = 10 SUMO seconds


def get_context(step: int, sudden: bool) -> str:
    if step < 360:  return "morning"
    if step < 720:  return "off"
    if step < 1080: return "evening"
    return "sudden" if sudden else "evening"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sudden", action="store_true",
                        help="Use sudden-burst scenario (12600s, 1260 steps)")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    cfg = SUDDEN_CFG if args.sudden else FULL_CFG
    env = SumoIntersectionEnv(cfg, gui=False)
    if args.sudden:
        env.max_steps = 1260   # 12600s / 10s

    # Fixed-time counter: switch phase every GREEN_DURATION / STEP_LENGTH steps
    steps_per_green = GREEN_DURATION // STEP_LENGTH   # = 6 steps

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "mean_reward", "avg_wait",
            "r_morning", "r_off", "r_evening",
            *(["r_sudden"] if args.sudden else []),
        ])

    for epoch in range(N_EPOCHS):
        obs, _ = env.reset()
        step_idx      = 0
        phase_step    = 0
        current_phase = 0   # 0 or 1 (index into green_phases)
        ep_rews, ep_waits = [], []
        ctx_rews = {c: [] for c in ("morning", "off", "evening", "sudden")}

        done = False
        while not done:
            # Fixed-time logic: switch every steps_per_green steps
            action = current_phase
            if phase_step >= steps_per_green:
                current_phase = 1 - current_phase
                phase_step = 0

            obs, rew, done, trunc, info = env.step(action)
            ep_rews.append(rew)

            # Average waiting time across all edges
            try:
                import traci
                wait = sum(
                    traci.edge.getWaitingTime(e)
                    for e in ["N2C", "S2C", "E2C", "W2C", "C2N", "C2S", "C2E", "C2W"]
                ) / 8.0
            except Exception:
                wait = 0.0
            ep_waits.append(wait)

            ctx = get_context(step_idx, args.sudden)
            ctx_rews[ctx].append(rew)
            step_idx  += 1
            phase_step += 1

            if done or trunc:
                break

        mean_r  = float(np.mean(ep_rews))
        avg_w   = float(np.mean(ep_waits))
        r_m = np.mean(ctx_rews["morning"]) if ctx_rews["morning"] else 0
        r_o = np.mean(ctx_rews["off"])     if ctx_rews["off"]     else 0
        r_e = np.mean(ctx_rews["evening"]) if ctx_rews["evening"] else 0

        row = [epoch, mean_r, avg_w, r_m, r_o, r_e]
        if args.sudden:
            r_s = np.mean(ctx_rews["sudden"]) if ctx_rews["sudden"] else 0
            row.append(r_s)

        print(f"[FixedTime] epoch={epoch:4d} | r̄={mean_r:7.2f} | "
              f"wait={avg_w:6.1f}s | "
              f"morning={r_m:5.1f} off={r_o:5.1f} evening={r_e:5.1f}"
              + (f" sudden={row[-1]:5.1f}" if args.sudden else ""))

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(row)

    env.close()
    print(f"\n[FixedTime] Done. Log → {LOG_PATH}")


if __name__ == "__main__":
    main()

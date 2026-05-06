"""
sumo_env.py
===========
Non-stationary SUMO Traffic Control Environment
for demonstrating Catastrophic Forgetting.

Context schedule (hardcoded):
  Steps     0 –  4 999 : morning_peak  (早峰)
  Steps  5 000 –  9 999 : off_peak     (離峰)
  Steps 10 000 – 14 999 : evening_peak (晚峰)

Observation space : Box(24,) — [queue_length, waiting_time] × 12 incoming lanes
Action space      : Discrete(2) — 0 = keep current green, 1 = switch to next green
Reward            : -sum(waiting_time) across all 12 incoming lanes per step
Episode length    : 15 000 steps (done=True at step 15 000)

Usage
-----
    from sumo_env import SumoEnv
    env = SumoEnv(use_gui=False)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.close()
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# ── SUMO path setup ────────────────────────────────────────────────────────────
SUMO_HOME  = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
if SUMO_TOOLS not in sys.path:
    sys.path.append(SUMO_TOOLS)

import traci  # noqa: E402  (must come after sys.path update)

# ── Project imports ────────────────────────────────────────────────────────────
from tdx_crawler      import get_traffic_data
from sumo_scene_builder import TURN_MOVEMENTS, TURN_RATIOS

# ── File paths ─────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
NETS_DIR   = os.path.join(_HERE, "nets")
NET_FILE   = os.path.join(NETS_DIR, "cross.net.xml")
ROUTE_FILE = os.path.join(NETS_DIR, "rl_env.rou.xml")
CFG_FILE   = os.path.join(NETS_DIR, "rl_env.sumocfg")

# ── Simulation constants ───────────────────────────────────────────────────────
TOTAL_STEPS = 15_000

#: Context windows — (begin_step, end_step_inclusive, context_name)
CONTEXT_WINDOWS: List[Tuple[int, int, str]] = [
    (0,      4_999,  "morning_peak"),
    (5_000,  9_999,  "off_peak"),
    (10_000, 14_999, "evening_peak"),
]

# ── Traffic-light constants ────────────────────────────────────────────────────
TL_ID           = "center"   # matches node id in cross.nod.xml
MIN_GREEN       = 10         # minimum steps before a switch is permitted
YELLOW_DURATION = 3          # steps of yellow inserted between green phases

#: Exact state strings from cross.net.xml <tlLogic>
GREEN_STATES: Dict[int, str] = {
    0: "GGGGgrrrrrGGGGgrrrrr",   # phase 0 — NS + WE green
    2: "rrrrrGGGGgrrrrrGGGGg",   # phase 2 — SN + EW green
}
YELLOW_STATES: Dict[int, str] = {
    0: "yyyyyrrrrryyyyyrrrrr",   # phase 1 — yellow after phase 0
    2: "rrrrryyyyyrrrrryyyyy",   # phase 3 — yellow after phase 2
}

# ── Observation constants ──────────────────────────────────────────────────────
INCOMING_EDGES = ["WE", "EW", "SN", "NS"]   # edges approaching the junction
NUM_LANES      = 3                           # lanes per edge (from cross.edg.xml)

#: All 12 incoming lane IDs in a fixed, reproducible order
LANE_IDS: List[str] = [
    f"{edge}_{i}"
    for edge in INCOMING_EDGES
    for i in range(NUM_LANES)
]
# → ["WE_0","WE_1","WE_2","EW_0","EW_1","EW_2",
#    "SN_0","SN_1","SN_2","NS_0","NS_1","NS_2"]


# ══════════════════════════════════════════════════════════════════════════════
class SumoEnv:
    """
    Non-stationary SUMO RL environment for a single 4-way signalised junction.

    The traffic context changes every 5 000 steps (morning_peak → off_peak →
    evening_peak) via time-windowed flows in a pre-built route file.  The
    traffic-light controller is entirely under the RL agent's authority; SUMO's
    internal phase timer is bypassed by calling ``setRedYellowGreenState`` on
    every simulation step.

    Parameters
    ----------
    use_gui : bool
        If True, launch ``sumo-gui.exe`` for visual debugging.  Defaults to
        False (headless ``sumo.exe``).
    """

    #: Gym-compatible space descriptors (read-only)
    observation_space_shape: Tuple[int, ...] = (24,)
    action_space_n: int = 2

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, use_gui: bool = False) -> None:
        self._use_gui = use_gui

        # TraCI connection state
        self._traci_open: bool = False

        # Simulation step counter (reset to 0 on each reset())
        self._sim_step: int = 0

        # Traffic-light state machine
        self._current_phase: int = 0    # current phase index (0 or 2 = green)
        self._green_time:    int = 0    # steps elapsed in the current green phase
        self._in_yellow:     bool = False
        self._yellow_steps:  int = 0

        # Current context label (informational)
        self._context: str = "morning_peak"

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers — file generation
    # ══════════════════════════════════════════════════════════════════════════

    def _build_route_file(self) -> None:
        """
        Generate ``nets/rl_env.rou.xml`` with time-windowed flows for all three
        traffic contexts.

        Structure
        ---------
        * 12 shared ``<route>`` elements (one per direction × turn type).
        * 36 ``<flow>`` elements (12 per context × 3 contexts), each carrying
          ``begin`` and ``end`` attributes that match the context windows.
        """
        # ── Fetch traffic data for every context ──────────────────────────────
        context_data: Dict[str, Dict[str, int]] = {}
        for _, _, ctx in CONTEXT_WINDOWS:
            raw = get_traffic_data(ctx)
            context_data[ctx] = {
                entry["direction"]: entry["volume_per_hour"]
                for entry in raw
                if entry.get("volume_per_hour", 0) > 0
            }

        lines: List[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append("<routes>")
        lines.append("")
        lines.append("    <!-- Vehicle type -->")
        lines.append(
            '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5"'
            ' length="5" maxSpeed="13.89"/>'
        )
        lines.append("")

        # ── Route definitions (shared; declared once) ─────────────────────────
        lines.append("    <!-- Route definitions (shared across all contexts) -->")
        for direction, turns in TURN_MOVEMENTS.items():
            for turn_name, (from_edge, to_edge) in turns.items():
                route_id = f"route_{direction}_{turn_name}"
                lines.append(
                    f'    <route id="{route_id}" edges="{from_edge} {to_edge}"/>'
                )
        lines.append("")

        # ── Time-windowed flows per context ───────────────────────────────────
        for begin, end, ctx in CONTEXT_WINDOWS:
            lines.append(f"    <!-- {ctx}: steps {begin}–{end} -->")
            vph_map = context_data.get(ctx, {})

            for direction, turns in TURN_MOVEMENTS.items():
                vph = vph_map.get(direction, 0)
                if vph <= 0:
                    continue
                for turn_name in turns:
                    turn_vph = vph * TURN_RATIOS[turn_name]
                    if turn_vph < 1:
                        continue
                    period   = round(3600.0 / turn_vph, 4)
                    flow_id  = f"flow_{ctx}_{direction}_{turn_name}"
                    route_id = f"route_{direction}_{turn_name}"
                    lines.append(
                        f'    <flow id="{flow_id}" route="{route_id}"'
                        f' begin="{begin}" end="{end}"'
                        f' period="{period}"'
                        f' departLane="best" departSpeed="max"/>'
                    )
            lines.append("")

        lines.append("</routes>")

        os.makedirs(NETS_DIR, exist_ok=True)
        with open(ROUTE_FILE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _build_config_file(self) -> None:
        """
        Generate ``nets/rl_env.sumocfg`` pointing to the existing network and
        the freshly-written route file.  Simulation end is ``TOTAL_STEPS``.
        """
        lines: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<configuration>",
            "    <input>",
            '        <net-file value="cross.net.xml"/>',
            '        <route-files value="rl_env.rou.xml"/>',
            "    </input>",
            "    <time>",
            '        <begin value="0"/>',
            f'        <end value="{TOTAL_STEPS}"/>',
            "    </time>",
            "    <report>",
            '        <verbose value="false"/>',
            '        <no-step-log value="true"/>',
            "    </report>",
            "</configuration>",
        ]
        os.makedirs(NETS_DIR, exist_ok=True)
        with open(CFG_FILE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers — simulation queries
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_context(step: int) -> str:
        """Return the context name that corresponds to ``step``."""
        for begin, end, name in CONTEXT_WINDOWS:
            if begin <= step <= end:
                return name
        return CONTEXT_WINDOWS[-1][2]   # fallback: last context

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation vector.

        Returns
        -------
        np.ndarray, shape (24,), dtype float32
            Interleaved [queue_length, waiting_time] for every incoming lane::

                [WE_0_q, WE_0_w, WE_1_q, WE_1_w, WE_2_q, WE_2_w,
                 EW_0_q, EW_0_w, ...,
                 NS_2_q, NS_2_w]
        """
        obs: List[float] = []
        for lane_id in LANE_IDS:
            queue   = float(traci.lane.getLastStepHaltingNumber(lane_id))
            waiting = float(traci.lane.getWaitingTime(lane_id))
            obs.append(queue)
            obs.append(waiting)
        return np.array(obs, dtype=np.float32)

    def _apply_action(self, action: int) -> None:
        """
        Update the traffic-light state machine and write the new signal state
        to SUMO via ``setRedYellowGreenState``.

        The agent is shielded from yellow phases: all yellow bookkeeping happens
        here.  While a yellow transition is in progress the agent's action is
        ignored.

        Parameters
        ----------
        action : int
            0 — keep the current green phase.
            1 — request a switch to the other green phase (subject to MIN_GREEN).
        """
        if self._in_yellow:
            # ── Yellow transition in progress — count down and ignore action ──
            self._yellow_steps += 1
            traci.trafficlight.setRedYellowGreenState(
                TL_ID, YELLOW_STATES[self._current_phase]
            )
            if self._yellow_steps >= YELLOW_DURATION:
                # Transition complete — activate the next green phase
                next_phase           = (self._current_phase + 2) % 4
                self._current_phase  = next_phase
                self._green_time     = 0
                self._in_yellow      = False
                self._yellow_steps   = 0
                traci.trafficlight.setRedYellowGreenState(
                    TL_ID, GREEN_STATES[self._current_phase]
                )
        else:
            # ── Green phase ───────────────────────────────────────────────────
            if action == 1 and self._green_time >= MIN_GREEN:
                # Initiate yellow transition
                self._in_yellow    = True
                self._yellow_steps = 0
                traci.trafficlight.setRedYellowGreenState(
                    TL_ID, YELLOW_STATES[self._current_phase]
                )
            else:
                # Keep current green (or action==1 but MIN_GREEN not yet reached)
                self._green_time += 1
                traci.trafficlight.setRedYellowGreenState(
                    TL_ID, GREEN_STATES[self._current_phase]
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Public RL interface
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self) -> np.ndarray:
        """
        (Re)start the SUMO simulation.

        Steps
        -----
        1. Close any existing TraCI connection.
        2. Regenerate ``rl_env.rou.xml`` and ``rl_env.sumocfg``.
        3. Reset all internal counters.
        4. Launch SUMO via ``traci.start``.
        5. Advance one simulation step (so lane statistics are populated).
        6. Override the traffic-light to phase 0.
        7. Return the initial observation.

        Returns
        -------
        np.ndarray, shape (24,)
        """
        self.close()

        # Rebuild simulation files (picks up latest TDX data if USE_MOCK=False)
        print("[SumoEnv] 產生 rl_env.rou.xml …")
        self._build_route_file()
        print("[SumoEnv] 產生 rl_env.sumocfg …")
        self._build_config_file()

        # Reset counters
        self._sim_step      = 0
        self._current_phase = 0
        self._green_time    = 0
        self._in_yellow     = False
        self._yellow_steps  = 0
        self._context       = "morning_peak"

        # Start SUMO
        binary = os.path.join(
            SUMO_HOME, "bin",
            "sumo-gui.exe" if self._use_gui else "sumo.exe",
        )
        sumo_cmd = [
            binary,
            "-c", CFG_FILE,
            "--waiting-time-memory", "3600",  # accumulate wait over full episode
            "--no-step-log",                  # suppress per-step console noise
        ]
        traci.start(sumo_cmd)
        self._traci_open = True

        # Advance one step so TraCI statistics are non-empty
        traci.simulationStep()
        self._sim_step = 1

        # Set initial TL state
        traci.trafficlight.setRedYellowGreenState(TL_ID, GREEN_STATES[0])

        print(f"[SumoEnv] 模擬啟動 ✓  總步數={TOTAL_STEPS}，obs shape={self.observation_space_shape}")
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Advance the simulation by one step.

        Parameters
        ----------
        action : int
            0 — keep current green phase.
            1 — switch to next green phase (MIN_GREEN constraint enforced).

        Returns
        -------
        obs : np.ndarray, shape (24,)
        reward : float
            Negative sum of per-lane waiting times.
        done : bool
            True when ``_sim_step >= TOTAL_STEPS``.
        info : dict
            ``{"step": int, "context": str, "green_time": int}``
        """
        if not self._traci_open:
            raise RuntimeError("Call reset() before step().")

        # 1. Apply traffic-light action
        self._apply_action(action)

        # 2. Advance SUMO by one second
        traci.simulationStep()
        self._sim_step += 1

        # 3. Reward: negative sum of waiting times across all incoming lanes
        reward = -sum(
            traci.lane.getWaitingTime(lid) for lid in LANE_IDS
        )

        # 4. Observation
        obs = self._get_obs()

        # 5. Terminal condition
        done = self._sim_step >= TOTAL_STEPS

        # 6. Update context label
        self._context = self._get_context(self._sim_step)

        info: Dict = {
            "step":       self._sim_step,
            "context":    self._context,
            "green_time": self._green_time,
        }

        return obs, reward, done, info

    def close(self) -> None:
        """Gracefully shut down the TraCI connection."""
        if self._traci_open:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_open = False


# ══════════════════════════════════════════════════════════════════════════════
# Quick sanity test — random agent
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import random

    print("=" * 60)
    print(" SumoEnv — random agent sanity test")
    print("=" * 60)

    env = SumoEnv(use_gui=True)

    try:
        obs = env.reset()
        print(f"\nobs shape : {obs.shape}")
        print(f"obs[:6]   : {obs[:6]}   (WE 前3車道 queue/wait)")
        print(f"action dim: {env.action_space_n}")
        print()

        total_reward = 0.0
        last_context = env._context

        while True:
            action = random.randint(0, env.action_space_n - 1)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # 印出 context 切換
            if info["context"] != last_context:
                print(
                    f"  ★ Context 切換  step={info['step']:5d} : "
                    f"{last_context} → {info['context']}"
                )
                last_context = info["context"]

            # 每 1000 步印進度
            if info["step"] % 1_000 == 0:
                print(
                    f"  step={info['step']:5d} | "
                    f"ctx={info['context']:15s} | "
                    f"green={info['green_time']:3d} | "
                    f"reward={reward:10.1f} | "
                    f"cum={total_reward:12.1f}"
                )

            if done:
                break

    except KeyboardInterrupt:
        print("\n[中斷]")
    finally:
        env.close()

    print()
    print("=" * 60)
    print(f" 完成  |  步數={env._sim_step}  |  total reward={total_reward:.1f}")
    print("=" * 60)

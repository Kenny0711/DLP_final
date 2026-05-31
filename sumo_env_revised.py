from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
if SUMO_TOOLS not in sys.path:
    sys.path.append(SUMO_TOOLS)

import traci

from tdx_crawler import get_traffic_data
from sumo_scene_builder import TURN_MOVEMENTS, TURN_RATIOS

_HERE = os.path.dirname(os.path.abspath(__file__))
NETS_DIR = os.path.join(_HERE, "nets")
NET_FILE = os.path.join(NETS_DIR, "cross.net.xml")
ROUTE_FILE = os.path.join(NETS_DIR, "rl_env.rou.xml")
CFG_FILE = os.path.join(NETS_DIR, "rl_env.sumocfg")

TL_ID = "center"
MIN_GREEN = 10
YELLOW_DURATION = 3

GREEN_STATES: Dict[int, str] = {
    0: "GGGGgrrrrrGGGGgrrrrr",
    2: "rrrrrGGGGgrrrrrGGGGg",
}
YELLOW_STATES: Dict[int, str] = {
    0: "yyyyyrrrrryyyyyrrrrr",
    2: "rrrrryyyyyrrrrryyyyy",
}

INCOMING_EDGES = ["WE", "EW", "SN", "NS"]
NUM_LANES = 3

LANE_IDS: List[str] = [
    f"{edge}_{i}"
    for edge in INCOMING_EDGES
    for i in range(NUM_LANES)
]

class SumoEnv:
    observation_space_shape: Tuple[int, ...] = (24,)
    action_space_n: int = 2

    def __init__(self, use_gui: bool = False, mode: str = "online") -> None:
        self._use_gui = use_gui
        self.mode = mode

        if self.mode == "warmup":
            self.TOTAL_STEPS = 360
            self.CONTEXT_WINDOWS = [(0, 359, "morning_peak")]
        else:
            self.TOTAL_STEPS = 1080
            self.CONTEXT_WINDOWS = [
                (0, 359, "morning_peak"),
                (360, 719, "off_peak"),
                (720, 1079, "evening_peak"),
            ]

        self._traci_open: bool = False
        self._sim_step: int = 0
        self._current_phase: int = 0
        self._green_time: int = 0
        self._in_yellow: bool = False
        self._yellow_steps: int = 0
        self._context: str = "morning_peak"

    def _build_route_file(self) -> None:
        context_data: Dict[str, Dict[str, int]] = {}
        for _, _, ctx in self.CONTEXT_WINDOWS:
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
        lines.append('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>')
        lines.append("")

        for direction, turns in TURN_MOVEMENTS.items():
            for turn_name, (from_edge, to_edge) in turns.items():
                route_id = f"route_{direction}_{turn_name}"
                lines.append(f'    <route id="{route_id}" edges="{from_edge} {to_edge}"/>')
        lines.append("")

        for begin, end, ctx in self.CONTEXT_WINDOWS:
            vph_map = context_data.get(ctx, {})
            for direction, turns in TURN_MOVEMENTS.items():
                vph = vph_map.get(direction, 0)
                if vph <= 0:
                    continue
                for turn_name in turns:
                    turn_vph = vph * TURN_RATIOS[turn_name]
                    if turn_vph < 1:
                        continue
                    period = round(3600.0 / turn_vph, 4)
                    flow_id = f"flow_{ctx}_{direction}_{turn_name}"
                    route_id = f"route_{direction}_{turn_name}"
                    lines.append(
                        f'    <flow id="{flow_id}" route="{route_id}" '
                        f'begin="{begin}" end="{end}" period="{period}" '
                        f'departLane="best" departSpeed="max"/>'
                    )
            lines.append("")

        lines.append("</routes>")
        os.makedirs(NETS_DIR, exist_ok=True)
        with open(ROUTE_FILE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def _build_config_file(self) -> None:
        lines: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<configuration>",
            "    <input>",
            '        <net-file value="cross.net.xml"/>',
            '        <route-files value="rl_env.rou.xml"/>',
            "    </input>",
            "    <time>",
            '        <begin value="0"/>',
            f'        <end value="{self.TOTAL_STEPS}"/>',
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

    def _get_context(self, step: int) -> str:
        for begin, end, name in self.CONTEXT_WINDOWS:
            if begin <= step <= end:
                return name
        return self.CONTEXT_WINDOWS[-1][2]

    def _get_obs(self) -> np.ndarray:
        obs: List[float] = []
        for lane_id in LANE_IDS:
            queue = float(traci.lane.getLastStepHaltingNumber(lane_id))
            waiting = float(traci.lane.getWaitingTime(lane_id))
            obs.append(queue)
            obs.append(waiting)
        return np.array(obs, dtype=np.float32)

    def _apply_action(self, action: int) -> None:
        if self._in_yellow:
            self._yellow_steps += 1
            traci.trafficlight.setRedYellowGreenState(TL_ID, YELLOW_STATES[self._current_phase])
            if self._yellow_steps >= YELLOW_DURATION:
                next_phase = (self._current_phase + 2) % 4
                self._current_phase = next_phase
                self._green_time = 0
                self._in_yellow = False
                self._yellow_steps = 0
                traci.trafficlight.setRedYellowGreenState(TL_ID, GREEN_STATES[self._current_phase])
        else:
            if action == 1 and self._green_time >= MIN_GREEN:
                self._in_yellow = True
                self._yellow_steps = 0
                traci.trafficlight.setRedYellowGreenState(TL_ID, YELLOW_STATES[self._current_phase])
            else:
                self._green_time += 1
                traci.trafficlight.setRedYellowGreenState(TL_ID, GREEN_STATES[self._current_phase])

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.close()
        self._build_route_file()
        self._build_config_file()
        self._sim_step = 0
        self._current_phase = 0
        self._green_time = 0
        self._in_yellow = False
        self._yellow_steps = 0
        self._context = "morning_peak"

        import platform

        if platform.system() == "Windows":
            # Windows
            exe_name = "sumo-gui.exe" if self._use_gui else "sumo.exe"
            binary = os.path.join(SUMO_HOME, "bin", exe_name)
        else:
            # Linux/Mac
            binary = "sumo-gui" if self._use_gui else "sumo"

        sumo_cmd = [
            binary,
            "-c", CFG_FILE,
            "--waiting-time-memory", "3600",
            "--no-step-log",
        ]
        traci.start(sumo_cmd)
        self._traci_open = True
        traci.simulationStep()
        self._sim_step = 1
        traci.trafficlight.setRedYellowGreenState(TL_ID, GREEN_STATES[0])
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not self._traci_open:
            raise RuntimeError("Call reset() before step().")

        self._apply_action(action)
        traci.simulationStep()
        self._sim_step += 1

        reward = -sum(traci.lane.getWaitingTime(lid) for lid in LANE_IDS)
        obs = self._get_obs()
        done = self._sim_step >= self.TOTAL_STEPS
        trunc = False
        self._context = self._get_context(self._sim_step)

        info: Dict = {
            "step": self._sim_step,
            "context": self._context,
            "green_time": self._green_time,
        }

        return obs, reward, done, trunc, info

    def close(self) -> None:
        if self._traci_open:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_open = False

if __name__ == "__main__":
    import random
    env = SumoEnv(use_gui=True, mode="online")
    try:
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action = random.randint(0, env.action_space_n - 1)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            if done or trunc:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
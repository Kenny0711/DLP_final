"""
Convert traffic cache JSON(s) -> SUMO route file for LCPO intersection env.

兩個路口的資料合併取中位數，產生代表性 intersection.rou.xml。
Two intersections' data are merged (per-direction median) into one XML.

Direction mapping (TDX -> SUMO edges):
  EB (Eastbound,  enters from West)  : W2C -> C2E
  WB (Westbound,  enters from East)  : E2C -> C2W
  NB (Northbound, enters from South) : S2C -> C2N
  SB (Southbound, enters from North) : N2C -> C2S

Output:
  nets/intersection.rou.xml       全段模擬 (morning→off→evening, 10800s)
  nets/warmup.rou.xml             僅 morning_peak (0~3600s) 供 warm-up
  nets/intersection.sumocfg
  nets/warmup.sumocfg
"""

import argparse
import json
import os
import statistics
from typing import Dict, List

HERE     = os.path.dirname(os.path.abspath(__file__))
NETS_DIR = os.path.join(HERE, "nets")

# 預設兩個路口的 cache 檔
DEFAULT_CACHES = [
    os.path.join(HERE, "traffic_cache.json"),           # 忠孝x復興南
    os.path.join(HERE, "traffic_cache_dunhua.json"),    # 忠孝x敦化
]

EDGE_MAP = {
    "EB": ("W2C", "C2E"),
    "WB": ("E2C", "C2W"),
    "NB": ("S2C", "C2N"),
    "SB": ("N2C", "C2S"),
}

# 全段模擬時窗 (morning→off→evening)
CONTEXTS = [
    ("morning_peak",  0,    3600),
    ("off_peak",      3600, 7200),
    ("evening_peak",  7200, 10800),
]

SIM_END  = 10800
MIN_VPH  = 10       # 丟棄低於此值的感測器錯誤資料


def load_averages(cache_files: List[str]) -> Dict[str, Dict[str, int]]:
    """
    從多個 cache 檔合併，計算每個 period × direction 的中位數 vph。
    Merge multiple cache files and compute median vph per period × direction.
    """
    bucket: Dict = {p: {d: [] for d in EDGE_MAP} for p, *_ in CONTEXTS}

    for path in cache_files:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        src = os.path.basename(path)
        for date, periods in cache.items():
            for period, _, __ in CONTEXTS:
                for entry in periods.get(period, []):
                    direction = entry.get("direction")
                    vph = entry.get("volume_per_hour", 0)
                    if direction in EDGE_MAP and vph >= MIN_VPH:
                        bucket[period][direction].append(vph)

    result: Dict = {}
    for period, _, __ in CONTEXTS:
        result[period] = {}
        for direction in EDGE_MAP:
            vals = bucket[period][direction]
            if vals:
                med = round(statistics.median(vals))
                result[period][direction] = med
                print(f"  {period:15s} {direction}: median={med:5d}  "
                      f"(n={len(vals)}, range=[{min(vals)}, {max(vals)}])")
            else:
                print(f"  {period:15s} {direction}: NO VALID DATA — skipped")
    return result


def _flow_lines(averages, contexts_subset, prefix="") -> List[str]:
    lines = []
    for period, begin, end in contexts_subset:
        lines.append(f"    <!-- {period} : {begin}~{end}s -->")
        for direction, (from_e, to_e) in EDGE_MAP.items():
            vph = averages.get(period, {}).get(direction, 0)
            if not vph:
                continue
            fid = f"{prefix}{period}_{direction}" if prefix else f"{period}_{direction}"
            lines.append(
                f'    <flow id="{fid}" type="car"'
                f' from="{from_e}" to="{to_e}"'
                f' begin="{begin}" end="{end}"'
                f' vehsPerHour="{vph}"/>'
            )
        lines.append("")
    return lines


def generate_rou(averages: Dict, out_path: str, contexts_subset, label: str) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        f"    <!-- {label} — merged from 忠孝x復興南 + 忠孝x敦化 -->",
        '    <vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="13.9" sigma="0.5"/>',
        "",
    ]
    lines.extend(_flow_lines(averages, contexts_subset))
    lines.append("</routes>")
    os.makedirs(NETS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[rou]  -> {out_path}")


def generate_cfg(rou_filename: str, end_time: int, out_path: str) -> None:
    xml = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<configuration>",
        "    <input>",
        '        <net-file value="intersection.net.xml"/>',
        f'        <route-files value="{rou_filename}"/>',
        "    </input>",
        "    <time>",
        '        <begin value="0"/>',
        f'        <end value="{end_time}"/>',
        '        <step-length value="1"/>',
        "    </time>",
        "    <processing>",
        '        <collision.action value="warn"/>',
        "    </processing>",
        "</configuration>",
    ]) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[cfg]  -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SUMO route XMLs from traffic cache files"
    )
    parser.add_argument(
        "--caches", nargs="+", default=DEFAULT_CACHES,
        help="Cache JSON file paths (default: both intersections)"
    )
    args = parser.parse_args()

    print(f"=== generate_rou.py ===")
    print(f"Sources: {[os.path.basename(c) for c in args.caches]}\n")
    print(f"[stats] Computing medians (threshold >= {MIN_VPH} vph):\n")

    averages = load_averages(args.caches)

    # 1. 全段模擬 XML (morning → off → evening)
    generate_rou(
        averages,
        out_path=os.path.join(NETS_DIR, "intersection.rou.xml"),
        contexts_subset=CONTEXTS,
        label="Full experiment (morning→off→evening)",
    )
    generate_cfg(
        rou_filename="intersection.rou.xml",
        end_time=SIM_END,
        out_path=os.path.join(NETS_DIR, "intersection.sumocfg"),
    )

    # 2. Warm-up XML (morning_peak only, 0~3600s)
    generate_rou(
        averages,
        out_path=os.path.join(NETS_DIR, "warmup.rou.xml"),
        contexts_subset=[CONTEXTS[0]],   # morning_peak only
        label="Warm-up (morning_peak only)",
    )
    generate_cfg(
        rou_filename="warmup.rou.xml",
        end_time=3600,
        out_path=os.path.join(NETS_DIR, "warmup.sumocfg"),
    )

    print("\nDone. Generated:")
    print("  nets/intersection.rou.xml  + intersection.sumocfg  (全段, 10800s)")
    print("  nets/warmup.rou.xml        + warmup.sumocfg         (warm-up, 3600s)")

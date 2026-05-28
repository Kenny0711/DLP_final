"""
Convert traffic_cache.json -> SUMO route file for LCPO intersection env.

Direction mapping (TDX -> SUMO edges):
  EB (Eastbound,  enters from West)  : W2C -> C2E
  WB (Westbound,  enters from East)  : E2C -> C2W
  NB (Northbound, enters from South) : S2C -> C2N
  SB (Southbound, enters from North) : N2C -> C2S

Output:
  nets/intersection.rou.xml   (compatible with LCPO sumo_env.py)
  nets/intersection.sumocfg
"""

import argparse
import json
import os
import statistics

HERE     = os.path.dirname(os.path.abspath(__file__))
CACHE    = os.path.join(HERE, "traffic_cache.json")
NETS_DIR = os.path.join(HERE, "nets")
OUT_ROU  = os.path.join(NETS_DIR, "intersection.rou.xml")
OUT_CFG  = os.path.join(NETS_DIR, "intersection.sumocfg")

EDGE_MAP = {
    "EB": ("W2C", "C2E"),
    "WB": ("E2C", "C2W"),
    "NB": ("S2C", "C2N"),
    "SB": ("N2C", "C2S"),
}

# Time windows in SUMO seconds (matches LCPO env: max_steps=1080, step_length=10)
CONTEXTS = [
    ("morning_peak",  0,    3600),
    ("off_peak",      3600, 7200),
    ("evening_peak",  7200, 10800),
]

SIM_END   = 10800
MIN_VPH   = 10      # discard placeholder / sensor-error values below this


def load_averages() -> dict:
    """Compute per-period median vph per direction across all valid cache dates."""
    with open(CACHE, encoding="utf-8") as f:
        cache = json.load(f)

    bucket: dict = {p: {d: [] for d in EDGE_MAP} for p, *_ in CONTEXTS}

    for date, periods in cache.items():
        for period, _, __ in CONTEXTS:
            for entry in periods.get(period, []):
                direction = entry.get("direction")
                vph = entry.get("volume_per_hour", 0)
                if direction in EDGE_MAP and vph >= MIN_VPH:
                    bucket[period][direction].append(vph)

    result: dict = {}
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


def generate_rou(averages: dict, junction_id: str) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        f"    <!-- Junction {junction_id} — generated from traffic_cache.json -->",
        '    <vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="13.9" sigma="0.5"/>',
        "",
    ]
    for period, begin, end in CONTEXTS:
        lines.append(f"    <!-- {period} : {begin}~{end}s -->")
        for direction, (from_e, to_e) in EDGE_MAP.items():
            vph = averages.get(period, {}).get(direction, 0)
            if not vph:
                continue
            lines.append(
                f'    <flow id="{period}_{direction}" type="car"'
                f' from="{from_e}" to="{to_e}"'
                f' begin="{begin}" end="{end}"'
                f' vehsPerHour="{vph}"/>'
            )
        lines.append("")
    lines.append("</routes>")

    os.makedirs(NETS_DIR, exist_ok=True)
    with open(OUT_ROU, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[rou]  -> {OUT_ROU}")


def generate_cfg() -> None:
    xml = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<configuration>",
        "    <input>",
        '        <net-file value="intersection.net.xml"/>',
        '        <route-files value="intersection.rou.xml"/>',
        "    </input>",
        "    <time>",
        '        <begin value="0"/>',
        f'        <end value="{SIM_END}"/>',
        '        <step-length value="1"/>',
        "    </time>",
        "    <processing>",
        '        <collision.action value="warn"/>',
        "    </processing>",
        "</configuration>",
    ]) + "\n"
    with open(OUT_CFG, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[cfg]  -> {OUT_CFG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SUMO route XML from traffic_cache.json"
    )
    parser.add_argument(
        "--junction", default="J1", choices=["J1", "J2", "J3"],
        help="Which intersection this machine represents (for labelling only)"
    )
    args = parser.parse_args()

    print(f"=== generate_rou.py  (junction={args.junction}) ===\n")
    print(f"[stats] Computing medians from {CACHE} (threshold >= {MIN_VPH} vph):\n")
    averages = load_averages()
    generate_rou(averages, args.junction)
    generate_cfg()
    print("\nDone.")

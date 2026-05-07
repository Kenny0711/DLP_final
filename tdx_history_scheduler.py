"""
TDX traffic route history scheduler.

Keep tdx_crawler.py unchanged. This script periodically fetches traffic data
from tdx_crawler and writes timestamped SUMO route XML files.

Usage:
    python3 tdx_history_scheduler.py --once
    python3 tdx_history_scheduler.py --interval 20
    python3 tdx_history_scheduler.py --mock --once
"""

import argparse
import datetime
import os
import time

import requests


def load_local_env() -> None:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_file):
        return

    with open(env_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()

import tdx_crawler


INTERSECTION_ROAD1 = "復興南路一段"
INTERSECTION_ROAD2 = "忠孝東路"
INTERSECTION_NAME = f"{INTERSECTION_ROAD1}_x_{INTERSECTION_ROAD2}"

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(HERE, "traffic_history")
SIM_DURATION = 3600

TURN_MOVEMENTS = {
    "EB": {
        "straight": ("WE", "CE"),
        "left": ("WE", "CN"),
        "right": ("WE", "CS"),
    },
    "WB": {
        "straight": ("EW", "CW"),
        "left": ("EW", "CS"),
        "right": ("EW", "CN"),
    },
    "NB": {
        "straight": ("SN", "CN"),
        "left": ("SN", "CW"),
        "right": ("SN", "CE"),
    },
    "SB": {
        "straight": ("NS", "CS"),
        "left": ("NS", "CE"),
        "right": ("NS", "CW"),
    },
}

TURN_RATIOS = {"straight": 0.70, "left": 0.10, "right": 0.20}


def has_valid_tdx_credentials() -> bool:
    return (
        bool(tdx_crawler.CLIENT_ID)
        and bool(tdx_crawler.CLIENT_SECRET)
        and tdx_crawler.CLIENT_ID != "YOUR_CLIENT_ID"
        and tdx_crawler.CLIENT_SECRET != "YOUR_CLIENT_SECRET"
    )


def print_credential_help() -> None:
    print("[TDX] 無法取得 token，請確認 TDX_CLIENT_ID / TDX_CLIENT_SECRET 是否正確。")
    print("[TDX] Linux/macOS 設定方式：")
    print('      export TDX_CLIENT_ID="你的 Client ID"')
    print('      export TDX_CLIENT_SECRET="你的 Client Secret"')
    print("[TDX] 若只想測試 XML 寫檔流程，可以加 --mock。")


def get_token_or_exit():
    if tdx_crawler.USE_MOCK:
        return None

    if not has_valid_tdx_credentials():
        print_credential_help()
        raise SystemExit(1)

    try:
        return tdx_crawler.get_tdx_token()
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        print(f"[TDX] 取得 token 失敗：HTTP {status_code}")
        print_credential_help()
        raise SystemExit(1) from exc


def safe_filename(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def get_current_period(now=None) -> str:
    now = now or datetime.datetime.now()
    hour = now.hour
    if 7 <= hour < 9:
        return "morning_peak"
    if 17 <= hour < 19:
        return "evening_peak"
    return "off_peak"


def has_valid_route_data(traffic_data: list) -> bool:
    for entry in traffic_data:
        direction = entry.get("direction")
        volume_per_hour = entry.get("volume_per_hour", 0)
        if direction in TURN_MOVEMENTS and volume_per_hour > 0:
            return True
    return False


def _positive_volume(value) -> int:
    try:
        volume = int(float(value))
    except (TypeError, ValueError):
        return 0
    return volume if volume > 0 else 0


def fetch_tdx_traffic_nonnegative(token: str) -> list:
    """
    Fetch TDX live traffic while ignoring negative sentinel values.

    Some TDX VD vehicle records report negative Volume values when live data is
    unavailable. Those are not real traffic counts and must not be summed into
    SUMO flows.
    """
    metadata = tdx_crawler.fetch_vd_metadata(token)
    if not metadata:
        print("[警告] 找不到目標路口的 VD 感測器，確認 TARGET_ROADS 設定是否正確")
        return []

    time.sleep(1)
    url = f"{tdx_crawler.TDX_API_BASE}/v2/Road/Traffic/Live/VD/City/Taipei?$format=JSON"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    raw = resp.json()

    live_list = raw.get("VDLives", [])
    dir_map = {"E": "EB", "W": "WB", "N": "NB", "S": "SB"}
    dir_volumes = {"EB": [], "WB": [], "NB": [], "SB": []}
    ignored_negative = 0

    for vd in live_list:
        vdid = vd.get("VDID", "")
        if vdid not in metadata:
            continue

        bearing = metadata[vdid].get("bearing")
        if bearing not in dir_map:
            continue
        direction = dir_map[bearing]

        total_volume = 0
        for link in vd.get("LinkFlows", []):
            for lane in link.get("Lanes", []):
                for vehicle in lane.get("Vehicles", []):
                    raw_volume = vehicle.get("Volume", 0)
                    if _positive_volume(raw_volume) == 0 and raw_volume not in (0, "0", None):
                        ignored_negative += 1
                    total_volume += _positive_volume(raw_volume)

        if total_volume > 0:
            dir_volumes[direction].append(total_volume * 12)

    if ignored_negative:
        print(f"[資料] 已忽略 {ignored_negative} 筆 TDX 負值/無效 Volume")

    if not dir_volumes["NB"] and dir_volumes["SB"]:
        avg_sb = sum(dir_volumes["SB"]) / len(dir_volumes["SB"])
        dir_volumes["NB"] = [int(avg_sb * 0.9)]
        print("[資料] NB 無感測器，以 SB 平均值 × 0.9 估算")

    results = []
    for direction, volumes in dir_volumes.items():
        if volumes:
            avg = int(sum(volumes) / len(volumes))
            results.append({"direction": direction, "volume_per_hour": avg})
        else:
            print(f"[警告] {direction} 無正流量資料，跳過")

    return results


def fetch_traffic_snapshot(token=None) -> list:
    now = datetime.datetime.now()

    if tdx_crawler.USE_MOCK:
        period = get_current_period(now)
        data = [
            dict(item)
            for item in tdx_crawler.MOCK_DATA.get(period, tdx_crawler.MOCK_DATA["off_peak"])
        ]
        for item in data:
            item["period"] = period
        return data

    token = token or tdx_crawler.get_tdx_token()
    return fetch_tdx_traffic_nonnegative(token)


def build_route_xml(traffic_data: list, timestamp=None) -> str:
    routes = []
    flows = []

    for direction, turns in TURN_MOVEMENTS.items():
        for turn, (from_edge, to_edge) in turns.items():
            route_id = f"route_{direction}_{turn}"
            routes.append(f'    <route id="{route_id}" edges="{from_edge} {to_edge}"/>')

    for entry in traffic_data:
        direction = entry.get("direction")
        volume_per_hour = entry.get("volume_per_hour", 0)

        if direction not in TURN_MOVEMENTS or volume_per_hour <= 0:
            continue

        for turn, (from_edge, to_edge) in TURN_MOVEMENTS[direction].items():
            turn_vph = volume_per_hour * TURN_RATIOS[turn]
            if turn_vph < 1:
                continue

            period = round(3600 / turn_vph, 4)
            route_id = f"route_{direction}_{turn}"
            flow_id = f"flow_{direction}_{turn}"

            flows.append(
                f'    <flow id="{flow_id}" route="{route_id}" '
                f'begin="0" end="{SIM_DURATION}" '
                f'period="{period}" '
                f'departLane="best" departSpeed="max"/>'
            )

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<routes>\n'
    xml += '    <!-- 車輛類型 -->\n'
    xml += '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>\n\n'
    xml += '    <!-- 路線定義 -->\n'
    xml += '\n'.join(routes) + '\n\n'
    xml += '    <!-- 車流（依 TDX/Mock 資料） -->\n'
    xml += '\n'.join(flows) + '\n'
    xml += '</routes>\n'
    return xml


def save_route_history(traffic_data: list, timestamp=None) -> str:
    timestamp = timestamp or datetime.datetime.now()
    date_key = timestamp.strftime("%Y%m%d")
    time_key = timestamp.strftime("%Y%m%d_%H%M%S")
    intersection_slug = safe_filename(INTERSECTION_NAME)

    out_dir = os.path.join(HISTORY_DIR, intersection_slug, date_key)
    os.makedirs(out_dir, exist_ok=True)

    route_file = os.path.join(out_dir, f"{time_key}_{intersection_slug}.rou.xml")

    with open(route_file, "w", encoding="utf-8") as f:
        f.write(build_route_xml(traffic_data, timestamp=timestamp))

    return route_file


def capture_once(token=None):
    now = datetime.datetime.now()
    data = fetch_traffic_snapshot(token=token)

    if not data:
        print("[歷史] Live 資料為空，本次不存檔")
        return None

    print(f"[歷史] 本次車流資料：{data}")
    if not has_valid_route_data(data):
        print("[歷史] 沒有可轉成 SUMO flow 的正流量資料，本次不存檔")
        return None

    out_file = save_route_history(data, timestamp=now)
    print(f"[歷史] 已存入 {out_file}")
    return out_file


def run_scheduler(interval_seconds: int = 20) -> None:
    print(
        f"[排程] 每 {interval_seconds} 秒更新一次，"
        f"路口：{INTERSECTION_NAME}，資料夾：{HISTORY_DIR}"
    )
    print("[排程] 按 Ctrl+C 停止")

    token = get_token_or_exit()

    while True:
        started_at = time.time()
        try:
            capture_once(token=token)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in (401, 403):
                print("[排程] Token 可能過期，重新取得 TDX token 後重試")
                token = get_token_or_exit()
                capture_once(token=token)
            else:
                print(f"[排程] HTTP 錯誤，本輪跳過：{exc}")
        except Exception as exc:
            print(f"[排程] 發生錯誤，本輪跳過：{exc}")

        elapsed = time.time() - started_at
        time.sleep(max(0, interval_seconds - elapsed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule TDX traffic route XML capture.")
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Seconds between captures. Default: 20.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one route XML snapshot and exit.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock traffic data from tdx_crawler.py without calling TDX.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mock:
        tdx_crawler.USE_MOCK = True

    if args.once:
        capture_once(token=get_token_or_exit())
    else:
        run_scheduler(interval_seconds=args.interval)

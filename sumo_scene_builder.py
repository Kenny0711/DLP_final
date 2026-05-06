"""
SUMO 場景建構器
把 tdx_crawler 的車流資料轉換成 SUMO 模擬所需的三個檔案：
  nets/cross.net.xml   → 路網（十字路口）
  nets/taipei.rou.xml  → 車輛路由（根據車流量）
  nets/taipei.sumocfg  → 模擬設定檔
"""

import os
import subprocess
from tdx_crawler import get_traffic_data

# ──────────────────────────────────────────
# 路徑設定
# ──────────────────────────────────────────
SUMO_HOME    = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
NETCONVERT   = os.path.join(SUMO_HOME, "bin", "netconvert.exe")
NETS_DIR     = os.path.join(os.path.dirname(__file__), "nets")
NET_FILE     = os.path.join(NETS_DIR, "cross.net.xml")
NOD_FILE     = os.path.join(NETS_DIR, "cross.nod.xml")
EDG_FILE     = os.path.join(NETS_DIR, "cross.edg.xml")
ROUTE_FILE   = os.path.join(NETS_DIR, "taipei.rou.xml")
CONFIG_FILE  = os.path.join(NETS_DIR, "taipei.sumocfg")

SIM_DURATION = 3600   # 模擬秒數（1 小時）

# 每個進入方向的三種轉向（台灣右側通行）
TURN_MOVEMENTS = {
    "EB": {                                 # 東行（由西入）
        "straight": ("WE", "CE"),           # 直行 → 東出
        "left":     ("WE", "CN"),           # 左轉 → 北出
        "right":    ("WE", "CS"),           # 右轉 → 南出
    },
    "WB": {                                 # 西行（由東入）
        "straight": ("EW", "CW"),           # 直行 → 西出
        "left":     ("EW", "CS"),           # 左轉 → 南出
        "right":    ("EW", "CN"),           # 右轉 → 北出
    },
    "NB": {                                 # 北行（由南入）
        "straight": ("SN", "CN"),           # 直行 → 北出
        "left":     ("SN", "CW"),           # 左轉 → 西出
        "right":    ("SN", "CE"),           # 右轉 → 東出
    },
    "SB": {                                 # 南行（由北入）
        "straight": ("NS", "CS"),           # 直行 → 南出
        "left":     ("NS", "CE"),           # 左轉 → 東出
        "right":    ("NS", "CW"),           # 右轉 → 西出
    },
}

# 台灣城市路口典型轉向比例
TURN_RATIOS = {"straight": 0.70, "left": 0.10, "right": 0.20}


# ──────────────────────────────────────────
# Step 1：產生路網
# ──────────────────────────────────────────
def write_node_file():
    """產生 .nod.xml（節點描述）"""
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n'
    xml += '    <node id="center" x="0"    y="0"    type="traffic_light"/>\n'
    xml += '    <node id="west"   x="-200" y="0"    type="dead_end"/>\n'
    xml += '    <node id="east"   x="200"  y="0"    type="dead_end"/>\n'
    xml += '    <node id="north"  x="0"    y="200"  type="dead_end"/>\n'
    xml += '    <node id="south"  x="0"    y="-200" type="dead_end"/>\n'
    xml += '</nodes>\n'
    with open(NOD_FILE, "w", encoding="utf-8") as f:
        f.write(xml)


def write_edge_file():
    """產生 .edg.xml（邊描述）"""
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n'
    # 進入路口
    xml += '    <edge id="WE" from="west"   to="center" numLanes="3" speed="13.89"/>\n'
    xml += '    <edge id="EW" from="east"   to="center" numLanes="3" speed="13.89"/>\n'
    xml += '    <edge id="SN" from="south"  to="center" numLanes="3" speed="13.89"/>\n'
    xml += '    <edge id="NS" from="north"  to="center" numLanes="3" speed="13.89"/>\n'
    # 離開路口
    xml += '    <edge id="CE" from="center" to="east"   numLanes="3" speed="13.89"/>\n'
    xml += '    <edge id="CW" from="center" to="west"   numLanes="3" speed="13.89"/>\n'
    xml += '    <edge id="CN" from="center" to="north"  numLanes="3" speed="13.89"/>\n'
    xml += '    <edge id="CS" from="center" to="south"  numLanes="3" speed="13.89"/>\n'
    xml += '</edges>\n'
    with open(EDG_FILE, "w", encoding="utf-8") as f:
        f.write(xml)


def build_network():
    os.makedirs(NETS_DIR, exist_ok=True)

    if os.path.exists(NET_FILE):
        print(f"[路網] 已存在：{NET_FILE}，跳過產生")
        return

    print("[路網] 產生節點與邊描述檔...")
    write_node_file()
    write_edge_file()

    print("[路網] 使用 netconvert 產生十字路口路網...")
    cmd = [
        NETCONVERT,
        f"--node-files={NOD_FILE}",
        f"--edge-files={EDG_FILE}",
        f"--output-file={NET_FILE}",
        "--no-turnarounds",
        "--tls.default-type=static",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[錯誤] netconvert 失敗：\n{result.stderr}")
        raise RuntimeError("netconvert failed")
    print(f"[路網] 產生完成：{NET_FILE}")


# ──────────────────────────────────────────
# Step 2：產生路由（.rou.xml）
# ──────────────────────────────────────────
def build_routes(traffic_data: list):
    print(f"[路由] 根據車流量產生路由檔（含左轉/右轉）...")

    routes = []
    flows  = []

    for entry in traffic_data:
        direction = entry["direction"]
        vph       = entry["volume_per_hour"]

        if direction not in TURN_MOVEMENTS or vph <= 0:
            continue

        for turn, (from_edge, to_edge) in TURN_MOVEMENTS[direction].items():
            turn_vph = vph * TURN_RATIOS[turn]
            if turn_vph < 1:
                continue
            period   = round(3600 / turn_vph, 4)
            route_id = f"route_{direction}_{turn}"
            flow_id  = f"flow_{direction}_{turn}"

            routes.append(
                f'    <route id="{route_id}" edges="{from_edge} {to_edge}"/>'
            )
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

    with open(ROUTE_FILE, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[路由] 產生完成：{ROUTE_FILE}")


# ──────────────────────────────────────────
# Step 3：產生模擬設定（.sumocfg）
# ──────────────────────────────────────────
def build_config():
    print(f"[設定] 產生模擬設定檔...")

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<configuration>\n'
    xml += '    <input>\n'
    xml += f'        <net-file value="cross.net.xml"/>\n'
    xml += f'        <route-files value="taipei.rou.xml"/>\n'
    xml += '    </input>\n'
    xml += '    <time>\n'
    xml += f'        <begin value="0"/>\n'
    xml += f'        <end value="{SIM_DURATION}"/>\n'
    xml += '    </time>\n'
    xml += '    <report>\n'
    xml += '        <verbose value="false"/>\n'
    xml += '        <no-step-log value="true"/>\n'
    xml += '    </report>\n'
    xml += '</configuration>\n'

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[設定] 產生完成：{CONFIG_FILE}")


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────
def build_scene(period: str = "morning_peak"):
    print(f"\n{'='*50}")
    print(f" 建構 SUMO 場景：忠孝東路 x 敦化南路（{period}）")
    print(f"{'='*50}")

    # 1. 抓車流資料
    print(f"\n[車流] 取得 {period} 車流資料...")
    traffic_data = get_traffic_data(period)
    for d in traffic_data:
        print(f"  {d['direction']}: {d['volume_per_hour']} 輛/小時")

    # 2. 建路網
    build_network()

    # 3. 建路由
    build_routes(traffic_data)

    # 4. 建設定
    build_config()

    print(f"\n✓ 場景建構完成！")
    print(f"  路網：{NET_FILE}")
    print(f"  路由：{ROUTE_FILE}")
    print(f"  設定：{CONFIG_FILE}")
    print(f"\n執行模擬：python run_simulation.py")
    print(f"視覺化：  python run_simulation.py --gui")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="morning_peak",
                        choices=["morning_peak", "off_peak", "evening_peak"],
                        help="車流時段")
    args = parser.parse_args()
    build_scene(args.period)

"""
TDX 車流量爬蟲
忠孝東路 x 敦化南路 路口

USE_MOCK = True  → 使用模擬資料（不需帳號）
USE_MOCK = False → 使用真實 TDX API（需填入 CLIENT_ID / CLIENT_SECRET）
"""

import json
import os
import sys
import time
import datetime
import requests

# ──────────────────────────────────────────
# 設定區
# ──────────────────────────────────────────
USE_MOCK = False           # 切換 Mock / 真實 TDX

# 從環境變數讀取 TDX 憑證（或直接填入字串，但請勿上傳到公開儲存庫）
CLIENT_ID     = os.environ.get("TDX_CLIENT_ID",     "YOUR_CLIENT_ID")
CLIENT_SECRET = os.environ.get("TDX_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

TDX_AUTH_URL  = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
TDX_API_BASE  = "https://tdx.transportdata.tw/api/basic"

# 忠孝東路 x 敦化南路 附近的 VD 感測器關鍵字
TARGET_ROADS  = ["忠孝東路", "敦化南路"]

# 快取檔案路徑（與 tdx_crawler.py 同目錄）
_HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE  = os.path.join(_HERE, "traffic_cache.json")

# 記憶體內 metadata 快取（同一個 Python session 只打一次靜態 API）
_metadata_cache: dict = {}

# 城市幹道各小時相對於全天平均的流量比例（典型值）
HOURLY_FACTORS = {
     0: 0.15,  1: 0.10,  2: 0.08,  3: 0.07,  4: 0.10,
     5: 0.25,  6: 0.55,  7: 1.20,  8: 1.35,  9: 1.05,
    10: 0.90, 11: 0.85, 12: 0.90, 13: 0.80, 14: 0.80,
    15: 0.95, 16: 1.15, 17: 1.40, 18: 1.50, 19: 1.20,
    20: 0.95, 21: 0.80, 22: 0.65, 23: 0.40,
}

# 各時段的代表因子（早峰 07–09、離峰 09–17、晚峰 17–19 的平均）
PERIOD_FACTORS = {
    "morning_peak": 1.275,   # (1.20 + 1.35) / 2
    "off_peak":     0.856,   # avg(1.05,0.90,0.85,0.90,0.80,0.80,0.95,1.15)
    "evening_peak": 1.450,   # (1.40 + 1.50) / 2
}

# ──────────────────────────────────────────
# Mock 資料（模擬三個時段）
# ──────────────────────────────────────────
MOCK_DATA = {
    "morning_peak": [   # 早尖峰 07:00–09:00
        {"direction": "EB", "volume_per_hour": 1200},
        {"direction": "WB", "volume_per_hour": 1000},
        {"direction": "NB", "volume_per_hour": 800},
        {"direction": "SB", "volume_per_hour": 700},
    ],
    "off_peak": [        # 離峰 09:00–17:00
        {"direction": "EB", "volume_per_hour": 550},
        {"direction": "WB", "volume_per_hour": 480},
        {"direction": "NB", "volume_per_hour": 380},
        {"direction": "SB", "volume_per_hour": 320},
    ],
    "evening_peak": [    # 晚尖峰 17:00–19:00
        {"direction": "EB", "volume_per_hour": 1400},
        {"direction": "WB", "volume_per_hour": 1300},
        {"direction": "NB", "volume_per_hour": 950},
        {"direction": "SB", "volume_per_hour": 880},
    ],
}


# ──────────────────────────────────────────
# TDX 認證
# ──────────────────────────────────────────
def get_tdx_token() -> str:
    resp = requests.post(TDX_AUTH_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


# ──────────────────────────────────────────
# Step A：靜態 VD 資料（路名 + 方向）
# ──────────────────────────────────────────
def fetch_vd_metadata(token: str, force_refresh: bool = False) -> dict:
    """
    取得台北市所有 VD 感測器的靜態資料，
    回傳 {VDID: {"road_name": str, "bearing": str}} 的 dict。
    只保留 TARGET_ROADS 附近的感測器。
    """
    global _metadata_cache
    if _metadata_cache and not force_refresh:
        return _metadata_cache

    url = f"{TDX_API_BASE}/v2/Road/Traffic/VD/City/Taipei?$format=JSON"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    raw = resp.json()

    vd_list = raw if isinstance(raw, list) else raw.get("VDs", [])
    dir_map  = {"E": "EB", "W": "WB", "N": "NB", "S": "SB"}
    metadata = {}

    for vd in vd_list:
        vdid      = vd.get("VDID", "")
        road_name = vd.get("RoadName", "")          # RoadName 在 VD 最上層
        if not any(r in road_name for r in TARGET_ROADS):
            continue
        for link in vd.get("DetectionLinks", []):
            bearing = link.get("Bearing", "")       # Bearing 在 DetectionLinks 裡
            if bearing in dir_map:
                metadata[vdid] = {
                    "road_name": road_name,
                    "bearing":   bearing,
                }
    print(f"[靜態] 找到 {len(metadata)} 個目標路口 VD：{list(metadata.keys())}")
    _metadata_cache = metadata   # 存進記憶體，下次不用重打
    return metadata


# ──────────────────────────────────────────
# Step B：即時車流（用 VDID join 靜態資料）
# ──────────────────────────────────────────
def fetch_tdx_traffic(token: str) -> list:
    """
    先抓靜態 VD metadata（路名+方向），再抓 Live 流量，用 VDID join。
    回傳與 mock 相同格式的 list。
    """
    metadata = fetch_vd_metadata(token)
    if not metadata:
        print("[警告] 找不到目標路口的 VD 感測器，確認 TARGET_ROADS 設定是否正確")
        return []

    time.sleep(1)   # 避免連續打 API 觸發 429
    url = f"{TDX_API_BASE}/v2/Road/Traffic/Live/VD/City/Taipei?$format=JSON"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    raw = resp.json()

    live_list = raw.get("VDLives", [])
    dir_map   = {"E": "EB", "W": "WB", "N": "NB", "S": "SB"}

    # 每個方向累積所有感測器的數值，最後取平均
    dir_volumes = {"EB": [], "WB": [], "NB": [], "SB": []}

    for vd in live_list:
        vdid = vd.get("VDID", "")
        if vdid not in metadata:
            continue
        direction = dir_map[metadata[vdid]["bearing"]]

        total_volume = 0
        for link in vd.get("LinkFlows", []):
            for lane in link.get("Lanes", []):
                for v in lane.get("Vehicles", []):
                    total_volume += v.get("Volume", 0)

        dir_volumes[direction].append(total_volume * 12)   # 輛/5min → 輛/hr

    # NB（北行）無感測器 → 用 SB 平均值估算
    if not dir_volumes["NB"] and dir_volumes["SB"]:
        avg_sb = sum(dir_volumes["SB"]) / len(dir_volumes["SB"])
        dir_volumes["NB"] = [int(avg_sb * 0.9)]   # 北行通常略少於南行
        print("[資料] NB 無感測器，以 SB 平均值 × 0.9 估算")

    results = []
    for direction, volumes in dir_volumes.items():
        if volumes:
            avg = int(sum(volumes) / len(volumes))
            results.append({"direction": direction, "volume_per_hour": avg})
        else:
            print(f"[警告] {direction} 無任何感測器資料，跳過")

    return results


# ──────────────────────────────────────────
# 快取：讀 / 寫 traffic_cache.json
# ──────────────────────────────────────────
def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def capture_period(period: str) -> list:
    """
    抓取**當下** Live 車流量，存入快取，並回傳資料。
    用法（在對應時段手動執行）：
        python tdx_crawler.py --capture morning_peak   # 早上 07–09 點執行
        python tdx_crawler.py --capture off_peak        # 白天 09–17 點執行
        python tdx_crawler.py --capture evening_peak    # 傍晚 17–19 點執行
    """
    today = datetime.date.today().isoformat()
    print(f"[快取] 抓取 {today} {period} 的即時車流...")

    token   = get_tdx_token()
    results = fetch_tdx_traffic(token)

    if not results:
        print("[快取] Live 資料為空，放棄存檔")
        return []

    cache = _load_cache()
    cache.setdefault(today, {})[period] = results
    _save_cache(cache)
    print(f"[快取] 已存入 {CACHE_FILE}  ({today} / {period})")
    return results


def estimate_period_from_live(token: str) -> dict:
    """
    抓一次 Live 車流，推算三個時段的估計值。

    原理：
      1. 取得現在的 Live 車流（current_vph）
      2. 查表目前時刻的 hourly factor → 反推全天基準值
         base = current_vph / current_factor
      3. 各時段估計 = base × period_factor

    回傳 {"morning_peak": [...], "off_peak": [...], "evening_peak": [...]}
    """
    live = fetch_tdx_traffic(token)
    if not live:
        return {}

    now_hour      = datetime.datetime.now().hour
    current_factor = HOURLY_FACTORS.get(now_hour, 1.0)
    print(f"[估算] 現在 {now_hour:02d}:XX，流量因子 = {current_factor}")

    result = {}
    for period, p_factor in PERIOD_FACTORS.items():
        scale = p_factor / current_factor
        period_data = []
        for entry in live:
            estimated_vph = max(1, int(entry["volume_per_hour"] * scale))
            period_data.append({
                "direction":       entry["direction"],
                "volume_per_hour": estimated_vph,
            })
        result[period] = period_data
        print(f"[估算] {period:15s} (×{scale:.2f})：{period_data}")

    return result


def get_cached_traffic(period: str) -> list:
    """
    取得指定時段的車流資料，優先序：
      1. 快取中昨天的資料
      2. 快取中最近有該 period 的資料
      3. 即時 Live → 時段估算（會把三個時段一起存入今天的快取）
      4. Fallback 到 Mock
    """
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    today     = datetime.date.today().isoformat()
    cache     = _load_cache()

    # 優先用昨天或已有快取
    for date_key in [yesterday] + sorted(cache.keys(), reverse=True):
        if date_key in cache and period in cache[date_key]:
            data = cache[date_key][period]
            print(f"[快取] 讀取 {date_key} / {period}：{data}")
            return data

    # 沒有快取 → 用 Live 估算三個時段並存入今天
    print(f"[快取] 無 {period} 快取，從 Live 估算三個時段...")
    token      = get_tdx_token()
    estimated  = estimate_period_from_live(token)
    if estimated:
        cache.setdefault(today, {}).update(estimated)
        _save_cache(cache)
        print(f"[快取] 已將估算結果存入 {today}")
        if period in estimated:
            return estimated[period]

    print(f"[快取] 估算失敗，使用 Mock 資料")
    return list(MOCK_DATA[period])


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────
def get_traffic_data(period: str = "morning_peak") -> list:
    """
    回傳指定時段的車流資料。

    USE_MOCK = True  → 回傳 Mock 固定數字
    USE_MOCK = False → 讀取本地快取（traffic_cache.json）；
                       若無快取則 fallback 到 Mock
    """
    if USE_MOCK:
        data = [dict(d) for d in MOCK_DATA.get(period, MOCK_DATA["morning_peak"])]
        for d in data:
            d["period"] = period
        return data
    else:
        return get_cached_traffic(period)


# ──────────────────────────────────────────
if __name__ == "__main__":
    # 用法：
    #   python tdx_crawler.py                       → 印出三個時段的資料
    #   python tdx_crawler.py --capture morning_peak → 抓 Live 存入快取
    #   python tdx_crawler.py --capture-all          → 三個時段都存（測試用）

    if "--capture-all" in sys.argv:
        # 把目前 Live 讀數存為三個時段（數值相同，僅供功能測試）
        token   = get_tdx_token()
        results = fetch_tdx_traffic(token)
        today   = datetime.date.today().isoformat()
        cache   = _load_cache()
        for p in ["morning_peak", "off_peak", "evening_peak"]:
            cache.setdefault(today, {})[p] = results
        _save_cache(cache)
        print(f"[快取] 已將 Live 資料存為三個時段（{today}）")
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif "--capture" in sys.argv:
        idx    = sys.argv.index("--capture")
        period = sys.argv[idx + 1]
        data   = capture_period(period)
        print(json.dumps(data, ensure_ascii=False, indent=2))

    else:
        print(f"USE_MOCK = {USE_MOCK}")
        for period in ["morning_peak", "off_peak", "evening_peak"]:
            print(f"\n{'='*40}")
            print(f"時段：{period}")
            print('='*40)
            data = get_traffic_data(period)
            print(json.dumps(data, ensure_ascii=False, indent=2))

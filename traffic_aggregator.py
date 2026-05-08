"""
TDX 車流資料匯總器
從 traffic_cache.json 讀取多天收集的資料，
計算每個時段（早峰/離峰/晚峰）每個方向的平均 VPH，
輸出 aggregate_traffic.json。

執行方式：
    python traffic_aggregator.py                  # 匯總所有可用資料
    python traffic_aggregator.py --min-days 7     # 至少 7 天才輸出
    python traffic_aggregator.py --show           # 只顯示統計，不寫入檔案
"""

import argparse
import datetime
import json
import os
from collections import defaultdict
from statistics import mean, stdev

# ──────────────────────────────────────────
# 路徑設定
# ──────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE     = os.path.join(_HERE, "traffic_cache.json")
AGGREGATE_FILE = os.path.join(_HERE, "aggregate_traffic.json")

PERIODS = ["morning_peak", "off_peak", "evening_peak"]


# ──────────────────────────────────────────
# 核心函式
# ──────────────────────────────────────────
def aggregate(min_days: int = 1, dry_run: bool = False) -> dict:
    """
    從 traffic_cache.json 計算各時段各方向的平均 VPH。

    Parameters
    ----------
    min_days : int
        最少需要幾天的資料（不足時印出警告，但仍繼續）。
    dry_run : bool
        True → 只計算不寫入 aggregate_traffic.json。

    Returns
    -------
    dict  匯總結果（與 aggregate_traffic.json 相同結構）
    """
    if not os.path.exists(CACHE_FILE):
        print(f"[錯誤] 找不到 {CACHE_FILE}")
        print(f"       請先執行：python collect_2weeks.py --test")
        raise SystemExit(1)

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache: dict = json.load(f)

    if not cache:
        print("[錯誤] traffic_cache.json 是空的，尚無收集資料。")
        raise SystemExit(1)

    print(f"[匯總] 讀取 {len(cache)} 天資料：{sorted(cache.keys())}")

    # ── 收集各時段各方向的 VPH 清單 ──────────────────────────────
    # dir_vphs[period][direction] = [vph1, vph2, ...]
    dir_vphs: dict = {p: defaultdict(list) for p in PERIODS}
    period_dates: dict = defaultdict(set)

    for date, day_data in cache.items():
        for period in PERIODS:
            if period not in day_data:
                continue
            for entry in day_data[period]:
                direction = entry.get("direction", "")
                vph       = entry.get("volume_per_hour", 0)
                if direction and vph > 0:
                    dir_vphs[period][direction].append(vph)
                    period_dates[period].add(date)

    # ── 檢查樣本數 ──────────────────────────────────────────────
    print()
    for period in PERIODS:
        n = len(period_dates[period])
        status = "✓" if n >= min_days else f"⚠ 不足（需要至少 {min_days} 天）"
        print(f"  {period:<15s}：{n:2d} 天資料  {status}")

    # ── 計算平均值 ──────────────────────────────────────────────
    traffic: dict = {}
    for period in PERIODS:
        entries = []
        for direction in sorted(dir_vphs[period].keys()):
            vphs  = dir_vphs[period][direction]
            avg   = int(mean(vphs))
            sigma = round(stdev(vphs), 1) if len(vphs) > 1 else 0.0
            entries.append({
                "direction":       direction,
                "volume_per_hour": avg,
                "samples":         len(vphs),
                "std":             sigma,
            })
        traffic[period] = entries

    # ── 組裝結果 ──────────────────────────────────────────────
    all_dates = sorted(cache.keys())
    result = {
        "generated_at":  datetime.datetime.now().isoformat(timespec="seconds"),
        "days_collected": all_dates,
        "date_range":    f"{all_dates[0]} ～ {all_dates[-1]}",
        "sample_counts": {p: len(period_dates[p]) for p in PERIODS},
        "traffic":       traffic,
    }

    # ── 印出摘要 ──────────────────────────────────────────────
    _print_summary(result)

    # ── 寫入檔案 ──────────────────────────────────────────────
    if not dry_run:
        with open(AGGREGATE_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[完成] 已寫入 {AGGREGATE_FILE}")
        print(f"[完成] 下一步：python tdx_crawler.py  （確認自動載入匯總資料）")
    else:
        print("\n[dry-run] 未寫入檔案。")

    return result


def _print_summary(result: dict) -> None:
    """印出易讀的匯總摘要。"""
    print(f"\n{'='*55}")
    print(f" 車流資料匯總結果")
    print(f"{'='*55}")
    print(f" 日期範圍：{result['date_range']}")
    print(f" 收集天數：{len(result['days_collected'])} 天")
    print()

    for period in PERIODS:
        entries = result["traffic"].get(period, [])
        n_days  = result["sample_counts"].get(period, 0)
        print(f" {period}  （{n_days} 天，共 {sum(e['samples'] for e in entries)} 筆）")
        for e in entries:
            bar = "█" * (e["volume_per_hour"] // 100)
            print(f"   {e['direction']}：{e['volume_per_hour']:5d} vph  "
                  f"σ={e['std']:6.1f}  {bar}")
        print()

    print(f"{'='*55}")


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="從 traffic_cache.json 匯總 14 天車流資料"
    )
    parser.add_argument(
        "--min-days", type=int, default=1,
        help="要求至少 N 天資料才產生匯總（預設 1）"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="只顯示統計，不寫入 aggregate_traffic.json"
    )
    args = parser.parse_args()

    aggregate(min_days=args.min_days, dry_run=args.show)

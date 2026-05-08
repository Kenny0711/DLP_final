"""
14 天 TDX 車流資料自動收集器
每天在 08:00 / 13:00 / 18:00 / 22:00 各抓一次 Live 車流，
存入 traffic_cache.json。

API 消耗：4 次/天 × 14 天 = 56 次（遠低於免費會員 500/天上限）

執行方式：
    python collect_2weeks.py             # 從現在開始，持續 14 天
    python collect_2weeks.py --days 7    # 只收集 7 天
    python collect_2weeks.py --test      # 測試：立刻各抓一次，不等待
"""

import argparse
import datetime
import time

from tdx_crawler import capture_period

# ──────────────────────────────────────────
# 每天收集的時段排程
# 22:00 歸類為 off_peak（夜間離峰），與 13:00 合併後平均，
# 讓 off_peak 涵蓋「白天」和「夜間」兩種低流量樣本。
# ──────────────────────────────────────────
SCHEDULE = [
    ("08:00", "morning_peak"),
    ("13:00", "off_peak"),
    ("18:00", "evening_peak"),
    ("22:00", "off_peak"),
]


def _next_occurrence(time_str: str, from_dt: datetime.datetime) -> datetime.datetime:
    """回傳 from_dt 之後第一次出現 HH:MM 的 datetime。"""
    h, m = map(int, time_str.split(":"))
    candidate = from_dt.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= from_dt:
        candidate += datetime.timedelta(days=1)
    return candidate


def _build_event_list(days: int) -> list:
    """
    預先計算 days × len(SCHEDULE) 個收集事件（datetime, period），
    從「現在」開始往後 days 天，依時間排序。
    """
    start = datetime.datetime.now()
    events = []
    for d in range(days):
        date = (start + datetime.timedelta(days=d)).date()
        for time_str, period in SCHEDULE:
            h, m = map(int, time_str.split(":"))
            dt = datetime.datetime(date.year, date.month, date.day, h, m)
            events.append((dt, period, time_str))
    events.sort(key=lambda x: x[0])
    return events


def run_collection(days: int = 14) -> None:
    start_date = datetime.date.today()
    end_date   = start_date + datetime.timedelta(days=days - 1)
    total      = days * len(SCHEDULE)

    print(f"{'='*55}")
    print(f" 14 天 TDX 車流資料收集器")
    print(f"{'='*55}")
    print(f" 收集期間：{start_date} ～ {end_date}（{days} 天）")
    print(f" 每天時段：{', '.join(t for t, _ in SCHEDULE)}")
    print(f" 預計次數：{total} 次 API 呼叫")
    print(f" 請讓此視窗持續開啟，或使用 Windows 工作排程器。")
    print(f"{'='*55}\n")

    events    = _build_event_list(days)
    completed = 0

    for event_dt, period, time_str in events:
        now  = datetime.datetime.now()
        wait = (event_dt - now).total_seconds()

        # 超過 5 分鐘的過去事件直接跳過（允許中途重啟腳本）
        if wait < -300:
            print(f"[跳過] {event_dt:%Y-%m-%d %H:%M}  {period:<13s}（已過時）")
            continue

        if wait > 0:
            hours, rem = divmod(int(wait), 3600)
            mins = rem // 60
            print(f"[等待] 下次：{event_dt:%Y-%m-%d %H:%M}  {period:<13s}"
                  f"（{hours:02d}h {mins:02d}m 後）")
            time.sleep(wait)

        # ── 執行抓取 ──────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[抓取] {ts}  →  {period}")
        try:
            data = capture_period(period)
            completed += 1
            print(f"[抓取] ✓  {len(data)} 個方向  "
                  f"（進度 {completed}/{total}）")
        except Exception as exc:
            print(f"[抓取] ✗  失敗：{exc}")

    print(f"\n{'='*55}")
    print(f" 收集完成！共完成 {completed}/{total} 次。")
    print(f" 下一步：python traffic_aggregator.py")
    print(f"{'='*55}\n")


def run_test() -> None:
    """測試模式：立刻對每個時段各抓一次，不等待。"""
    print("[測試] 立刻對所有時段各抓一次...\n")
    seen_periods = set()
    for _, period, time_str in _build_event_list(1):
        if period in seen_periods:
            continue          # 同一個 period 只測試一次
        seen_periods.add(period)
        print(f"[測試] 抓取 {period}（對應時間 {time_str}）")
        try:
            data = capture_period(period)
            for entry in data:
                print(f"         {entry['direction']}: "
                      f"{entry['volume_per_hour']} vph")
        except Exception as exc:
            print(f"[測試] ✗  失敗：{exc}")
    print("\n[測試] 完成。請確認 traffic_cache.json 有今天的記錄。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="14 天 TDX 車流資料自動收集器"
    )
    parser.add_argument(
        "--days", type=int, default=14,
        help="收集天數（預設 14）"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="測試模式：立刻各抓一次，不等待"
    )
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        run_collection(days=args.days)

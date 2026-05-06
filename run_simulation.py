"""
執行 SUMO 模擬
用法：
  python run_simulation.py           # 無 GUI，印出統計
  python run_simulation.py --gui     # 開啟 SUMO 視覺化介面
  python run_simulation.py --period off_peak   # 指定車流時段
"""

import os
import sys
import argparse
import subprocess

# ──────────────────────────────────────────
# 路徑設定
# ──────────────────────────────────────────
SUMO_HOME   = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_BIN    = os.path.join(SUMO_HOME, "bin", "sumo.exe")
SUMO_GUI    = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
NETS_DIR    = os.path.join(os.path.dirname(__file__), "nets")
CONFIG_FILE = os.path.join(NETS_DIR, "taipei.sumocfg")

# 加入 SUMO tools 到 Python 路徑（讓 traci 可以 import）
SUMO_TOOLS  = os.path.join(SUMO_HOME, "tools")
if SUMO_TOOLS not in sys.path:
    sys.path.append(SUMO_TOOLS)


# ──────────────────────────────────────────
# 確認場景檔存在
# ──────────────────────────────────────────
def check_scene():
    if not os.path.exists(CONFIG_FILE):
        print("[錯誤] 找不到場景設定檔，請先執行：")
        print("  python sumo_scene_builder.py")
        sys.exit(1)


# ──────────────────────────────────────────
# 用 TraCI 跑模擬並收集統計
# ──────────────────────────────────────────
def run_with_traci(use_gui: bool = False):
    try:
        import traci
    except ImportError:
        print("[錯誤] 無法 import traci，請確認 SUMO_HOME 設定正確")
        print(f"  目前 SUMO_HOME = {SUMO_HOME}")
        sys.exit(1)

    sumo_binary = SUMO_GUI if use_gui else SUMO_BIN
    sumo_cmd = [sumo_binary, "-c", CONFIG_FILE, "--waiting-time-memory", "3600"]

    print(f"\n{'='*50}")
    print(f" 啟動 SUMO {'(GUI)' if use_gui else '(無 GUI)'}")
    print(f"{'='*50}")
    print(f"設定檔：{CONFIG_FILE}")

    traci.start(sumo_cmd)

    step          = 0
    total_vehicles = 0
    total_waiting  = 0.0
    arrived        = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

            # 每 100 步印一次進度
            if step % 100 == 0:
                vehicles_now = traci.vehicle.getIDCount()
                print(f"  步驟 {step:4d}s │ 路上車輛：{vehicles_now:4d} 輛")

            # 累計統計
            for vid in traci.vehicle.getIDList():
                total_waiting += traci.vehicle.getWaitingTime(vid)

            arrived += traci.simulation.getArrivedNumber()

    except KeyboardInterrupt:
        print("\n[中斷] 使用者中止模擬")
    finally:
        traci.close()

    # 印出統計
    print(f"\n{'='*50}")
    print(f" 模擬結果統計")
    print(f"{'='*50}")
    print(f"  模擬時長：       {step} 秒")
    print(f"  總通過車輛數：   {arrived} 輛")
    print(f"  累計等待時間：   {total_waiting:.1f} 秒")
    if arrived > 0:
        print(f"  平均等待時間：   {total_waiting / arrived:.2f} 秒/輛")
    print(f"{'='*50}\n")


# ──────────────────────────────────────────
# 直接呼叫 sumo（不用 traci，最簡單）
# ──────────────────────────────────────────
def run_simple(use_gui: bool = False):
    sumo_binary = SUMO_GUI if use_gui else SUMO_BIN
    cmd = [sumo_binary, "-c", CONFIG_FILE]

    print(f"\n[執行] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[錯誤] SUMO 執行失敗")
    else:
        print("[完成] 模擬結束")


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",    action="store_true", help="開啟 SUMO 視覺化介面")
    parser.add_argument("--period", default="morning_peak",
                        choices=["morning_peak", "off_peak", "evening_peak"],
                        help="若需重新建場景，指定時段")
    parser.add_argument("--rebuild", action="store_true", help="重新建構場景")
    args = parser.parse_args()

    # 若需要重建場景
    if args.rebuild or not os.path.exists(CONFIG_FILE):
        from sumo_scene_builder import build_scene
        build_scene(args.period)

    check_scene()

    # 嘗試用 traci 跑（有統計資料），失敗則用簡單模式
    try:
        run_with_traci(use_gui=args.gui)
    except Exception as e:
        print(f"[警告] TraCI 模式失敗：{e}")
        print("[切換] 使用簡單模式...")
        run_simple(use_gui=args.gui)

"""
Auto-detect and add SUMO binary to PATH.
自動找 SUMO 執行檔並加入 PATH，讓 traci.start(["sumo",...]) 可以找到它。
Import this before importing sumo_env or traci.
"""
import glob
import os
import shutil


def ensure_sumo_in_path() -> None:
    if shutil.which("sumo"):
        return

    candidates = glob.glob(
        os.path.join(os.path.expanduser("~"), "**", "sumo", "bin", "sumo"),
        recursive=True,
    )
    # Also check SUMO_HOME env var
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidates = [os.path.join(sumo_home, "bin", "sumo")] + candidates

    for path in candidates:
        if os.path.isfile(path):
            os.environ["PATH"] = os.path.dirname(path) + ":" + os.environ.get("PATH", "")
            print(f"[sumo_path] Added to PATH: {os.path.dirname(path)}")
            return

    print("[sumo_path] WARNING: sumo binary not found. "
          "Set SUMO_HOME or add sumo/bin to PATH manually.")


ensure_sumo_in_path()

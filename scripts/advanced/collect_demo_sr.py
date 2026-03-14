"""Measure replay success rate for libero-10 tasks.

Run from project root:
    python scripts/advanced/collect_demo_sr.py

Success rate = num_success / (num_success + num_skip)
where num_skip = tot_give_up (demos that timed out / failed)
"""
import os
import sys
import subprocess
from pathlib import Path

PYTHON = sys.executable
LOG_DIR = Path("claude/log")
OUT_DIR = Path("claude/out/sr_test")
NUM_SUCCESS = 10

LIBERO_10_TASKS = [
    "libero.pick_alphabet_soup",
    "libero.pick_bbq_sauce",
    "libero.pick_butter",
    "libero.pick_chocolate_pudding",
    "libero.pick_cream_cheese",
    "libero.pick_milk",
    "libero.orange_juice",
    "libero.pick_salad_dressing",
    "libero.pick_tomato_sauce",
]

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

results = []

for i, task in enumerate(LIBERO_10_TASKS):
    save_dir = str(OUT_DIR / task)
    log_file = str(LOG_DIR / f"sr_{task}.log")

    cmd = (
        f"MUJOCO_GL=egl {PYTHON} scripts/advanced/collect_demo.py "
        f"--sim=mujoco --task={task} --num_envs=1 --headless "
        f"--num_demo_success={NUM_SUCCESS} --run_all "
        f"--custom_save_dir={save_dir}"
    )

    print(f"\n🚀 [{i+1}/{len(LIBERO_10_TASKS)}] {task}", flush=True)
    print(f"📋 Log: {log_file}", flush=True)

    with open(log_file, "w") as f:
        subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

    # Count results from saved files
    success_dir = Path(save_dir) / "success"
    failed_dir = Path(save_dir) / "failed"
    n_success = len(list(success_dir.iterdir())) if success_dir.exists() else 0
    n_failed = len(list(failed_dir.iterdir())) if failed_dir.exists() else 0
    sr = n_success / (n_success + n_failed) * 100 if (n_success + n_failed) > 0 else 0

    results.append((task, n_success, n_failed, sr))
    print(f"✅ success={n_success}  skip={n_failed}  SR={sr:.1f}%", flush=True)

# Summary
print("\n" + "=" * 62)
print("📊 Libero-10 Replay Success Rate Summary")
print("=" * 62)
print(f"{'Task':<40} {'Succ':>6} {'Skip':>6} {'SR':>8}")
print("-" * 62)
for task, n_success, n_failed, sr in results:
    short = task.replace("libero.", "")
    print(f"{short:<40} {n_success:>6} {n_failed:>6} {sr:>7.1f}%")
print("=" * 62)

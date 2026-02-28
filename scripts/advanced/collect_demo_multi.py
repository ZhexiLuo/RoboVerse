import os
from pathlib import Path

import yaml

NUM_DEMO_SUCCESS = 100
LIBERO_SUITES = ["libero", "libero_90"]

with open("dashboard/conf.yml") as f:
    all_tasks = yaml.load(f, Loader=yaml.FullLoader)

tasks_list = []
for suite in LIBERO_SUITES:
    tasks_list.extend(all_tasks["tasks"].get(suite, []))

print(f"📋 Total tasks: {len(tasks_list)}")


def is_done(task: str) -> bool:
    # Check success dir under demo_mujoco/{task}/robot-franka/success/
    success_dir = Path(f"roboverse_demo/demo_mujoco/{task}/robot-franka/success")
    return success_dir.exists() and len(list(success_dir.iterdir())) >= NUM_DEMO_SUCCESS


for task in tasks_list:
    if is_done(task):
        print(f"✅ [skip] {task}")
        continue
    cmd = (
        f"MUJOCO_GL=egl python scripts/advanced/collect_demo.py "
        f"--sim=mujoco --task={task} --num_envs=1 --headless "
        f"--num_demo_success={NUM_DEMO_SUCCESS} --run_unfinished"
    )
    print(f"🚀 {cmd}")
    os.system(cmd)

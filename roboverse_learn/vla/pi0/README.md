# π₀ on RoboVerse

See the unified pipeline doc: [`../../README.md`](../../README.md) (§ π₀ section)

## Quick Reference

```bash
# 1. Install openpi
cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && cd ../..

# 2. Convert demos to LeRobot
cd third_party/openpi
uv run ../../roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py \
  --input-root ../../roboverse_demo --repo-id local/roboverse-libero10 --overwrite

# 3. Train (one-click)
./roboverse_learn/vla/pi0/train_pi0.sh -i ./roboverse_demo -r local/roboverse-libero10

# 4. Serve + eval
cd third_party/openpi && uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_roboverse_lora --policy.dir=<checkpoint>
MUJOCO_GL=egl .venv311/bin/python roboverse_learn/vla/pi0/pi_eval.py \
  --task libero.pick_butter --robot franka --sim mujoco
```

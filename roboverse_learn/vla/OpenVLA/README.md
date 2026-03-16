# OpenVLA Fine-tuning for RoboVerse

See the unified pipeline doc: [`../../README.md`](../../README.md) (§ OpenVLA section)

## Quick Reference

```bash
# 1. Setup (one-time)
bash roboverse_learn/vla/OpenVLA/setup_env.sh

# 2. Convert demos to RLDS
cd roboverse_learn/vla/rlds_utils/roboverse && conda run -n rlds_env tfds build --overwrite

# 3. Fine-tune
conda activate openvla && bash roboverse_learn/vla/OpenVLA/finetune.sh

# 4. Eval
MUJOCO_GL=osmesa conda run -n openvla python roboverse_learn/vla/OpenVLA/vla_eval.py \
  --model_path runs/<checkpoint> --task libero.pick_butter --robot franka --sim mujoco
```

## Key Hyperparams (finetune.sh)

| Param | Default |
|-------|---------|
| `LORA_RANK` | 32 |
| `BATCH_SIZE` | 8 |
| `LEARNING_RATE` | 5e-4 |
| `MAX_STEPS` | 5000 |
| `NPROC_PER_NODE` | 1 (multi-GPU via env var) |

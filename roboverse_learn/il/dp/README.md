# Diffusion Policy — RoboVerse

See the unified pipeline doc: [`../README.md`](../README.md) (§ Diffusion Policy section)

## Quick Reference

```bash
# Batch train + eval (9 libero tasks)
bash roboverse_learn/il/dp/dp_run.sh

# Single task train
export algo_model="ddpm_dit_model"
MUJOCO_GL=egl python roboverse_learn/il/dp/main.py \
  --config-name=dp_runner.yaml task_name=libero.pick_butter \
  dataset_config.zarr_path="./data_policy/libero.pick_butterFrankaL0_obs:joint_pos_act:joint_pos_99.zarr" \
  train_enable=True eval_enable=False
```

## Available Models

| `algo_model` | Backbone | Scheduler |
|---|---|---|
| `ddpm_dit_model` | DiT | DDPM (recommended) |
| `fm_dit_model` | DiT | Flow Matching |
| `ddpm_unet_model` | UNet | DDPM |
| `ddim_unet_model` | UNet | DDIM |
| `fm_unet_model` | UNet | Flow Matching |
| `vita_model` | MLP | Flow Matching |

## Key Files

| File | Description |
|------|-------------|
| `main.py` | Hydra entry point (train + eval) |
| `dp_run.sh` | Batch train+eval script |
| `runner/dp_runner.py` | Core training/eval loop |
| `configs/dp_runner.yaml` | Master Hydra config |
| `configs/model_config/` | Per-model architecture configs |

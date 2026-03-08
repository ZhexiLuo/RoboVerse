# Diffusion Policy — LIBERO-10 Pipeline

## 1. Environment Setup

```bash
# Install diffusion_policy package
cd roboverse_learn/il/utils/diffusion_policy && pip install -e . && cd ../../../../

# Install remaining deps
pip install pandas wandb

# Python runtime (always use venv311 for MuJoCo)
export PYTHON=".venv311/bin/python3"
```

Register a [Weights & Biases](https://wandb.ai) account before training.

---

## 2. Demo Collection

### 2.1 Single task

```bash
MUJOCO_GL=egl .venv311/bin/python3 scripts/advanced/collect_demo.py \
  --task=libero.pick_chocolate_pudding \
  --sim=mujoco --num_envs=1 \
  --num_demo_success=100 --run_unfinished --headless
```

Demos saved to: `roboverse_demo/demo_mujoco/{task}/robot-franka/success/`

### 2.2 Batch collection (all libero-10 tasks)

```bash
MUJOCO_GL=egl .venv311/bin/python3 scripts/advanced/collect_demo_multi.py
```

Task list read from `dashboard/conf.yml`. Skips tasks already at 99+ demos.

---

## 3. Zarr Conversion

```bash
.venv311/bin/python3 roboverse_learn/il/data2zarr_dp.py \
  --task_name="libero.pick_chocolate_puddingFrankaL0_obs:joint_pos_act:joint_pos" \
  --expert_data_num=99 \
  --metadata_dir="roboverse_demo/demo_mujoco/libero.pick_chocolate_pudding/robot-franka/success" \
  --observation_space=joint_pos \
  --action_space=joint_pos
```

Output: `data_policy/{task}FrankaL0_obs:joint_pos_act:joint_pos_99.zarr`

---

## 4. Train + Eval (all 9 tasks)

```bash
# Edit variables at top of script, then run:
bash roboverse_learn/il/dp/dp_run.sh
```

Key variables in `dp_run.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `algo_model` | `ddpm_dit_model` | Algorithm (see §6) |
| `num_epochs` | `100` | Training epochs |
| `eval_max_step` | `350` | Max steps per eval episode |
| `train_enable` | `True` | Run training |
| `eval_enable` | `True` | Run evaluation after training |
| `eval_ckpt_name` | `100` | Checkpoint epoch to evaluate |
| `gpu` | `0` | GPU device id |

Logs: `roboverse_learn/claude/log/dp_{task}.log`
Checkpoints: `info/outputs/DP/{task}/checkpoints/{epoch}.ckpt`
Eval results: `tmp/{task}/diffusion_policy/franka/{ckpt}/final_stats.txt`

---

## 5. Eval Only (from existing checkpoint)

```bash
MUJOCO_GL=egl .venv311/bin/python3 ./roboverse_learn/il/dp/main.py \
  --config-name=dp_runner.yaml \
  task_name=libero.pick_chocolate_pudding \
  train_enable=False eval_enable=True \
  eval_path=./info/outputs/DP/libero.pick_chocolate_pudding/checkpoints/100.ckpt \
  eval_config.eval_args.task=libero.pick_chocolate_pudding \
  eval_config.eval_args.max_step=350 \
  eval_config.eval_args.num_envs=1 \
  eval_config.eval_args.sim=mujoco \
  +eval_config.eval_args.max_demo=99 \
  logging.mode=online
```

---

## 6. Generate Report

After all evals finish:

```bash
# Libero-10 summary report
.venv311/bin/python3 roboverse_learn/il/dp/gen_report.py

# Randomization-level ablation report (single task)
.venv311/bin/python3 roboverse_learn/il/dp/gen_rand_level_report.py \
  --task=libero.pick_chocolate_pudding
```

Reports saved to `roboverse_learn/claude/out/`.

> ⚠️ **Note**: Both report scripts parse `final_stats.txt` under `tmp/{task}/diffusion_policy/franka/*/`.
> They will show `❌ not found` for tasks whose `tmp/` directory is absent.

---

## 7. Supported Algorithms

| `$algo_model` | Backbone | Scheduler |
|---|---|---|
| `ddpm_dit_model` | DiT | DDPM |
| `fm_dit_model` | DiT | Flow Matching |
| `vita_model` | MLP | Flow Matching |
| `ddpm_unet_model` | UNet | DDPM |
| `ddim_unet_model` | UNet | DDIM |
| `fm_unet_model` | UNet | Flow Matching |
| `score_model` | UNet | Score-based SDE |

`algo_model` must be set as an **environment variable** (not CLI arg):

```bash
export algo_model=fm_dit_model
bash roboverse_learn/il/dp/dp_run.sh
```

---

## 8. LIBERO-10 Benchmark Results

Task: ddpm_dit_model, 100 epochs, max_step=350, joint_pos obs/action.

| Task | Success Rate | Status |
|------|-------------|--------|
| `libero.orange_juice` | **93.94%** | ✅ |
| `libero.pick_milk` | **86.87%** | ✅ |
| `libero.pick_chocolate_pudding` | **79.80%** | ✅ |
| `libero.pick_cream_cheese` | **73.74%** | ✅ |
| `libero.pick_alphabet_soup` | pending re-eval | ⏳ |
| `libero.pick_bbq_sauce` | pending re-eval | ⏳ |
| `libero.pick_butter` | pending re-eval | ⏳ |
| `libero.pick_salad_dressing` | train failed (zarr empty) | ❌ |
| `libero.pick_tomato_sauce` | train failed (zarr empty) | ❌ |

---

## 9. Common Pitfalls

- **`algo_model` must be env var**, not Hydra CLI arg — set with `export algo_model=...`
- **Use `.venv311/bin/python3`** — bare `python` or system Python will fail on torch/zarr imports
- **`MUJOCO_GL=egl`** required for headless rendering
- **max_step=350** recommended for LIBERO tasks (250 causes heavy timeouts → ~14% success)
- **Checkpoint path**: `./info/outputs/DP/{task_name}/checkpoints/{epoch}.ckpt`

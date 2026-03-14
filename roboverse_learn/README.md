# RoboVerse Learn — IL / VLA Pipeline

Full pipeline for imitation learning and VLA evaluation on Libero-10 benchmark with MuJoCo.

---

## Environment

| Env | Python | Used for |
|-----|--------|---------|
| `.venv311` | 3.11 | DP, ACT, π₀ server |
| `openvla` (conda) | 3.10 | OpenVLA eval, π₀ eval (has pyroki + jax) |

```bash
# .venv311
source /home/zhexi/project/RoboVerse/.venv311/bin/activate

# openvla conda env
conda activate openvla
# Or in tmux (non-login shell):
conda run -n openvla <command>
```

> ⚠️ MuJoCo headless rendering requires `MUJOCO_GL=egl` (server) or `MUJOCO_GL=osmesa` (conda run).

---

## Libero-10 Tasks (9 tasks, 99 demos each)

```
libero.pick_alphabet_soup   libero.pick_bbq_sauce
libero.pick_butter          libero.pick_chocolate_pudding
libero.pick_cream_cheese    libero.pick_milk
libero.orange_juice         libero.pick_salad_dressing
libero.pick_tomato_sauce
```

> `libero.pick_ketchup` removed: only 50 init states, not viable for training.

---

## Step 1 — Collect Demonstrations

> ⚠️ Libero must use `--sim=mujoco`. Trajectory files were recorded in MuJoCo; cross-simulator replay fails.

**Single task:**

```bash
MUJOCO_GL=egl python scripts/advanced/collect_demo.py \
  --sim=mujoco --task=libero.pick_alphabet_soup \
  --num_envs=1 --headless \
  --num_demo_success=100 --cust_name=v1 \
  2>&1 | tee claude/log/collect_alphabet_soup.log
```

**Batch (all libero tasks):**

```bash
python scripts/advanced/collect_demo_multi_zhexi.py
```

**Output**: `roboverse_demo/demo_mujoco/{task}-{cust_name}/robot-franka/success/demo_XXXX/`

---

## Step 2 — Convert to ZARR

```bash
python roboverse_learn/il/data2zarr_dp.py \
  --task_name libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos \
  --expert_data_num 99 \
  --metadata_dir ./roboverse_demo/demo_mujoco/libero.pick_alphabet_soup/robot-franka/success \
  --observation_space joint_pos \
  --action_space joint_pos
```

**Output**: `data_policy/{task}FrankaL0_obs:joint_pos_act:joint_pos_99.zarr`

Or use the combined collect + convert script:

```bash
bash roboverse_learn/il/collect_demo.sh   # edit task_name_set first
```

---

## Diffusion Policy (DP) ✅

Fully verified end-to-end on Libero-10 with MuJoCo.

### Train + Eval (batch, 9 tasks)

```bash
# Edit task list and params at top of script, then:
bash roboverse_learn/il/dp/dp_run.sh
# Log: roboverse_learn/claude/log/dp_{task}.log
```

### Train (single task)

```bash
export algo_model="ddpm_dit_model"   # must be env var, not CLI arg
MUJOCO_GL=egl python roboverse_learn/il/dp/main.py \
  --config-name=dp_runner.yaml \
  task_name=libero.pick_alphabet_soup \
  dataset_config.zarr_path="./data_policy/libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_99.zarr" \
  eval_config.policy_runner.obs.obs_type=joint_pos \
  eval_config.policy_runner.action.action_type=joint_pos \
  eval_config.policy_runner.action.delta=False \
  train_config.training_params.num_epochs=1000 \
  train_enable=True eval_enable=False \
  2>&1 | tee claude/log/dp_train.log
```

**Checkpoint**: `info/outputs/DP/{task}/checkpoints/{epoch}.ckpt`

### Eval (single task)

```bash
export algo_model="ddpm_dit_model"
MUJOCO_GL=egl python roboverse_learn/il/dp/main.py \
  --config-name=dp_runner.yaml \
  task_name=libero.pick_alphabet_soup \
  dataset_config.zarr_path="./data_policy/libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_99.zarr" \
  eval_config.eval_args.task=libero.pick_alphabet_soup \
  eval_config.eval_args.sim=mujoco \
  eval_config.eval_args.num_envs=1 \
  eval_config.eval_args.max_step=350 \
  eval_config.policy_runner.obs.obs_type=joint_pos \
  eval_config.policy_runner.action.action_type=joint_pos \
  eval_config.policy_runner.action.delta=False \
  train_enable=False eval_enable=True \
  eval_path="./info/outputs/DP/libero.pick_alphabet_soup/checkpoints/100.ckpt"
```

**Output**: `tmp/{task}/diffusion_policy/franka/{ckpt_name}/final_stats.txt`

### Available Models

| `algo_model` env var | Backbone | Notes |
|----------------------|----------|-------|
| `ddpm_dit_model` | DiT | Recommended |
| `fm_dit_model` | DiT | Fast convergence |
| `ddpm_unet_model` | UNet | Classic DP |
| `fm_unet_model` | UNet | Flow Matching |
| `ddim_unet_model` | UNet | Fast inference |
| `vita_model` | MLP | Lightest |

---

## ACT (Action Chunking Transformer) ✅

Fully verified end-to-end on Libero-10. 4/9 tasks reach 100% SR at 300 epochs.

### One-time setup

```bash
.venv311/bin/pip install -e roboverse_learn/il/utils/act/detr/
```

### Train + Eval (batch, 9 tasks)

```bash
tmux new-session -d -s act_libero
tmux send-keys -t act_libero \
  'bash roboverse_learn/il/act/act_run_libero.sh 0 2>&1 | tee claude/log/act_libero.log' Enter
```

Hyperparams: `chunk_size=20`, `kl_weight=10`, `hidden_dim=512`, `dim_feedforward=3200`, `lr=1e-5`, `batch=8`

### Train (single task)

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
.venv311/bin/python -m roboverse_learn.il.utils.act.train \
  --task_name libero.pick_alphabet_soup_obs:joint_pos_act:joint_pos_chunk20 \
  --num_episodes 99 \
  --dataset_dir data_policy/libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_99.zarr \
  --policy_class ACT --kl_weight 10 --chunk_size 20 \
  --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 300 --lr 1e-5 --state_dim 9 --seed 42 \
  2>&1 | tee claude/log/act_train.log
```

**Checkpoint**: `info/outputs/ACT/{date}/{time}_{task}_obs:joint_pos_act:joint_pos_chunk20_99/policy_best.ckpt`

> `--ckpt_path` takes the **directory** — `policy_best.ckpt` is appended internally.

### Eval (batch, all 9 tasks, uses fixed 300ep checkpoints)

```bash
CUDA_VISIBLE_DEVICES=0 bash roboverse_learn/il/act/eval_all_fixed.sh \
  2>&1 | tee claude/log/act_eval_fixed.log
# Log per task: claude/log/act_eval_fixed_{task}.log
```

### Eval (single task)

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
.venv311/bin/python -m roboverse_learn.il.act.act_eval_runner \
  --task libero.pick_alphabet_soup \
  --robot franka --sim mujoco \
  --algo act \
  --ckpt_path ./info/outputs/ACT/2026.03.09/03.22.10_libero.pick_alphabet_soup_obs:joint_pos_act:joint_pos_chunk20_99 \
  --headless True \
  --num_eval 20 \
  --temporal_agg True \
  --chunk_size 20 \
  2>&1 | tee claude/log/act_eval.log
```

**Output**: `tmp/act/{task}/{ckpt_name}/success_rate.txt`

### ACT Results (300 epochs, 20 evals each)

| Task | Best Val Loss | Success Rate |
|------|--------------|-------------|
| libero.pick_alphabet_soup | 0.166 | 0% |
| libero.pick_bbq_sauce | 0.185 | 0% |
| libero.pick_butter | 0.179 | **100%** ✅ |
| libero.pick_chocolate_pudding | 0.185 | 0% |
| libero.pick_cream_cheese | 0.209 | **100%** ✅ |
| libero.pick_milk | ~0.210 | 0% |
| libero.orange_juice | ~0.190 | 0% |
| libero.pick_salad_dressing | 0.182 | **100%** ✅ |
| libero.pick_tomato_sauce | ~0.300 | **100%** ✅ |

wandb project: `RoboVerse_ACT`

---

## π₀ (pi0) — LoRA Fine-tuning + Eval

LoRA fine-tune π₀/π₀.₅ on existing libero-10 demos (9-dim joint_pos), then evaluate.

> ⚠️ `pi0_libero` pre-trained checkpoint outputs 7-dim EEF delta — incompatible with `pi_eval.py` (expects 9-dim joint_pos).
> Must LoRA fine-tune with RoboVerse data to get a compatible `pi0_roboverse_lora` / `pi05_roboverse_lora` checkpoint.

### Step 1 — One-time setup

```bash
# Install openpi (in third_party/openpi, uses uv)
cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && cd ../..

# Install openpi_client in RoboVerse eval env
.venv311/bin/pip install openpi_client

# Install lerobot + imageio (for data conversion, in openpi's uv env)
cd third_party/openpi && uv pip install lerobot imageio-ffmpeg && cd ../..
```

### Step 2 — Convert libero-10 demos to LeRobot format

Existing demos at `roboverse_demo/demo_mujoco/libero.pick_*/robot-franka/success/` → LeRobot dataset.

```bash
cd third_party/openpi
uv run ../../roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py \
  --input-root ../../roboverse_demo \
  --repo-id local/roboverse-libero10 \
  --overwrite
cd ../..
```

Dataset saved to `~/.cache/huggingface/lerobot/local/roboverse-libero10/`.

### Step 3 — Train (one-click script)

```bash
./roboverse_learn/vla/pi0/train_pi0.sh \
  -i ./roboverse_demo \
  -r local/roboverse-libero10 \
  -c pi05_roboverse_lora \
  -e libero10_pi05_lora \
  --skip-data-conversion \
  --skip-registration \
  --skip-norm-stats \
  2>&1 | tee claude/log/pi0_train.log
```

The script auto-registers `roboverse_policy.py` + configs in openpi, computes norm stats, and launches training.

> ⚠️ If registration and norm stats are already done (second run+), use `--skip-registration --skip-norm-stats` to skip directly to training.

**Options**: `-c pi0_roboverse_lora` for π₀ (instead of π₀.₅), `--overwrite-training` to restart.

<details>
<summary>Manual training steps (if not using one-click script)</summary>

```bash
# 2a. Register policy + config in openpi
cp roboverse_learn/vla/pi0/roboverse_policy.py \
   third_party/openpi/src/openpi/policies/roboverse_policy.py
# Then add LeRobotRoboVerseDataConfig + TrainConfig to
# third_party/openpi/src/openpi/training/config.py
# (see roboverse_learn/vla/pi0/README.md §3-4 for exact snippets)

# 2b. Compute norm stats (run from third_party/openpi, with .venv311 deactivated)
deactivate 2>/dev/null; cd third_party/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_roboverse_lora
cd ../..

# 2c. Launch training (run from third_party/openpi, with .venv311 deactivated)
deactivate 2>/dev/null; cd third_party/openpi
mkdir -p ../../claude/log
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_roboverse_lora \
  --exp-name=libero10_pi05_lora --overwrite \
  2>&1 | tee ../../claude/log/pi0_train.log
cd ../..
```

</details>

### Step 4 — Start server with trained checkpoint

```bash
tmux new-session -d -s pi0_server
tmux send-keys -t pi0_server \
  'cd third_party/openpi && uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_roboverse_lora \
    --policy.dir=<checkpoint_path>' Enter
# Wait for "server listening on 0.0.0.0:8000" before eval
```

Replace `<checkpoint_path>` with actual checkpoint dir (e.g., `checkpoints/pi05_roboverse_lora/libero10_pi05_lora/...`).

### Step 5 — Run eval (single task)

```bash
MUJOCO_GL=egl .venv311/bin/python roboverse_learn/vla/pi0/pi_eval.py \
  --task libero.pick_butter \
  --robot franka --sim mujoco \
  --policy-host localhost --policy-port 8000 \
  --num_episodes 10 --max_steps 250 \
  --output-dir claude/out/pi0_eval/libero.pick_butter \
  2>&1 | tee claude/log/pi0_pick_butter.log
```

### Step 5b — Run eval (all 9 libero tasks)

```bash
for task in libero.pick_alphabet_soup libero.pick_bbq_sauce libero.pick_butter \
  libero.pick_chocolate_pudding libero.pick_cream_cheese libero.pick_milk \
  libero.orange_juice libero.pick_salad_dressing libero.pick_tomato_sauce; do
  echo "=== Evaluating ${task} ==="
  MUJOCO_GL=egl .venv311/bin/python roboverse_learn/vla/pi0/pi_eval.py \
    --task ${task} --robot franka --sim mujoco \
    --policy-host localhost --policy-port 8000 \
    --num_episodes 10 --max_steps 250 \
    --output-dir claude/out/pi0_eval/${task} \
    2>&1 | tee claude/log/pi0_${task}.log
done
```

> Full training details: `roboverse_learn/vla/pi0/README.md`

---

## OpenVLA — LoRA Fine-tuning + Eval

LoRA fine-tune `openvla-7b` on RoboVerse libero-10 demos, then evaluate.

> ⚠️ Base `openvla-7b` outputs EEF delta actions normalized with `bridge_orig` stats (WidowX robot).
> Direct zero-shot eval on Franka/LIBERO will produce wrong-scale actions — **must fine-tune** to get the correct `unnorm_key` and action distribution.

### Env: `openvla` conda (Python 3.10)

```bash
# conda env 'openvla': Python 3.10, transformers==4.40.1, flash-attn==2.5.5,
# pyroki, metasim+mujoco — created by setup_env.sh
bash roboverse_learn/vla/OpenVLA/setup_env.sh   # one-time
```

### Step 1 — Save weights locally (one-time, ~15GB)

> Needed by `finetune.sh`; skipped if `third_party/openvla/openvla-7b/` already exists.

```bash
# Runs in background, notifies when done
tmux new-session -d -s openvla_download
tmux send-keys -t openvla_download \
  'conda activate openvla && cd third_party/openvla && python3 -c "
from transformers import AutoModelForVision2Seq, AutoProcessor; import torch
p=AutoProcessor.from_pretrained(\"openvla/openvla-7b\",trust_remote_code=True)
m=AutoModelForVision2Seq.from_pretrained(\"openvla/openvla-7b\",torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True,attn_implementation=\"flash_attention_2\")
p.save_pretrained(\"openvla-7b\"); m.save_pretrained(\"openvla-7b\")
print(\"✅ Done → third_party/openvla/openvla-7b/\")
"' Enter
```

### Step 2 — Convert demos to RLDS (tensorflow_datasets format)

```bash
# Symlink demos, then build RLDS dataset via tfds
cd roboverse_learn/vla/rlds_utils/roboverse
ln -sf ../../../../roboverse_demo/demo_mujoco/libero.pick_butter demo/libero.pick_butter-
conda run -n rlds_env tfds build --overwrite
# Output: ~/tensorflow_datasets/bridge_orig/
```

To include all 9 libero-10 tasks, symlink each task dir before building:
```bash
for task in libero.pick_alphabet_soup libero.pick_bbq_sauce libero.pick_butter \
  libero.pick_chocolate_pudding libero.pick_cream_cheese libero.pick_milk \
  libero.orange_juice libero.pick_salad_dressing libero.pick_tomato_sauce; do
  ln -sf "$(pwd)/roboverse_demo/demo_mujoco/${task}" \
    roboverse_learn/vla/rlds_utils/roboverse/demo/${task}-
done
cd roboverse_learn/vla/rlds_utils/roboverse
conda run -n rlds_env tfds build --overwrite
cd ../../../..
```

### Step 3 — LoRA fine-tune (one-click)

```bash
tmux new-session -d -s openvla_train
tmux send-keys -t openvla_train \
  'conda activate openvla && bash roboverse_learn/vla/OpenVLA/finetune.sh \
    2>&1 | tee claude/log/openvla_finetune.log' Enter
# Checkpoint: roboverse_learn/vla/OpenVLA/runs/
# Adapter:    roboverse_learn/vla/OpenVLA/adapters/
```

Key hyperparams in `finetune.sh` (edit before running):

| Param | Default | Notes |
|-------|---------|-------|
| `LORA_RANK` | 32 | LoRA rank |
| `BATCH_SIZE` | 8 | Per-GPU batch |
| `LEARNING_RATE` | 5e-4 | |
| `MAX_STEPS` | 5000 | |
| `DATASET_NAME` | `bridge_orig` | matches RLDS output |

### Step 4 — Eval with fine-tuned checkpoint

```bash
# Single task
MUJOCO_GL=osmesa conda run -n openvla \
  python roboverse_learn/vla/OpenVLA/vla_eval.py \
    --model_path roboverse_learn/vla/OpenVLA/runs/<checkpoint_dir> \
    --task libero.pick_butter \
    --robot franka --sim mujoco \
    --num_episodes 10 --max_steps 250 \
    --output_dir claude/out/openvla_eval/libero.pick_butter \
    2>&1 | tee claude/log/openvla_pick_butter.log
```

```bash
# All 9 libero tasks
for task in libero.pick_alphabet_soup libero.pick_bbq_sauce libero.pick_butter \
  libero.pick_chocolate_pudding libero.pick_cream_cheese libero.pick_milk \
  libero.orange_juice libero.pick_salad_dressing libero.pick_tomato_sauce; do
  echo "=== Evaluating ${task} ==="
  MUJOCO_GL=osmesa conda run -n openvla \
    python roboverse_learn/vla/OpenVLA/vla_eval.py \
      --model_path roboverse_learn/vla/OpenVLA/runs/<checkpoint_dir> \
      --task ${task} --robot franka --sim mujoco \
      --num_episodes 10 --max_steps 250 \
      --output_dir claude/out/openvla_eval/${task} \
      2>&1 | tee claude/log/openvla_${task}.log
done
```

> ⚠️ `conda activate` fails in non-login shells (tmux). Always use `conda run -n openvla`.
> ⚠️ Use `MUJOCO_GL=osmesa` to avoid EGL errors in conda environments.
> Full details: `roboverse_learn/vla/OpenVLA/README.md`

---

## Output Directory Reference

```
claude/log/                             # all terminal logs redirected here
claude/out/pi0_eval/{task}/             # π₀ eval videos + JSON reports
claude/out/openvla_eval/{task}/         # OpenVLA eval videos + JSON reports

~/.cache/huggingface/lerobot/{repo_id}/ # LeRobot dataset (π₀ training data)
third_party/openpi/checkpoints/         # π₀ trained checkpoints

roboverse_demo/demo_mujoco/{task}/      # raw demo files
  robot-franka/success/demo_XXXX/
    metadata.json                       # per-step joint states + actions
    rgb.mp4                             # head camera video

data_policy/{task}FrankaL0_obs:joint_pos_act:joint_pos_99.zarr  # training data

info/outputs/DP/{task}/checkpoints/     # DP checkpoints (by epoch)
info/outputs/ACT/{date}/{time}_{task}_*/
  policy_best.ckpt                      # best val loss checkpoint
  policy_last.ckpt
  dataset_stats.pkl                     # normalization stats

tmp/{task}/diffusion_policy/franka/{ckpt}/final_stats.txt  # DP eval results
tmp/act/{task}/{ckpt}/success_rate.txt                     # ACT eval results
```

---

## Key Source Files

| File | Description |
|------|-------------|
| `scripts/advanced/collect_demo.py` | Demo collection entry point |
| `roboverse_learn/il/data2zarr_dp.py` | Convert demos → zarr dataset |
| `roboverse_learn/il/dp/main.py` | DP train/eval entry (Hydra) |
| `roboverse_learn/il/dp/dp_run.sh` | DP batch train+eval script |
| `roboverse_learn/il/act/act_run_libero.sh` | ACT batch train+eval (9 tasks) |
| `roboverse_learn/il/act/eval_all_fixed.sh` | ACT batch eval with saved checkpoints |
| `roboverse_learn/vla/pi0/pi_eval.py` | π₀ eval client (WebSocket → action decode) |
| `roboverse_learn/vla/pi0/train_pi0.sh` | π₀ one-click LoRA training script |
| `roboverse_learn/vla/pi0/convert_roboverse_to_lerobot.py` | Convert RoboVerse demos → LeRobot format |
| `roboverse_learn/vla/pi0/roboverse_policy.py` | π₀ data transforms (RoboVerse ↔ openpi) |
| `roboverse_learn/vla/OpenVLA/vla_eval.py` | OpenVLA eval with pyroki IK |
| `roboverse_learn/vla/OpenVLA/finetune.sh` | OpenVLA LoRA fine-tuning script |
| `roboverse_learn/vla/OpenVLA/run_pipeline.sh` | One-click RLDS convert + fine-tune |
| `roboverse_learn/vla/rlds_utils/roboverse/roboverse.py` | RoboVerse → RLDS (Bridge V2 format) converter |

---

## Current Status

| Algorithm | Train | Eval | Best SR | Date |
|-----------|-------|------|---------|------|
| **Diffusion Policy** | ✅ | ✅ | TBD | 2026-03 |
| **ACT** | ✅ | ✅ | 4/9 tasks 100% (300ep) | 2026-03-09 |
| **π₀ (LoRA fine-tune)** | ⏳ pending | ⏳ pending | — | — |
| **OpenVLA (LoRA fine-tune)** | ⏳ pending | ⏳ pending | — | — |

> Previous π₀/OpenVLA eval results were obtained with buggy code (reverted in commit 81097c88).
> Re-evaluation needed with upstream-aligned code.

For detailed results and timeline: see `roboverse_learn/libero_act_pi0_openvla.md`

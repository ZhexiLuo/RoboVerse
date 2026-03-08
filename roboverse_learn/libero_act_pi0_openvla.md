# Libero-10 Multi-Policy Benchmark

Goal: evaluate ACT (trained, 100 epochs) + π₀ (zero-shot) + OpenVLA (zero-shot) on 9 libero-10 tasks.

## Task List (9 tasks, 99 demos each)

| Task | Zarr |
|------|------|
| libero.pick_alphabet_soup | ✅ `data_policy/libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_99.zarr` |
| libero.pick_bbq_sauce | ✅ |
| libero.pick_butter | ✅ |
| libero.pick_chocolate_pudding | ✅ |
| libero.pick_cream_cheese | ✅ |
| libero.pick_milk | ✅ |
| libero.orange_juice | ✅ |
| libero.pick_salad_dressing | ✅ |
| libero.pick_tomato_sauce | ✅ |

---

## Pipeline Commands

### ACT: Train

```bash
# Train all 9 tasks (100 epochs each, ~18h total on RTX 4090)
tmux new-session -d -s act_libero
tmux send-keys -t act_libero \
  'bash roboverse_learn/il/act/act_run_libero.sh 0 2>&1 | tee claude/log/act_libero.log' Enter
```

Hyperparams: chunk_size=20, kl_weight=10, hidden_dim=512, dim_feedforward=3200, lr=1e-5, batch=8, wandb=RoboVerse_ACT

**Checkpoint paths** (per task):
```
info/outputs/ACT/2026.03.08/{HH.MM.SS}_{task}_obs:joint_pos_act:joint_pos_chunk20_99/
  policy_best.ckpt   # best val loss checkpoint
  policy_last.ckpt   # final epoch checkpoint
  cfg.yaml           # training config
  dataset_stats.pkl  # normalization stats
```

**Training log**: `claude/log/act_libero.log`

**Actual checkpoint dirs (2026-03-08 run)**:
| Task | Checkpoint Dir |
|------|---------------|
| libero.pick_alphabet_soup | `info/outputs/ACT/2026.03.08/22.21.08_libero.pick_alphabet_soup_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_bbq_sauce | `info/outputs/ACT/2026.03.08/22.22.03_libero.pick_bbq_sauce_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_butter | `info/outputs/ACT/2026.03.08/22.22.57_libero.pick_butter_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_chocolate_pudding | `info/outputs/ACT/2026.03.08/22.23.53_libero.pick_chocolate_pudding_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_cream_cheese | `info/outputs/ACT/2026.03.08/22.24.51_libero.pick_cream_cheese_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_milk | `info/outputs/ACT/2026.03.08/22.25.48_libero.pick_milk_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.orange_juice | `info/outputs/ACT/2026.03.08/22.26.45_libero.orange_juice_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_salad_dressing | `info/outputs/ACT/2026.03.08/23.09.24_libero.pick_salad_dressing_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_tomato_sauce | `info/outputs/ACT/2026.03.08/23.10.22_libero.pick_tomato_sauce_obs:joint_pos_act:joint_pos_chunk20_99/` |

### ACT: Eval

```bash
# Eval single task — pass DIRECTORY to --ckpt_path (runner appends policy_best.ckpt)
CKPT_DIR="info/outputs/ACT/2026.03.08/22.21.08_libero.pick_alphabet_soup_obs:joint_pos_act:joint_pos_chunk20_99"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
.venv311/bin/python -m roboverse_learn.il.act.act_eval_runner \
  --task libero.pick_alphabet_soup \
  --robot franka --num_envs 1 --sim mujoco \
  --algo act --ckpt_path ./${CKPT_DIR} \
  --headless True --num_eval 99 \
  --temporal_agg True --chunk_size 20 \
  2>&1 | tee claude/log/act_eval_libero.pick_alphabet_soup.log
```

**Eval output paths**:
- Videos: `tmp/act/{task}/{ckpt_name}/{episode_idx}.mp4`
- Success rate: `tmp/act/{task}/{ckpt_name}/success_rate.txt`
- Logs: `claude/log/act_eval_{task}.log`

---

### π₀: Zero-shot Eval

**Step 1: Install openpi** (one-time)
```bash
cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && cd ../..
.venv311/bin/pip install openpi_client
python roboverse_learn/vla/pi0/openpi_config_patch.py   # register pi0_roboverse_lora config
```

**Step 2: Download checkpoint** (one-time, ~9.8GB)
```bash
.venv311/bin/gsutil -m cp -r \
  gs://openpi-assets/checkpoints/pi0_libero \
  third_party/openpi/checkpoints/pi0_libero
```

**Step 3: Start server** (tmux window: pi0_server)
```bash
bash roboverse_learn/vla/pi0/start_server.sh
# Default config: pi0_libero (LIBERO norm stats, 8-dim state, 7-dim action)
# Wait for "server listening on 0.0.0.0:8000" before running eval
```

**Step 4: Run eval** (tmux window: pi0_eval)
```bash
bash roboverse_learn/vla/pi0/eval_libero.sh
# Passes --state-dim 8 to truncate RoboVerse 9-dim state → LIBERO 8-dim
```

**Eval output paths**:
- Videos: `claude/out/pi0_eval/{task}/episode_NNN.mp4`
- JSON report: `claude/out/pi0_eval/{task}/pi_eval_{task}_{timestamp}.json`
- Logs: `claude/log/pi0_{task}.log`

> **Notes on pi0_libero compatibility**:
> - Server uses `pi0_libero` config (LIBERO norm stats, 8-dim state, 7-dim action output)
> - `pi_eval.py` truncates state to 8-dim via `--state-dim 8`
> - Action decode: 7-dim arm joints, gripper binary inferred from sign of last action value

---

### OpenVLA: Zero-shot Eval

**Setup** (one-time):
```bash
# conda env 'openvla' with: Python 3.10, transformers==4.40.1,
# flash-attn==2.5.5, pyroki, metasim+mujoco, numpy==1.26.4
# IMPORTANT: conda activate does not work in non-login shells
# Use conda run instead:
conda run -n openvla bash roboverse_learn/vla/OpenVLA/eval_libero.sh
```

**Run eval**:
```bash
# Recommended: use conda run (avoids conda init issues in tmux/scripts)
tmux new-session -d -s openvla_eval
tmux send-keys -t openvla_eval \
  'conda run -n openvla bash roboverse_learn/vla/OpenVLA/eval_libero.sh 0 2>&1 | tee claude/log/openvla_eval_all.log' Enter
# Model auto-downloads from HuggingFace: openvla/openvla-7b (~15GB, cached in ~/.cache/huggingface/)
```

**Eval output paths**:
- Videos: `claude/out/openvla_eval/{task}/episode_NNN.mp4`
- JSON report: `claude/out/openvla_eval/{task}/openvla_{task}_{timestamp}.json`
- Logs: `claude/log/openvla_{task}.log` (per task), `claude/log/openvla_eval_all.log` (all tasks)

> **Known issue**: `conda activate` fails in non-login shells (tmux default). Use `conda run -n openvla` instead.

---

## Eval Results

### ACT (trained, 100 epochs, wandb: RoboVerse_ACT)

| Task | Best Val Loss | Eval Status | Success Rate | Videos |
|------|--------------|-------------|-------------|--------|
| libero.pick_alphabet_soup | 0.280 (ep97) | ✅ done (99/99) | **0.0%** | `tmp/act/libero.pick_alphabet_soup/22.21.08_.../` |
| libero.pick_bbq_sauce | 0.277 (ep91) | ✅ done (99/99) | **0.0%** | `tmp/act/libero.pick_bbq_sauce/22.22.03_.../` |
| libero.pick_butter | 0.317 (ep82) | ✅ done (99/99) | **0.0%** | `tmp/act/libero.pick_butter/22.22.57_.../` |
| libero.pick_chocolate_pudding | 0.305 (ep93) | ✅ done (99/99) | **0.0%** | `tmp/act/libero.pick_chocolate_pudding/22.23.53_.../` |
| libero.pick_cream_cheese | 0.292 (ep88) | ✅ done (99/99) | **0.0%** | `tmp/act/libero.pick_cream_cheese/22.24.51_.../` |
| libero.pick_milk | 0.269 (ep93) | ⏳ stopped at 51/99 | — | — |
| libero.orange_juice | 0.309 (ep90) | ⏳ not started | — | — |
| libero.pick_salad_dressing | 0.301 (ep93) | ⏳ not started | — | — |
| libero.pick_tomato_sauce | 0.331 (ep93) | ⏳ not started | — | — |

> ⚠️ **100 epochs insufficient**: All completed tasks time out (max_steps=800, task timeout=250 steps). Policy does not converge to meaningful behavior at 100 epochs. **Recommend 300–500 epochs** for non-trivial success rates.

### π₀ (zero-shot, pi0_libero checkpoint)

| Task | Eval Status | Success Rate | Output Dir |
|------|------------|-------------|-----------|
| libero.pick_alphabet_soup | ✅ done (99/99) | **100.0%** | `claude/out/pi0_eval/libero.pick_alphabet_soup/` |
| libero.pick_bbq_sauce | ✅ done (99/99) | **100.0%** | `claude/out/pi0_eval/libero.pick_bbq_sauce/` |
| libero.pick_butter | ⏳ stopped at 2/99 | — | — |
| libero.pick_chocolate_pudding | ⏳ not started | — | — |
| libero.pick_cream_cheese | ⏳ not started | — | — |
| libero.pick_milk | ⏳ not started | — | — |
| libero.orange_juice | ⏳ not started | — | — |
| libero.pick_salad_dressing | ⏳ not started | — | — |
| libero.pick_tomato_sauce | ⏳ not started | — | — |

### OpenVLA (zero-shot, openvla-7b)

| Task | Eval Status | Success Rate |
|------|------------|-------------|
| (all tasks) | ⏳ not started — EGL init issue in conda env | — |

> **Note**: OpenVLA eval has EGL rendering issue when run via `conda run` in the openvla env. Need to investigate `MUJOCO_GL` or EGL device config for that conda env.

---

## Current Status & TODO

### ✅ Done
- [x] zarr data ready for all 9 tasks (99 demos each)
- [x] ACT training completed for all 9 tasks (100 epochs, wandb logged)
- [x] ACT eval completed for 5/9 tasks (alphabet_soup, bbq_sauce, butter, chocolate_pudding, cream_cheese) → all 0%
- [x] Videos generated: `tmp/act/{task}/*/` (99 videos per task, 5 tasks)
- [x] π₀ server running (`pi0_libero` config, port 8000)
- [x] π₀ eval completed for 2/9 tasks → both 100%!
- [x] OpenVLA model downloaded (openvla-7b, ~15GB cached)

### 🔧 TODO: Resume Evals

**ACT** — resume eval for 4 remaining tasks:
```bash
# Run from RoboVerse root, sequentially (reuse existing tmux act_eval)
for task_info in \
  "libero.pick_milk:22.25.48_libero.pick_milk_obs:joint_pos_act:joint_pos_chunk20_99" \
  "libero.orange_juice:22.26.45_libero.orange_juice_obs:joint_pos_act:joint_pos_chunk20_99" \
  "libero.pick_salad_dressing:23.09.24_libero.pick_salad_dressing_obs:joint_pos_act:joint_pos_chunk20_99" \
  "libero.pick_tomato_sauce:23.10.22_libero.pick_tomato_sauce_obs:joint_pos_act:joint_pos_chunk20_99"; do
  task="${task_info%%:*}"
  ckpt="info/outputs/ACT/2026.03.08/${task_info##*:}"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
  .venv311/bin/python -m roboverse_learn.il.act.act_eval_runner \
    --task ${task} --robot franka --num_envs 1 --sim mujoco \
    --algo act --ckpt_path ./${ckpt} \
    --headless True --num_eval 99 --temporal_agg True --chunk_size 20 \
    2>&1 | tee claude/log/act_eval_${task}.log
done
```

**π₀** — resume eval for 7 remaining tasks:
```bash
# Server must be running: bash roboverse_learn/vla/pi0/start_server.sh
bash roboverse_learn/vla/pi0/eval_libero.sh
# (script loops all 9 tasks; tasks with existing JSON will be skipped if modified)
```

**OpenVLA** — fix EGL issue then run:
```bash
# Investigate: conda run -n openvla python -c "import mujoco; print(mujoco.__version__)"
# Option A: set MUJOCO_GL=osmesa if EGL not available in openvla env
# Option B: install mujoco EGL support in openvla conda env
conda run -n openvla bash roboverse_learn/vla/OpenVLA/eval_libero.sh 0 2>&1 | tee claude/log/openvla_eval_all.log
```

### 🔮 Future: ACT Longer Training
- 100 epochs → 0% success rate (timeout). Recommend 300–500 epochs.
- Retrain with `--num_epochs 300` and re-eval.

---

## Progress Timeline

| Date | Event |
|------|-------|
| 2026-03-08 | zarr data ready for all 9 tasks |
| 2026-03-08 | ACT infra verified (zarr v3 fix, 1-epoch dry run) |
| 2026-03-08 | ACT training completed for 9/9 tasks (100 epochs each, wandb: RoboVerse_ACT) |
| 2026-03-08 | ACT eval done for 5/9 tasks — all 0% (100 epochs insufficient) |
| 2026-03-08 | Videos generated: `tmp/act/` (5 tasks × 99 episodes) |
| 2026-03-08 | openpi installed, pi0_libero downloaded (9.8GB) |
| 2026-03-08 | π₀ eval running — alphabet_soup 100%, bbq_sauce 100% |
| 2026-03-08 | OpenVLA model downloaded (openvla-7b, 15GB), EGL issue blocking eval |

---

## Known Issues (resolved)

1. ~~**π₀ state-dim mismatch**~~ → Fixed: `--state-dim 8` in eval_libero.sh
2. ~~**ACT eval curobo dependency**~~ → Fixed: removed unused `get_curobo_models()` call
3. ~~**ACT eval CUDA tensor**~~ → Fixed: `.cpu()` before passing actions to MuJoCo
4. ~~**π₀ pyroki/jax version conflict**~~ → Fixed: removed unused `setup_ik_solver()` call
5. ~~**π₀ action decode mismatch**~~ → Fixed: pi0_libero outputs 7-dim arm joints (not 9-dim)
6. ~~**ACT zarr empty (salad/tomato)**~~ → Fixed: re-ran data2zarr_dp.py; all 9 tasks now have 99 demos
7. ~~**numpy 2.x conflict in .venv311**~~ → Fixed: downgraded to numpy==1.26.4

## Known Issues (active)

1. **OpenVLA EGL error** (`EGL_NOT_INITIALIZED`) when running `conda run -n openvla`. The openvla conda env may lack EGL libraries or needs `MUJOCO_GL=osmesa` fallback.
2. **ACT 0% success rate** — 100 epochs insufficient for joint_pos policy to learn picking behavior. Need 300–500 epochs.

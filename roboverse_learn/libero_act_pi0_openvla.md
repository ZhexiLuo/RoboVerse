# Libero-10 Multi-Policy Benchmark

Goal: evaluate ACT (trained, 300 epochs) + π₀ (zero-shot) + OpenVLA (zero-shot) on 9 libero-10 tasks.

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
# Train all 9 tasks (300 epochs each)
tmux new-session -d -s act_libero
tmux send-keys -t act_libero \
  'bash roboverse_learn/il/act/act_run_libero.sh 0 2>&1 | tee claude/log/act_libero.log' Enter
```

Hyperparams: chunk_size=20, kl_weight=10, hidden_dim=512, dim_feedforward=3200, lr=1e-5, batch=8, wandb=RoboVerse_ACT

**Checkpoint paths** (per task):
```
info/outputs/ACT/2026.03.09/{HH.MM.SS}_{task}_obs:joint_pos_act:joint_pos_chunk20_99/
  policy_best.ckpt   # best val loss checkpoint
  policy_last.ckpt   # final epoch checkpoint
  cfg.yaml           # training config
  dataset_stats.pkl  # normalization stats
```

**Actual checkpoint dirs (2026-03-09, 300 epochs)**:
| Task | Checkpoint Dir |
|------|---------------|
| libero.pick_alphabet_soup | `info/outputs/ACT/2026.03.09/03.22.10_libero.pick_alphabet_soup_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_bbq_sauce | `info/outputs/ACT/2026.03.09/03.31.46_libero.pick_bbq_sauce_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_butter | `info/outputs/ACT/2026.03.09/03.38.58_libero.pick_butter_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_chocolate_pudding | `info/outputs/ACT/2026.03.09/03.42.08_libero.pick_chocolate_pudding_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_cream_cheese | `info/outputs/ACT/2026.03.09/03.45.13_libero.pick_cream_cheese_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_milk | `info/outputs/ACT/2026.03.09/03.48.24_libero.pick_milk_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.orange_juice | `info/outputs/ACT/2026.03.09/03.51.40_libero.orange_juice_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_salad_dressing | `info/outputs/ACT/2026.03.09/03.54.56_libero.pick_salad_dressing_obs:joint_pos_act:joint_pos_chunk20_99/` |
| libero.pick_tomato_sauce | `info/outputs/ACT/2026.03.09/03.58.08_libero.pick_tomato_sauce_obs:joint_pos_act:joint_pos_chunk20_99/` |

### ACT: Eval

```bash
# Eval all 9 tasks (20 evals each) — uses fixed joint order + correct camera
bash roboverse_learn/il/act/eval_all_fixed.sh
```

**Eval output paths**:
- Videos: `tmp/act/{task}/{ckpt_name}/{episode_idx}.mp4`
- Success rate: `tmp/act/{task}/{ckpt_name}/success_rate.txt`
- Logs: `claude/log/act_eval_fixed_{task}.log`

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

### ACT (trained, 300 epochs, wandb: RoboVerse_ACT)

Eval: 20 demos each, with fixed joint order + correct camera pos `(1.0, 0, 0.75)`.

| Task | Best Val Loss | Eval (20x) | Success Rate |
|------|--------------|------------|-------------|
| libero.pick_alphabet_soup | 0.166 (ep292) | ✅ done | **0%** |
| libero.pick_bbq_sauce | 0.185 (ep278) | ✅ done | **0%** |
| libero.pick_butter | 0.179 (ep292) | ✅ done | **100%** ✅ |
| libero.pick_chocolate_pudding | 0.185 (ep292) | ✅ done | **0%** |
| libero.pick_cream_cheese | 0.209 (ep210) | ✅ done | **100%** ✅ |
| libero.pick_milk | ~0.210 | ✅ done | **0%** |
| libero.orange_juice | ~0.190 | ✅ done | **0%** |
| libero.pick_salad_dressing | 0.182 (ep276) | ✅ done | **100%** ✅ |
| libero.pick_tomato_sauce | ~0.300 | ✅ done | **100%** ✅ |

> 4/9 tasks achieve 100% SR, 5/9 tasks still 0%. May need more epochs or hyperparameter tuning for the 0% tasks.

### π₀ (zero-shot, pi0_libero checkpoint)

| Task | Eval Status | Success Rate | Output Dir |
|------|------------|-------------|-----------|
| libero.pick_alphabet_soup | ✅ done (99/99) | **100%** | `claude/out/pi0_eval/libero.pick_alphabet_soup/` |
| libero.pick_bbq_sauce | ✅ done (99/99) | **100%** | `claude/out/pi0_eval/libero.pick_bbq_sauce/` |
| libero.pick_butter | ⏳ not started | — | — |
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

> **Note**: OpenVLA eval has EGL rendering issue when run via `conda run` in the openvla env. Need to investigate `MUJOCO_GL` or EGL device config.

---

## Current Status & TODO

### ✅ Done
- [x] zarr data ready for all 9 tasks (99 demos each)
- [x] ACT training completed for all 9 tasks (**300 epochs**, wandb: RoboVerse_ACT)
- [x] ACT eval completed for all 9/9 tasks (20 evals each, fixed bugs)
- [x] π₀ server running (`pi0_libero` config, port 8000)
- [x] π₀ eval completed for 2/9 tasks → both 100%
- [x] OpenVLA model downloaded (openvla-7b, ~15GB cached)

### 🔧 TODO
1. **π₀ eval** — resume for remaining 7 tasks:
   ```bash
   bash roboverse_learn/vla/pi0/start_server.sh   # start server first
   bash roboverse_learn/vla/pi0/eval_libero.sh    # eval all 9 tasks
   ```

2. **OpenVLA eval** — fix EGL issue then run:
   ```bash
   # Option A: set MUJOCO_GL=osmesa
   # Option B: install EGL support in openvla conda env
   conda run -n openvla bash roboverse_learn/vla/OpenVLA/eval_libero.sh 0
   ```

3. **ACT 0% tasks** — investigate why 5 tasks still fail at 300 epochs:
   - Candidates: pick_alphabet_soup, pick_bbq_sauce, pick_chocolate_pudding, pick_milk, orange_juice
   - Options: more epochs (500+), tune lr/chunk_size, check demo quality

---

## Progress Timeline

| Date | Event |
|------|-------|
| 2026-03-08 | zarr data ready for all 9 tasks |
| 2026-03-08 | ACT infra verified (zarr v3 fix, eval curobo fix) |
| 2026-03-08 | ACT training completed for 9/9 tasks (100 epochs, wandb: RoboVerse_ACT) |
| 2026-03-08 | ACT eval (100ep) — all 0%, camera bug + joint order bug found |
| 2026-03-08 | π₀ eval: alphabet_soup 100%, bbq_sauce 100% |
| 2026-03-08 | OpenVLA model downloaded, EGL issue blocking eval |
| 2026-03-09 | ACT retrained 300 epochs for all 9 tasks (wandb: RoboVerse_ACT) |
| 2026-03-09 | Bug fix: joint order double-reindex (commit c80b4fc8) |
| 2026-03-09 | Bug fix: camera pos mismatch (1.5,0,1.5)→(1.0,0,0.75) (commit f0477c94) |
| 2026-03-09 | ACT eval (300ep, fixed) — 4/9 tasks 100%, 5/9 tasks 0% |

---

## Known Issues (resolved)

1. ~~**π₀ state-dim mismatch**~~ → Fixed: `--state-dim 8` in eval_libero.sh
2. ~~**ACT eval curobo dependency**~~ → Fixed: removed unused `get_curobo_models()` call
3. ~~**ACT eval CUDA tensor**~~ → Fixed: `.cpu()` before passing actions to MuJoCo
4. ~~**π₀ pyroki/jax version conflict**~~ → Fixed: removed unused `setup_ik_solver()` call
5. ~~**π₀ action decode mismatch**~~ → Fixed: pi0_libero outputs 7-dim arm joints (not 9-dim)
6. ~~**ACT zarr empty (salad/tomato)**~~ → Fixed: re-ran data2zarr_dp.py; all 9 tasks now have 99 demos
7. ~~**numpy 2.x conflict in .venv311**~~ → Fixed: downgraded to numpy==1.26.4
8. ~~**ACT joint order double-reindex**~~ → Fixed (commit c80b4fc8): zip action directly with `sorted(joint_names)`
9. ~~**ACT camera pos mismatch**~~ → Fixed (commit f0477c94): eval now uses `(1.0, 0, 0.75)` matching demo collection

## Known Issues (active)

1. **OpenVLA EGL error** — `conda run -n openvla` triggers `EGL_NOT_INITIALIZED`. Try `MUJOCO_GL=osmesa`.
2. **ACT 5/9 tasks 0% SR** — pick_alphabet_soup, pick_bbq_sauce, pick_chocolate_pudding, pick_milk, orange_juice still fail at 300 epochs. Root cause unclear.

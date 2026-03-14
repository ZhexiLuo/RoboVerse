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

**⚠️ 重要 Bug 修复记录 (2026-03-10)**:
- Bug 1: SR 计算错误 — `truncated`（超时）被误算为成功
- Bug 2: state 格式错误 — 发送 joint_pos 而非 world-frame EE 位姿（xyz+axis_angle+gripper=8dim）
- Bug 3: action 解码错误 — `extra_delta_transform=True` + `AbsoluteActions` 后输出是绝对 EE 目标（非 delta）
- Bug 4: IK target 坐标系错误 — world-frame 目标需转换到 robot local frame 再 IK
- Bug 5: pyroki 在 .venv311 中 JAX 冲突 → 改用 openvla conda env（JAX 0.6.2 + numpy 1.26.4 兼容）

**Eval cmd (修复后)**:
```bash
bash roboverse_learn/vla/pi0/start_server.sh        # tmux pi0_server
conda run -n openvla bash roboverse_learn/vla/pi0/eval_libero_ik.sh  # tmux pi0_eval
```

**结果 (2026-03-10, 10 episodes each, pi0_libero zero-shot)**:

| Task | SR | Output Dir |
|------|-----|-----------|
| libero.pick_alphabet_soup | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_alphabet_soup/` |
| libero.pick_bbq_sauce | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_bbq_sauce/` |
| libero.pick_butter | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_butter/` |
| libero.pick_chocolate_pudding | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_chocolate_pudding/` |
| libero.pick_cream_cheese | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_cream_cheese/` |
| libero.pick_milk | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_milk/` |
| libero.orange_juice | **0/10 (0%)** | `claude/out/pi0_eval/libero.orange_juice/` |
| libero.pick_salad_dressing | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_salad_dressing/` |
| libero.pick_tomato_sauce | **0/10 (0%)** | `claude/out/pi0_eval/libero.pick_tomato_sauce/` |

> SR=0% 可能原因：①RoboVerse 场景与 LIBERO 原版存在差异（桌面布局/相机视角），②pi0_libero 为 zero-shot，未经 RoboVerse 数据微调

### OpenVLA (zero-shot, openvla-7b)

**Bug 修复 (2026-03-10)**: `model.norm_stats` 被空字典覆盖 → 保留模型内置 norm_stats（在 config.json 中）

**Eval cmd**:
```bash
# 注意：需先停止 pi0 server 释放 GPU（OpenVLA 需要 ~15GB）
conda run -n openvla bash roboverse_learn/vla/OpenVLA/eval_libero_10ep.sh 0 \
  2>&1 | tee claude/log/openvla_eval_v2.log
```

**结果 (2026-03-13, 10 episodes each, openvla-7b zero-shot)**:

| Task | SR | Output Dir |
|------|-----|-----------|
| libero.pick_alphabet_soup | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_alphabet_soup/` |
| libero.pick_bbq_sauce | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_bbq_sauce/` |
| libero.pick_butter | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_butter/` |
| libero.pick_chocolate_pudding | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_chocolate_pudding/` |
| libero.pick_cream_cheese | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_cream_cheese/` |
| libero.pick_milk | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_milk/` |
| libero.orange_juice | **0/10 (0%)** | `claude/out/openvla_eval/libero.orange_juice/` |
| libero.pick_salad_dressing | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_salad_dressing/` |
| libero.pick_tomato_sauce | **0/10 (0%)** | `claude/out/openvla_eval/libero.pick_tomato_sauce/` |

> SR=0% 原因：① openvla-7b 为 zero-shot，未见 RoboVerse 数据；② 所有 episode 均达 250 步超时（任务 max），机器人有运动但未完成抓放；③ RoboVerse 场景布局与 LIBERO 原版训练数据存在差异

---

## Current Status & TODO

### ✅ Done
- [x] zarr data ready for all 9 tasks (99 demos each)
- [x] ACT training completed for all 9 tasks (300 epochs, wandb: RoboVerse_ACT)
- [x] ACT eval completed for all 9/9 tasks (20 evals each) — 4/9 tasks 100% SR
- [x] π₀ eval completed for all 9/9 tasks (10 evals each, 2026-03-10) — all 0% (zero-shot)
- [x] OpenVLA model downloaded (openvla-7b, ~15GB cached)
- [x] OpenVLA eval completed for all 9/9 tasks (2026-03-13) — all 0% (zero-shot)

### 🔧 TODO
1. **ACT 0% tasks** — 5/9 tasks (pick_alphabet_soup, pick_bbq_sauce, pick_chocolate_pudding, pick_milk, orange_juice) 仍 0%，可尝试：
   - 更多 epoch (500+)
   - 调整 lr/chunk_size
   - 检查 demo 质量

2. **π₀ SR=0% 分析** — 所有 episode 均为 250 步超时，机器人动作输出但无法完成抓放。需要微调 pi0_libero checkpoint 或分析 IK 执行轨迹视频。

3. **OpenVLA SR=0% 分析** — 所有 episode 均为 250 步超时。openvla-7b 为 zero-shot，场景差异导致泛化失败。需要微调或使用 LIBERO action space fine-tuned 版本。

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
| 2026-03-10 | π₀ 5 bugs fixed (SR calc, state format, action decode, IK frame, jax conflict) |
| 2026-03-10 | π₀ eval completed for all 9/9 tasks — all 0% (zero-shot, scene mismatch) |
| 2026-03-10 | OpenVLA norm_stats bug fixed (commit e7df0332) |
| 2026-03-13 | OpenVLA eval completed for all 9/9 tasks — all 0% (zero-shot, scene mismatch) |

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
10. ~~**OpenVLA norm_stats overwritten**~~ → Fixed (commit e7df0332): keep model's built-in norm_stats
11. ~~**OpenVLA CUDA OOM (pi0 server running)**~~ → Fixed: stop pi0 server before running OpenVLA eval

## Known Issues (active)

1. **π₀ SR=0%** — zero-shot pi0_libero; all episodes timeout at 250 steps. Root cause: scene/camera domain gap between RoboVerse and original LIBERO training data. Requires fine-tuning.
2. **OpenVLA SR=0%** — zero-shot openvla-7b; all episodes timeout at 250 steps. Same domain gap issue. Requires fine-tuning on RoboVerse LIBERO data.
3. **ACT 5/9 tasks 0% SR** — pick_alphabet_soup, pick_bbq_sauce, pick_chocolate_pudding, pick_milk, orange_juice still fail at 300 epochs. Root cause unclear (may need 500+ epochs or hyperparameter tuning).

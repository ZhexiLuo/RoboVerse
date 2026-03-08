# Libero-10 Multi-Policy Benchmark

Goal: evaluate ACT (trained) + π₀ (zero-shot) + OpenVLA (zero-shot) on 9 libero-10 tasks.

## Task List (9 tasks, 99 demos each)

| Task | Zarr |
|------|------|
| libero.pick_alphabet_soup | ✅ `data_policy/...99.zarr` |
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

### ACT: Train + Eval

```bash
# Train all 9 tasks (100 epochs each, ~18h total on RTX 4090)
tmux new-session -d -s act_libero
tmux send-keys -t act_libero \
  'bash roboverse_learn/il/act/act_run_libero.sh 0 2>&1 | tee claude/log/act_libero.log' Enter
```

Hyperparams: chunk_size=20, kl_weight=10, hidden_dim=512, dim_feedforward=3200, lr=1e-5, batch=8
Eval: 99 episodes per task, max_steps=350, temporal_agg=True

Checkpoint path: `info/outputs/ACT/{date}/{task}/checkpoints/policy_best.ckpt`

---

### π₀: Zero-shot Eval

**Step 1: Install openpi**
```bash
cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && cd ../..
python roboverse_learn/vla/pi0/openpi_config_patch.py   # register pi0_roboverse_lora config
```

**Step 2: Download checkpoint**
```bash
.venv311/bin/gsutil -m cp -r \
  gs://openpi-assets/checkpoints/pi0_libero \
  third_party/openpi/checkpoints/pi0_libero
```

**Step 3: Start server (tmux window pi0_server)**
```bash
# Terminal 1
bash roboverse_learn/vla/pi0/start_server.sh pi0_libero
# Wait for "Server listening..." before running eval
```

**Step 3: Start server (tmux window pi0_server)**
```bash
# Terminal 1 — default config is now pi0_libero (8-dim state)
bash roboverse_learn/vla/pi0/start_server.sh
# Wait for "server listening on 0.0.0.0:8000" before running eval
```

**Step 4: Run eval (tmux window pi0_eval)**
```bash
# Terminal 2 — pass --state-dim 8 to truncate RoboVerse 9-dim → LIBERO 8-dim
bash roboverse_learn/vla/pi0/eval_libero.sh
# Logs: claude/log/pi0_{task}.log
# Output: claude/out/pi0_eval/{task}/
```

> **Notes on pi0_libero compatibility**:
> - Server uses `pi0_libero` config (LIBERO norm stats, 8-dim state, 7-dim action output)
> - `pi_eval.py` truncates state to 8-dim via `--state-dim 8`
> - Action decode: 7-dim arm joints, gripper binary inferred from sign of last action value

---

### OpenVLA: Zero-shot Eval

**Setup** (one-time):
```bash
# conda env 'openvla' already created with: Python 3.10, transformers==4.40.1,
# flash-attn==2.5.5, pyroki, metasim+mujoco, numpy==1.26.4
conda activate openvla
```

**Run eval**:
```bash
conda activate openvla
bash roboverse_learn/vla/OpenVLA/eval_libero.sh
# Model auto-downloads from HuggingFace: openvla/openvla-7b (~15GB, first run only)
# Logs: claude/log/openvla_{task}.log
# Output: claude/out/openvla_eval/{task}/
```

Custom model path:
```bash
MODEL_PATH=openvla/openvla-7b bash roboverse_learn/vla/OpenVLA/eval_libero.sh
```

---

## Eval Results

### ACT (trained, 100 epochs)

| Task | Val Loss | Status | Success Rate |
|------|----------|--------|-------------|
| libero.pick_alphabet_soup | 0.280 (ep97) | 🔄 eval running | — |
| libero.pick_bbq_sauce | 0.277 (ep91) | 🔄 eval running | — |
| libero.pick_butter | 0.317 (ep82) | 🔄 eval running | — |
| libero.pick_chocolate_pudding | 0.305 (ep93) | 🔄 eval running | — |
| libero.pick_cream_cheese | 0.292 (ep88) | 🔄 eval running | — |
| libero.pick_milk | 0.269 (ep93) | 🔄 eval running | — |
| libero.orange_juice | 0.309 (ep90) | 🔄 eval running | — |
| libero.pick_salad_dressing | — | 🔄 training | — |
| libero.pick_tomato_sauce | — | 🔄 training | — |

### π₀ (zero-shot, pi0_libero checkpoint)

| Task | Status | Success Rate |
|------|--------|-------------|
| libero.pick_alphabet_soup | 🔄 45/99 eps | — |
| libero.pick_bbq_sauce | ⏳ queued | — |
| libero.pick_butter | ⏳ queued | — |
| libero.pick_chocolate_pudding | ⏳ queued | — |
| libero.pick_cream_cheese | ⏳ queued | — |
| libero.pick_milk | ⏳ queued | — |
| libero.orange_juice | ⏳ queued | — |
| libero.pick_salad_dressing | ⏳ queued | — |
| libero.pick_tomato_sauce | ⏳ queued | — |

### OpenVLA (zero-shot, openvla-7b)

| Task | Status | Success Rate |
|------|--------|-------------|
| — | ⏳ model downloading (2/3 shards) | — |

---

## Progress Timeline

| Date | Event |
|------|-------|
| 2026-03-08 | zarr data ready for all 9 tasks |
| 2026-03-08 | ACT infra verified (zarr v3 fix, 1-epoch dry run) |
| 2026-03-08 | ACT training completed for 7/9 tasks (salad+tomato zarr was empty, fixed+retraining) |
| 2026-03-08 | ACT eval running (fixed: curobo→removed, CUDA tensor→.cpu()) |
| 2026-03-08 | openpi installed, pi0_libero downloaded (9.8GB) |
| 2026-03-08 | π₀ eval running (fixed: pyroki→removed, 7-dim action decode, --state-dim 8) |
| 2026-03-08 | OpenVLA model downloading (~15GB, 2/3 shards done) |

---

## Known Issues (resolved)

1. ~~**π₀ state-dim mismatch**~~ → Fixed: `--state-dim 8` in eval_libero.sh
2. ~~**ACT eval curobo dependency**~~ → Fixed: removed unused `get_curobo_models()` call
3. ~~**ACT eval CUDA tensor**~~ → Fixed: `.cpu()` before passing actions to MuJoCo
4. ~~**π₀ pyroki/jax version conflict**~~ → Fixed: removed unused `setup_ik_solver()` call
5. ~~**π₀ action decode mismatch**~~ → Fixed: pi0_libero outputs 7-dim arm joints (not 9-dim)

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

**Step 4: Run eval (tmux window pi0_eval)**
```bash
# Terminal 2
bash roboverse_learn/vla/pi0/eval_libero.sh
# Logs: claude/log/pi0_{task}.log
# Output: claude/out/pi0_eval/{task}/
```

> ⚠️ **Known issue**: `pi0_libero` uses 8-dim state (libero_policy), but `pi_eval.py` sends
> 9-dim (roboverse_policy). Need to resolve before running: either use `pi0_libero` config
> on server (which normalizes based on LIBERO stats), or adapt pi_eval.py to truncate state.
> **Status: TODO — discuss approach before running.**

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

| Task | Checkpoint | Success Rate | Notes |
|------|-----------|-------------|-------|
| libero.pick_alphabet_soup | 🔄 training | — | Epoch ~30/100 |
| libero.pick_bbq_sauce | ⏳ pending | — | |
| libero.pick_butter | ⏳ pending | — | |
| libero.pick_chocolate_pudding | ⏳ pending | — | |
| libero.pick_cream_cheese | ⏳ pending | — | |
| libero.pick_milk | ⏳ pending | — | |
| libero.orange_juice | ⏳ pending | — | |
| libero.pick_salad_dressing | ⏳ pending | — | |
| libero.pick_tomato_sauce | ⏳ pending | — | |

### π₀ (zero-shot, pi0_libero checkpoint)

| Task | Success Rate | Notes |
|------|-------------|-------|
| — | ⏳ blocked (state-dim mismatch, see Known Issues) | Download: 4.8G/11.2G |

### OpenVLA (zero-shot, openvla-7b)

| Task | Success Rate | Notes |
|------|-------------|-------|
| — | 🔄 dry run in progress | |

---

## Progress Timeline

| Date | Event |
|------|-------|
| 2026-03-08 | zarr data ready for all 9 tasks |
| 2026-03-08 | ACT infra verified (zarr v3 fix, 1-epoch dry run) |
| 2026-03-08 | ACT training started in tmux:act_libero |
| 2026-03-08 | openpi installed, pi0_roboverse_lora config registered |
| 2026-03-08 | OpenVLA conda env ready |
| 2026-03-08 | pi0_libero download in progress (4.8G/11.2G) |

---

## Known Issues

1. **π₀ state-dim mismatch**: `pi0_libero` checkpoint expects 8-dim state (LIBERO format),
   `pi_eval.py` sends 9-dim RoboVerse format. Options:
   - (A) Use `pi0_libero` config on server + modify `pi_eval.py` to send 8-dim state
   - (B) Use `pi0_roboverse_lora` config (but needs pi0_base checkpoint, not pi0_libero)
   - **Recommended: Option A** — truncate last dim in pi_eval.py (gripper=1 in LIBERO format)

2. **ACT eval**: `act_eval_runner.py` uses `get_curobo_models` which may require curobo;
   verify before eval starts.

3. **OpenVLA VRAM**: openvla-7b needs ~20GB VRAM; RTX 4090 (24GB) should be sufficient
   but leaves little headroom if ACT also runs. Consider scheduling sequentially.

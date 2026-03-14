# Train and Evaluate Diffusion Policy on Libero in RoboVerse

## Pipeline Overview

```
STEP 1: Collect Demos  →  STEP 2: Convert Data  →  STEP 3: Train  →  STEP 4: Evaluate
  collect_demo.py           data2zarr_dp.py          dp/main.py        dp/main.py
```

---

## Available Libero Tasks

The LIBERO benchmark defines 4 standard evaluation suites. Current RoboVerse availability:

| Suite | Available | Task Count | Location |
|-------|-----------|------------|----------|
| **LIBERO-Object** | ✅ | 10 | `roboverse_pack/tasks/libero/` |
| **LIBERO-90** | ✅ | 65 | `roboverse_pack/tasks/libero_90/` |
| **LIBERO-Spatial** | ❌ not implemented | — | — |
| **LIBERO-Goal** | ❌ not implemented | — | — |
| **LIBERO-Long** | ❌ not implemented | — | — |

---

### LIBERO-Object (9 tasks) — `libero/`

Pick different objects and place them in the basket. Same scene layout, different target objects.

```
libero.pick_alphabet_soup       libero.pick_bbq_sauce
libero.pick_butter              libero.pick_chocolate_pudding
libero.pick_cream_cheese        libero.pick_milk
libero.orange_juice             libero.pick_salad_dressing
libero.pick_tomato_sauce
```

> ⚠️ `libero.pick_ketchup` removed: trajectory file has only 50 init states (vs 3000 for other tasks), success rate ~14%, max collectable demos ≈ 7. Not viable for training.

- Max steps: **250**
- Base class: `LiberoBaseTask`
- Success check: `RelativeBboxDetector` (object inside basket)

---

### LIBERO-90 (65 tasks) — `libero_90/`

Multi-step manipulation across kitchen and living room scenes. Includes articulated objects (cabinets, drawers, microwave).

```
libero_90.kitchen_scene{1-10}_*       # 46 kitchen tasks
libero_90.living_room_scene{1-4}_*    # 19 living room tasks
```

- Max steps: **300**
- Base class: `Libero90BaseTask`
- Success check: `JointPosChecker` or `RelativeBboxDetector`

<details>
<summary>Full LIBERO-90 task list (click to expand)</summary>

**Kitchen Scenes**
```
libero_90.kitchen_scene1_open_bottom_drawer
libero_90.kitchen_scene1_open_drawer_put_bowl
libero_90.kitchen_scene1_open_top_drawer
libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate
libero_90.kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet
libero_90.kitchen_scene2_open_the_top_drawer_of_the_cabinet
libero_90.kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate
libero_90.kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate
libero_90.kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate
libero_90.kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet
libero_90.kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle
libero_90.kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl
libero_90.kitchen_scene3_put_the_frying_pan_on_the_stove
libero_90.kitchen_scene3_put_the_moka_pot_on_the_stove
libero_90.kitchen_scene3_turn_on_the_stove
libero_90.kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it
libero_90.kitchen_scene4_close_the_bottom_drawer_of_the_cabinet
libero_90.kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer
libero_90.kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet
libero_90.kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet
libero_90.kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet
libero_90.kitchen_scene4_put_the_wine_bottle_on_the_wine_rack
libero_90.kitchen_scene5_close_the_top_drawer_of_the_cabinet
libero_90.kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet
libero_90.kitchen_scene5_put_the_black_bowl_on_the_plate
libero_90.kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet
libero_90.kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet
libero_90.kitchen_scene6_close_the_microwave
libero_90.kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug
libero_90.kitchen_scene7_open_the_microwave
libero_90.kitchen_scene7_put_the_white_bowl_on_the_plate
libero_90.kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate
libero_90.kitchen_scene8_put_the_right_moka_pot_on_the_stove
libero_90.kitchen_scene8_turn_off_the_stove
libero_90.kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf
libero_90.kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet
libero_90.kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf
libero_90.kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet
libero_90.kitchen_scene9_turn_on_the_stove
libero_90.kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it
libero_90.kitchen_scene10_close_the_top_drawer_of_the_cabinet
libero_90.kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it
libero_90.kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet
libero_90.kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it
libero_90.kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it
libero_90.kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it
```

**Living Room Scenes**
```
libero_90.living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket
libero_90.living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket
libero_90.living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket
libero_90.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket
libero_90.living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket
libero_90.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket
libero_90.living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket
libero_90.living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket
libero_90.living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket
libero_90.living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray
libero_90.living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray
libero_90.living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray
libero_90.living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray
libero_90.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray
libero_90.living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray
libero_90.living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray
libero_90.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray
libero_90.living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray
libero_90.living_room_scene4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray
```

</details>

---

## Step-by-Step Commands

> Example task: `libero.pick_alphabet_soup`, simulator: `mujoco`, observation/action space: `joint_pos`

### STEP 1 — Collect Demonstrations

> ⚠️ **Libero must use `--sim=mujoco`**. Trajectory files were recorded in MuJoCo; cross-simulator replay causes physics mismatch (all-giveup). MuJoCo is also hardcoded to `num_envs=1`. See [issue #81](https://github.com/RoboVerseOrg/RoboVerse/issues/81).

**Single task:**

```bash
MUJOCO_GL=egl python scripts/advanced/collect_demo.py \
  --sim=mujoco --task=libero.pick_alphabet_soup \
  --num_envs=1 --headless \
  --num_demo_success=100 --cust_name=v1
```
run muti-tasks:
`python /home/zhexi/project/RoboVerse/scripts/advanced/collect_demo_multi.py`

**Output**: `roboverse_demo/demo_mujoco/{task}-{cust_name}/robot-franka/success/demo_XXXX/`

---

### STEP 2 — Convert to ZARR Format

```bash
python roboverse_learn/il/data2zarr_dp.py \
  --task_name libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos \
  --expert_data_num 100 \
  --metadata_dir ./roboverse_demo/demo_mujoco/libero.pick_alphabet_soup-test/robot-franka/success \
  --observation_space joint_pos \
  --action_space joint_pos
```

**Output**: `data_policy/libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_100.zarr`

ZARR structure:
```
data_policy/{task_name}.zarr
├── data/
│   ├── head_camera      # (N, 3, 256, 256) image array
│   ├── state            # (N, 9) joint positions or EE state
│   ├── action           # (N, 9) action targets
│   └── episode_ends     # episode boundary indices
└── meta/
```

---

### STEP 3 — Train Diffusion Policy

> ⚠️ **必须先设置 `algo_model` 环境变量**（小写），否则 Hydra 会找不到默认的 `ddpm_model` 配置（该名称不存在）。
>
> ⚠️ **必须在训练时同时指定 `eval_config.policy_runner` 参数**。`dp_eval_runner.py` 在评估时从 checkpoint 的保存 cfg 读取 policy_runner 设置（不是命令行参数），若训练时未指定则保存的是默认值 `action_type=ee`，导致评估时触发未安装的 cuRobo IK 求解器。

```bash
export algo_model="ddpm_dit_model"   # 必须在运行前设置，注意是小写
python roboverse_learn/il/dp/main.py \
  --config-name=dp_runner.yaml \
  task_name=libero.pick_alphabet_soup \
  dataset_config.zarr_path="./data_policy/libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_100.zarr" \
  eval_config.policy_runner.obs.obs_type=joint_pos \
  eval_config.policy_runner.action.action_type=joint_pos \
  eval_config.policy_runner.action.delta=False \
  train_config.training_params.num_epochs=1000 \
  train_enable=True \
  eval_enable=False
```

**Checkpoints**: `info/outputs/DP/libero.pick_alphabet_soup/checkpoints/{epoch}.ckpt`

---

### STEP 4 — Evaluate Policy

> ⚠️ **必须设置 `MUJOCO_GL=egl`**，否则 MuJoCo 渲染会报 `OpenGL platform library not loaded` 错误。
> 默认 eval config 是 `sim=isaacsim` + `obs_type=ee`，需显式覆盖为 `mujoco` + `joint_pos`。

```bash
export algo_model="ddpm_dit_model" && MUJOCO_GL=egl python roboverse_learn/il/dp/main.py \
  --config-name=dp_runner.yaml \
  task_name=libero.pick_alphabet_soup \
  dataset_config.zarr_path="./data_policy/libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_100.zarr" \
  eval_config.eval_args.task=libero.pick_alphabet_soup \
  eval_config.eval_args.sim=mujoco \
  eval_config.eval_args.num_envs=1 \
  eval_config.eval_args.max_step=250 \
  eval_config.policy_runner.obs.obs_type=joint_pos \
  eval_config.policy_runner.action.action_type=joint_pos \
  eval_config.policy_runner.action.delta=False \
  train_enable=False \
  eval_enable=True \
  eval_path="./info/outputs/DP/libero.pick_alphabet_soup/checkpoints/100.ckpt"
```

**输出**:
- `tmp/{ckpt_name}/{demo_idx}.mp4` — 每条轨迹的推理视频
- `tmp/{ckpt_name}/final_stats.txt` — 最终成功率汇总

---

## Key Configuration Parameters

### Data Collection

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--sim` | Simulator backend | `mujoco` (Libero), `isaacsim` (other tasks) |
| `--num_demo_success` | Target successful demo count | e.g., `100`, `200` |
| `--num_envs` | Parallel environments | `1`, `4`, `8` |
| `--enable_randomization` | Domain randomization | flag |
| `--cust_name` | Custom label for output dir | any string |

### Data Conversion

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--observation_space` | Observation representation | `joint_pos`, `ee` |
| `--action_space` | Action representation | `joint_pos`, `ee` |
| `--delta_ee` | Use delta EE actions | `0` (absolute), `1` (relative) |
| `--expert_data_num` | Number of demos to process | e.g., `100` |

**Observation/Action space details**:

| Space | Obs shape | Action shape | Notes |
|-------|-----------|--------------|-------|
| `joint_pos` | `[9]` joint qpos | `[9]` qpos target | Simple, fast |
| `ee` | `[9]` = pos(3)+quat(4)+gripper(2) | `[9]` EE target | Better generalization |

### Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_epochs` | Training epochs | `1000` |
| `batch_size` | Dataloader batch size | `32` |
| `lr` | Learning rate | `1e-4` |
| `rollout_every` | Eval frequency (epochs) | `50` |
| `checkpoint_every` | Save frequency (epochs) | `50` |

### Model Selection

| Model (`algo_model`) | Backbone | Speed | Quality |
|----------------------|----------|-------|---------|
| `ddpm_dit_model` | DiT | Medium | ⭐⭐⭐⭐⭐ Recommended |
| `fm_dit_model` | DiT | Fast | ⭐⭐⭐⭐⭐ Fast convergence |
| `ddpm_unet_model` | UNet | Slow | ⭐⭐⭐⭐ Classic DP |
| `fm_unet_model` | UNet | Fast | ⭐⭐⭐⭐ |
| `ddim_unet_model` | UNet | Fast inference | ⭐⭐⭐⭐ |
| `vita_model` | MLP | Fastest | ⭐⭐⭐ Lightweight |

Set model via environment variable **before** running (note: **lowercase** `algo_model`, as defined in `dp_runner.yaml`):
```bash
export algo_model="ddpm_dit_model"   # lowercase! oc.env:algo_model,ddpm_model
```

> ⚠️ The default fallback `ddpm_model` does **not** exist — always set this variable explicitly.

---

## Using the Shell Scripts (Recommended)

The IL directory provides convenience scripts that wrap both steps:

```bash
# Edit task name and parameters, then run
vim roboverse_learn/il/collect_demo.sh   # set task_name_set=libero.pick_alphabet_soup
bash roboverse_learn/il/collect_demo.sh  # runs collect + data2zarr

vim roboverse_learn/il/dp/dp_run.sh     # set task, model, epochs
bash roboverse_learn/il/dp/dp_run.sh    # runs train + eval
```

---

## Output Directory Structure

```
roboverse_demo/
└── demo_mujoco/
    └── libero.pick_alphabet_soup-test/
        └── robot-franka/success/
            ├── demo_0000/
            │   ├── metadata.json
            │   └── rgb.mp4
            └── demo_0001/ ...

data_policy/
└── libero.pick_alphabet_soupFrankaL0_obs:joint_pos_act:joint_pos_100.zarr

info/outputs/DP/
└── libero.pick_alphabet_soup/
    ├── checkpoints/
    │   ├── 50.ckpt
    │   ├── 100.ckpt
    │   └── last.ckpt
    └── logs.json.txt
```

---

## Key Source Files

### Demo Collection

| File | Description |
|------|-------------|
| `scripts/advanced/collect_demo.py` | Main entry point for demo collection. Loads a task, runs the simulator, records successful trajectories as `metadata.json` + `rgb.mp4` per episode. |
| `roboverse_learn/il/collect_demo.sh` | Shell wrapper for collection + conversion. Edit `task_name_set`, `sim_set`, `num_envs` here, then run to produce the zarr dataset in one step. |

### Demo Output Files (per `demo_XXXX/`)

| File | Description |
|------|-------------|
| `metadata.json` | Per-step robot state: joint positions (`qpos`), end-effector pose, gripper state, and action targets. Shape: `(T, 9)` per field. |
| `rgb.mp4` | Head camera video of the episode. Frame size: 256×256, used for visual debugging. |

### Data Processing

| File | Description |
|------|-------------|
| `roboverse_learn/il/data2zarr_dp.py` | Converts `metadata.json` + `rgb.mp4` into a ZARR dataset. Stacks all episodes, writes `data/state`, `data/action`, `data/head_camera`, and `meta/episode_ends`. |

### Training & Evaluation

| File | Description |
|------|-------------|
| `roboverse_learn/il/dp/main.py` | Hydra entry point. Instantiates `DPRunner` from config and calls `runner.run()`. |
| `roboverse_learn/il/dp/runner/dp_runner.py` | Core training loop (`train()` at L233) and evaluation loop (`evaluate()` at L484). Loads dataset, builds model, runs epochs, saves checkpoints. |
| `roboverse_learn/il/dp/eval_runner/dp_eval_runner.py` | Evaluation-only runner. Uses EMA model weights, maintains a sliding `deque` of `n_obs_steps` observations, queries policy for actions in rollout. |
| `roboverse_learn/il/dp/dp_run.sh` | Shell wrapper for train + eval. Set `task_name_set`, `sim_set`, `algo_model`, `num_epochs` here. |

### Configuration

| File | Description |
|------|-------------|
| `roboverse_learn/il/dp/configs/dp_runner.yaml` | Master Hydra config. Defines observation shape (`[3,256,256]` + `[9]`), action shape `[9]`, horizon=8, n_obs_steps=3, n_action_steps=4. Selects model via `$ALGO_MODEL` env var. |
| `roboverse_learn/il/dp/configs/model_config/` | Per-model configs: `ddpm_dit_model.yaml`, `fm_dit_model.yaml`, `ddpm_unet_model.yaml`, etc. Each defines network architecture and noise scheduler. |

### Task Definitions

| File | Description |
|------|-------------|
| `roboverse_pack/tasks/libero/libero_base.py` | Base class for all LIBERO-Object tasks. Defines camera setup, scene layout, and `RelativeBboxDetector` success check (object inside basket). |
| `roboverse_pack/tasks/libero_90/libero_90_base.py` | Base class for LIBERO-90 tasks. Supports articulated objects (`ArticulationObjCfg`) and `JointPosChecker` for drawer/cabinet success detection. |

---

## Environment Setup

### Required: `.venv311` (Python 3.11)

IsaacSim requires Python 3.11. Use the project's pre-configured virtual environment:

```bash
source /home/zhexi/project/RoboVerse/.venv311/bin/activate
```

### Installed Components (verified)

| Component | Version | Status |
|-----------|---------|--------|
| IsaacSim | 5.0.0.0 | ✅ |
| IsaacLab | 0.45.9 | ✅ (`third_party/IsaacLab221`) |
| torch | 2.7.0+cu128 | ✅ |
| zarr | v3 | ✅ (v3 compat patches applied) |
| diffusers | 0.36 | ✅ (scheduler device fix applied) |

### Verify IsaacLab

```bash
source .venv311/bin/activate
python -c "import isaaclab; print(f'IsaacLab {isaaclab.__version__} ready')"
# Expected: IsaacLab 0.45.9 ready
```

### W&B Login (required before training)

```bash
wandb login
```

---


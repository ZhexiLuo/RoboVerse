#!/bin/bash
# Collect demos + convert zarr for randomization level ablation experiment
# Task: libero.pick_chocolate_pudding, Levels: L0~L3
# Usage: bash scripts/advanced/rand_level_collect.sh

set -e

TASK="libero.pick_chocolate_pudding"
LEVELS=(0 1 2 3)
PYTHON=".venv311/bin/python3"
NUM_DEMOS=99
OBS_SPACE=joint_pos
ACT_SPACE=joint_pos

mkdir -p claude/log

# ── Randomization flags per level ──────────────────────────────────────────
# Defaults in collect_demo.py: physics=True materials=True cameras=True lights=False
rand_flags() {
  case $1 in
    0) echo "" ;;
    1) echo "--enable_randomization --no-randomize_materials --no-randomize_cameras" ;;
    2) echo "--enable_randomization --no-randomize_cameras" ;;
    3) echo "--enable_randomization" ;;
  esac
}

# ── Skip if 99+ demos already collected ────────────────────────────────────
demos_done() {
  local dir="roboverse_demo/demo_mujoco/${1}/robot-franka/success"
  [[ -d "$dir" ]] && [[ $(ls "$dir" | wc -l) -ge $NUM_DEMOS ]]
}

for level in "${LEVELS[@]}"; do
  TASK_L="${TASK}-L${level}"
  ZARR_PATH="data_policy/${TASK}FrankaL${level}_obs:${OBS_SPACE}_act:${ACT_SPACE}_${NUM_DEMOS}.zarr"
  LOG="claude/log/rand_level_collect_L${level}.log"

  echo "════════════════════════════════════════════════════"
  echo "🚀 Level ${level}: ${TASK_L}"
  echo "════════════════════════════════════════════════════"

  # ── L0: reuse existing demo dir (collected without -L suffix) ──────────
  if [[ $level -eq 0 ]]; then
    DEMO_DIR="roboverse_demo/demo_mujoco/${TASK}/robot-franka/success"
    echo "ℹ️  L0: reusing existing demo dir → ${DEMO_DIR}"
  else
    DEMO_DIR="roboverse_demo/demo_mujoco/${TASK_L}/robot-franka/success"
  fi

  # ── STEP 1: Collect Demos ───────────────────────────────────────────────
  if demos_done "${TASK_L}" || ([[ $level -eq 0 ]] && demos_done "${TASK}"); then
    echo "✅ [skip collect] demos already done"
  else
    echo "📦 Collecting demos L${level}..."
    MUJOCO_GL=egl $PYTHON scripts/advanced/collect_demo.py \
      --sim=mujoco \
      --task=${TASK} \
      --num_envs=1 \
      --headless \
      --num_demo_success=${NUM_DEMOS} \
      --custom_save_dir="roboverse_demo/demo_mujoco/${TASK_L}/robot-franka" \
      --run_unfinished \
      $(rand_flags $level) \
      2>&1 | tee -a "${LOG}"
    echo "✅ Collection done"
  fi

  # ── STEP 2: Convert to Zarr ─────────────────────────────────────────────
  if [[ -d "${ZARR_PATH}" ]]; then
    echo "✅ [skip zarr] ${ZARR_PATH} already exists"
  else
    echo "🔄 Converting to zarr..."
    $PYTHON roboverse_learn/il/data2zarr_dp.py \
      --task_name="${TASK}FrankaL${level}_obs:${OBS_SPACE}_act:${ACT_SPACE}" \
      --expert_data_num=${NUM_DEMOS} \
      --metadata_dir="${DEMO_DIR}" \
      --observation_space=${OBS_SPACE} \
      --action_space=${ACT_SPACE} \
      2>&1 | tee -a "${LOG}"
    echo "✅ Zarr done → ${ZARR_PATH}"
  fi

  echo "✅ Level ${level} complete. Log: ${LOG}"
done

echo ""
echo "🎉 All levels collected! Next step:"
echo "   bash roboverse_learn/il/dp/dp_random_level_run.sh"

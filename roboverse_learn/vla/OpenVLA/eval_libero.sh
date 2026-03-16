#!/bin/bash
# OpenVLA eval on LIBERO-10 tasks
# Usage: bash roboverse_learn/vla/OpenVLA/eval_libero.sh [gpu_id] [num_episodes] [max_steps]
#
# ⚠️ MUST use openvla conda env (torch 2.2.0+cu121).
#    torch 2.10+cu128 (.venv311) causes EGL segfault when combined with transformers.

set -e

ROBOVERSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROBOVERSE_ROOT"

GPU_ID=${1:-4}
NUM_EPISODES=${2:-5}
MAX_STEPS=${3:-500}
CKPT="${OPENVLA_CKPT:-roboverse_learn/vla/OpenVLA/runs/openvla-7b+bridge_orig+b16+lr-0.0005+lora-r32+dropout-0.0}"
OUT_ROOT="/mnt2/rbv/out/openvla"
LOG_DIR="$ROBOVERSE_ROOT/claude/log"
PYTHON="/home/zhexi/.conda/envs/openvla/bin/python"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$ROBOVERSE_ROOT:$PYTHONPATH"

TASKS=(
    "libero.pick_alphabet_soup"
    "libero.pick_bbq_sauce"
    "libero.pick_butter"
    "libero.pick_chocolate_pudding"
    "libero.pick_cream_cheese"
    "libero.pick_milk"
    "libero.orange_juice"
    "libero.pick_salad_dressing"
    "libero.pick_tomato_sauce"
)

echo "🤖 OpenVLA Eval (LIBERO-10)"
echo "  GPU: $GPU_ID | Episodes: $NUM_EPISODES | Max steps: $MAX_STEPS"
echo "  Checkpoint: $CKPT"
echo "  Output: $OUT_ROOT"
echo ""

mkdir -p "$OUT_ROOT" "$LOG_DIR"

for task in "${TASKS[@]}"; do
    task_out="$OUT_ROOT/$task"
    mkdir -p "$task_out"
    echo "📊 [$(date +%H:%M:%S)] $task"

    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u \
        roboverse_learn/vla/OpenVLA/vla_eval.py \
        --model_path "$CKPT" \
        --task "$task" \
        --robot franka --sim mujoco \
        --num_episodes $NUM_EPISODES --max_steps $MAX_STEPS \
        --output_dir "$task_out" \
        2>&1 | tee "$LOG_DIR/openvla_eval_${task}.log"

    sr=$(grep -oP "Success rate: \K[0-9.]+%" "$LOG_DIR/openvla_eval_${task}.log" | tail -1 || echo "N/A")
    echo "  ✅ $task → $sr"
done

echo ""
echo "🎉 OpenVLA eval complete! Results: $OUT_ROOT"

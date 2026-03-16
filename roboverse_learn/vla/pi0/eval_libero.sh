#!/bin/bash
# Pi0.5 eval on LIBERO-10 tasks
# Usage: bash roboverse_learn/vla/pi0/eval_libero.sh [num_episodes] [max_steps]
# Requires: policy server running (auto-started via tmux)

set -e

ROBOVERSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_ROOT="$ROBOVERSE_ROOT/third_party/openpi"
cd "$ROBOVERSE_ROOT"

NUM_EPISODES=${1:-5}
MAX_STEPS=${2:-500}
CONFIG="pi05_roboverse_lora"
CKPT_BASE="$OPENPI_ROOT/checkpoints/$CONFIG/libero10_pi05_lora"
OUT_ROOT="/mnt2/rbv/out/pi05"
LOG_DIR="$ROBOVERSE_ROOT/claude/log"
PYTHON="$ROBOVERSE_ROOT/.venv311/bin/python"

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

# Auto-detect latest checkpoint step dir
CKPT_DIR=$(ls -d "$CKPT_BASE"/[0-9]* 2>/dev/null | sort -t/ -k$(echo "$CKPT_BASE" | tr -cd '/' | wc -c | xargs -I{} expr {} + 2) -n | tail -1)
[ -z "$CKPT_DIR" ] && CKPT_DIR="$CKPT_BASE"

echo "🧠 Pi0.5 Eval (LIBERO-10)"
echo "  Episodes: $NUM_EPISODES | Max steps: $MAX_STEPS"
echo "  Checkpoint: $CKPT_DIR"
echo "  Output: $OUT_ROOT"
echo ""

# Start policy server
echo "📡 Starting policy server..."
tmux kill-session -t pi05_server 2>/dev/null || true
tmux new-session -d -s pi05_server \
    "cd $OPENPI_ROOT && CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/serve_policy.py policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT_DIR 2>&1 | tee $LOG_DIR/pi05_serve.log"
echo "⏳ Waiting 90s for server..."
sleep 90

mkdir -p "$OUT_ROOT" "$LOG_DIR"

for task in "${TASKS[@]}"; do
    task_out="$OUT_ROOT/$task"
    mkdir -p "$task_out"
    echo "📊 [$(date +%H:%M:%S)] $task"

    CUDA_VISIBLE_DEVICES=0 $PYTHON -u \
        roboverse_learn/vla/pi0/pi_eval.py \
        --task "$task" \
        --robot franka --sim mujoco \
        --policy-host localhost --policy-port 8000 \
        --num_episodes $NUM_EPISODES --max_steps $MAX_STEPS \
        --output-dir "$task_out" \
        2>&1 | tee "$LOG_DIR/pi05_eval_${task}.log"

    sr=$(grep -oP "Success rate: \K[0-9.]+%" "$LOG_DIR/pi05_eval_${task}.log" | tail -1 || echo "N/A")
    echo "  ✅ $task → $sr"
done

echo "🧹 Stopping policy server..."
tmux kill-session -t pi05_server 2>/dev/null || true
echo "🎉 Pi0.5 eval complete! Results: $OUT_ROOT"

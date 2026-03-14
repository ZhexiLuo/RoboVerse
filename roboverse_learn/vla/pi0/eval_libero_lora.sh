#!/bin/bash
# Eval pi0 LoRA checkpoint on all libero-10 tasks
# Usage: bash eval_libero_lora.sh [checkpoint_dir]
# If checkpoint_dir not given, auto-detects latest checkpoint

set -e

ROBOVERSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_ROOT="$ROBOVERSE_ROOT/third_party/openpi"
OUTPUT_ROOT="$ROBOVERSE_ROOT/claude/out/pi0"
LOG_DIR="$ROBOVERSE_ROOT/claude/log"
CONFIG="pi05_roboverse_lora"
CKPT_BASE="$OPENPI_ROOT/checkpoints/$CONFIG/libero10_pi05_lora"

# Auto-detect checkpoint dir (latest numbered step dir)
if [ -n "$1" ]; then
    CKPT_DIR="$1"
else
    CKPT_DIR="$CKPT_BASE"
fi

echo "🚀 Starting pi0 LoRA eval"
echo "  Checkpoint: $CKPT_DIR"
echo "  Output:     $OUTPUT_ROOT"
echo ""

# Step 1: Start policy server
echo "📡 Starting policy server..."
tmux kill-session -t pi0_server 2>/dev/null || true
tmux new-session -d -s pi0_server
tmux send-keys -t pi0_server \
    "cd $OPENPI_ROOT && uv run scripts/serve_policy.py policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT_DIR" \
    Enter

echo "⏳ Waiting 60s for server to start..."
sleep 60

# Step 2: Run eval for all 9 tasks
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

cd "$ROBOVERSE_ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

for task in "${TASKS[@]}"; do
    echo "=== Evaluating $task ==="
    MUJOCO_GL=egl .venv311/bin/python roboverse_learn/vla/pi0/pi_eval.py \
        --task "$task" \
        --robot franka --sim mujoco \
        --policy-host localhost --policy-port 8000 \
        --num_episodes 10 --max_steps 250 \
        --output-dir "$OUTPUT_ROOT/$task" \
        2>&1 | tee "$LOG_DIR/pi0_lora_${task}.log"
done

echo ""
echo "✅ Eval complete! Results in $OUTPUT_ROOT"

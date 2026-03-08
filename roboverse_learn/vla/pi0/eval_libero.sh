#!/bin/bash
# Batch π₀ zero-shot eval on 9 libero-10 tasks
# Uses pi0_libero checkpoint (trained on LIBERO, 8-dim state)
# Pre-requisite: start the π₀ policy server first:
#   cd third_party/openpi
#   uv run scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi0_libero \
#     --policy.dir=./checkpoints/pi0_libero
# Then run this script from RoboVerse root:
#   bash roboverse_learn/vla/pi0/eval_libero.sh

policy_host="${POLICY_HOST:-localhost}"
policy_port="${POLICY_PORT:-8000}"
gpu_id="${GPU_ID:-0}"

task_names=(
  libero.pick_alphabet_soup
  libero.pick_bbq_sauce
  libero.pick_butter
  libero.pick_chocolate_pudding
  libero.pick_cream_cheese
  libero.pick_milk
  libero.orange_juice
  libero.pick_salad_dressing
  libero.pick_tomato_sauce
)

mkdir -p claude/log claude/out/pi0_eval

# Wait for server to be ready
echo "⏳ Waiting for π₀ server at ${policy_host}:${policy_port}..."
for i in $(seq 1 30); do
  if nc -z ${policy_host} ${policy_port} 2>/dev/null; then
    echo "✅ Server ready!"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "❌ Server not reachable after 30s. Start the server first."
    exit 1
  fi
  sleep 1
done

for task in "${task_names[@]}"; do
  echo ""
  echo "========================================"
  echo "🤖 π₀ eval: ${task}"
  echo "========================================"

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${gpu_id} \
  .venv311/bin/python roboverse_learn/vla/pi0/pi_eval.py \
    --task ${task} \
    --robot franka --sim mujoco \
    --policy-host ${policy_host} \
    --policy-port ${policy_port} \
    --num_episodes 99 --max_steps 350 \
    --state-dim 8 \
    --output-dir claude/out/pi0_eval/${task} \
    2>&1 | tee claude/log/pi0_${task}.log

  echo "✅ Done: ${task}"
done

echo ""
echo "🎉 All π₀ evaluations completed!"
echo "Results in: claude/out/pi0_eval/"

#!/bin/bash
# Batch OpenVLA zero-shot eval on 9 libero-10 tasks
# Usage: conda activate openvla && bash roboverse_learn/vla/OpenVLA/eval_libero.sh [gpu_id]
# Note: Run from RoboVerse project root

gpu_id=${1:-0}
model_path="${MODEL_PATH:-openvla/openvla-7b}"

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

mkdir -p claude/log claude/out/openvla_eval

for task in "${task_names[@]}"; do
  echo ""
  echo "========================================"
  echo "🤖 OpenVLA eval: ${task}"
  echo "========================================"

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${gpu_id} \
  python roboverse_learn/vla/OpenVLA/vla_eval.py \
    --model_path ${model_path} \
    --task ${task} \
    --robot franka --sim mujoco \
    --solver pyroki \
    --num_envs 1 --num_episodes 99 --max_steps 350 \
    --output_dir claude/out/openvla_eval/${task} \
    2>&1 | tee claude/log/openvla_${task}.log

  echo "✅ Done: ${task}"
done

echo ""
echo "🎉 All OpenVLA evaluations completed!"
echo "Results in: claude/out/openvla_eval/"

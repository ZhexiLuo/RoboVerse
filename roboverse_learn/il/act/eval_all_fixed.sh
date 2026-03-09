#!/bin/bash
# Re-eval all 9 libero-10 ACT tasks with fixed joint order (20 evals each)
# Usage: bash roboverse_learn/il/act/eval_all_fixed.sh

BASE="info/outputs/ACT/2026.03.09"

declare -A TASKS_CKPTS
TASKS_CKPTS["libero.pick_alphabet_soup"]="${BASE}/03.22.10_libero.pick_alphabet_soup_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.pick_bbq_sauce"]="${BASE}/03.31.46_libero.pick_bbq_sauce_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.pick_butter"]="${BASE}/03.38.58_libero.pick_butter_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.pick_chocolate_pudding"]="${BASE}/03.42.08_libero.pick_chocolate_pudding_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.pick_cream_cheese"]="${BASE}/03.45.13_libero.pick_cream_cheese_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.pick_milk"]="${BASE}/03.48.24_libero.pick_milk_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.orange_juice"]="${BASE}/03.51.40_libero.orange_juice_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.pick_salad_dressing"]="${BASE}/03.54.56_libero.pick_salad_dressing_obs:joint_pos_act:joint_pos_chunk20_99"
TASKS_CKPTS["libero.pick_tomato_sauce"]="${BASE}/03.58.08_libero.pick_tomato_sauce_obs:joint_pos_act:joint_pos_chunk20_99"

for task in "${!TASKS_CKPTS[@]}"; do
  ckpt="${TASKS_CKPTS[$task]}"
  echo ""
  echo "========================================"
  echo "📊 Eval: ${task} (20x)"
  echo "========================================"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
  .venv311/bin/python -m roboverse_learn.il.act.act_eval_runner \
    --task ${task} \
    --robot franka \
    --num_envs 1 \
    --sim mujoco \
    --algo act \
    --ckpt_path ./${ckpt} \
    --headless True \
    --num_eval 20 \
    --temporal_agg True \
    --chunk_size 20 \
    2>&1 | tee claude/log/act_eval_fixed_${task}.log

  sr=$(grep "Success Rate:" claude/log/act_eval_fixed_${task}.log | tail -1)
  echo "✅ ${task}: ${sr}"
done

echo ""
echo "🎉 All re-evals done!"

#!/bin/bash
# Train remaining 7 libero-10 tasks with ACT 300 epochs
# Usage: bash roboverse_learn/il/act/act_run_libero_remaining.sh [gpu_id]

gpu_id=${1:-0}

task_names=(
  libero.pick_butter
  libero.pick_chocolate_pudding
  libero.pick_cream_cheese
  libero.pick_milk
  libero.orange_juice
  libero.pick_salad_dressing
  libero.pick_tomato_sauce
)

# ACT hyperparameters
expert_data_num=99
chunk_size=20
kl_weight=10
hidden_dim=512
lr=1e-5
batch_size=8
dim_feedforward=3200
num_epochs=300
obs_space=joint_pos
act_space=joint_pos
sim=mujoco

extra="obs:${obs_space}_act:${act_space}"

for task in "${task_names[@]}"; do
  echo ""
  echo "========================================"
  echo "🚀 Task: ${task}"
  echo "========================================"

  zarr_path="data_policy/${task}FrankaL0_obs:joint_pos_act:joint_pos_99.zarr"
  if [ ! -d "${zarr_path}" ]; then
    echo "⚠️  Zarr not found: ${zarr_path}, skipping"
    continue
  fi

  echo "=== 🏋️  Training ==="
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${gpu_id} \
  .venv311/bin/python -m roboverse_learn.il.utils.act.train \
    --task_name ${task}_${extra}_chunk${chunk_size} \
    --num_episodes ${expert_data_num} \
    --dataset_dir ${zarr_path} \
    --policy_class ACT --kl_weight ${kl_weight} --chunk_size ${chunk_size} \
    --hidden_dim ${hidden_dim} --batch_size ${batch_size} --dim_feedforward ${dim_feedforward} \
    --num_epochs ${num_epochs} --lr ${lr} --state_dim 9 --seed 42

  if [ $? -ne 0 ]; then
    echo "❌ Training failed for ${task}"
    continue
  fi

  echo "✅ Done: ${task}"
done

echo ""
echo "🎉 All remaining ACT tasks completed!"

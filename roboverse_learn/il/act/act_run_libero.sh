#!/bin/bash
# Batch ACT train + eval for all 9 libero-10 tasks
# Usage: bash roboverse_learn/il/act/act_run_libero.sh [gpu_id]

gpu_id=${1:-0}

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

# ACT hyperparameters
expert_data_num=99
chunk_size=20
kl_weight=10
hidden_dim=512
lr=1e-5
batch_size=8
dim_feedforward=3200
num_epochs=100
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

  # Training
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
    echo "❌ Training failed for ${task}, skipping eval"
    continue
  fi

  # Evaluation
  echo "=== 📊 Evaluation ==="
  ckpt_path=$(cat ./roboverse_learn/il/act/ckpt_dir_path.txt 2>/dev/null)
  if [ -z "${ckpt_path}" ]; then
    echo "❌ ckpt_dir_path.txt not found, skipping eval"
    continue
  fi

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${gpu_id} \
  .venv311/bin/python -m roboverse_learn.il.act.act_eval_runner \
    --task ${task} \
    --robot franka \
    --num_envs 1 \
    --sim ${sim} \
    --algo act \
    --ckpt_path ./${ckpt_path} \
    --headless True \
    --num_eval 99 \
    --temporal_agg True \
    --chunk_size ${chunk_size}

  echo "✅ Done: ${task}"
done

echo ""
echo "🎉 All ACT tasks completed!"

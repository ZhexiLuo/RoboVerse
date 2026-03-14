## Randomization level ablation: train + eval one task across L0~L3
## Usage: bash roboverse_learn/il/dp/dp_random_level_run.sh
## Prerequisite: collect demos for each level first (see rand_level_collect.sh)

train_enable=True
eval_enable=True

## Target task
task_name="libero.pick_chocolate_pudding"

## Randomization levels to sweep
levels=(0 1 2 3)

config_name=dp_runner
num_epochs=100
port=50010
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=350         # 250 causes heavy timeout (~14% success); use 350
expert_data_num=99
sim_set=mujoco
eval_ckpt_name=100

## Choose training algorithm
# Supported: ddpm_unet_model, ddpm_dit_model, ddim_unet_model, fm_unet_model, fm_dit_model, score_model, vita_model
export algo_model="ddpm_dit_model"

echo "🎯 Task: ${task_name}"
echo "🔀 Levels: ${levels[*]}"
echo "🤖 Model: $algo_model"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

## Create log directory
mkdir -p roboverse_learn/claude/log

## Sweep all randomization levels sequentially
for level in "${levels[@]}"; do
  task_name_l="${task_name}-L${level}"
  zarr_path="./data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr"
  log_file="roboverse_learn/claude/log/dp_${task_name_l}.log"
  eval_path="./info/outputs/DP/${task_name_l}/checkpoints/${eval_ckpt_name}.ckpt"

  echo "=========================================="
  echo "🚀 Level ${level}: ${task_name_l}"
  echo "   zarr: ${zarr_path}"
  echo "=========================================="

  ## Guard: skip if zarr not found
  if [ ! -d "${zarr_path}" ]; then
    echo "⚠️  [skip] zarr not found: ${zarr_path}"
    echo "   Run rand_level_collect.sh first to collect L${level} demos."
    continue
  fi

  MUJOCO_GL=egl .venv311/bin/python3 ./roboverse_learn/il/dp/main.py \
    --config-name=${config_name}.yaml \
    task_name=${task_name_l} \
    dataset_config.zarr_path="${zarr_path}" \
    train_config.training_params.seed=${seed} \
    train_config.training_params.num_epochs=${num_epochs} \
    train_config.training_params.device=${gpu} \
    eval_config.policy_runner.obs.obs_type=${obs_space} \
    eval_config.policy_runner.action.action_type=${act_space} \
    eval_config.policy_runner.action.delta=${delta_ee} \
    eval_config.eval_args.task=${task_name} \
    eval_config.eval_args.max_step=${eval_max_step} \
    eval_config.eval_args.num_envs=${eval_num_envs} \
    eval_config.eval_args.sim=${sim_set} \
    +eval_config.eval_args.max_demo=${expert_data_num} \
    train_enable=${train_enable} \
    eval_enable=${eval_enable} \
    eval_path=${eval_path} \
    logging.mode=online \
    2>&1 | tee "${log_file}"

  echo "✅ Level ${level} done. Log: ${log_file}"
done

echo ""
echo "🎉 All levels finished!"
echo "📊 Generate comparison report:"
echo "   python roboverse_learn/il/dp/gen_rand_level_report.py --task=${task_name}"

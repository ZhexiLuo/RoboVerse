## Separate training and evaluation
train_enable=True
eval_enable=True

## Batch task list (9 libero-10 pick tasks)
task_names=(
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

level=0
config_name=dp_runner
num_epochs=100
port=50010
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=350         # experimental: 250 causes heavy timeout (14% success), try 350
expert_data_num=99
sim_set=mujoco
eval_ckpt_name=100

## Choose training or inference algorithm
# Supported: ddpm_unet_model, ddpm_dit_model, ddim_unet_model, fm_unet_model, fm_dit_model, score_model, vita_model
export algo_model="ddpm_dit_model"

echo "Selected model: $algo_model"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

## Create log directory
mkdir -p roboverse_learn/claude/log

## Run all tasks sequentially
for task_name_set in "${task_names[@]}"; do
  echo "=========================================="
  echo "🚀 Starting task: ${task_name_set}"
  echo "=========================================="

  log_file="roboverse_learn/claude/log/dp_${task_name_set}.log"
  eval_path="./info/outputs/DP/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"

  MUJOCO_GL=egl python ./roboverse_learn/il/dp/main.py \
    --config-name=${config_name}.yaml \
    task_name=${task_name_set} \
    dataset_config.zarr_path="./data_policy/${task_name_set}FrankaL${level}_${extra}_${expert_data_num}.zarr" \
    train_config.training_params.seed=${seed} \
    train_config.training_params.num_epochs=${num_epochs} \
    train_config.training_params.device=${gpu} \
    eval_config.policy_runner.obs.obs_type=${obs_space} \
    eval_config.policy_runner.action.action_type=${act_space} \
    eval_config.policy_runner.action.delta=${delta_ee} \
    eval_config.eval_args.task=${task_name_set} \
    eval_config.eval_args.max_step=${eval_max_step} \
    eval_config.eval_args.num_envs=${eval_num_envs} \
    eval_config.eval_args.sim=${sim_set} \
    +eval_config.eval_args.max_demo=${expert_data_num} \
    train_enable=${train_enable} \
    eval_enable=${eval_enable} \
    eval_path=${eval_path} \
    logging.mode=online \
    2>&1 | tee "${log_file}"

  echo "✅ Task ${task_name_set} done. Log: ${log_file}"
done

echo ""
echo "🎉 All tasks finished!"
echo "📊 Run gen_report.py to generate report:"
echo "   python roboverse_learn/il/dp/gen_report.py"

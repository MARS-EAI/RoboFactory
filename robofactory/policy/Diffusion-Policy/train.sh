# Examples:
# bash train.sh robot_dp dual_bottles_pick_easy 0 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer_pointcloud 0112 0 0


task_name=${1}
load_num=${2}
agent_id=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

alg_name=robot_dp
# task choices: See TASK.md
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
                                            task.name=${task_name} \
                                            task.dataset.zarr_path="data/zarr_data/${task_name}_Agent${agent_id}_${load_num}.zarr" \
                                            training.debug=$DEBUG \
                                            training.seed=${seed} \
                                            training.device="cuda:0" \
                                            exp_name=${exp_name} \
                                            logging.mode=${wandb_mode}
                                            # checkpoint.save_ckpt=${save_ckpt}
                                            # hydra.run.dir=${run_dir} \
set -x
export CUDA_VISIBLE_DEVICES=1,2
cd /inspire/hdd/global_user/zhangjinghao-240108110057/code/verl
# --- Control WandB Usage ---
# Set USE_WANDB to "false" to disable WandB logging.
USE_WANDB=${USE_WANDB:-"false"}

# Basic Project Settings
WANDB_PRJ_NAME=rlpr
OUTPUT_DIR=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/running/LM/Qwen3-4B/RL
project_name=verl_grpo_example_MATH-lighteval
experiment_name=qwen3_4b_function_rm
MODEL=/inspire/hdd/global_user/zhangjinghao-240108110057/running/LM/Qwen3-4B/RL/verl_grpo_example_MATH-lighteval/qwen3_8b_function_rm/global_step_105/hf
N_GPUS_PER_NODE=2

# Judge Settings
# You can choose one of the following options:

# Rule as Judge (Default)
unset OPENAI_API_KEY
unset OPENAI_API_BASE
export USED_MODEL=${USED_MODEL:-"no_api"}

# # OpenSouce Model as Judge
# export USED_MODEL=Qwen/Qwen2.5-72B-Instruct
# export CLIENT_IP=http://127.0.0.1:8001

# # API-Based Model as Judge
# export OPENAI_API_KEY=your_openai_api_key
# export OPENAI_API_BASE=your_openai_api_base
# export USED_MODEL=gpt-4.1 # or gpt-4o


# Train and Validation Files
TRAIN_FILES=dataset/train/MATH-lighteval/train.parquet
VAL_DIR=${VAL_DIR:-"dataset/eval"}
VAL_FILES=[${VAL_DIR}'/aime24.parquet',${VAL_DIR}'/aime25.parquet',${VAL_DIR}'/amc23.parquet',${VAL_DIR}'/minerva.parquet',${VAL_DIR}'/olympiad.parquet']

# Logging and Checkpointing
VAL_SAVE_RESULTS_DIR=outputs/test_generations/${EXP_NAME}
mkdir -p "${VAL_SAVE_RESULTS_DIR}"

# --- Conditional WandB Setup ---
TRAINER_LOGGER_CONFIG="['console']" # Default logger
declare -a WANDB_PARAMETERS # Array to hold WandB specific parameters

if [ "$USE_WANDB" = "true" ]; then
    echo "WandB logging ENABLED. Make sure you have logged in."
    export WANDB_MODE=online
    export WANDB_DIR_PATH=./wandb # Define path for WandB data
    mkdir -p "${WANDB_DIR_PATH}"

    export WANDB_DIR=${WANDB_DIR_PATH}

    TRAINER_LOGGER_CONFIG="['console','wandb']"
    WANDB_PARAMETERS=(
        "trainer.project_name=$WANDB_PRJ_NAME"
        "trainer.val_generations_to_log_to_wandb=10"
        "+trainer.train_generations_to_log_to_wandb=1"
        "+trainer.train_generations_to_log_to_wandb_2=50"
        "+wandb_dir=${WANDB_DIR}" # Use the exported WANDB_DIR which is ./wandb
    )
else
    echo "WandB logging DISABLED."
    # WANDB_PARAMETERS array remains empty
    # TRAINER_LOGGER_CONFIG remains ['console']
    # Unset WANDB_DIR if you don't want it in the environment when WandB is disabled
    unset WANDB_DIR
fi


# export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

nnodes=${VERL_N_TRAIN_NODE:-1}
KL_COEF=0


# Main Training Command and Configuration
python -m verl.trainer.main_ppo \
    data.val_files=$VAL_FILES \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=32768 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.temperature=0 \
    trainer.experiment_name=$EXP_NAME \
    "${WANDB_PARAMETERS[@]}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=$nnodes \
    +trainer.val_save_results_dir=${VAL_SAVE_RESULTS_DIR} \
    +trainer.val_only=True \
    "$@"
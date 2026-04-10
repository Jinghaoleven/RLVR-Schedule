#!/usr/bin/env bash
set -x
# source /mnt/public/users/zhangjinghao/miniconda3/bin/activate /mnt/public/users/zhangjinghao/miniconda3/envs/verl
ROOT_DIR=/mnt/public/users/zhangjinghao
export WORKING_DIR=/mnt/public/users/zhangjinghao/code/verl
export TENSORBOARD_DIR=/mnt/public/users/zhangjinghao/code/verl/tensorboard_log
cd $WORKING_DIR
export HYDRA_FULL_ERROR=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=ERROR
export NCCL_IB_GID_INDEX=5
export NCCL_IB_TC=138
export NCCL_IB_QPS_PER_CONNECTION=8

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
################## 自动检测主节点 ##################
echo "[INFO] Launching multi-node training"
echo "       Master node: ${MASTER_ADDR}"
echo "       Nodes total: ${WORLD_SIZE}"
echo "       GPUs per node: ${NPROC_PER_NODE}"
echo "       NODE_RANK: ${RANK}"

############################ Quick Config ############################
ROLLOUT_NAME="vllm" # sglang or vllm

STUDENT_MODEL=$ROOT_DIR/models/Qwen3-4B
TEACHER_MODEL=/mnt/public/users/zhuyongfu/model/Qwen3/Qwen3-32B

STUDENT_WORLD_SIZE=4
TEACHER_WORLD_SIZE=4
TEACHER_RESOURCE_POOL=True

# USE_POLICY_GRADIENT=False
# DISTILLATION_LOSS_MODE="k3"
# DISTILLATION_LOSS_MODE="forward_kl_topk"
# USE_FUSED_KERNELS=False

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=32768
MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_BSZ=256
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

ENFORCE_EAGER=False # true for faster debugging

############################ Paths ############################
xcombined_train_path=$WORKING_DIR/dataset/train/XCombined/train.parquet 
MATH500_test_path=$WORKING_DIR/dataset/train/MATH-lighteval/test.parquet 

TRAIN_FILES="['$xcombined_train_path']"
TEST_FILES="['$MATH500_test_path']"

PROJECT_NAME='Qwen3-4B-new'
EXP_NAME="OPD-Qwen3-32B-4B-loss-${DISTILLATION_LOSS_MODE}-pg-${USE_POLICY_GRADIENT}"
############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT_LENGTH
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path=$STUDENT_MODEL
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.teacher_model.nnodes=$WORLD_SIZE
    distillation.teacher_model.model_path=$TEACHER_MODEL
    distillation.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=0.58
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=512
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=1
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=8
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console", "tensorboard"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.default_local_dir=$WORKING_DIR/result/LM/$PROJECT_NAME/$EXP_NAME \
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=$WORLD_SIZE
    trainer.save_freq=25
    trainer.test_freq=300
    trainer.total_epochs=2
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=auto
)


############################ Launch ############################
ray stop
# ================================
# Head node
# ================================
if [ "${RANK}" == "0" ]; then
    echo "Starting Ray HEAD node..."
    ray start --head \
        --node-ip-address=$MASTER_ADDR \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --num-gpus=8 

# ================================
# Worker nodes
# ================================
else
    echo "Starting Ray WORKER node..."
    sleep 20
    ray start \
        --address="$MASTER_ADDR:6379" \
        --num-gpus=8
fi

sleep 50
if [ "${RANK}" == "0" ]; then
    echo "[INFO] Submitting Ray job..."
    ray job submit --address="http://$MASTER_ADDR:8265" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- \
        python3 -m verl.trainer.main_ppo \
        --config-path=config \
        --config-name='ppo_trainer.yaml' \
        "${DATA[@]}" \
        "${ALGORITHM[@]}" \
        "${MODEL[@]}" \
        "${DISTILLATION[@]}" \
        "${ROLLOUT[@]}" \
        "${STUDENT[@]}" \
        "${TRAINER[@]}" \
        "$@"
else
    sleep 10
    echo "Starting Ray worker node..."
    sleep inf
fi
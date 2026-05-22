set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WORKING_DIR=${WORKING_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$WORKING_DIR/tensorboard_log}
echo "WORKING_DIR is set to: $WORKING_DIR"

project_name=qwen3_4b
experiment_name=grpo_tp_schedule_debug
MODEL_PATH=/mnt/public/users/zhangjinghao/models/Qwen3-4B
TRAIN_FILE=${TRAIN_FILE:-$WORKING_DIR/dataset/OpenReasoning/train.parquet}
VAL_FILE=${VAL_FILE:-$WORKING_DIR/dataset/aime25.parquet}
LOG_DIR="$WORKING_DIR/training_logs/$project_name/$experiment_name"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

cd "$WORKING_DIR"
if [ ! -f verl/trainer/runtime_env.yaml ]; then
    echo "[ERROR] WORKING_DIR must point to the RLVR_schedule repo root: $WORKING_DIR" >&2
    exit 1
fi
export HYDRA_FULL_ERROR=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=ERROR
export NCCL_IB_GID_INDEX=5
export NCCL_IB_TC=138
export NCCL_IB_QPS_PER_CONNECTION=8

NPROC_PER_NODE=${NPROC_PER_NODE:-2}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=39936 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.response_key=solution \
    data.schedule_strategy=tp_schedule \
    data.initial_schedule_magnitude=0.99 \
    data.min_schedule_ratio=0 \
    data.max_schedule_ratio=0.8 \
    data.schedule_mode=linear \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=3600 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=False \
    trainer.default_local_dir="$WORKING_DIR/result/LM/$project_name/$experiment_name" \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=25 \
    trainer.test_freq=300 \
    trainer.total_epochs=1
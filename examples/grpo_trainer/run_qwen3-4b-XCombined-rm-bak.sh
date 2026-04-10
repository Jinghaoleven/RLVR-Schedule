# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
# source /mnt/public/users/zhangjinghao/miniconda3/bin/activate /mnt/public/users/zhangjinghao/miniconda3/envs/verl
set -x
ROOT_DIR=/mnt/public/users/zhangjinghao
project_name=qwen3_4b_new
experiment_name=xcombined-rm-grpo

export WORKING_DIR=/mnt/public/users/zhangjinghao/code/verl
export TENSORBOARD_DIR=/mnt/public/users/zhangjinghao/code/verl/tensorboard_log
LOG_DIR=$WORKING_DIR/training_logs/$project_name/$experiment_name
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/$(date +%Y%m%d_%H%M%S).log
cd $WORKING_DIR

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export HF_HUB_OFFLINE=1
export NCCL_DEBUG=ERROR
export NCCL_IB_GID_INDEX=5
export NCCL_IB_TC=138
export NCCL_IB_QPS_PER_CONNECTION=8

MODEL=$ROOT_DIR/models/Qwen3-4B
REWARD_MODEL=$ROOT_DIR/models/Qwen3-4B
# REWARD_MODEL=/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b
# For GPT-OSS
# export VERL_USE_GPT_OSS=1
# export TIKTOKEN_RS_CACHE_DIR=/mnt/public/users/zhangjinghao/code/vllm-cap/models/gpt-oss

ray stop

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$WORKING_DIR/dataset/train/XCombined/train.parquet \
    data.val_files=$WORKING_DIR/dataset/train/MATH-lighteval/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=3600 \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=1 \
    reward.reward_model.model_path=$REWARD_MODEL \
    reward.custom_reward_function.path=tests/experimental/reward_loop/reward_fn.py \
    reward.custom_reward_function.name=compute_score_gt \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.gpu_memory_utilization=0.7 \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.prompt_length=40000 \
    reward.reward_model.rollout.response_length=256 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$WORKING_DIR/result/LM/$project_name/$experiment_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=300 \
    trainer.total_epochs=2 "$@" 2>&1 | tee "$LOG_FILE"

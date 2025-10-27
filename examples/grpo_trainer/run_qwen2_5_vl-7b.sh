set -x
ENGINE=${1:-vllm}
PUB_DIR=/inspire/hdd/global_user/zhangjinghao-240108110057/models
DATA_DIR=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/dataset/LM_data/Verl_data
OUTPUT_DIR=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/running/LVM/Qwen2_5VL-3B/RL
project_name=verl_grpo_geo3k
experiment_name=qwen2_5_vl_3b_function_rm

export WORKING_DIR=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/code/verl
cd $WORKING_DIR
export HYDRA_FULL_ERROR=1

pip install mathruler
pip install sandbox_fusion==0.3.7
pip install math-verify -U

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/geo3k/train.parquet \
    data.val_files=$DATA_DIR/geo3k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$PUB_DIR/Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$OUTPUT_DIR/$project_name/$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

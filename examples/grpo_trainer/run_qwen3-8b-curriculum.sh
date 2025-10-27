# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
PUB_DIR=/inspire/hdd/global_user/zhangjinghao-240108110057/models
OUTPUT_DIR=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/running/LM/Qwen3-4B/RL
project_name=verl_grpo_example_MATH-lighteval
experiment_name=qwen3_4b_function_rm_progress

export WORKING_DIR=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/code/verl
cd $WORKING_DIR
export HYDRA_FULL_ERROR=1

ray stop
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=dataset/train//MATH-lighteval/train.parquet \
    data.val_files=dataset/train//MATH-lighteval/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_response=curriculum \
    data.max_response_ratio=0.8 \
    data.max_curriculum_epoch=5 \
    actor_rollout_ref.model.path=$PUB_DIR/Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$OUTPUT_DIR/$project_name/$experiment_name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
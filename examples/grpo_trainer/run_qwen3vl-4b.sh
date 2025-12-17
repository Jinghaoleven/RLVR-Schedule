# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_FLASH_ATTN_VERSION=3
# ['FLASH_ATTN', 'TRITON_ATTN', 'XFORMERS', 'ROCM_FLASH', 'ROCM_AITER_MLA', 'ROCM_AITER_FA', 'TORCH_SDPA', 'FLASHINFER', 'FLASHINFER_MLA', 'TRITON_MLA', 'CUTLASS_MLA', 'FLASHMLA', 'FLASH_ATTN_MLA', 'PALLAS', 'IPEX', 'DUAL_CHUNK_FLASH_ATTN', 'DIFFERENTIAL_FLASH_ATTN', 'NO_ATTENTION', 'FLEX_ATTENTION', 'TREE_ATTN', 'ROCM_ATTN']

set -x
# export CUDA_VISIBLE_DEVICES=0,1,2,3
ENGINE=${1:-vllm}
ROOT_DIR=/mnt/public/users/zhangjinghao
project_name=qwen3vl_4b
experiment_name=DRIFT-TL-Distill-4K-base-grpo

export WORKING_DIR=/mnt/public/users/zhangjinghao/code/verl
cd $WORKING_DIR
export HYDRA_FULL_ERROR=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# pip install mathruler
# pip install sandbox_fusion==0.3.7
# pip install math-verify -U

ray stop
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=dataset/train/DRIFT-TL-Distill-4K/train.parquet \
    data.val_files=dataset/train/Mathvista/test_mini_box.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.filter_overlong_prompts_workers=16 \
    actor_rollout_ref.model.path=$ROOT_DIR/models/Qwen3-VL-4B-Thinking \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=result/VLM/$project_name/$experiment_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 $@
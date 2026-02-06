# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
ROOT_DIR=/mnt/public/users/zhangjinghao
project_name=qwen3_4b
experiment_name=xcombined-pro-wpg-fk3-advshift-grpo-v0.001

# export CUDA_VISIBLE_DEVICES=0,1,2,3
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

sleep 30
if [ "${RANK}" == "0" ]; then
    echo "[INFO] Submitting Ray job..."
    ray job submit --address="http://$MASTER_ADDR:8265" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- \
        python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$WORKING_DIR/dataset/train/XCombined/train.parquet \
        data.val_files=$WORKING_DIR/dataset/train/MATH-lighteval/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=1024 \
        data.max_response_length=39936 \
        data.filter_overlong_prompts=True \
        data.truncation='right' \
        data.response_key=solution \
        data.trust_response=curriculum \
        data.max_response_ratio=0.99 \
        data.max_curriculum_epoch=130 \
        actor_rollout_ref.model.path=$ROOT_DIR/models/Qwen3-4B \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_sft_loss=True \
        actor_rollout_ref.actor.sft_loss_coef=0.001 \
        actor_rollout_ref.actor.sft_mode=forward_kl_policy_advshift \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=k3 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.4 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.nccl_timeout=3600 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","tensorboard"]' \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.default_local_dir=$WORKING_DIR/result/LM/$project_name/$experiment_name \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=$NPROC_PER_NODE \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=25 \
        trainer.test_freq=300 \
        trainer.total_epochs=2 $@
else
    sleep 10
    echo "Starting Ray worker node..."
    sleep inf
fi
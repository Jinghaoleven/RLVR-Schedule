set -x
ROOT_DIR=/mnt/public/users/zhangjinghao
project_name=qwen3_4b
experiment_name=XCombined-sft

export WORKING_DIR=/mnt/public/users/zhangjinghao/code/verl
cd $WORKING_DIR

export HYDRA_FULL_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nproc_per_node=8

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$WORKING_DIR/dataset/train/XCombined/train.parquet \
    data.val_files=$WORKING_DIR/dataset/train/QuestA-50/train.parquet\
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['solution'] \
    data.train_batch_size=256 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=40960 \
    model.partial_pretrain=$ROOT_DIR/models/Qwen3-4B \
    optim.lr=1e-6 \
    optim.warmup_steps_ratio=0 \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$WORKING_DIR/result/LM/$project_name/$experiment_name \
    trainer.logger='["console","tensorboard"]' \
    trainer.total_epochs=5 $@ \
    trainer.save_freq=117 \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true

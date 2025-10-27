actor_dir=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/running/LM/Qwen3-4B/RL/verl_grpo_example_MATH-lighteval/qwen3_4b_function_rm_progress/global_step_105/actor
target_dir=/inspire/hdd/project/longmemory/zhangjinghao-240108110057/running/LM/Qwen3-4B/RL/verl_grpo_example_MATH-lighteval/qwen3_4b_function_rm_progress/global_step_105/hf
cd /inspire/hdd/project/longmemory/zhangjinghao-240108110057/code/verl
python -m verl.model_merger merge --backend fsdp --local_dir $actor_dir --target_dir $target_dir --private
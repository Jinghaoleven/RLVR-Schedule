actor_dir=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b_instruct/QuestA-25-pro-wsft1.0-grpo/global_step_123/actor
target_dir=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b_instruct/QuestA-25-pro-wsft1.0-grpo/global_step_123/hf
cd /mnt/public/users/zhangjinghao/code/verl
source /mnt/public/users/zhangjinghao/.verl-venv/bin/activate
python -m verl.model_merger merge --backend fsdp --local_dir $actor_dir --target_dir $target_dir --private
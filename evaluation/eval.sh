export CUDA_VISIBLE_DEVICES=1,2
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /inspire/hdd/global_user/zhangjinghao-240108110057/code/verl
model_path=/inspire/hdd/global_user/zhangjinghao-240108110057/running/LM/Qwen3-4B/RL/verl_grpo_example_MATH-lighteval/qwen3_8b_function_rm/global_step_105/hf
topp=0.95
topk=20
temperature=0.6
max_response_length=38912
n=1
python evaluation/eval_vllm_suite.py \
  --model $model_path \
  --n $n \
  --max_length $max_response_length \
  --p $topp \
  --k $topk \
  --t $temperature \
  --experiment_name GRPO-base

python evaluation/grade.py --file_dir evaluation/results/Base
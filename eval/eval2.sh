cd /mnt/public/users/zhangjinghao/code/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_DEEP_GEMM=1
MODEL_NAME=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/xcombined-pro-wsft-grpo-v0.001/global_step_234/hf
EXP_NAME=Qwen3-4B-xcombined-pro-wsft-grpo

python eval/evaluate_parallel.py \
--model_name $MODEL_NAME \
--exp_name $EXP_NAME \
--dataset_name dataset/evaluation_suite \
--temperature 0 \
--top_p 0.95 \
--max_tokens 40960 \
--template qwen_math_box \
--n_samples 1 \
--eval_type pass@k 
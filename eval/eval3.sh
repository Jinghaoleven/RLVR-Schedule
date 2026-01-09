# cd /mnt/public/users/zhangjinghao/code/verl
# export CUDA_VISIBLE_DEVICES=2
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export VLLM_DISABLE_DEEP_GEMM=2

# $PUB_DIR/Qwen2.5-Math-7B
# $PUB_DIR/Qwen2.5-Math-7B-Instruct
# $ROOT_DIR/running/Qwen2.5-Math-7B-GRPO-mathlv3to5_8k
# $ROOT_DIR/running/Qwen2.5-Math-7B-GRPO-Base7BRef-mathlv3to5_8k
# $ROOT_DIR/running/LM/Qwen2.5-Math-7B/Qwen2.5-Math-7B-math3t5-GRPO-Base-KL/ckpt/global_step150_hf
# $ROOT_DIR/running/Qwen2.5-7B-Instruct-math12k-GRPO-Base-KL/ckpt/global_step200_hf
# $ROOT_DIR/running/LM/Qwen2.5-Math-7B/Qwen2.5-Math-7B-math3t5-GRPO-Base-KL/ckpt/global_step150_hf
# $ROOT_DIR/running/Qwen2.5-Math-7B-math3t5-GRPO-idmatchingtag+rm-allnorm-0.8noise-.01-.6clip-cdetach/ckpt/global_step150_hf
# $ROOT_DIR/running/LM/Qwen2.5-Math-7B/Qwen2.5-Math-7B-math3t5-GRPO-idmatchingtag-rm-allnorm-0.8noise-.01-.6clip-30topk-cdetach-mtrain-ccfilter/ckpt/global_step150_hf
# $ROOT_DIR/running/LM/Qwen2_5-Math-7B/RL/Qwen2.5-Math-7B-math_lvl3to5_8k/GRPO-tkprefix-rp-allnorm-seq01-.8tar-.01coef-.6clip-15topk-cdetach-mtrain-bc.0-4lr/ckpt/global_step200_hf
# $ROOT_DIR/running/LM/Qwen2_5-Math-7B/RL/Qwen2.5-Math-7B-math_lvl3to5_8k/GRPO-tkprefix-rp-allnorm-seq11-.8tar-.01coef-.7clip-30topk-cdetach-mtrain-bc.0-4lr-klpln/ckpt/global_step200_hf
# $ROOT_DIR/running/LM/Qwen2_5-Math-7B/RL/Qwen2.5-Math-7B-math_lvl3to5_8k/GRPO-tkprefix-rp-allnorm-seq01-.8tar-.01coef-.7clip-30topk-cdetach-mtrain-bc.0-4lr-klp/ckpt/global_step200_hf
# $ROOT_DIR/running/LM/Qwen2_5-Math-7B/RL/Qwen2.5-Math-7B-math_lvl3to5_8k/Qwen2.5-Math-7B-math3t5-GRPO-Base-KL/ckpt/global_step200_hf
# $ROOT_DIR/running/LM/Qwen2_5-Math-7B/RL/Qwen2.5-Math-7B-math_lvl3to5_8k/Qwen2.5-Math-7B-math3t5-GRPO-entropy-adv/ckpt/global_step200_hf
# $ROOT_DIR/running/LM/Qwen2.5-Math-1.5B/RL/Qwen2.5-Math-1.5B-math_lvl3to5_8k/GRPO-tkprefix-rp-allnorm-seq01-.8tar-.01coef-.6clip-15topk-cdetach-mtrain-bc.0-4lr/ckpt/global_step200_hf
# python $ROOT_DIR/code/RL_code/understand-r1-zero/evaluate_model.py \
# --model_name $ROOT_DIR/running/LM/Qwen2.5-Math-1.5B/RL/Qwen2.5-Math-1.5B-math_lvl3to5_8k/GRPO-tkprefix-rp-allnorm-seq01-.8tar-.01coef-.6clip-15topk-cdetach-mtrain-bc.0-4lr/ckpt/global_step200_hf \
# --dataset_name $ROOT_DIR/code/RL_code/understand-r1-zero/datasets/evaluation_suite \
# --temperature 0.7 \
# --max_tokens 8192 \
# --template qwen_math_box \
# --n_samples 32 \
# --eval_type pass@k 
# --save true \
# --save_path /inspire/hdd/project/longmemory/zhangjinghao-240108110057/running_eval/lm_response

# $ROOT_DIR/running/LM/Qwen2_5-Math-7B/RL/Qwen2.5-Math-7B-math_lvl3to5_8k/Qwen2.5-Math-7B-math3t5-GRPO-Base-KL/ckpt/global_step200_hf
# $PUB_DIR/Qwen/Qwen2.5-Math-7B
# $ROOT_DIR/running/LM/Qwen2_5-Math-7B/RL/Qwen2.5-Math-7B-math_lvl3to5_8k/Qwen2.5-Math-7B-math3t5-GRPO-entropy-adv/ckpt/global_step200_hf
# $PUB_DIR/Qwen/Qwen2.5-Math-1.5B
# $PUB_DIR/meta-llama/Llama-3.1-8B/LLM-Research/Meta-Llama-3.1-8B
# $ROOT_DIR/running/LM/Qwen2.5-Math-1.5B/RL/Qwen2.5-Math-1.5B-math_lvl3to5_8k/GRPO-tkprefix-rp-allnorm-seq01-.8tar-.01coef-.6clip-15topk-cdetach-mtrain-bc.0-4lr/ckpt/global_step200_hf

# source /mnt/public/users/zhangjinghao/.lmvenv/bin/activate
# source /mnt/public/users/zhangjinghao/.verl-venv/bin/activate
# source /mnt/public/users/zhangjinghao/code/project/RLFR/.rlfrvenv/bin/activate

cd /mnt/public/users/zhangjinghao/code/verl
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_DEEP_GEMM=1
MODEL_NAME=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/xcombined-base-grpo/global_step_234/hf
EXP_NAME=Qwen3-4B-xcombined-base-grpo

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
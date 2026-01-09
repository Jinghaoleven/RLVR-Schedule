cd /mnt/public/users/zhangjinghao/code/verl
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_DEEP_GEMM=1
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

# source /mnt/public/users/zhangjinghao/.verl-venv/bin/activate
# source /mnt/public/users/zhangjinghao/.lmvenv/bin/activate
# source /mnt/public/users/zhangjinghao/code/project/RLFR/.rlfrvenv/bin/activate
python eval/evaluate_model.py \
--model_name /mnt/public/users/zhangjinghao/models/Qwen3-4B \
--dataset_name dataset/evaluation_suite \
--temperature 0 \
--top_p 0.95 \
--max_tokens 40960 \
--template qwen_math_box \
--n_samples 1 \
--eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_24_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_36_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_48_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_60_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_72_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_84_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_96_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_108_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_120_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_132_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_144_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_156_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_168_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

# python eval/evaluate_model.py \
# --model_name /mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/limov2-sft-2/global_step_180_hf \
# --dataset_name dataset/evaluation_suite \
# --temperature 0 \
# --max_tokens 24576 \
# --template qwen_math_box \
# --n_samples 1 \
# --eval_type pass@k 

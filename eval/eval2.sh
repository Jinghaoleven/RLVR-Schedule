cd /mnt/public/users/zhangjinghao/code/verl
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_DEEP_GEMM=1
# MODEL_NAME=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/xcombined-pro-wpg-fk1-ISdt-grpo-v0.001/global_step_234/hf
# EXP_NAME=Qwen3-4B-xcombined-pro-wpg-fk1-ISdt-grpo-v0.001
# MODEL_NAME=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/xcombined-pro-on-policy-grpo/global_step_234/actor
# MERGE_PATH="${MODEL_NAME%/actor}/hf"
# EXP_NAME=Qwen3-4B-xcombined-pro-on-policy-grpo-ve
# MERGE_PATH="/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/xcombined-pro-grpo/global_step_234/hf"
# EXP_NAME=Qwen3-4B-xcombined-pro-grpo-v1
# MERGE_PATH=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/xcombined-pro-grpo-v2.4/global_step_234/hf
# EXP_NAME=Qwen3-4B-xcombined-pro-grpo-v2.4-ev1
MERGE_PATH=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b/xcombined-pro-grpo/global_step_234/hf
EXP_NAME=Qwen3-4B-xcombined-pro-grpo-ev2
# MERGE_PATH=/mnt/public/users/zhangjinghao/models/Qwen3-4B 
# EXP_NAME=Qwen3-4B-pass32
source /mnt/public/users/zhangjinghao/miniconda3/bin/activate /mnt/public/users/zhangjinghao/miniconda3/envs/verl

# python -m verl.model_merger merge --backend fsdp --local_dir $MODEL_NAME --target_dir $MERGE_PATH --private
python eval/evaluate_parallel.py \
    --model_name $MERGE_PATH \
    --exp_name $EXP_NAME \
    --dataset_name dataset/evaluation_suite \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 40960 \
    --template qwen_math_box \
    --n_samples 1 \
    --eval_type pass@k
    # --save_path /mnt/public/users/zhangjinghao/code/verl/eval/model_results
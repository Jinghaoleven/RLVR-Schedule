export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# source /mnt/public/users/zhangjinghao/.lmvenv/bin/activate
cd /mnt/public/users/zhangjinghao/code/verl
model_path=/mnt/public/users/zhangjinghao/code/verl/result/LM/qwen3_4b_instruct/QuestA-25-sft/global_step_664_hf
# topp=0.95
# topk=20
# temperature=0.6
# max_response_length=38912
# n=1
topp=1
topk=-1
temperature=0.0
max_response_length=32768
n=1
project_name=qwen3-4b-instruct
# experiment_name=SFT-QuestA-25 
experiment_name=GRPO-Pro-wSFT-QuestA-25 
# python evaluation/eval_vllm_suite3.py \
#   --model $model_path \
#   --n $n \
#   --max_length $max_response_length \
#   --p $topp \
#   --k $topk \
#   --t $temperature \
#   --project_name $project_name \
#   --experiment_name $experiment_name \

python evaluation/grade.py --file_dir evaluation/results/$project_name/$experiment_name
# python evaluation/grade.py --file_dir evaluation/results/$experiment_name
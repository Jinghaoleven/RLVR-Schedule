cd /mnt/public/users/zhangjinghao/code/verl/examples/collect_process
# python dedup_hf_by_problem.py \
#   --input /mnt/public/users/zhangjinghao/dataset/LM_data/QuestA/OpenR1-25-0-4.jsonl \
#   --problem_col problem \
#   --output_hf /mnt/public/users/zhangjinghao/dataset/LM_data/QuestA-25-dedup

# python dedup_hf_by_problem.py \
#   --input "/mnt/public/users/zhangjinghao/dataset/LM_data/OpenMathReasoning/data/cot-*.parquet" \
#   --problem_col problem \
#   --output_hf /mnt/public/users/zhangjinghao/dataset/LM_data/OpenMathReasoning-dedup

# python dedup_hf_by_problem.py \
#   --input "/mnt/public/users/zhangjinghao/code/verl/dataset/train-collect/combined_train.parquet" \
#   --problem_col extra_info \
#   --output_hf /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect

# python data_filter.py \
#   --dataset /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect/DeepMath-103K-dedup/train.parquet \
#   --split train \
#   --difficulty_threshold 5 \
#   --output_dir /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect/filter/DeepMath-103K-dedup

# python data_combine.py \
#   --dataset /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect/filter/DeepMath-103K-dedup/train.parquet,/mnt/public/users/zhangjinghao/code/verl/dataset/train/limo-v2/train.parquet,/mnt/public/users/zhangjinghao/code/verl/dataset/train/QuestA-25/train.parquet \
#   --output_dir /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect

python data_shard.py \
  --dataset /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect/combined_train.parquet \
  --output_dir /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect/combined_sharded \
  --shard_size 1000
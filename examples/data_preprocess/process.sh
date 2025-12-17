cd /mnt/public/users/zhangjinghao/code/verl
source /mnt/public/users/zhangjinghao/.verl-venv/bin/activate

# python examples/data_preprocess/aime24.py \
#  --local_dataset_path /mnt/public/users/zhangjinghao/code/verl/dataset/eval/aime24.parquet\
#  --local_save_dir /mnt/public/users/zhangjinghao/code/verl/dataset/eval\

# python examples/data_preprocess/quest.py \
#  --local_dataset_path /mnt/public/users/zhangjinghao/dataset/LM_data/QuestA/OpenR1-50-0-4.jsonl \
#  --local_save_dir /mnt/public/users/zhangjinghao/code/verl/dataset/train/QuestA-50 \


python examples/data_preprocess/openmathreasoning.py \
 --local_dataset_path /mnt/public/users/zhangjinghao/dataset/LM_data/OpenMathReasoning-dedup \
 --local_save_dir /mnt/public/users/zhangjinghao/code/verl/dataset/train-collect/OpenMathReasoning-dedup \

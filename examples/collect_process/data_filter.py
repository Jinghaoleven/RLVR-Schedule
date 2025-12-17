import argparse
from datasets import load_dataset
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset name or local path")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--difficulty_threshold", type=float, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def make_filter_fn(threshold):
    def filter_fn(example):
        difficulty = example.get("extra_info", {}).get("difficulty", None)
        if difficulty is None:
            return False
        return difficulty > threshold
    return filter_fn

# def make_filter_fn(threshold):
#     def filter_fn(example):
#         problem_type = example.get("extra_info", {}).get("problem_type", None)
#         expected_answer = example.get("extra_info", {}).get("expected_answer", None)
#         verify = len(expected_answer) <= 163
#         if problem_type == "has_answer_extracted":
#             return True
#         else:
#             return False
#     return filter_fn
    
def main():
    args = parse_args()

    # 1. 读取数据集
    dataset = load_dataset(
        "parquet",
        data_files = args.dataset,
        split=args.split,
    )

    print(f"Loaded dataset: {dataset}")
    print("Example:", dataset[0])

    # 3. 过滤
    dataset = dataset.filter(
        make_filter_fn(args.difficulty_threshold),
        num_proc=8,   # 数据大时强烈建议
        desc="Filtering by difficulty",
    )

    print(f"Filtered dataset: {dataset}")

    # 4. 保存为 parquet
    dataset.to_parquet(
        os.path.join(args.output_dir, "train.parquet")
    )

    print(f"Saved filtered dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
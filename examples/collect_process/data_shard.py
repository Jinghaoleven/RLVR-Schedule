from datasets import load_dataset, concatenate_datasets, Dataset
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset name or local path")
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def save_shard_parquet(dataset: Dataset, num_shards: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        file_name = f"train-{i:05d}-of-{num_shards:05d}.parquet"
        output_path = os.path.join(output_dir, file_name)
        print(f"Saving shard {i+1}/{num_shards} to {output_path}...")
        shard.to_parquet(output_path)

def main():
    args = parse_args()

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(
        "parquet",
        data_files=args.dataset,
    )["train"]
    print(f"Loaded dataset: {dataset}")

    num_shards = (len(dataset) + args.shard_size - 1) // args.shard_size
    save_shard_parquet(dataset, num_shards, args.output_dir)

if __name__ == "__main__":
    main()
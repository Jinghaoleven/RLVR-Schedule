from datasets import load_dataset, concatenate_datasets
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset name or local path")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    dataset_paths = args.dataset.split(",")
    datasets_list = []
    for dataset_path in dataset_paths:
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_dataset(
            "parquet",
            data_files=dataset_path,
        )["train"]
        datasets_list.append(dataset)
        print(f"Loaded dataset: {dataset}")
    combined_dataset = concatenate_datasets(datasets_list)
    print(f"Combined dataset: {combined_dataset}")
    combined_dataset.to_parquet(
        os.path.join(args.output_dir, "combined_train.parquet")
    )
    print(f"Saved combined dataset to {os.path.join(args.output_dir, 'combined_train.parquet')}")

if __name__ == "__main__":
    main()
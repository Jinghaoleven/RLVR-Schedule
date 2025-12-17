from datasets import load_dataset

path = "/mnt/public/users/zhangjinghao/code/verl/dataset/train/Mathvista/test_mini.parquet"
save_path = "/mnt/public/users/zhangjinghao/code/verl/dataset/train/Mathvista/test_mini_box.parquet"
ds = load_dataset("parquet", data_files=path)["train"]

instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

def make_map_fn(split):
    def process_fn(example, idx):
        prompt = example["prompt"][0]["content"]
        question = prompt.split("Question: ")[1]
        question = "<image>" + question + " " + instruction_following
        example["prompt"][0]["content"] = question
        return example

    return process_fn

ds = ds.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
ds.to_parquet(save_path)
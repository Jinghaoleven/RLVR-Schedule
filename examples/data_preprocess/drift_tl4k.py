# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
from PIL import Image
import datasets
import re
import io

from verl.utils.hdfs_io import copy, makedirs


def convert_bytes(image, buffer):
    buffer.seek(0)
    buffer.truncate(0)
    image.save(buffer, format=image.format or 'PNG')
    image_bytes = buffer.getvalue()
    return image_bytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/geo3k")
    parser.add_argument("--save_dir", default="~/data/geo3k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    media_dir = "/mnt/public/users/zhangjinghao/dataset/LVM_data"
    data_source = "ChaoHuangCS/DRIFT-TL-Distill-4K"
    dataset = datasets.load_dataset("/mnt/public/users/zhangjinghao/dataset/LVM_data/DRIFT-TL-Distill-4K")

    train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # instruction_following = (
    #     r"You should first thinks about the reasoning process in the mind and then provides the user with the answer. "
    #     r"Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    #     r"i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.\n"
    #     r"Question:\n"
    # )
    # add a row to each data item that represents a unique id
    buffer = io.BytesIO()
    def make_map_fn(split):
        def process_fn(example, idx):
            messages = example.pop("messages")
            images = example.pop("images")
            prompt = messages[0]["content"] + " " + instruction_following
            solution_match = re.search(r"<think>\s*(.*?)\s*</think>", messages[1]["content"], re.DOTALL)
            solution = solution_match.group(1).strip() if solution_match else None

            answer_part = messages[1]["content"].split("</think>")[-1]  # 取 </think> 之后的文本
            answer_match = re.search(r"\\boxed\{(.*?)\}", answer_part)
            answer = answer_match.group(1).strip() if answer_match else None

            # # remove \\number{21} -> 21
            # answer = re.sub(r"\\number\s*\{\s*(?P<latex>[^{}]+)\s*\}", r"\1", answer)

            images_values = []
            for img_path in images:
                img = Image.open(os.path.join(media_dir,img_path.strip("../")))
                image_bytes = convert_bytes(img, buffer)
                images_values.append({"bytes":image_bytes})

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images_values,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": prompt,
                    "solution": solution
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    # test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    makedirs(args.save_dir)
    train_dataset = train_dataset.filter(
    lambda example: example["reward_model"]["ground_truth"] is not None,
    num_proc = 4,
    desc=f"Filter example without verifiable answer"
)
    train_dataset.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)

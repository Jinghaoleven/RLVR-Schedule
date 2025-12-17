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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/geo3k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    media_dir = "Reasoning_data/MM-Eureka"
    data_source = "FanqingM/MMK12"
    # dataset = datasets.load_dataset(data_source)
    dataset = datasets.load_dataset("json",data_files="Reasoning_data/MM-Eureka/dataset_prompt.jsonl")

    train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    instruction_following = (
        r"You should first thinks about the reasoning process in the mind and then provides the user with the answer. "
        r"Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
        r"i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.\n"
        r"Question:\n"
    )
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("prompt")
            prompt = instruction_following + "<image>" + problem
            answer = example.pop("answer")
            images = example.pop("images")

            # remove \\number{21} -> 21
            answer = re.sub(r"\\number\s*\{\s*(?P<latex>[^{}]+)\s*\}", r"\1", answer)

            images_values = []
            for img_path in images:
                img = Image.open(os.path.join(media_dir,img_path))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_bytes = buffer.getvalue()
                images_values.append({"bytes":img_bytes})

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
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    # test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)

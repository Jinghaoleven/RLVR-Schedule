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
    
    data_source = "AI4Math/MathVista"
    # dataset = datasets.load_dataset(data_source)
    dataset = datasets.load_dataset("Benchmark/MLLM_single/MathVista")

    test_dataset = dataset["testmini"]
    # test_dataset = dataset["test"]

    instruction_following = (
        r"Solve the question. The user asks a question, and you solves it. You first thinks about "
        r"the reasoning process in the mind and then provides the user with the answer. The answer "
        r"is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} "
        r"command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
        r"respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is "
        r"$\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>.\n"
    )
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            query = example.pop("query")
            query = query[query.find('Question: '):]
            prompt = instruction_following + "<image>" + query
            problem = example.pop("question")
            answer = example.pop("answer")
            images = example.pop("decoded_image")
            metadata = example.pop("metadata")

            example.pop("image")
            example.pop("unit")
            example.pop("precision")
            example.pop("question_type")
            example.pop("answer_type")


            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": [images],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "category": metadata.pop("category"),
                    "skills": metadata.pop("skills"),
                    "source": metadata.pop("source"),
                    "task": metadata.pop("task"),
                    "grade": metadata.pop("grade"),
                    "context": metadata.pop("context"),
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test_mini"), with_indices=True, num_proc=8)
    # test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test_mini.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)

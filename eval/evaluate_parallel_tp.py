# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import os
import multiprocessing as mp

import fire
import numpy as np
import vllm
from jinja2 import Template
# import multiprocessing

from datasets import load_from_disk
from math_grader import (answer_tag_reward_fn,
                                            boxed_reward_fn)


def apply_qwen_math_box_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_llama3_box_template(question: str):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        + question
        + "\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

def apply_qwen_tag_template(question: str):
    return (
        "<|im_start|>system\nYou should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


# The following two templates are used to evaluate baselines from other projects.
def apply_prime_zero_template(question: str):
    """https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME-Zero"""
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    return f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"


def apply_open_reasoner_zero_template(question: str):
    "https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/e008f6d95f0b9a0e992f6b8bac912515b50a4634/playground/zero_setting_base.py"
    prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""
    prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""
    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=question)
    prompt_template = Template(prompt_template_jinja)
    return prompt_template.render(bos_token="", prompt=prompt_instruction)

def setup_template_and_reward(template, model_name, sampling_params):
    if template in ["qwen_math_box", "no"]:
        math_reward_fn = boxed_reward_fn
        if template == "qwen_math_box":
            apply_template = apply_qwen_math_box_template
        else:
            apply_template = lambda x: x
    elif template == "qwen_math_tag":
        math_reward_fn = answer_tag_reward_fn
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True
        apply_template = apply_qwen_tag_template
    elif template == "llama_box":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_llama3_box_template
    elif template == "r1":
        math_reward_fn = answer_tag_reward_fn
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True
        apply_template = apply_r1_template
    elif template == "prime-zero":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_prime_zero_template
    elif template == "open-reasoner-zero":
        from understand_r1_zero.math_grader import answer_tag_reward_fn_for_orz

        math_reward_fn = answer_tag_reward_fn_for_orz
        apply_template = apply_open_reasoner_zero_template
    elif template == "llama-instruct":

        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n",
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

    elif template == "r1d":  # r1-distill
        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )

    else:
        raise ValueError

    return apply_template, math_reward_fn

def _evaluate_worker(
    gpu_ids,
    dataset_name,
    tasks,
    task_indices,
    model_name,
    template,
    temperature,
    top_p,
    max_tokens,
    max_model_len,
    n_samples,
    tensor_parallel_size,
    eval_type,
    result_queue,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=int(time.time_ns()),
    )

    apply_template, math_reward_fn = setup_template_and_reward(
        template,
        model_name,
        sampling_params,
    )

    model = vllm.LLM(
        model_name,
        tensor_parallel_size=tensor_parallel_size,
        # swap_space=32,
        # max_model_len=max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )

    results = {}
    datasets = load_from_disk(dataset_name)

    for task_name in tasks:
        indices = task_indices.get(task_name, [])
        if not indices:
            continue
        dataset = datasets[task_name]
        prompts = [dataset["problem"][i] for i in indices]
        targets = [dataset["answer"][i] for i in indices]
        prompts = list(map(apply_template, prompts))

        outputs = model.generate(prompts, sampling_params)
        batch_scores = []
        batch_formatted = []
        batch_lengths = []
        to_be_saved = []
        for k in range(len(outputs)):
            output = outputs[k]
            gt_repeated = [targets[k]] * sampling_params.n
            rewards, infos = [], []
            for model_output, gt in zip([o.text for o in output.outputs], gt_repeated):
                info, r = math_reward_fn(model_output, gt, fast=False)
                rewards.append(r)
                infos.append(info)
            rewards = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            if eval_type == "avg@k":
                batch_scores.append(rewards.mean())
            elif eval_type == "pass@k":
                batch_scores.append(1 if np.any(rewards == 1) else 0)

            if infos[0] is not {}:
                batch_formatted.append(np.array([i["formatted"] for i in infos]).sum())

            to_be_saved.append(
                {
                    "task_name": task_name,
                    "prompt": output.prompt,
                    "gt": gt_repeated,
                    "model_output": [o.text for o in output.outputs],
                    "reward": [r for r in rewards],
                }
            )

        results[task_name] = {
            "batch_scores": batch_scores,
            "batch_formatted": batch_formatted,
            "batch_lengths": batch_lengths,
            "to_be_saved": to_be_saved,
        }

    result_queue.put(results)

def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime", "aime25", "amc", "math", "minerva", "olympiad_bench"],
    # tasks: list = ["aime", "aime25", "amc", "math", "olympiad_bench"],
    # tasks: list = ["aime", "aime25", "amc", "math"],
    # tasks: list = ["minerva", "olympiad_bench"],
    template: str = "no",
    dataset_name: str = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangjinghao-240108110057/code/RL_code/understand-r1-zero/datasets/evaluation_suite",
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 3000,
    max_model_len: int = 40960,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 1,
    tensor_parallel_size: int = 1,
    eval_type: str = "avg@k",
    max_test: int = 999999,
    save_path: str = None,
    exp_name: str = "exp",
):
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        available_gpus = [gpu for gpu in cuda_visible.split(",") if gpu]
    else:
        available_gpus = ["0"]
    print(f"available_gpus: {available_gpus}")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if len(available_gpus) % tensor_parallel_size != 0:
        raise ValueError(
            "Number of visible GPUs must be divisible by tensor_parallel_size"
        )
    gpu_groups = [
        available_gpus[i : i + tensor_parallel_size]
        for i in range(0, len(available_gpus), tensor_parallel_size)
    ]
    print(f"gpu_groups: {gpu_groups}")

    if "prime" in model_name.lower():
        template = "prime-zero"
    if "open-reasoner-zero" in model_name.lower():
        template = "open-reasoner-zero"

    # if "instruct" in model_name.lower() and "instruct" not in template:
    #     input(
    #         f"{model_name}\n{template}\ninstruct model but not instruct template! continue?"
    #     )

    print("Using template:", template)

    results = {}
    avg_lens = {}
    max_lens = {}
    formatted = {}
    to_be_saved = []
    datasets = load_from_disk(dataset_name)
    task_indices_per_gpu = [dict() for _ in available_gpus]

    for task_name in tasks:
        if task_name not in datasets:
            continue
        dataset = datasets[task_name]
        total = min(len(dataset["problem"]), max_test)
        indices = list(range(total))
        shards = np.array_split(indices, len(available_gpus))
        for rank, shard in enumerate(shards):
            task_indices_per_gpu[rank][task_name] = shard.tolist()

    gpu_to_rank = {gpu: rank for rank, gpu in enumerate(available_gpus)}
    task_indices_per_group = []
    for group in gpu_groups:
        merged = {}
        for gpu in group:
            per_gpu = task_indices_per_gpu[gpu_to_rank[gpu]]
            for task_name, indices in per_gpu.items():
                merged.setdefault(task_name, []).extend(indices)
        task_indices_per_group.append(merged)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = []
    for rank, gpu_group in enumerate(gpu_groups):
        p = ctx.Process(
            target=_evaluate_worker,
            args=(
                gpu_group,
                dataset_name,
                tasks,
                task_indices_per_group[rank],
                model_name,
                template,
                temperature,
                top_p,
                max_tokens,
                max_model_len,
                n_samples,
                tensor_parallel_size,
                eval_type,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)

    aggregated = {}
    for _ in workers:
        worker_result = result_queue.get()
        for task_name, data in worker_result.items():
            if task_name not in aggregated:
                aggregated[task_name] = {
                    "batch_scores": [],
                    "batch_formatted": [],
                    "batch_lengths": [],
                    "to_be_saved": [],
                }
            aggregated[task_name]["batch_scores"].extend(data["batch_scores"])
            aggregated[task_name]["batch_formatted"].extend(data["batch_formatted"])
            aggregated[task_name]["batch_lengths"].extend(data["batch_lengths"])
            aggregated[task_name]["to_be_saved"].extend(data["to_be_saved"])

    for p in workers:
        p.join()

    for task_name, data in aggregated.items():
        results[task_name] = np.mean(data["batch_scores"])
        avg_lens[task_name] = np.mean(data["batch_lengths"])
        if data["batch_formatted"]:
            formatted[task_name] = np.mean(data["batch_formatted"]) / n_samples
        max_lens[task_name] = np.max(data["batch_lengths"])
        to_be_saved.extend(data["to_be_saved"])

    print(results)
    print("avg:", np.mean(list(results.values())))
    print("avg_lens:", avg_lens)
    print("max_lens:", max_lens)
    print("formatted:", formatted)

    result_summary = {
        "exp_name": exp_name,
        "model_name": model_name,
        "template": template,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n_samples": n_samples,
        "eval_type": eval_type,
        "tasks": tasks,
        "results": _to_float_dict(results),
        "avg": float(np.mean(list(results.values()))) if results else None,
        "avg_lens": _to_float_dict(avg_lens),
        "max_lens": _to_float_dict(max_lens),
        "formatted": _to_float_dict(formatted),
    }
    summary_dir = "/mnt/public/users/zhangjinghao/code/verl/eval/results"
    summary_fn = os.path.join(
        summary_dir,
        f"{exp_name}_max_tokens{max_tokens}_temperature{temperature}_top_p{top_p}.json",
    )
    print(f"saving eval summary at {summary_fn}")
    json.dump(
        result_summary,
        open(
            summary_fn,
            "w",
        ),
        indent=4,
    )

    if save_path:
        def _to_float_dict(data):
            return {key: float(value) for key, value in data.items()}

        # fn = "model_eval_out_" + model_name.replace("/", "_") + str(int(time.time()))
        fn = os.path.join(save_path,model_name.split("/")[-3])
        fn = f"{fn}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json"
        print(f"saving model outputs at {fn}")
        json.dump(
            to_be_saved,
            open(
                fn,
                "w",
            ),
            indent=4,
        )

# multiprocessing.set_start_method('spawn')
if __name__ == '__main__':
    fire.Fire(main)

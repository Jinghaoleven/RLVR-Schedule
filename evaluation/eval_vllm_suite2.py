#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU batch inference for AIME-24 with persistent workers.

• 主进程：
  - 读取 evaluation_suite 下所有 parquet 任务
  - 每个任务生成 N 个随机种子，并按 GPU 均分
  - 把 (task_name, samples, seeds_for_gpu) 丢到任务队列 job_queue
  - 从 result_queue 收集所有 GPU 的结果，写入一个 JSONL 文件
• Worker 进程（数量 = GPU 数量）：
  - 绑定到单个 GPU（通过 CUDA_VISIBLE_DEVICES）
  - 初始化自己的 vLLM.LLM（只初始化一次）
  - 循环从 job_queue 取任务并执行
"""

import os
import json
import random
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse


# --------------------------------------------------------------------------- #
#                               Argument parser                               #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/path/to/model")
parser.add_argument("--evaluation_suite", type=str, default="dataset/eval")
parser.add_argument("--t", type=float, default=1.4)
parser.add_argument("--p", type=float, default=1.0)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--n", type=int, default=32)          # roll-outs / prompt
parser.add_argument("--max_length", type=int, default=90000)
parser.add_argument("--project_name", type=str, default="qwen3-4b")
parser.add_argument("--experiment_name", type=str, default="Polaris-4B")
parser.add_argument("--output", type=str, default="evaluation/results")
args = parser.parse_args()

# --------------------------------------------------------------------------- #
#                   Global constants (read-only in workers)                   #
# --------------------------------------------------------------------------- #
PROJECT     = args.project_name
NAME        = args.experiment_name
N           = args.n                          # roll-outs per prompt
MODEL_PATH  = args.model
MAX_TOKENS  = args.max_length
TEMPERATURE = args.t
TOP_P       = args.p
TOP_K       = args.k


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return obj


def load_samples(filepath: str):
    """Read parquet file and return a list of prompts."""
    df = pd.read_parquet(filepath)
    samples = [
        {
            "example_id": i,
            "prompt": df.at[i, "prompt"][0]["content"],
            "answer": df.at[i, "reward_model"]["ground_truth"],
        }
        for i in range(len(df))
    ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_seeds(seeds: list[int], num_workers: int):
    """Round-robin split of the seed list into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, s in enumerate(seeds):
        chunks[idx % num_workers].append(s)
    return chunks


# --------------------------------------------------------------------------- #
#                               Worker function                               #
# --------------------------------------------------------------------------- #
def worker_loop(gpu_id: int, job_queue: mp.Queue, result_queue: mp.Queue):
    """
    每个 worker 绑定到一个 GPU，并在一个循环里处理多个任务。

    job = {
        "task_name": str,
        "samples": list[dict],
        "seeds": list[int],
    }
    """
    # 绑定到单个 GPU（必须在 LLM 初始化前设置）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker GPU {gpu_id}] Starting, binding to CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)

    # 初始化 vLLM 模型（只初始化一次）
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        enforce_eager=False,
        dtype="bfloat16",
        max_model_len=MAX_TOKENS,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
    )
    print(f"[Worker GPU {gpu_id}] LLM initialized.", flush=True)

    while True:
        job = job_queue.get()
        if job is None:
            # 收到结束信号
            print(f"[Worker GPU {gpu_id}] Received stop signal. Exiting.", flush=True)
            break

        task_name = job["task_name"]
        samples   = job["samples"]
        seeds     = job["seeds"]

        if not seeds:
            # 理论上不会收到空 job，防御一下
            continue

        print(f"[Worker GPU {gpu_id}] Task={task_name}, seeds={seeds}", flush=True)

        local_results = []
        for seed in seeds:
            sampling = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_TOKENS,
                seed=seed,
            )
            messages = [[{"role": "user", "content": s["prompt"]}] for s in samples]

            # 子进程里 tqdm 会比较乱，关掉进度条
            outputs = llm.chat(messages, sampling, use_tqdm=False)

            for sample, out in zip(samples, outputs):
                local_results.append(
                    {
                        "task_name": task_name,
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "seed": seed,
                        "response": out.outputs[0].text,
                    }
                )

        # 把当前 task + seeds 的结果返回给主进程
        result_queue.put(
            {
                "task_name": task_name,
                "results": local_results,
            }
        )
        print(
            f"[Worker GPU {gpu_id}] Finished task={task_name}, "
            f"seeds={seeds}, generations={len(local_results)}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    # 1. 解析可用 GPU
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set.")
    available_gpus = [g for g in visible.split(",") if g != ""]
    num_gpus = len(available_gpus)
    print(f"Detected {num_gpus} GPUs: {available_gpus}")

    # 2. 创建 multiprocessing 上下文 & 队列
    ctx = mp.get_context("spawn")
    job_queue    = ctx.Queue()
    result_queue = ctx.Queue()

    # 3. 启动 worker 进程（每个 GPU 一个）
    workers = []
    for g in available_gpus:
        gpu_id = int(g)
        p = ctx.Process(
            target=worker_loop,
            args=(gpu_id, job_queue, result_queue),
            daemon=False,
        )
        p.start()
        workers.append(p)

    # 4. 遍历 evaluation_suite 下的所有 parquet 任务
    tasks = sorted(os.listdir(args.evaluation_suite))
    for task in tasks:
        if not task.endswith(".parquet"):
            continue

        data_path = os.path.join(args.evaluation_suite, task)
        data_name = task.rsplit(".parquet", 1)[0]

        out_path = Path(
            f"{args.output}/{NAME}/{data_name}-{TEMPERATURE}-{N}-{MAX_TOKENS}-{TOP_K}.jsonl"
        )
        if out_path.exists():
            print(f"[Skip] {out_path} already exists.")
            continue

        print(f"\n=== Evaluation: {data_name} ===")
        samples = load_samples(data_path)

        # 5. 为当前 task 生成种子并按 GPU 均分
        random_seeds = random.sample(range(2**31 - 1), N)
        seed_chunks = split_seeds(random_seeds, num_gpus)

        # 构造任务，发给每个有 seed 的 GPU
        jobs_for_task = 0
        for idx, g in enumerate(available_gpus):
            seeds_for_gpu = seed_chunks[idx]
            if not seeds_for_gpu:
                continue
            job = {
                "task_name": data_name,
                "samples": samples,
                "seeds": seeds_for_gpu,
            }
            job_queue.put(job)
            jobs_for_task += 1

        print("Seed assignment for task:", data_name)
        for idx, g in enumerate(available_gpus):
            seeds_for_gpu = seed_chunks[idx]
            if seeds_for_gpu:
                print(f"  GPU {g}: {len(seeds_for_gpu)} seeds -> {seeds_for_gpu}")

        # 6. 收集当前 task 的所有结果
        all_results = []
        for _ in tqdm(range(jobs_for_task), desc=f"Collect {data_name}"):
            msg = result_queue.get()
            assert msg["task_name"] == data_name
            all_results.extend(msg["results"])

        print(
            f"[Main] Task={data_name}: collected {len(all_results)} generations "
            f"(expected ~ {len(samples) * N})."
        )

        # 7. 写入 JSONL
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(to_jsonable(item), ensure_ascii=False) + "\n")
        print(f"[Main] Saved results to {out_path}")

    # 8. 所有 task 完成，给 worker 发送停止信号
    print("\nAll tasks done. Sending stop signal to workers...")
    for _ in workers:
        job_queue.put(None)

    for p in workers:
        p.join()
    print("All workers exited.")


if __name__ == "__main__":
    main()
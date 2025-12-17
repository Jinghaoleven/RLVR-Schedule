#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# --------------------------- Argument parser --------------------------------
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

PROJECT     = args.project_name
NAME        = args.experiment_name
N           = args.n
MODEL_PATH  = args.model
MAX_TOKENS  = args.max_length
TEMPERATURE = args.t
TOP_P       = args.p
TOP_K       = args.k


# ------------------------------ Helpers -------------------------------------
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
    df = pd.read_parquet(filepath)
    samples = [
        {
            "example_id": i,
            "prompt": df.at[i, "prompt"][0]["content"],
            "answer": df.at[i, "reward_model"]["ground_truth"],
        }
        for i in range(len(df))
    ]
    print(f"Total unique samples in {filepath}: {len(samples)}")
    return samples


# ------------------------------ Worker --------------------------------------
def worker_loop(gpu_id: int, job_queue: mp.Queue, result_queue: mp.Queue):
    # 绑 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker GPU {gpu_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    # init vLLM（每个 worker 一次）
    from vllm import LLM, SamplingParams
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
            print(f"[Worker GPU {gpu_id}] Stop signal received.", flush=True)
            break

        task_name  = job["task_name"]
        samples    = job["samples"]
        seed       = job["seed"]

        sampling = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_TOKENS,
            seed=seed,
        )
        messages = [[{"role": "user", "content": s["prompt"]}] for s in samples]

        outputs = llm.chat(messages, sampling, use_tqdm=False)

        local_results = []
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

        result_queue.put(
            {
                "task_name": task_name,
                "seed": seed,
                "results": local_results,
            }
        )
        print(f"[Worker GPU {gpu_id}] Done task={task_name}, seed={seed}, "
              f"gens={len(local_results)}", flush=True)


# ------------------------------ main ----------------------------------------
def main():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set.")
    available_gpus = [g for g in visible.split(",") if g != ""]
    num_gpus = len(available_gpus)
    print(f"Detected {num_gpus} GPUs: {available_gpus}")

    ctx = mp.get_context("spawn")
    job_queue    = ctx.Queue()
    result_queue = ctx.Queue()

    # 启动 worker（每 GPU 一个）
    workers = []
    for g in available_gpus:
        gpu_id = int(g)
        p = ctx.Process(target=worker_loop, args=(gpu_id, job_queue, result_queue))
        p.start()
        workers.append(p)

    # 收集所有 task 的 meta，用于知道每个 task 需要多少 job
    tasks = []
    for fname in sorted(os.listdir(args.evaluation_suite)):
        if not fname.endswith(".parquet"):
            continue
        if fname.startswith("olympiad") or fname.startswith("minerva") or fname.startswith("aime24-system"):
            continue
        data_path = os.path.join(args.evaluation_suite, fname)
        data_name = fname.rsplit(".parquet", 1)[0]

        out_path = Path(
            f"{args.output}/{PROJECT}/{NAME}/{data_name}-{TEMPERATURE}-{N}-{MAX_TOKENS}-{TOP_K}.jsonl"
        )
        if out_path.exists():
            print(f"[Skip] {out_path} exists.")
            continue

        samples = load_samples(data_path)
        tasks.append({
            "data_name": data_name,
            "data_path": data_path,
            "out_path": out_path,
            "samples": samples,
        })

    # 统计总 job 数：每个 task 有 N 个 seed，每个 seed 一个 job
    total_jobs = len(tasks) * N
    print(f"Total tasks: {len(tasks)}, total jobs (task * N seeds): {total_jobs}")

    # 把所有 job 丢进全局队列（fine-grained）
    for t in tasks:
        data_name = t["data_name"]
        samples   = t["samples"]
        seeds     = random.sample(range(2**31 - 1), N)
        print(f"[Main] Task={data_name}, seeds={seeds}")
        for seed in seeds:
            job_queue.put(
                {
                    "task_name": data_name,
                    "samples": samples,
                    "seed": seed,
                }
            )

    # 收集结果：按 task 分桶
    results_by_task = {t["data_name"]: [] for t in tasks}

    for _ in tqdm(range(total_jobs), desc="Collect all jobs"):
        msg = result_queue.get()
        task_name = msg["task_name"]
        results_by_task[task_name].extend(msg["results"])

    # 写出每个 task 的 JSONL
    for t in tasks:
        data_name = t["data_name"]
        out_path  = t["out_path"]
        all_results = results_by_task[data_name]
        print(f"[Main] Task={data_name}, total generations={len(all_results)}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(to_jsonable(item), ensure_ascii=False) + "\n")
        print(f"[Main] Saved to {out_path}")

    # 停 worker
    print("All jobs done, sending stop signals...")
    for _ in workers:
        job_queue.put(None)
    for p in workers:
        p.join()
    print("All workers exited.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
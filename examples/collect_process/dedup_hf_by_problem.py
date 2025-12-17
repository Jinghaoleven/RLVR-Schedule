#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import hashlib
import os
import re
import unicodedata

from datasets import load_dataset, Dataset


_WS_RE = re.compile(r"\s+")


def normalize_problem(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def fingerprint_problem(text: str) -> str:
    # 用 hex 字符串，方便 debug
    return hashlib.blake2b(
        normalize_problem(text).encode("utf-8", errors="ignore"),
        digest_size=8,
    ).hexdigest()


def collect_parquet_files(inputs, split):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)
        # elif os.path.isfile(p) and p.endswith(".parquet"):
        elif os.path.isfile(p) and p.endswith(".jsonl"):
            files.append(p)
        else:
            files += glob.glob(p, recursive=True)

    files = sorted(set(files))
    if not files:
        raise ValueError(f"No parquet files found from {inputs}")

    if split:
        s = split.strip().strip("/")
        token = f"/{s}/"
        kept = [f for f in files if token in f.replace("\\", "/")]
        if not kept:
            kept = [f for f in files if s in f.replace("\\", "/")]
        if not kept:
            raise ValueError(f"No parquet files match split={split}")
        files = kept

    return files


def main():
    ap = argparse.ArgumentParser("HF dataset dedup by problem (non-streaming)")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--problem_col", default="problem")
    ap.add_argument("--keep_columns", nargs="*", default=None)
    ap.add_argument("--output_hf", required=True)
    ap.add_argument("--num_proc", type=int, default=1,
                    help="Processes for map(). Use >1 if CPU allows.")
    args = ap.parse_args()

    parquet_files = collect_parquet_files(args.input, args.split)
    print(f"[info] loading {len(parquet_files)} parquet files")

    ds = load_dataset(
        "parquet",
        data_files={"train": parquet_files},
        split="train",
        streaming=False,
    )

    if args.problem_col not in ds.column_names:
        raise ValueError(
            f"{args.problem_col!r} not found. Columns: {ds.column_names}"
        )

    # 可选裁剪列（省内存）
    if args.keep_columns:
        keep = set(args.keep_columns)
        keep.add(args.problem_col)
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    print(ds)

    seen = set()
    keep_indices = []

    for i, p in enumerate(ds[args.problem_col]["question"]):
        h = fingerprint_problem(p)
        if h in seen:
            # keep_indices.append(i)
            continue
        seen.add(h)
        keep_indices.append(i)

        if len(keep_indices) % 200_000 == 0:
            print(f"[progress] kept={len(keep_indices):,}")

    dedup_ds = ds.select(keep_indices)
    import pdb; pdb.set_trace()
    dedup_ds.to_parquet(os.path.join(args.output_hf, "combined_train_dedup.parquet"))
    # dedup_ds.save_to_disk(args.output_hf)

    print(f"[done] kept={len(dedup_ds):,} unique problems")
    print(f"[saved] {args.output_hf}")


if __name__ == "__main__":
    main()
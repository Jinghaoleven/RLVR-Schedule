import argparse
import glob
import json
import os


def _load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    results = data.get("results", {})
    cleaned = {}
    for key, value in results.items():
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            cleaned[key] = value
    return cleaned


def _format_table(df):
    headers = ["exp"] + list(df.columns)
    rows = []
    for exp_name, row in df.iterrows():
        rows.append([exp_name] + [row.get(col) for col in df.columns])

    def _cell(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    str_rows = [[_cell(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(sep, fill):
        parts = [fill * (w + 2) for w in widths]
        return sep + sep.join(parts) + sep

    def _row(values):
        cells = [f" {v.ljust(w)} " for v, w in zip(values, widths)]
        return "|" + "|".join(cells) + "|"

    lines = [_line("+", "-"), _row(headers), _line("+", "=")]
    for row in str_rows:
        lines.append(_row(row))
        lines.append(_line("+", "-"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate eval results into an Excel sheet."
    )
    parser.add_argument(
        "--results_dir",
        default="/mnt/public/users/zhangjinghao/code/verl/eval/results",
        help="Directory containing evaluation json files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern to match result files.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/public/users/zhangjinghao/code/verl/eval/bench_results.xlsx",
        help="Output Excel filename or absolute path.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Show the aggregated table in the terminal.",
    )
    args = parser.parse_args()

    result_glob = os.path.join(args.results_dir, args.pattern)
    files = sorted(glob.glob(result_glob))
    if not files:
        print(f"No result files found for pattern: {result_glob}")
        return

    rows = {}
    tasks = set()
    for path in files:
        exp_name = os.path.splitext(os.path.basename(path))[0]
        results = _load_results(path)
        rows[exp_name] = results
        tasks.update(results.keys())

    tasks = sorted(tasks)
    records = []
    for exp_name, results in rows.items():
        row = {"exp": exp_name}
        for task in tasks:
            row[task] = results.get(task)
        records.append(row)

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to write results.xlsx") from exc

    df = pd.DataFrame(records).set_index("exp")
    df = df[tasks]
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    df["average"] = numeric_df.mean(axis=1, skipna=True)
    df = df.sort_values(by="average", ascending=False)

    if args.table:
        print(_format_table(df))

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.results_dir, output_path)
    def _excel_cell(value):
        if isinstance(value, float):
            return round(value, 3) * 100
        return value

    df_export = df.map(_excel_cell)
    df_export.to_excel(output_path, engine="openpyxl")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize benchmark smoke runs below results/benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "benchmark"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "summary" / "smoke_summary.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_tool_calls(messages: list[dict[str, Any]], name: str) -> int:
    count = 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == name:
                count += 1
        content = message.get("content") or ""
        count += content.count(f'"name": "{name}"')
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for status_path in sorted(args.results_root.glob("*/*/*/run_status.json")):
        run_root = status_path.parent
        config_path = run_root / "run_config.json"
        if not config_path.exists():
            continue
        config = load_json(config_path)
        status = load_json(status_path)
        result_path = Path(status["result_file"])
        result_rows = load_jsonl(result_path) if result_path.exists() else []
        dataset_path = Path(config["DATASET"])
        dataset_rows = load_jsonl(dataset_path) if dataset_path.exists() else []
        source_item = dataset_rows[0] if dataset_rows else {}
        trajectory_path = Path(status["subagent_trajectory_file"])
        trajectory_rows = load_jsonl(trajectory_path) if trajectory_path.exists() else []
        result = result_rows[0] if result_rows else {}
        messages = result.get("messages") or []
        judge_path = result_path.with_name(
            result_path.stem + "_judge_results" + result_path.suffix
        )
        judge_rows = load_jsonl(judge_path) if judge_path.exists() else []
        judgement = judge_rows[0].get("judgement", "") if judge_rows else "not run"
        tool_names = ("search", "visit", "google_scholar", "PythonInterpreter")
        tool_counts = {
            name: count_tool_calls(messages, name) for name in tool_names
        }
        rows.append(
            {
                "dataset": run_root.parents[1].name,
                "setting": run_root.parent.name,
                "run_id": run_root.name,
                "file_name": source_item.get("file_name", ""),
                "ground_truth": result.get("answer", ""),
                "prediction": result.get("prediction", ""),
                "termination": result.get("termination", result.get("error", "")),
                "judgement": judgement,
                "elapsed_seconds": status.get("elapsed_seconds", ""),
                "subagent_calls": len(trajectory_rows),
                "tool_counts": tool_counts,
                "result_file": str(result_path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# Benchmark smoke summary\n\n")
        handle.write(
            "| Dataset | Setting | File | Ground truth | Prediction | Status | "
            "Judge | Seconds | Sub-agents | Search | Visit | Scholar | Python |\n"
        )
        handle.write(
            "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            prediction = str(row["prediction"]).replace("|", "\\|").replace("\n", " ")
            if len(prediction) > 120:
                prediction = prediction[:117] + "..."
            ground_truth = str(row["ground_truth"]).replace("|", "\\|")
            tools = row["tool_counts"]
            handle.write(
                f"| {row['dataset']} | {row['setting']} | {row['file_name']} | "
                f"{ground_truth} | {prediction} | {row['termination']} | "
                f"{row['judgement']} | {row['elapsed_seconds']} | "
                f"{row['subagent_calls']} | {tools['search']} | "
                f"{tools['visit']} | {tools['google_scholar']} | "
                f"{tools['PythonInterpreter']} |\n"
            )
        handle.write("\n## Result files\n\n")
        for row in rows:
            handle.write(
                f"- `{row['dataset']}` / `{row['setting']}`: "
                f"`{row['result_file']}`\n"
            )

    print(f"Wrote summary for {len(rows)} run(s): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

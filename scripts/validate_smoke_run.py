#!/usr/bin/env python3
"""Validate one benchmark wrapper run without invoking a judge."""

import argparse
import json
import re
from pathlib import Path

_TEXT_TOOL_CALL_RE = re.compile(
    r"(?:<\s*(?:tool_call|tool_calls|invoke)\b"
    r"|<(?:\|｜){2}DSML(?:\|｜){2}(?:tool_calls|invoke|parameter)\b"
    r"|DSML.*tool_calls)",
    re.IGNORECASE | re.DOTALL,
)
_UNUSABLE_SUBAGENT_CONTENT = {
    "",
    "(Sub-agent returned no usable content.)",
    "(Sub-agent returned no output.)",
}


def _read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _usable_subagent_trajectory(record, allow_fallback=False):
    status = str(record.get("status") or "").strip().lower()
    content = str(record.get("content") or "").strip()
    return (
        status != "error"
        and status != "timeout"
        and (allow_fallback or not status.endswith("_fallback"))
        and content not in _UNUSABLE_SUBAGENT_CONTENT
        and not _TEXT_TOOL_CALL_RE.search(content)
    )


def validate_smoke_run(result_path, trajectory_path, setting,
                       require_sub_agent=False):
    errors = []
    records = []
    trajectory_records = []

    if not result_path.is_file() or result_path.stat().st_size == 0:
        errors.append(f"missing or empty result file: {result_path}")
    else:
        try:
            records = _read_jsonl(result_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid result JSONL: {exc}")
        if not records:
            errors.append("result JSONL contains no records")
        for index, record in enumerate(records, 1):
            prediction = record.get("prediction")
            if not isinstance(prediction, str) or not prediction.strip():
                errors.append(f"result record {index} has an empty prediction")
            elif prediction.strip().lower().startswith("no answer found"):
                errors.append(
                    f"result record {index} contains a no-answer sentinel")
            termination = str(record.get("termination") or "").lower()
            if (
                "exceed available llm calls" in termination
                or "no answer found after" in termination
            ):
                errors.append(
                    f"result record {index} failed with termination: "
                    f"{record.get('termination')}"
                )

    if trajectory_path.is_file() and trajectory_path.stat().st_size > 0:
        try:
            trajectory_records = _read_jsonl(trajectory_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid sub-agent trajectory JSONL: {exc}")

    if setting == "single" and trajectory_records:
        errors.append(
            "single-agent run unexpectedly produced sub-agent trajectories")

    if setting == "swarm" and require_sub_agent:
        if not trajectory_records:
            errors.append(
                "SearchSwarm smoke did not produce a sub-agent trajectory")
        elif not any(
            _usable_subagent_trajectory(record)
            for record in trajectory_records
        ):
            errors.append(
                "all SearchSwarm sub-agent trajectories ended in error, "
                "timeout, fallback-only delivery, empty output, or "
                "text-encoded tool calls"
            )

    return errors, len(records), len(trajectory_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--setting", choices=("single", "swarm"), required=True)
    parser.add_argument("--require-sub-agent", action="store_true")
    args = parser.parse_args()

    errors, result_count, trajectory_count = validate_smoke_run(
        args.result,
        args.trajectory,
        args.setting,
        require_sub_agent=args.require_sub_agent,
    )
    if errors:
        print("Smoke validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(
        "Smoke validation passed: "
        f"results={result_count}, subagent_trajectories={trajectory_count}"
    )


if __name__ == "__main__":
    main()

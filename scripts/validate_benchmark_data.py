#!/usr/bin/env python3
"""Validate SearchSwarm benchmark subset JSONL files."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "eval_data" / "benchmark"
EXPECTED = {
    "browsecomp_subset_20.jsonl": 20,
    "gaia_subset_20.jsonl": 20,
    "xbench_deepsearch_subset_20.jsonl": 20,
    "self_deepsearch_qa_20.jsonl": 20,
    "all_80.jsonl": 80,
}
REQUIRED_FIELDS = ("task_question", "ground_truth", "file_name", "metadata")


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return [], [f"missing file: {path.relative_to(ROOT)}"]
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                errors.append(f"{path.name}:{line_number}: blank line")
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(item, dict):
                errors.append(f"{path.name}:{line_number}: row is not an object")
                continue
            records.append(item)
    return records, errors


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def obvious_answer_leak(question: str, answer: str) -> bool:
    question_norm = normalized_text(question)
    answer_norm = normalized_text(answer)
    if len(answer_norm) < 6:
        return False
    if answer_norm in question_norm:
        return True
    compact_answer = re.sub(r"[\W_]+", "", answer_norm)
    compact_question = re.sub(r"[\W_]+", "", question_norm)
    return len(compact_answer) >= 8 and compact_answer in compact_question


def validate_records(
    path: Path, records: list[dict[str, Any]], expected_count: int
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if len(records) != expected_count:
        errors.append(
            f"{path.name}: expected {expected_count} rows, found {len(records)}"
        )
    file_names: list[str] = []
    for row_number, item in enumerate(records, 1):
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            errors.append(
                f"{path.name}:{row_number}: missing fields {', '.join(missing)}"
            )
            continue
        question = str(item.get("task_question") or "").strip()
        answer = str(item.get("ground_truth") or "").strip()
        file_name = str(item.get("file_name") or "").strip()
        metadata = item.get("metadata")
        if not question:
            errors.append(f"{path.name}:{row_number}: empty task_question")
        if not answer:
            errors.append(f"{path.name}:{row_number}: empty ground_truth")
        if not file_name:
            errors.append(f"{path.name}:{row_number}: empty file_name")
        else:
            file_names.append(file_name)
        if not isinstance(metadata, dict):
            errors.append(f"{path.name}:{row_number}: metadata is not an object")
        elif not str(metadata.get("dataset") or "").strip():
            errors.append(f"{path.name}:{row_number}: metadata.dataset is missing")
        if question and answer and obvious_answer_leak(question, answer):
            warnings.append(
                f"{path.name}:{row_number}: ground_truth appears verbatim in "
                "task_question"
            )
    duplicates = sorted(
        name for name, count in Counter(file_names).items() if count > 1
    )
    if duplicates:
        errors.append(
            f"{path.name}: duplicate file_name values: {', '.join(duplicates[:10])}"
        )
    return errors, warnings


def distribution(
    records: list[dict[str, Any]], metadata_key: str
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(item.get("metadata", {}).get(metadata_key) or "unspecified")
                for item in records
            ).items()
        )
    )


def main() -> int:
    all_errors: list[str] = []
    all_warnings: list[str] = []
    loaded: dict[str, list[dict[str, Any]]] = {}

    for filename, expected_count in EXPECTED.items():
        path = DATA_DIR / filename
        records, parse_errors = read_jsonl(path)
        loaded[filename] = records
        errors, warnings = validate_records(path, records, expected_count)
        all_errors.extend(parse_errors)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    print("Benchmark data summary")
    print("======================")
    for filename, expected_count in EXPECTED.items():
        records = loaded[filename]
        status = "OK" if len(records) == expected_count else "FAIL"
        print(f"{filename}: {len(records)}/{expected_count} [{status}]")

    print("\nDataset/category distributions")
    for filename in (
        "browsecomp_subset_20.jsonl",
        "gaia_subset_20.jsonl",
        "xbench_deepsearch_subset_20.jsonl",
        "self_deepsearch_qa_20.jsonl",
    ):
        records = loaded[filename]
        datasets = distribution(records, "dataset")
        categories = distribution(records, "category")
        print(f"{filename}: datasets={datasets}; categories={categories}")

    gaia_records = loaded["gaia_subset_20.jsonl"]
    print(f"\nGAIA level distribution: {distribution(gaia_records, 'level')}")
    self_records = loaded["self_deepsearch_qa_20.jsonl"]
    print(f"Self QA category distribution: {distribution(self_records, 'category')}")

    all_file_names = [
        str(item.get("file_name") or "")
        for filename, records in loaded.items()
        if filename != "all_80.jsonl"
        for item in records
    ]
    duplicate_names = sorted(
        name
        for name, count in Counter(all_file_names).items()
        if name and count > 1
    )
    print(
        "Duplicate file_name across four component subsets:",
        "none" if not duplicate_names else ", ".join(duplicate_names),
    )
    if duplicate_names:
        all_errors.append(
            "duplicate file_name values across component subsets: "
            + ", ".join(duplicate_names)
        )

    component_records = []
    for filename in (
        "browsecomp_subset_20.jsonl",
        "gaia_subset_20.jsonl",
        "xbench_deepsearch_subset_20.jsonl",
        "self_deepsearch_qa_20.jsonl",
    ):
        component_records.extend(loaded[filename])
    combined = loaded["all_80.jsonl"]
    if combined and component_records != combined:
        all_errors.append(
            "all_80.jsonl is not the exact ordered concatenation of the four subsets"
        )

    if all_warnings:
        print("\nWarnings")
        for warning in all_warnings:
            print(f"- {warning}")
    if all_errors:
        print("\nErrors", file=sys.stderr)
        for error in all_errors:
            print(f"- {error}", file=sys.stderr)
        print(
            f"\nVALIDATION FAILED: {len(all_errors)} error(s), "
            f"{len(all_warnings)} warning(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nVALIDATION PASSED: {sum(EXPECTED.values())} checked rows across "
        f"{len(EXPECTED)} files, {len(all_warnings)} warning(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

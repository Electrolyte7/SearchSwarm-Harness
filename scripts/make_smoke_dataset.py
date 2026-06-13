#!/usr/bin/env python3
"""Copy the first N valid JSON objects from a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1")
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")

    selected: list[dict] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                parser.error(f"{args.input}:{line_number}: invalid JSON: {exc}")
            if not isinstance(item, dict):
                parser.error(f"{args.input}:{line_number}: row is not an object")
            selected.append(item)
            if len(selected) == args.n:
                break

    if len(selected) != args.n:
        parser.error(
            f"requested {args.n} rows but {args.input} contains only "
            f"{len(selected)} non-empty rows"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} row(s): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

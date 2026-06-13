#!/usr/bin/env python3
"""Prepare fixed 20-item benchmark subsets for the SearchSwarm harness."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SEED = 42
SUBSET_SIZE = 20

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "eval_data" / "benchmark"
CACHE_DIR = OUTPUT_DIR / ".cache"

BROWSECOMP_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/"
    "browse_comp_test_set.csv"
)
BROWSECOMP_EXPECTED_SHA256 = (
    "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
)

XBENCH_REPO = "xbench/DeepSearch"
XBENCH_REVISION = "436bbed79aef5b19c857047650ab528be33c6680"
XBENCH_FILENAME = "DeepSearch.csv"
XBENCH_EXPECTED_SHA256 = (
    "10bdb81321e3d919c052c2c9a7095868d8bc9036f719fb25d9223043aa28c118"
)

GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"
GAIA_FILENAME = "2023/validation/metadata.parquet"

OUTPUTS = {
    "browsecomp": OUTPUT_DIR / "browsecomp_subset_20.jsonl",
    "gaia": OUTPUT_DIR / "gaia_subset_20.jsonl",
    "xbench": OUTPUT_DIR / "xbench_deepsearch_subset_20.jsonl",
    "self": OUTPUT_DIR / "self_deepsearch_qa_20.jsonl",
    "all": OUTPUT_DIR / "all_80.jsonl",
}


class PreparationError(RuntimeError):
    """A benchmark could not be downloaded or converted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, destination)
        except Exception as exc:
            raise PreparationError(f"download failed for {url}: {exc}") from exc
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise PreparationError(
            f"SHA-256 mismatch for {destination}: expected {expected_sha256}, "
            f"got {actual}. Remove the cache file and review the upstream change."
        )
    return destination


def hf_download(repo_id: str, filename: str, revision: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PreparationError(
            "huggingface-hub is required; install dependencies from requirements.txt"
        ) from exc

    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                cache_dir=CACHE_DIR / "huggingface",
                token=os.getenv("HF_TOKEN") or None,
            )
        )
    except Exception as exc:
        chain = []
        current: BaseException | None = exc
        while current is not None and current not in chain:
            chain.append(current)
            current = current.__cause__ or current.__context__
        message = " | ".join(str(error) for error in chain)
        message_lower = message.lower()
        if "fine-grained token" in message_lower or (
            "403 forbidden" in message_lower
            and "public gated repositories" in message_lower
        ):
            raise PreparationError(
                f"{repo_id} access is approved, but the active fine-grained "
                "Hugging Face token cannot read public gated repositories. "
                "Enable that permission in the token settings or log in with "
                f"a read token, then rerun this script. Original error: {message}"
            ) from exc
        if "gated" in message_lower or "401" in message:
            raise PreparationError(
                f"{repo_id} is gated. Accept access at "
                f"https://huggingface.co/datasets/{repo_id}, authenticate with "
                "`hf auth login` or set HF_TOKEN, then rerun this script. "
                f"Original error: {message}"
            ) from exc
        raise PreparationError(
            f"failed to download {repo_id}/{filename} at {revision}: {message}"
        ) from exc


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt_browsecomp(value: str, canary: str) -> str:
    encrypted = base64.b64decode(value)
    key = derive_key(canary, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")


def decrypt_xbench(value: str, canary: str) -> str:
    encrypted = base64.b64decode(value)
    key = canary.encode("utf-8")
    return bytes(
        byte ^ key[index % len(key)] for index, byte in enumerate(encrypted)
    ).decode("utf-8")


def largest_remainder_quotas(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(group_sizes.values())
    raw = {key: total * size / population for key, size in group_sizes.items()}
    quotas = {key: min(group_sizes[key], math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        group_sizes,
        key=lambda key: (raw[key] - math.floor(raw[key]), group_sizes[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < group_sizes[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise PreparationError("could not allocate stratified sample quotas")
    return quotas


def stratified_sample(
    records: list[dict[str, Any]], field: str, size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(field) or "unspecified")].append(record)
    quotas = largest_remainder_quotas(
        {key: len(values) for key, values in groups.items()}, size
    )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], quotas[key]))
    selected.sort(key=lambda item: int(item["_row_index"]))
    return selected, quotas


def standard_record(
    task_id: str,
    question: str,
    answer: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_question": question.strip(),
        "file_name": task_id,
        "ground_truth": answer.strip(),
        "metadata": metadata,
    }


def prepare_browsecomp() -> list[dict[str, Any]]:
    path = download_url(
        BROWSECOMP_URL,
        CACHE_DIR / "browsecomp" / "browse_comp_test_set.csv",
        BROWSECOMP_EXPECTED_SHA256,
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_records = []
        for row_index, row in enumerate(csv.DictReader(handle)):
            row["_row_index"] = row_index
            raw_records.append(row)
    selected, quotas = stratified_sample(
        raw_records, "problem_topic", SUBSET_SIZE, SEED
    )
    output = []
    for row in selected:
        original_id = int(row["_row_index"])
        output.append(
            standard_record(
                f"browsecomp-{original_id:04d}",
                decrypt_browsecomp(row["problem"], row["canary"]),
                decrypt_browsecomp(row["answer"], row["canary"]),
                {
                    "dataset": "browsecomp",
                    "category": row.get("problem_topic") or "unspecified",
                    "difficulty": None,
                    "source": BROWSECOMP_URL,
                    "original_dataset": "OpenAI BrowseComp",
                    "original_split": "test",
                    "original_id": original_id,
                    "original_category": row.get("problem_topic"),
                    "sample_seed": SEED,
                    "sampling_method": "stratified",
                    "source_file_sha256": BROWSECOMP_EXPECTED_SHA256,
                    "stratum_sample_size": quotas[
                        row.get("problem_topic") or "unspecified"
                    ],
                },
            )
        )
    print(
        "BrowseComp source:",
        len(raw_records),
        "rows; sampled topic distribution:",
        dict(Counter(item["metadata"]["category"] for item in output)),
    )
    return output


def prepare_xbench() -> list[dict[str, Any]]:
    path = hf_download(XBENCH_REPO, XBENCH_FILENAME, XBENCH_REVISION)
    actual = sha256_file(path)
    if actual != XBENCH_EXPECTED_SHA256:
        raise PreparationError(
            f"SHA-256 mismatch for pinned xBench file: expected "
            f"{XBENCH_EXPECTED_SHA256}, got {actual}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_records = list(csv.DictReader(handle))
    if len(raw_records) < SUBSET_SIZE:
        raise PreparationError(
            f"xBench has only {len(raw_records)} rows; need {SUBSET_SIZE}"
        )
    rng = random.Random(SEED)
    selected = rng.sample(raw_records, SUBSET_SIZE)
    selected.sort(key=lambda item: int(item["id"]))
    output = []
    for row in selected:
        original_id = str(row["id"])
        output.append(
            standard_record(
                f"xbench-deepsearch-{int(original_id):03d}",
                decrypt_xbench(row["prompt"], row["canary"]),
                decrypt_xbench(row["answer"], row["canary"]),
                {
                    "dataset": "xbench_deepsearch",
                    "category": "unspecified",
                    "difficulty": None,
                    "source": f"https://huggingface.co/datasets/{XBENCH_REPO}",
                    "original_dataset": XBENCH_REPO,
                    "original_split": XBENCH_FILENAME,
                    "original_id": original_id,
                    "sample_seed": SEED,
                    "sampling_method": "random",
                    "source_revision": XBENCH_REVISION,
                    "source_file_sha256": XBENCH_EXPECTED_SHA256,
                    "reference_steps_available": bool(row.get("reference_steps")),
                },
            )
        )
    print(f"xBench source: {len(raw_records)} rows; random sample seed={SEED}")
    return output


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def find_column(columns: Iterable[str], aliases: Iterable[str]) -> str:
    lookup = {column.casefold(): column for column in columns}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    raise PreparationError(
        f"none of the expected columns {list(aliases)} were found; "
        f"available columns: {list(columns)}"
    )


def normalize_level(value: Any) -> int:
    match = re.search(r"[123]", str(value))
    if not match:
        raise PreparationError(f"unrecognized GAIA level value: {value!r}")
    return int(match.group(0))


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def gaia_stratified_sample(
    records: list[dict[str, Any]], target: dict[int, int], seed: int
) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["_level"]].append(record)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for level in (1, 2, 3):
        take = min(target[level], len(groups[level]))
        level_items = rng.sample(groups[level], take)
        selected.extend(level_items)
        selected_ids.update(str(item["_original_id"]) for item in level_items)
    if len(selected) < SUBSET_SIZE:
        remaining = [
            item
            for item in records
            if str(item["_original_id"]) not in selected_ids
        ]
        selected.extend(rng.sample(remaining, SUBSET_SIZE - len(selected)))
    selected.sort(key=lambda item: str(item["_original_id"]))
    return selected


def prepare_gaia() -> list[dict[str, Any]]:
    path = hf_download(GAIA_REPO, GAIA_FILENAME, GAIA_REVISION)
    try:
        import pandas as pd
    except ImportError as exc:
        raise PreparationError(
            "pandas and pyarrow are required to read GAIA parquet metadata"
        ) from exc

    frame = pd.read_parquet(path)
    question_col = find_column(frame.columns, ["Question", "question"])
    answer_col = find_column(
        frame.columns, ["Final answer", "final_answer", "answer", "Answer"]
    )
    id_col = find_column(frame.columns, ["task_id", "id", "Task ID"])
    level_col = find_column(frame.columns, ["Level", "level", "difficulty"])
    file_col = find_column(frame.columns, ["file_name", "File name", "file"])
    annotator_col = next(
        (
            column
            for column in frame.columns
            if column.casefold()
            in {"annotator metadata", "annotator_metadata", "metadata"}
        ),
        None,
    )

    raw_records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        if not is_blank(row.get(file_col)):
            continue
        item = dict(row)
        item["_level"] = normalize_level(row[level_col])
        item["_original_id"] = str(row[id_col])
        raw_records.append(item)
    if len(raw_records) < SUBSET_SIZE:
        raise PreparationError(
            f"only {len(raw_records)} attachment-free GAIA dev rows are available"
        )
    selected = gaia_stratified_sample(raw_records, {1: 8, 2: 8, 3: 4}, SEED)
    output = []
    for row in selected:
        original_id = str(row["_original_id"])
        annotator_metadata = (
            json_safe(row.get(annotator_col)) if annotator_col else None
        )
        category = "unspecified"
        if isinstance(annotator_metadata, dict):
            category = str(
                annotator_metadata.get("category")
                or annotator_metadata.get("task_type")
                or "unspecified"
            )
        output.append(
            standard_record(
                f"gaia-{original_id}",
                str(row[question_col]),
                str(row[answer_col]),
                {
                    "dataset": "gaia",
                    "category": category,
                    "difficulty": f"Level {row['_level']}",
                    "level": row["_level"],
                    "source": f"https://huggingface.co/datasets/{GAIA_REPO}",
                    "original_dataset": GAIA_REPO,
                    "original_split": "2023_all/validation",
                    "original_id": original_id,
                    "original_level": row["_level"],
                    "original_file_name": None,
                    "annotator_metadata": annotator_metadata,
                    "sample_seed": SEED,
                    "sampling_method": "filtered_stratified",
                    "filter": "file_name is empty/null (no local attachment)",
                    "source_revision": GAIA_REVISION,
                    "source_file_sha256": sha256_file(path),
                },
            )
        )
    print(
        f"GAIA source: {len(frame)} dev rows; "
        f"{len(raw_records)} attachment-free; sampled levels:",
        dict(Counter(item["metadata"]["level"] for item in output)),
    )
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PreparationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
    return records


def validate_self_qa(records: list[dict[str, Any]]) -> None:
    if len(records) != SUBSET_SIZE:
        raise PreparationError(
            f"self QA must contain {SUBSET_SIZE} rows, found {len(records)}"
        )
    categories = Counter(
        item.get("metadata", {}).get("category") for item in records
    )
    expected = {
        "多约束实体识别": 5,
        "时间线追踪": 5,
        "候选比较排除": 5,
        "学术 / 技术检索": 5,
    }
    if categories != Counter(expected):
        raise PreparationError(
            f"self QA category distribution must be {expected}, got {dict(categories)}"
        )
    for index, item in enumerate(records, 1):
        metadata = item.get("metadata") or {}
        missing = [
            key
            for key in (
                "dataset",
                "category",
                "difficulty",
                "expected_reasoning_type",
            )
            if not metadata.get(key)
        ]
        if not metadata.get("source_hint") and not metadata.get("evidence_hint"):
            missing.append("source_hint/evidence_hint")
        if missing:
            raise PreparationError(
                f"self QA row {index} missing metadata: {', '.join(missing)}"
            )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {path.relative_to(ROOT)}: {len(records)} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("browsecomp", "gaia", "xbench", "all"),
        default="all",
        help="prepare one official subset or all subsets (default: all)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    self_records = read_jsonl(OUTPUTS["self"])
    validate_self_qa(self_records)
    print(
        "Self QA:",
        len(self_records),
        "rows;",
        dict(Counter(item["metadata"]["category"] for item in self_records)),
    )

    builders = {
        "browsecomp": prepare_browsecomp,
        "gaia": prepare_gaia,
        "xbench": prepare_xbench,
    }
    targets = list(builders) if args.only == "all" else [args.only]
    prepared: dict[str, list[dict[str, Any]]] = {"self": self_records}
    failures: dict[str, str] = {}

    for name in targets:
        try:
            records = builders[name]()
            if len(records) != SUBSET_SIZE:
                raise PreparationError(
                    f"{name} produced {len(records)} rows, expected {SUBSET_SIZE}"
                )
            write_jsonl(OUTPUTS[name], records)
            prepared[name] = records
        except Exception as exc:
            failures[name] = str(exc)
            print(f"ERROR [{name}]: {exc}", file=sys.stderr)

    if args.only == "all":
        for name in ("browsecomp", "gaia", "xbench"):
            if name not in prepared and OUTPUTS[name].exists():
                try:
                    existing = read_jsonl(OUTPUTS[name])
                    if len(existing) == SUBSET_SIZE:
                        prepared[name] = existing
                        print(
                            f"Using existing {OUTPUTS[name].relative_to(ROOT)} "
                            "because fresh preparation failed."
                        )
                except Exception:
                    pass
        if all(name in prepared for name in ("browsecomp", "gaia", "xbench", "self")):
            combined = []
            for name in ("browsecomp", "gaia", "xbench", "self"):
                combined.extend(prepared[name])
            write_jsonl(OUTPUTS["all"], combined)
        else:
            if OUTPUTS["all"].exists():
                OUTPUTS["all"].unlink()
            missing = sorted(
                {"browsecomp", "gaia", "xbench", "self"} - prepared.keys()
            )
            print(
                "all_80.jsonl was not generated because these subsets are "
                f"missing: {', '.join(missing)}",
                file=sys.stderr,
            )

    for name, path in OUTPUTS.items():
        if path.exists():
            try:
                count = len(read_jsonl(path))
            except Exception:
                count = "invalid"
            print(f"OUTPUT {path.relative_to(ROOT)}: {count}")
        else:
            print(f"OUTPUT {path.relative_to(ROOT)}: MISSING")

    if failures:
        print("\nPreparation completed with failures:", file=sys.stderr)
        for name, reason in failures.items():
            print(f"- {name}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

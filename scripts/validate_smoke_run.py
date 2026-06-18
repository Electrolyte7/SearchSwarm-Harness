#!/usr/bin/env python3
"""Validate one benchmark wrapper run without invoking a judge."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_safety import (
    contains_pseudo_tool_call,
    is_failed_placeholder_prediction,
    is_suppressed_prediction,
    is_usable_prediction,
    pseudo_tool_call_reasons,
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
        and not status.startswith("skipped")
        and (allow_fallback or not status.endswith("_fallback"))
        and content not in _UNUSABLE_SUBAGENT_CONTENT
        and not contains_pseudo_tool_call(content)
    )


def build_subagent_patch_summary(trajectory_records):
    statuses = [
        str(record.get("status") or "").strip().lower()
        for record in trajectory_records
        if isinstance(record, dict)
    ]
    numeric_steps = [
        float(record.get("steps", record.get("llm_calls", 0)) or 0)
        for record in trajectory_records
        if isinstance(record, dict)
    ]
    numeric_tool_calls = [
        float(record.get("tool_calls", 0) or 0)
        for record in trajectory_records
        if isinstance(record, dict)
    ]
    duplicate_pairs = []
    for record in trajectory_records:
        if not isinstance(record, dict):
            continue
        if record.get("duplicate_subagent_skipped"):
            duplicate_pairs.append({
                "prompt": record.get("prompt", ""),
                "matched_prompt": record.get("duplicate_matched_prompt", ""),
                "similarity": record.get("duplicate_similarity"),
            })
    return {
        "patch_enabled": any(
            bool(record.get("patch_enabled"))
            for record in trajectory_records
            if isinstance(record, dict)
        ),
        "subagent_early_stop_count": sum(
            1 for record in trajectory_records
            if isinstance(record, dict) and record.get("early_stop_triggered")
        ),
        "avg_subagent_steps": (
            round(sum(numeric_steps) / len(numeric_steps), 2)
            if numeric_steps else 0
        ),
        "avg_subagent_tool_calls": (
            round(sum(numeric_tool_calls) / len(numeric_tool_calls), 2)
            if numeric_tool_calls else 0
        ),
        "subagent_completed_count": statuses.count("completed"),
        "subagent_fallback_count": sum(
            1 for status in statuses if status.endswith("_fallback")
        ),
        "subagent_max_calls_count": sum(
            1 for status in statuses if status.startswith("max_calls")
        ),
        "duplicate_subagent_skipped_count": sum(
            1 for record in trajectory_records
            if isinstance(record, dict) and record.get("duplicate_subagent_skipped")
        ),
        "duplicate_subagent_brief_pairs": duplicate_pairs,
        "subagent_brief_count_before_filter": sum(
            int(record.get("brief_count_before_filter") or 0)
            for record in trajectory_records
            if isinstance(record, dict)
            and record.get("duplicate_subagent_skipped")
        ),
        "subagent_brief_count_after_filter": sum(
            int(record.get("brief_count_after_filter") or 0)
            for record in trajectory_records
            if isinstance(record, dict)
            and record.get("duplicate_subagent_skipped")
        ),
        "low_quality_report_count": sum(
            1 for record in trajectory_records
            if isinstance(record, dict) and record.get("low_quality_report") is True
        ),
        "high_quality_report_count": sum(
            1 for record in trajectory_records
            if isinstance(record, dict) and record.get("low_quality_report") is False
        ),
        "report_with_candidate_count": sum(
            1 for record in trajectory_records
            if isinstance(record, dict) and record.get("report_has_candidate")
        ),
        "report_with_evidence_count": sum(
            1 for record in trajectory_records
            if isinstance(record, dict) and record.get("report_has_evidence")
        ),
    }


def build_patch_v2_result_summary(records):
    result_records = [
        record for record in records
        if isinstance(record, dict)
    ]

    def sum_int(key):
        return sum(int(record.get(key) or 0) for record in result_records)

    return {
        "patch_v2_enabled": any(
            bool(record.get("patch_v2_enabled"))
            for record in result_records
        ),
        "candidate_ledger_enabled": any(
            bool(record.get("candidate_ledger_enabled"))
            for record in result_records
        ),
        "candidate_count": sum_int("candidate_count"),
        "candidate_from_main_count": sum_int("candidate_from_main_count"),
        "candidate_from_subagent_count": sum_int("candidate_from_subagent_count"),
        "candidate_from_low_quality_report_count": sum_int(
            "candidate_from_low_quality_report_count"),
        "candidate_deduplicated_count": sum_int("candidate_deduplicated_count"),
        "final_verifier_used_count": sum_int("final_verifier_used_count"),
        "final_verifier_changed_answer_count": sum_int(
            "final_verifier_changed_answer_count"),
        "final_verifier_kept_answer_count": sum_int(
            "final_verifier_kept_answer_count"),
        "final_verifier_rejected_candidate_count": sum_int(
            "final_verifier_rejected_candidate_count"),
        "final_verifier_low_confidence_count": sum_int(
            "final_verifier_low_confidence_count"),
        "final_verifier_empty_or_failed_count": sum_int(
            "final_verifier_empty_or_failed_count"),
        "adaptive_router_enabled": any(
            bool(record.get("adaptive_router_enabled"))
            for record in result_records
        ),
        "router_decision_count": sum_int("router_decision_count"),
        "router_skip_delegation_count": sum_int("router_skip_delegation_count"),
        "router_allow_delegation_count": sum_int("router_allow_delegation_count"),
        "router_force_diverse_brief_count": sum_int(
            "router_force_diverse_brief_count"),
        "router_stop_delegation_count": sum_int("router_stop_delegation_count"),
        "diverse_brief_generated_count": sum_int("diverse_brief_generated_count"),
        "duplicate_brief_rewritten_count": sum_int(
            "duplicate_brief_rewritten_count"),
        "main_agent_early_finalize_count": sum_int(
            "main_agent_early_finalize_count"),
    }


def build_validation_summary(records, trajectory_records, run_exit_code=None,
                             validation_status="unknown",
                             validation_executed=True):
    predictions = [
        record.get("prediction")
        for record in records
        if isinstance(record, dict)
    ]
    statuses = [
        str(record.get("status") or "").strip().lower()
        for record in trajectory_records
        if isinstance(record, dict)
    ]
    summary = {
        "run_success": run_exit_code == 0 if run_exit_code is not None else None,
        "run_exit_code": run_exit_code,
        "validation_status": validation_status,
        "validation_executed": validation_executed,
        "result_count": len(records),
        "prediction_empty_count": sum(
            1 for prediction in predictions
            if not isinstance(prediction, str) or not prediction.strip()
        ),
        "prediction_dsml_count": sum(
            1 for prediction in predictions
            if contains_pseudo_tool_call(prediction)
        ),
        "prediction_no_answer_count": sum(
            1 for prediction in predictions
            if isinstance(prediction, str)
            and prediction.strip().lower().startswith("no answer found")
        ),
        "prediction_suppressed_count": sum(
            1 for prediction in predictions
            if is_suppressed_prediction(prediction)
        ),
        "prediction_failed_placeholder_count": sum(
            1 for prediction in predictions
            if is_failed_placeholder_prediction(prediction)
        ),
        "usable_prediction_count": sum(
            1 for prediction in predictions
            if is_usable_prediction(prediction)
        ),
        "subagent_total": len(trajectory_records),
        "subagent_completed": statuses.count("completed"),
        "subagent_fallback": sum(
            1 for status in statuses if status.endswith("_fallback")
        ),
        "subagent_max_calls": sum(
            1 for status in statuses if status.startswith("max_calls")
        ),
    }
    summary.update(build_subagent_patch_summary(trajectory_records))
    summary.update(build_patch_v2_result_summary(records))
    return summary


def validate_smoke_run(result_path, trajectory_path, setting,
                       require_sub_agent=False, expected_count=None,
                       run_exit_code=None, strict_usable=False):
    errors = []
    records = []
    trajectory_records = []

    if run_exit_code is not None and run_exit_code != 0:
        errors.append(f"run process failed with exit code {run_exit_code}")

    if not result_path.is_file() or result_path.stat().st_size == 0:
        errors.append(f"missing or empty result file: {result_path}")
    else:
        try:
            records = _read_jsonl(result_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid result JSONL: {exc}")
        if not records:
            errors.append("result JSONL contains no records")
        elif expected_count is not None and len(records) != expected_count:
            errors.append(
                f"result JSONL contains {len(records)} records; "
                f"expected {expected_count}"
            )
        for index, record in enumerate(records, 1):
            prediction = record.get("prediction")
            if not isinstance(prediction, str) or not prediction.strip():
                errors.append(f"result record {index} has an empty prediction")
            else:
                if contains_pseudo_tool_call(prediction):
                    errors.append(
                        f"result record {index} contains pseudo tool-call "
                        "text in prediction: "
                        f"{', '.join(pseudo_tool_call_reasons(prediction))}"
                    )
                if prediction.strip().lower().startswith("no answer found"):
                    errors.append(
                        f"result record {index} contains a no-answer sentinel")
                if strict_usable and not is_usable_prediction(prediction):
                    errors.append(
                        f"result record {index} does not contain a usable "
                        "prediction"
                    )
            if record.get("error"):
                errors.append(
                    f"result record {index} contains error: {record.get('error')}"
                )
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
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--run-exit-code", type=int)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--strict-usable", action="store_true")
    args = parser.parse_args()

    errors, result_count, trajectory_count = validate_smoke_run(
        args.result,
        args.trajectory,
        args.setting,
        require_sub_agent=args.require_sub_agent,
        expected_count=args.expected_count,
        run_exit_code=args.run_exit_code,
        strict_usable=args.strict_usable,
    )
    records = _read_jsonl(args.result) if args.result.is_file() else []
    trajectory_records = (
        _read_jsonl(args.trajectory) if args.trajectory.is_file() else []
    )
    summary = build_validation_summary(
        records,
        trajectory_records,
        run_exit_code=args.run_exit_code,
        validation_status="failed" if errors else "success",
        validation_executed=True,
    )
    if args.summary_json:
        with args.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    if errors:
        print("Smoke validation failed:")
        for error in errors:
            print(f"- {error}")
        print("Validation summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print(
        "Smoke validation passed: "
        f"results={result_count}, subagent_trajectories={trajectory_count}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

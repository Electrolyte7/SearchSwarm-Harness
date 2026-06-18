import os
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_smoke_run import (
    build_validation_summary,
    validate_smoke_run,
)
from final_safety import contains_pseudo_tool_call
from tool_call_utils import normalize_tool_args
import patch_v2


class ToolArgumentNormalizationTests(unittest.TestCase):
    def test_unwraps_params_only_object(self):
        wrapped = {"params": {"prompts": [{"prompt": "p", "goal": "g"}]}}
        self.assertEqual(
            normalize_tool_args(wrapped),
            {"prompts": [{"prompt": "p", "goal": "g"}]},
        )

    def test_preserves_params_when_other_keys_exist(self):
        value = {"params": {"query": ["a"]}, "trace_id": "one"}
        self.assertEqual(normalize_tool_args(value), value)


class ApiCallBudgetTests(unittest.TestCase):
    def test_api_token_counter_reserves_last_call_for_final_answer(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        agent._model_uses_reasoning = False
        agent._tokenizer = None

        calls = []

        def fake_call_api(messages, max_tokens=None, use_tools=True):
            calls.append(use_tools)
            if use_tools:
                return (
                    "",
                    [{
                        "id": "search-1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": ["evidence"]}',
                        },
                    }],
                    {"prompt_tokens": 100},
                    "",
                    "tool_calls",
                )
            return (
                "<answer>final answer</answer>",
                None,
                {"prompt_tokens": 200},
                "",
                "stop",
            )

        agent.call_api = fake_call_api
        agent.custom_call_tool = lambda *args, **kwargs: "evidence"

        data = {"item": {"task_question": "question", "ground_truth": "truth"}}
        with patch.object(react_agent, "MAX_LLM_CALL_PER_RUN", 2), \
                patch.object(react_agent, "TOKEN_COUNTER", "api"), \
                patch.object(react_agent, "RUN_TIMEOUT_MINUTES", 1):
            result = agent._run_api(data, "test-model")

        self.assertEqual(result["prediction"], "final answer")
        self.assertEqual(
            result["termination"],
            "generate an answer as llm call limit reached",
        )
        self.assertEqual(calls, [True, False])

    def test_no_tool_final_dsml_is_repaired_before_prediction_write(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        agent._model_uses_reasoning = False
        agent._tokenizer = None
        agent.model_mode = "api"

        calls = []

        def fake_call_api(messages, max_tokens=None, use_tools=True):
            calls.append(use_tools)
            if use_tools:
                return (
                    '<｜tool▁calls▁begin｜>{"name":"search","arguments":{}}',
                    None,
                    {"prompt_tokens": 100},
                    "",
                    "stop",
                )
            return (
                "<answer>clean final answer</answer>",
                None,
                {"prompt_tokens": 120},
                "",
                "stop",
            )

        agent.call_api = fake_call_api
        data = {"item": {"task_question": "question", "ground_truth": "truth"}}
        with patch.object(react_agent, "MAX_LLM_CALL_PER_RUN", 5), \
                patch.object(react_agent, "TOKEN_COUNTER", "api"), \
                patch.object(react_agent, "RUN_TIMEOUT_MINUTES", 1):
            result = agent._run_api(data, "test-model")

        self.assertEqual(result["prediction"], "clean final answer")
        self.assertFalse(contains_pseudo_tool_call(result["prediction"]))
        self.assertEqual(calls, [True, False])

    def test_final_dsml_repair_failure_is_suppressed(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        agent._model_uses_reasoning = False
        agent._tokenizer = None
        agent.model_mode = "api"

        def fake_call_api(messages, max_tokens=None, use_tools=True):
            bad = '<tool_call>{"name":"search","arguments":{}}</tool_call>'
            return bad, None, {"prompt_tokens": 100}, "", "stop"

        agent.call_api = fake_call_api
        data = {"item": {"task_question": "question", "ground_truth": "truth"}}
        with patch.object(react_agent, "MAX_LLM_CALL_PER_RUN", 5), \
                patch.object(react_agent, "TOKEN_COUNTER", "api"), \
                patch.object(react_agent, "RUN_TIMEOUT_MINUTES", 1):
            result = agent._run_api(data, "test-model")

        self.assertIn("invalid tool-call-like final answer suppressed",
                      result["prediction"])
        self.assertFalse(contains_pseudo_tool_call(result["prediction"]))

    def test_post_verifier_tool_call_output_is_sanitized(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        agent._model_uses_reasoning = False
        agent._tokenizer = None
        agent.model_mode = "api"

        calls = []

        def fake_call_api(messages, max_tokens=None, use_tools=True):
            calls.append(use_tools)
            if use_tools:
                return (
                    "<answer>draft answer</answer>",
                    None,
                    {"prompt_tokens": 100},
                    "",
                    "stop",
                )
            return (
                "<answer>short answer</answer>",
                None,
                {"prompt_tokens": 120},
                "",
                "stop",
            )

        agent.call_api = fake_call_api
        data = {"item": {"task_question": "question", "ground_truth": "truth"}}
        with patch.object(react_agent, "MAX_LLM_CALL_PER_RUN", 5), \
                patch.object(react_agent, "TOKEN_COUNTER", "api"), \
                patch.object(react_agent, "RUN_TIMEOUT_MINUTES", 1), \
                patch.object(
                    agent,
                    "_patch_v2_verify_prediction",
                    return_value=(
                        '<tool_call>{"name":"search","arguments":{}}</tool_call>',
                        {"verifier_changed_answer": True},
                    ),
                ):
            result = agent._run_api(data, "test-model")

        self.assertEqual(result["prediction"], "short answer")
        self.assertFalse(contains_pseudo_tool_call(result["prediction"]))
        self.assertEqual(calls, [True, False])

    def test_post_verifier_long_explanation_is_compacted(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        agent._model_uses_reasoning = False
        agent._tokenizer = None
        agent.model_mode = "api"

        def fake_call_api(messages, max_tokens=None, use_tools=True):
            return (
                "<answer>draft answer</answer>",
                None,
                {"prompt_tokens": 100},
                "",
                "stop",
            )

        agent.call_api = fake_call_api
        data = {"item": {"task_question": "question", "ground_truth": "truth"}}
        long_answer = (
            "TD Insurance Meloche Monnex Program (the home and auto insurance "
            "affinity program for Brock University alumni, also referred to as "
            "the exclusive group insurance program)."
        )
        with patch.object(react_agent, "MAX_LLM_CALL_PER_RUN", 5), \
                patch.object(react_agent, "TOKEN_COUNTER", "api"), \
                patch.object(react_agent, "RUN_TIMEOUT_MINUTES", 1), \
                patch.object(
                    agent,
                    "_patch_v2_verify_prediction",
                    return_value=(
                        long_answer,
                        {"verifier_changed_answer": True},
                    ),
                ):
            result = agent._run_api(data, "test-model")

        self.assertEqual(
            result["prediction"], "TD Insurance Meloche Monnex Program")
        self.assertIn("compacted long final answer", result["termination"])

    def test_required_sub_agent_bootstrap_runs_when_model_does_not_delegate(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        agent._model_uses_reasoning = False
        agent._tokenizer = None
        agent.model_mode = "api"

        calls = []

        def fake_call_tool(tool_name, tool_args, **kwargs):
            calls.append((tool_name, tool_args, kwargs))
            return (
                'A sub-agent dispatched for goal "bootstrap smoke sub-agent '
                'report" returned the following report:\n\n'
                '<report>\nanswer: candidate\n'
                'evidence:\n- evidence\nconfidence: low\n</report>'
            )

        def fake_call_api(messages, max_tokens=None, use_tools=True):
            return (
                "<answer>final answer</answer>",
                None,
                {"prompt_tokens": 100},
                "",
                "stop",
            )

        agent.custom_call_tool = fake_call_tool
        agent.call_api = fake_call_api
        data = {"item": {"task_question": "question", "ground_truth": "truth"}}
        with patch.object(react_agent, "ENABLE_SUB_AGENT", True), \
                patch.object(react_agent, "REQUIRE_SUB_AGENT_CALL", True), \
                patch.object(react_agent, "TOOL_MAP", {"call_sub_agent": object()}), \
                patch.object(react_agent, "MAX_LLM_CALL_PER_RUN", 5), \
                patch.object(react_agent, "TOKEN_COUNTER", "api"), \
                patch.object(react_agent, "RUN_TIMEOUT_MINUTES", 10):
            result = agent._run_api(data, "test-model")

        self.assertEqual(result["prediction"], "final answer")
        self.assertEqual(calls[0][0], "call_sub_agent")
        self.assertIn("parent_deadline", calls[0][2])
        tool_messages = [
            m for m in result["messages"]
            if m.get("role") == "tool"
            and m.get("tool_call_id") == "bootstrap_sub_agent_1"
        ]
        bootstrap_assistant = [
            m for m in result["messages"]
            if m.get("role") == "assistant"
            and any(
                tc.get("id") == "bootstrap_sub_agent_1"
                for tc in (m.get("tool_calls") or [])
            )
        ]
        self.assertEqual(len(bootstrap_assistant), 1)
        self.assertIn("reasoning_content", bootstrap_assistant[0])
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("<report>", tool_messages[0]["content"])

    def test_main_synthetic_tool_call_gets_reasoning_backstop(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        agent._model_uses_reasoning = True

        message = agent._make_assistant_msg(
            "synthetic delegation",
            tool_calls=[{
                "id": "synthetic-1",
                "type": "function",
                "function": {
                    "name": "call_sub_agent",
                    "arguments": "{}",
                },
            }],
        )

        self.assertEqual(message["reasoning_content"], ".")
        self.assertEqual(message["tool_calls"][0]["id"], "synthetic-1")

    def test_main_normalizes_missing_or_duplicate_tool_call_ids(self):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import react_agent

        agent = react_agent.MultiTurnReactAgent.__new__(
            react_agent.MultiTurnReactAgent)
        calls = [
            {
                "id": None,
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            },
            {
                "id": None,
                "type": "function",
                "function": {"name": "visit", "arguments": "{}"},
            },
        ]
        normalized = agent._normalize_tool_calls_for_history(calls, 3)

        self.assertEqual(normalized[0]["id"], "synthetic_tool_call_3_1")
        self.assertEqual(normalized[1]["id"], "synthetic_tool_call_3_2")
        self.assertNotEqual(normalized[0]["id"], normalized[1]["id"])


class SmokeValidationTests(unittest.TestCase):
    def test_rejects_incomplete_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            result.write_text(
                json.dumps({"prediction": "answer"}) + "\n",
                encoding="utf-8",
            )
            errors, result_count, _ = validate_smoke_run(
                result,
                root / "subagent_trajectories.jsonl",
                "single",
                expected_count=20,
            )
            self.assertEqual(result_count, 1)
            self.assertTrue(any(
                "expected 20" in error for error in errors))

    def test_rejects_no_answer_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            result.write_text(
                json.dumps({
                    "prediction": "No answer found.",
                    "termination": "exceed available llm calls",
                }) + "\n",
                encoding="utf-8",
            )
            errors, _, _ = validate_smoke_run(
                result,
                root / "subagent_trajectories.jsonl",
                "single",
            )
            self.assertTrue(any("no-answer sentinel" in error for error in errors))
            self.assertTrue(any("termination" in error for error in errors))

    def test_rejects_pseudo_tool_call_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            result.write_text(
                json.dumps({
                    "prediction": (
                        '<｜tool▁calls▁begin｜>{"name":"search",'
                        '"arguments":{}}'
                    ),
                }) + "\n",
                encoding="utf-8",
            )
            errors, _, _ = validate_smoke_run(
                result,
                root / "subagent_trajectories.jsonl",
                "single",
            )
            self.assertTrue(any("pseudo tool-call" in error for error in errors))

    def test_nonzero_run_exit_code_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            result.write_text(
                json.dumps({"prediction": "answer"}) + "\n",
                encoding="utf-8",
            )
            errors, _, _ = validate_smoke_run(
                result,
                root / "subagent_trajectories.jsonl",
                "single",
                run_exit_code=1,
            )
            self.assertTrue(any("exit code 1" in error for error in errors))

    def test_suppressed_placeholder_summary_and_strict_usable(self):
        prediction = (
            "[Failed: invalid tool-call-like final answer suppressed "
            "(dsml_tool_call)]"
        )
        summary = build_validation_summary(
            [{"prediction": prediction}],
            [],
            run_exit_code=0,
            validation_status="success",
            validation_executed=True,
        )
        self.assertEqual(summary["prediction_suppressed_count"], 1)
        self.assertEqual(summary["prediction_failed_placeholder_count"], 1)
        self.assertEqual(summary["usable_prediction_count"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            result.write_text(
                json.dumps({"prediction": prediction}) + "\n",
                encoding="utf-8",
            )
            errors, _, _ = validate_smoke_run(
                result,
                root / "subagent_trajectories.jsonl",
                "single",
                strict_usable=True,
            )
            self.assertTrue(any("usable prediction" in error for error in errors))

    def test_swarm_requires_usable_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            trajectory = root / "subagent_trajectories.jsonl"
            result.write_text(
                json.dumps({"prediction": "answer"}) + "\n",
                encoding="utf-8",
            )

            errors, _, _ = validate_smoke_run(
                result, trajectory, "swarm", require_sub_agent=True)
            self.assertTrue(any("did not produce" in error for error in errors))

            trajectory.write_text(
                json.dumps({"status": "completed", "content": "report"}) + "\n",
                encoding="utf-8",
            )
            errors, result_count, trajectory_count = validate_smoke_run(
                result, trajectory, "swarm", require_sub_agent=True)
            self.assertEqual(errors, [])
            self.assertEqual((result_count, trajectory_count), (1, 1))

    def test_swarm_rejects_text_encoded_tool_call_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            trajectory = root / "subagent_trajectories.jsonl"
            result.write_text(
                json.dumps({"prediction": "answer"}) + "\n",
                encoding="utf-8",
            )
            trajectory.write_text(
                json.dumps({
                    "status": "max_calls",
                    "content": (
                        '<｜｜DSML｜｜tool_calls>'
                        '<｜｜DSML｜｜invoke name="search">'
                    ),
                }) + "\n",
                encoding="utf-8",
            )
            errors, _, _ = validate_smoke_run(
                result, trajectory, "swarm", require_sub_agent=True)
            self.assertTrue(any(
                "text-encoded tool calls" in error for error in errors))

    def test_strict_swarm_rejects_fallback_only_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            trajectory = root / "subagent_trajectories.jsonl"
            result.write_text(
                json.dumps({"prediction": "answer"}) + "\n",
                encoding="utf-8",
            )
            trajectory.write_text(
                json.dumps({
                    "status": "max_calls_fallback",
                    "content": (
                        "[Fallback report: recovered evidence.]\n\n"
                        "Useful evidence."
                    ),
                }) + "\n",
                encoding="utf-8",
            )
            errors, _, _ = validate_smoke_run(
                result, trajectory, "swarm", require_sub_agent=True)
            self.assertTrue(any(
                "fallback-only delivery" in error for error in errors))

    def test_single_rejects_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "iter1.jsonl"
            trajectory = root / "subagent_trajectories.jsonl"
            result.write_text(
                json.dumps({"prediction": "answer"}) + "\n",
                encoding="utf-8",
            )
            trajectory.write_text(
                json.dumps({"status": "completed", "content": "report"}) + "\n",
                encoding="utf-8",
            )
            errors, _, _ = validate_smoke_run(
                result, trajectory, "single", require_sub_agent=False)
            self.assertTrue(any("unexpectedly produced" in error for error in errors))


class PromptModeTests(unittest.TestCase):
    def test_required_dispatch_instruction_is_opt_in(self):
        import prompt

        with patch.dict(os.environ, {
            "ENABLE_SUB_AGENT": "1",
            "REQUIRE_SUB_AGENT_CALL": "1",
        }):
            self.assertIn("MUST call", prompt.get_preamble())

        with patch.dict(os.environ, {
            "ENABLE_SUB_AGENT": "1",
            "REQUIRE_SUB_AGENT_CALL": "0",
        }):
            self.assertNotIn("MUST call", prompt.get_preamble())


class SubAgentForceAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {
            "API_KEY": "offline-test-key",
            "OPENAI_API_KEY": "offline-test-key",
        }):
            import tool_sub_agent
        cls.module = tool_sub_agent

    def _agent(self):
        agent = self.module.SubAgent.__new__(self.module.SubAgent)
        agent._model_uses_reasoning = True
        agent._searched_queries = []
        agent._llm_calls_used = 0
        return agent

    def test_detects_xml_and_dsml_text_tool_calls(self):
        detect = self.module._contains_text_tool_call
        self.assertTrue(detect("<tool_call>{}</tool_call>"))
        self.assertTrue(detect(
            '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="search">'))
        self.assertFalse(detect("<report>usable evidence</report>"))

    def test_sub_agent_synthetic_tool_call_gets_reasoning_backstop(self):
        agent = self._agent()
        message = agent._make_assistant_msg(
            "",
            tool_calls=[{
                "id": "sub-synthetic-1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": ["x"]}',
                },
            }],
        )

        self.assertEqual(message["reasoning_content"], ".")
        self.assertEqual(message["tool_calls"][0]["id"], "sub-synthetic-1")

    def test_structured_force_answer_retries_dsml_then_accepts_report(self):
        agent = self._agent()
        responses = iter([
            (
                '<｜｜DSML｜｜tool_calls>'
                '<｜｜DSML｜｜invoke name="PythonInterpreter">',
                None,
                {},
                "I should run another tool.",
                "stop",
            ),
            (
                "<report>Recovered final report.</report>",
                None,
                {},
                None,
                "stop",
            ),
        ])
        agent._call_llm_structured = lambda *args, **kwargs: next(responses)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "tool", "content": "Useful gathered evidence."},
        ]

        with patch.object(
            self.module, "SUB_AGENT_FORCE_ANSWER_ATTEMPTS", 2
        ):
            result = agent._force_answer_structured(
                messages, object(), "model", True, status="max_calls")

        self.assertEqual(
            result["content"],
            "<report>\nRecovered final report.\n</report>",
        )
        self.assertEqual(result["status"], "max_calls")
        self.assertEqual(result["llm_calls"], 2)
        self.assertIn(
            "Tool execution is disabled.",
            result["messages"][-2]["content"],
        )

    def test_structured_force_answer_falls_back_to_existing_evidence(self):
        agent = self._agent()
        pseudo = (
            '<｜｜DSML｜｜tool_calls>'
            '<｜｜DSML｜｜invoke name="search">'
        )
        agent._call_llm_structured = lambda *args, **kwargs: (
            pseudo, None, {}, "Let me try another search.", "stop")
        messages = [
            {"role": "system", "content": "system"},
            {
                "role": "tool",
                "content": "Verified evidence: the answer is 100 million.",
            },
        ]

        with patch.object(
            self.module, "SUB_AGENT_FORCE_ANSWER_ATTEMPTS", 1
        ):
            result = agent._force_answer_structured(
                messages, object(), "model", True, status="max_calls")

        self.assertEqual(result["status"], "max_calls_fallback")
        self.assertIn("<report>", result["content"])
        self.assertIn("Fallback report", result["content"])
        self.assertIn("100 million", result["content"])
        self.assertNotIn("DSML", result["content"])

    def test_fallback_skips_action_intent_and_empty_tool_output(self):
        messages = [
            {
                "role": "tool",
                "content": "Evidence in page: dinosaurs dominated for 100 million years.",
            },
            {
                "role": "tool",
                "content": "stdout:\nStatus: 200\nEmpty response",
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": (
                    "I've been unable to retrieve it. Let me try a "
                    "different approach."
                ),
            },
        ]
        report = self.module._fallback_report(
            messages,
            (messages[-1]["reasoning_content"],),
        )
        self.assertIn("100 million years", report)
        self.assertIn("<report>", report)
        self.assertNotIn("different approach", report)
        self.assertNotIn("Empty response", report)

    def test_xml_force_answer_retries_tool_call_then_accepts_report(self):
        agent = self._agent()
        responses = iter([
            ("<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>", 10),
            ("<report>XML path recovered report.</report>", 20),
        ])
        agent._call_llm_xml = lambda *args, **kwargs: next(responses)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "tool", "content": "Useful gathered evidence."},
        ]

        with patch.object(
            self.module, "SUB_AGENT_FORCE_ANSWER_ATTEMPTS", 2
        ), patch.object(
            self.module, "count_messages_tokens", return_value=(100, 10000)
        ):
            result = agent._force_answer_xml(
                messages, object(), "model", status="max_calls")

        self.assertEqual(
            result["content"],
            "<report>\nXML path recovered report.\n</report>",
        )
        self.assertEqual(result["status"], "max_calls")
        self.assertEqual(result["llm_calls"], 2)

    def test_call_sub_agent_clips_timeout_to_parent_remaining_budget(self):
        tool = self.module.CallSubAgent(tool_map={})
        captured = {}

        def fake_run(self_agent, prompt, main_model=None, timeout_seconds=None):
            captured["timeout_seconds"] = timeout_seconds
            return {
                "content": "<report>\nanswer: ok\nevidence:\n- e\nconfidence: high\n</report>",
                "messages": [],
                "queries": [],
                "llm_calls": 0,
                "status": "completed",
                "duration_ms": 0,
            }

        with patch.object(self.module.SubAgent, "run", fake_run), \
                patch.object(self.module, "SUB_AGENT_TIMEOUT_MINUTES", 2.0), \
                patch.object(self.module, "PARENT_FINAL_RESERVE_MINUTES", 1.5):
            output = tool.call(
                {
                    "prompts": [{
                        "prompt": "verify one fact",
                        "goal": "verify",
                    }]
                },
                model="model",
                question="parent",
                parent_deadline=time.time() + 200,
            )

        self.assertIn("<report>", output)
        self.assertGreater(captured["timeout_seconds"], 100)
        self.assertLessEqual(captured["timeout_seconds"], 120)

    def test_call_sub_agent_skips_when_parent_budget_too_low(self):
        tool = self.module.CallSubAgent(tool_map={})
        records = []
        with patch.object(self.module.SubAgent, "run") as run_mock, \
                patch.object(self.module, "_write_trajectory",
                             lambda record: records.append(record)), \
                patch.object(self.module, "PARENT_FINAL_RESERVE_MINUTES", 1.5), \
                patch.object(self.module, "SUB_AGENT_MIN_TIMEOUT_SECONDS", 30):
            output = tool.call(
                {
                    "prompts": [{
                        "prompt": "verify one fact",
                        "goal": "verify",
                    }]
                },
                model="model",
                question="parent",
                parent_deadline=time.time() + 100,
            )

        self.assertIn("dispatch skipped", output)
        run_mock.assert_not_called()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "skipped_due_to_parent_budget")
        self.assertIn("<report>", records[0]["content"])

    def test_patch_duplicate_filter_skips_similar_brief(self):
        tool = self.module.CallSubAgent(tool_map={})
        records = []
        calls = []

        def fake_run(self_agent, prompt, main_model=None, timeout_seconds=None):
            calls.append(prompt)
            return {
                "content": (
                    "<report>\n"
                    "candidate_answer: ok\n"
                    "supporting_evidence:\n"
                    "- evidence\n"
                    "confidence: high\n"
                    "</report>"
                ),
                "messages": [],
                "queries": [],
                "llm_calls": 1,
                "tool_calls": 0,
                "steps": 1,
                "status": "completed",
                "duration_ms": 0,
                "early_stop_triggered": False,
                "low_quality_report": False,
                "report_quality_reason": "candidate and evidence present",
                "report_has_candidate": True,
                "report_has_evidence": True,
            }

        brief_a = "Search for Mesut Ozil retirement bodybuilding transformation clue"
        brief_b = "Search Mesut Ozil retired bodybuilding body transformation"
        brief_c = "Search from the article date, source phrase, and team history"

        with patch.object(self.module, "SEARCHSWARM_PATCH_DUPLICATE_FILTER", True), \
                patch.object(self.module, "SEARCHSWARM_PATCH_V1", True), \
                patch.object(self.module.SubAgent, "run", fake_run), \
                patch.object(self.module, "_write_trajectory",
                             lambda record: records.append(record)):
            output = tool.call(
                {
                    "prompts": [
                        {"prompt": brief_a, "goal": "a"},
                        {"prompt": brief_b, "goal": "b"},
                        {"prompt": brief_c, "goal": "c"},
                    ]
                },
                model="model",
                question="q13",
                parent_deadline=time.time() + 300,
            )

        self.assertIn("<report>", output)
        self.assertEqual(calls, [brief_a, brief_c])
        skipped = [
            record for record in records
            if record.get("duplicate_subagent_skipped")
        ]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["status"], "skipped_duplicate")

    def test_patch_early_stop_structured_generates_report(self):
        agent = self._agent()
        responses = []

        def fake_call(*args, **kwargs):
            if not kwargs.get("use_tools", True):
                return (
                    "<report>\n"
                    "candidate_answer: Mesut Ozil\n"
                    "supporting_evidence:\n"
                    "- Search result snippet names Mesut Ozil and a URL.\n"
                    "confidence: medium\n"
                    "uncertainty_or_missing_evidence: needs source visit\n"
                    "early_stop_triggered: true\n"
                    "</report>",
                    None,
                    {},
                    None,
                    "stop",
                )
            responses.append("tool")
            return (
                "I will check a result.",
                [{
                    "id": f"search-{len(responses)}",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": ["Mesut Ozil body transformation"]}',
                    },
                }],
                {"prompt_tokens": 100},
                None,
                "tool_calls",
            )

        agent._call_llm_structured = fake_call
        agent._execute_tool = lambda name, args: (
            "Search result snippet: possible answer: Mesut Ozil. "
            "URL: https://example.com/ozil"
        )

        with patch.object(self.module, "SEARCHSWARM_PATCH_BUDGET_AWARE", True), \
                patch.object(self.module, "SEARCHSWARM_PATCH_REPORT_QUALITY", True), \
                patch.object(self.module, "SUB_AGENT_MAX_LLM_CALLS", 3), \
                patch.object(self.module, "SEARCHSWARM_PATCH_EARLY_STOP_RATIO", 0.66):
            result = agent._run_structured(
                object(), "model", "prompt", is_api=True, timeout_seconds=300)

        self.assertEqual(result["status"], "early_stop")
        self.assertTrue(result["early_stop_triggered"])
        self.assertIn("candidate_answer: Mesut Ozil", result["content"])
        self.assertIn("low_quality_report: false", result["content"])
        self.assertEqual(result["llm_calls"], 3)
        self.assertEqual(result["tool_calls"], 2)

    def test_patch_early_stop_requires_evidence_signal(self):
        messages = [
            {"role": "assistant", "content": "I need to search more."},
            {"role": "user", "content": "No usable observation yet."},
        ]
        with patch.object(self.module, "SEARCHSWARM_PATCH_BUDGET_AWARE", True), \
                patch.object(self.module, "SEARCHSWARM_PATCH_EARLY_STOP_RATIO", 0.66):
            self.assertFalse(self.module._should_early_stop(2, 3, messages))

    def test_patch_report_quality_flags_missing_candidate_or_evidence(self):
        report_1 = (
            "<report>\n"
            "candidate_answer: TD Insurance Meloche Monnex\n"
            "supporting_evidence:\n"
            "- Source page names the sponsor.\n"
            "confidence: high\n"
            "</report>"
        )
        report_2 = (
            "<report>\n"
            "candidate_answer: peaksaver\n"
            "supporting_evidence:\n"
            "confidence: low\n"
            "</report>"
        )
        report_3 = (
            "<report>\n"
            "candidate_answer: \n"
            "supporting_evidence:\n"
            "- vague clue only\n"
            "confidence: low\n"
            "</report>"
        )

        self.assertFalse(self.module._report_quality(report_1)["low_quality_report"])
        self.assertTrue(self.module._report_quality(report_2)["low_quality_report"])
        self.assertTrue(self.module._report_quality(report_3)["low_quality_report"])

    def test_patch_summary_fields(self):
        summary = build_validation_summary(
            [{"prediction": "Mesut Ozil"}],
            [
                {
                    "status": "early_stop",
                    "patch_enabled": True,
                    "early_stop_triggered": True,
                    "steps": 3,
                    "tool_calls": 2,
                    "low_quality_report": False,
                    "report_has_candidate": True,
                    "report_has_evidence": True,
                },
                {
                    "status": "skipped_duplicate",
                    "patch_enabled": True,
                    "duplicate_subagent_skipped": True,
                    "duplicate_similarity": 0.82,
                    "prompt": "b",
                    "duplicate_matched_prompt": "a",
                    "brief_count_before_filter": 3,
                    "brief_count_after_filter": 2,
                    "steps": 0,
                    "tool_calls": 0,
                    "low_quality_report": True,
                    "report_has_candidate": False,
                    "report_has_evidence": False,
                },
            ],
            run_exit_code=0,
            validation_status="success",
        )

        self.assertTrue(summary["patch_enabled"])
        self.assertEqual(summary["subagent_early_stop_count"], 1)
        self.assertEqual(summary["duplicate_subagent_skipped_count"], 1)
        self.assertEqual(summary["low_quality_report_count"], 1)
        self.assertEqual(summary["high_quality_report_count"], 1)
        self.assertEqual(summary["avg_subagent_steps"], 1.5)
        self.assertEqual(summary["avg_subagent_tool_calls"], 1.0)


class PatchV2OfflineTests(unittest.TestCase):
    def test_relation_extraction_brought_to_you_by_clean_candidate(self):
        evidence = "Brought to you by TD Insurance Meloche Monnex."
        candidates = patch_v2.extract_relation_candidates(evidence)

        self.assertIn(
            "TD Insurance Meloche Monnex",
            [item["candidate"] for item in candidates],
        )
        self.assertNotIn(
            "Brought to you by TD Insurance Meloche Monnex",
            [item["candidate"] for item in candidates],
        )
        td = next(
            item for item in candidates
            if item["candidate"] == "TD Insurance Meloche Monnex")
        self.assertEqual(td["answer_role"], "sponsor_or_advertiser")

    def test_relation_extraction_group_insurance_program_through(self):
        evidence = (
            "As a Brock graduate, you enjoy a privileged status through an "
            "exclusive group insurance program through TD Insurance Meloche "
            "Monnex."
        )
        candidates = patch_v2.extract_relation_candidates(evidence)

        self.assertIn(
            "TD Insurance Meloche Monnex",
            [item["candidate"] for item in candidates],
        )
        td = next(
            item for item in candidates
            if item["candidate"] == "TD Insurance Meloche Monnex")
        self.assertEqual(td["answer_role"], "program_provider")

    def test_peaksaver_role_mismatch_for_advertised_program_question(self):
        question = "What is the advertised sponsor program in the article?"
        ledger = patch_v2.CandidateLedger(enabled=True)
        ledger.add(
            "peaksaver",
            "main_observation",
            "Ask about Ontario's peaksaver program. This voluntary program "
            "helps homeowners reduce electricity use and conserve energy.",
            confidence="medium",
        )
        items = ledger.as_list()
        score, _, _, conflicts = patch_v2._score_candidate(
            "peaksaver", question, items)

        self.assertEqual(items[0]["answer_role"], "article_subject")
        self.assertLess(score, 0)
        self.assertTrue(any("article subject" in item for item in conflicts))

    def test_relation_aware_verifier_replaces_peaksaver_with_td(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = (
            "What is the name of the program advertised in the article about "
            "reducing air conditioning use and conserving energy?"
        )
        ledger.add(
            "peaksaver",
            "main_observation",
            "Ask about Ontario's peaksaver program. This voluntary program "
            "helps homeowners reduce electricity use and conserve energy.",
            confidence="medium",
        )
        ledger.add_text(
            "Brought to you by TD Insurance Meloche Monnex. As a Brock "
            "graduate, you enjoy a privileged status through an exclusive "
            "group insurance program through TD Insurance Meloche Monnex.",
            "main_observation",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(question, "peaksaver", ledger)

        self.assertEqual(
            result["selected_candidate"], "TD Insurance Meloche Monnex")
        self.assertTrue(result["verifier_changed_answer"])
        rejected = {item["candidate"]: item for item in result["rejected_candidates"]}
        self.assertIn("peaksaver", rejected)
        self.assertIn("article subject", rejected["peaksaver"]["rejection_reason"])

    def test_relation_aware_keeps_brock_news_from_final(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = "What is the name of the advertised program?"
        ledger.add(
            "Brock News",
            "main_observation",
            "Brock News is the university news source.",
            confidence="high",
        )
        ledger.add_text(
            "Brought to you by TD Insurance Meloche Monnex.",
            "main_observation",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(
            question, "TD Insurance Meloche Monnex", ledger)

        rejected = {item["candidate"]: item for item in result["rejected_candidates"]}
        self.assertEqual(result["selected_candidate"], "TD Insurance Meloche Monnex")
        self.assertEqual(rejected["Brock News"]["answer_role"], "source_or_publication")
        self.assertFalse(rejected["Brock News"]["allowed_for_final"])

    def test_verifier_prefers_advertised_program_over_article_subject(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = (
            "What is the name of the program advertised in the article about "
            "reducing air conditioning use? It was brought to you by a sponsor."
        )
        ledger.add(
            "peaksaver",
            "main_observation",
            "The article正文 mentions Ontario's peaksaver energy-saving program "
            "as an air conditioning conservation tip.",
            confidence="medium",
        )
        ledger.add(
            "TD Insurance Meloche Monnex",
            "main_observation",
            "The page says Brought to you by TD Insurance Meloche Monnex and "
            "describes an alumni insurance program.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(question, "peaksaver", ledger)

        self.assertEqual(
            result["selected_candidate"], "TD Insurance Meloche Monnex")
        self.assertTrue(result["verifier_changed_answer"])
        self.assertIn(
            "peaksaver",
            [item["candidate"] for item in result["rejected_candidates"]],
        )
        self.assertTrue(result["verifier_changed_answer"])

    def test_verifier_rejects_clue_object_for_book_title(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = (
            "Discovered before 1920, this artifact is inscribed in an unknown "
            "language. What is the title of the book published in 1985 that "
            "attempts to decipher it?"
        )
        ledger.add(
            "Phaistos Disc",
            "main_observation",
            "The Phaistos Disc is the artifact clue object, not the requested "
            "book title.",
            confidence="medium",
        )
        ledger.add(
            "The Genius of the Few",
            "main_observation",
            "The Genius of the Few is a 1985 book title that attempts to "
            "decipher the Phaistos Disc symbols.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(question, "Phaistos Disc", ledger)

        self.assertEqual(result["selected_candidate"], "The Genius of the Few")
        self.assertIn(
            "Phaistos Disc",
            [item["candidate"] for item in result["rejected_candidates"]],
        )

    def test_low_quality_subagent_report_does_not_override_main_evidence(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = "What advertised program is named by the article sponsor?"
        ledger.add(
            "wrong candidate",
            "subagent_report",
            "low_quality_report: true; candidate_answer: wrong candidate",
            confidence="medium",
            from_low_quality_report=True,
        )
        ledger.add(
            "correct program",
            "main_observation",
            "The page sponsor line advertises the correct program.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(question, "wrong candidate", ledger)

        self.assertEqual(result["selected_candidate"], "correct program")
        self.assertTrue(result["verifier_changed_answer"])

    def test_verifier_keeps_draft_without_strong_replacement(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = "Who won the match?"
        ledger.add(
            "weak related entity",
            "main_observation",
            "A related entity appeared in a broad search result.",
            confidence="low",
        )
        ledger.add(
            "another weak clue",
            "subagent_report",
            "A vague report mentioned another weak clue without direct support.",
            confidence="low",
        )
        result = patch_v2.verify_final_answer(question, "Correct Draft", ledger)

        self.assertEqual(result["selected_candidate"], "Correct Draft")
        self.assertFalse(result["verifier_changed_answer"])

    def test_verifier_rejects_based_on_explanation_candidate(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = "What is the advertised program?"
        explanation = (
            "Based on the evidence, the answer is TD Insurance Meloche Monnex."
        )
        ledger.add(
            explanation,
            "main_observation",
            explanation,
            confidence="high",
        )
        ledger.add(
            "TD Insurance Meloche Monnex",
            "main_observation",
            "Brought to you by TD Insurance Meloche Monnex.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(
            question, "TD Insurance Meloche Monnex", ledger)

        rejected = {item["candidate"]: item for item in result["rejected_candidates"]}
        cleaned_explanation = patch_v2._clean_candidate(explanation)
        self.assertEqual(result["selected_candidate"], "TD Insurance Meloche Monnex")
        self.assertFalse(result["verifier_changed_answer"])
        self.assertEqual(
            rejected[cleaned_explanation]["candidate_type"], "generic_phrase")
        self.assertFalse(patch_v2.is_candidate_allowed_for_final(
            rejected[cleaned_explanation], question))

    def test_verifier_keeps_supported_correct_draft_answer(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = "What city was the author born in?"
        ledger.add(
            "Boston",
            "main_observation",
            "The biography states the author was born in Boston.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(question, "Boston", ledger)

        self.assertEqual(result["selected_candidate"], "Boston")
        self.assertFalse(result["verifier_changed_answer"])

    def test_verifier_does_not_replace_supported_draft_with_generic_source(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = (
            "Based on fight statistics, provide the nickname and date of birth "
            "of the winner."
        )
        ledger.add(
            "The Cuban Missile Crisis, 1990 May 08",
            "main_observation",
            "Sherdog and fight statistics identify Julian Marquez as winner; "
            "his nickname is The Cuban Missile Crisis and birth date is May 8, "
            "1990.",
            confidence="high",
        )
        ledger.add(
            "ambiguous",
            "draft_answer",
            "ambiguous",
            confidence="medium",
        )
        ledger.add(
            "open to interpretation",
            "main_observation",
            "open to interpretation",
            confidence="medium",
        )
        ledger.add(
            "Significant Strikes Attempted",
            "main_observation",
            "Significant Strikes Attempted was a statistic label.",
            confidence="medium",
        )
        result = patch_v2.verify_final_answer(
            question, "The Cuban Missile Crisis, 1990 May 08", ledger)

        self.assertEqual(
            result["selected_candidate"],
            "The Cuban Missile Crisis, 1990 May 08",
        )
        self.assertFalse(result["verifier_changed_answer"])
        rejected = {item["candidate"]: item for item in result["rejected_candidates"]}
        self.assertIn("ambiguous", rejected)
        self.assertFalse(patch_v2.is_candidate_allowed_for_final(
            rejected["ambiguous"], question))
        self.assertEqual(rejected["ambiguous"]["candidate_type"], "generic_phrase")
        self.assertEqual(
            rejected["Significant Strikes Attempted"]["candidate_type"],
            "tool_artifact",
        )

    def test_source_name_cannot_replace_answer_when_question_asks_program(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = (
            "What is the name of the program advertised in the article about "
            "reducing air conditioning use?"
        )
        ledger.add(
            "TD Insurance Meloche Monnex",
            "main_observation",
            "The article says Brought to you by TD Insurance Meloche Monnex "
            "and describes an alumni insurance program.",
            confidence="high",
        )
        ledger.add(
            "Brock News",
            "main_observation",
            "Brock News is the university news source where the page appeared.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(
            question, "TD Insurance Meloche Monnex", ledger)

        rejected = {item["candidate"]: item for item in result["rejected_candidates"]}
        self.assertEqual(result["selected_candidate"], "TD Insurance Meloche Monnex")
        self.assertFalse(result["verifier_changed_answer"])
        self.assertEqual(rejected["Brock News"]["candidate_type"], "source_name")
        self.assertFalse(patch_v2.is_candidate_allowed_for_final(
            rejected["Brock News"], question))

    def test_source_name_rejected_when_source_is_only_contextual_clue(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = (
            "What is the name of the program advertised in the article about "
            "reducing air conditioning use? Readers can subscribe to the "
            "university's online news source where the article was published."
        )
        ledger.add(
            "TD Insurance Meloche Monnex",
            "main_observation",
            "The article says Brought to you by TD Insurance Meloche Monnex.",
            confidence="high",
        )
        ledger.add(
            "Brock News",
            "main_observation",
            "Brock News is the online news source where the article appeared.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(
            question, "TD Insurance Meloche Monnex", ledger)

        rejected = {item["candidate"]: item for item in result["rejected_candidates"]}
        self.assertEqual(result["selected_candidate"], "TD Insurance Meloche Monnex")
        self.assertEqual(rejected["Brock News"]["candidate_type"], "source_name")
        self.assertFalse(rejected["Brock News"]["allowed_for_final"])
        self.assertIn("candidate failed final-answer gate",
                      rejected["Brock News"]["rejection_reason"])

    def test_verifier_ignores_bootstrap_smoke_report_candidate(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        question = "What is the title of the 1985 book?"
        ledger.add(
            "bootstrap smoke sub-agent report [date_or_timeline_brief]",
            "subagent_report",
            "bootstrap smoke sub-agent report [date_or_timeline_brief]",
            confidence="medium",
        )
        ledger.add(
            "The Genius of the Few",
            "main_observation",
            "The Genius of the Few is the 1985 book title.",
            confidence="high",
        )
        result = patch_v2.verify_final_answer(
            question,
            "bootstrap smoke sub-agent report [date_or_timeline_brief]",
            ledger,
        )

        self.assertEqual(result["selected_candidate"], "The Genius of the Few")
        rejected = {
            item["candidate"]: item for item in result["rejected_candidates"]
        }
        self.assertIn(
            "bootstrap smoke sub-agent report [date_or_timeline_brief]",
            rejected,
        )
        self.assertEqual(
            rejected[
                "bootstrap smoke sub-agent report [date_or_timeline_brief]"
            ]["candidate_type"],
            "tool_artifact",
        )

    def test_adaptive_router_simple_vs_complex(self):
        simple = {"prompts": [
            {"prompt": "Verify the direct answer.", "goal": "verify"}
        ]}
        complex_params = {"prompts": [
            {"prompt": "Search the entity retirement clue.", "goal": "entity"},
            {"prompt": "Search the article source and date clue.", "goal": "source"},
            {"prompt": "Verify all constraints and counterevidence.", "goal": "check"},
        ]}
        with patch.object(patch_v2, "PATCH_ADAPTIVE_ROUTER", True):
            simple_decision = patch_v2.route_delegation(
                "Who wrote Hamlet?", simple, previous_delegations=1)
            complex_decision = patch_v2.route_delegation(
                "A long article clue asks which program was advertised, by the "
                "same author five months later, after a gold medal coach story "
                "between 2008 and 2011.",
                complex_params,
                previous_delegations=0,
            )

        self.assertFalse(simple_decision["allow"])
        self.assertTrue(complex_decision["allow"])
        self.assertLessEqual(len(complex_decision["params"]["prompts"]), 2)
        self.assertGreaterEqual(len(complex_decision["brief_types"]), 1)

    def test_main_agent_early_finalize_trigger(self):
        ledger = patch_v2.CandidateLedger(enabled=True)
        ledger.add(
            "TD Insurance Meloche Monnex",
            "main_observation",
            "Brought to you by TD Insurance Meloche Monnex; alumni insurance "
            "program.",
            confidence="high",
        )
        question = "What program was advertised and brought to you by the sponsor?"
        with patch.object(patch_v2, "PATCH_MAIN_EARLY_FINALIZE", True):
            result = patch_v2.should_main_early_finalize(
                question, ledger, round_num=12, max_calls=20)

        self.assertTrue(result["trigger"])
        self.assertEqual(
            result["verifier"]["selected_candidate"],
            "TD Insurance Meloche Monnex",
        )


if __name__ == "__main__":
    unittest.main()

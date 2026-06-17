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


if __name__ == "__main__":
    unittest.main()

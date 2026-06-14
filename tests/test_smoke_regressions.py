import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_smoke_run import validate_smoke_run
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


class SmokeValidationTests(unittest.TestCase):
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

        self.assertEqual(result["content"], "Recovered final report.")
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

        self.assertEqual(result["content"], "XML path recovered report.")
        self.assertEqual(result["status"], "max_calls")
        self.assertEqual(result["llm_calls"], 2)


if __name__ == "__main__":
    unittest.main()

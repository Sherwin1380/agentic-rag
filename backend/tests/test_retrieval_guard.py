"""Tests that one chat turn cannot repeatedly execute retrieval."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import agent


class _Span:
    def end(self, **kwargs):
        return None


class _Trace:
    id = None

    def __init__(self, **kwargs):
        pass

    def span(self, **kwargs):
        return _Span()

    def generation(self, **kwargs):
        return _Span()

    def update(self, **kwargs):
        pass

    def flush(self):
        pass


class RetrievalGuardTests(unittest.TestCase):
    def test_reuses_first_successful_retrieval(self):
        chunks = [
            {
                "id": "12CFR-229.10::c0",
                "text": "Electronic-payment availability rule.",
                "metadata": {
                    "title": "§ 229.10 Next-day availability",
                    "source": "12CFR-229.10",
                    "url": "https://www.ecfr.gov/current/title-12/section-229.10",
                },
                "dense_score": 0.91,
                "sparse_score": 4.2,
            }
        ]
        registry = {}

        with patch.object(agent.retriever, "hybrid_search", return_value=chunks) as search:
            first, first_summary = agent._execute_tool(
                "search_documentation",
                {"query": "electronic payments"},
                registry,
                _Trace(),
            )
            second, second_summary = agent._execute_tool(
                "search_documentation",
                {"query": "electronic payments Regulation CC"},
                registry,
                _Trace(),
            )

        self.assertEqual(len(first["chunks"]), 1)
        self.assertEqual(first_summary, "retrieved 1 chunks")
        self.assertEqual(second["chunks"], [])
        self.assertEqual(second_summary, "reused previous retrieval")
        self.assertIn("answer now", second["message"])
        search.assert_called_once_with("electronic payments")

    def test_agent_disables_tools_after_retrieval(self):
        tool_call = types.SimpleNamespace(
            id="call-1",
            function=types.SimpleNamespace(
                name="search_documentation",
                arguments='{"query": "electronic payments"}',
            ),
        )
        completions = [
            types.SimpleNamespace(
                usage=None,
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="",
                            tool_calls=[tool_call],
                        )
                    )
                ],
            ),
            types.SimpleNamespace(
                usage=None,
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="Electronic payments are available next day [1].",
                            tool_calls=None,
                        )
                    )
                ],
            ),
        ]
        chunks = [
            {
                "id": "12CFR-229.10::c0",
                "text": "Electronic-payment availability rule.",
                "metadata": {
                    "title": "§ 229.10 Next-day availability",
                    "source": "12CFR-229.10",
                },
                "dense_score": 0.91,
                "sparse_score": None,
            }
        ]
        settings = types.SimpleNamespace(
            enable_web_search=False,
            max_agent_steps=3,
            groq_model="test-primary",
            fallback_model="test-fallback",
        )
        offered_tools = []

        def fake_chat(messages, tools=None, **kwargs):
            offered_tools.append(
                [schema["function"]["name"] for schema in tools or []]
            )
            return completions.pop(0)

        with (
            patch.object(agent, "get_settings", return_value=settings),
            patch.object(agent, "Trace", _Trace),
            patch.object(agent.llm, "chat", side_effect=fake_chat),
            patch.object(agent.retriever, "hybrid_search", return_value=chunks) as search,
            patch.object(agent.llmops, "log_response"),
            patch.object(agent.governance, "audit_log"),
        ):
            result = agent.run_agent("Explain electronic payments", [])

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0].tool, "search_documentation")
        self.assertIn("search_documentation", offered_tools[0])
        self.assertEqual(offered_tools[1], [])
        search.assert_called_once_with("electronic payments")

    def test_agent_recovers_when_final_answer_is_only_private_markup(self):
        tool_call = types.SimpleNamespace(
            id="call-1",
            function=types.SimpleNamespace(
                name="search_documentation",
                arguments='{"query": "Regulation Z adverse action"}',
            ),
        )
        completions = [
            types.SimpleNamespace(
                usage=None,
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="", tool_calls=[tool_call])
                    )
                ],
            ),
            types.SimpleNamespace(
                usage=None,
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=(
                                "<think>I should search again.</think>"
                                "<tool_call>search_documentation</tool_call>"
                            ),
                            tool_calls=None,
                        )
                    )
                ],
            ),
            types.SimpleNamespace(
                usage=None,
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="The retrieved text does not answer that question [1].",
                            tool_calls=None,
                        )
                    )
                ],
            ),
        ]
        chunks = [
            {
                "id": "12CFR-1026::c0",
                "text": "A retrieved Regulation Z passage.",
                "metadata": {
                    "title": "Regulation Z",
                    "source": "12CFR-1026",
                },
                "dense_score": 0.8,
                "sparse_score": None,
            }
        ]
        settings = types.SimpleNamespace(
            enable_web_search=False,
            max_agent_steps=3,
            groq_model="test-primary",
            fallback_model="test-fallback",
        )
        called_models = []

        def fake_chat(messages, tools=None, model=None, **kwargs):
            called_models.append(model)
            return completions.pop(0)

        with (
            patch.object(agent, "get_settings", return_value=settings),
            patch.object(agent, "Trace", _Trace),
            patch.object(agent.llm, "chat", side_effect=fake_chat),
            patch.object(agent.retriever, "hybrid_search", return_value=chunks) as search,
            patch.object(agent.llmops, "log_response"),
            patch.object(agent.governance, "audit_log"),
        ):
            result = agent.run_agent("What is the Regulation Z rule?", [])

        self.assertEqual(
            result["answer"],
            "The retrieved text does not answer that question [1].",
        )
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["model_used"], "test-fallback")
        self.assertEqual(called_models, [None, None, "test-fallback"])
        self.assertEqual(len(result["steps"]), 1)
        search.assert_called_once_with("Regulation Z adverse action")


if __name__ == "__main__":
    unittest.main()

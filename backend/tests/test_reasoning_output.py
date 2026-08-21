"""Tests for model-specific reasoning controls and output sanitization."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import agent, llm


class _Completions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return "completion"


class _Client:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_Completions())


class ReasoningOutputTests(unittest.TestCase):
    def test_removes_private_reasoning_and_tool_blocks(self):
        raw = (
            "<think>private chain of thought</think>\n"
            "<tool_call><function=search_documentation>again</function></tool_call>\n"
            "The supported answer [1]."
        )

        self.assertEqual(agent._sanitize_answer(raw), "The supported answer [1].")

    def test_returns_empty_when_response_contains_only_control_markup(self):
        raw = "<think>search again</think><tool_call>search_documentation</tool_call>"

        self.assertEqual(agent._sanitize_answer(raw), "")

    def test_removes_unclosed_private_block_after_visible_answer(self):
        raw = "The supported answer [1].\n<think>I should search again"

        self.assertEqual(agent._sanitize_answer(raw), "The supported answer [1].")

    def test_removes_escaped_and_encoded_private_blocks(self):
        escaped = r"\<think>private\</think>Visible answer."
        encoded = "&lt;tool_call&gt;search again&lt;/tool_call&gt;Visible answer."

        self.assertEqual(agent._sanitize_answer(escaped), "Visible answer.")
        self.assertEqual(agent._sanitize_answer(encoded), "Visible answer.")

    def test_gpt_oss_excludes_reasoning(self):
        client = _Client()
        settings = types.SimpleNamespace(
            groq_model="openai/gpt-oss-20b",
            llm_temperature=0.1,
        )

        with (
            patch.object(llm, "_get_client", return_value=client),
            patch.object(llm, "get_settings", return_value=settings),
        ):
            result = llm.chat([{"role": "user", "content": "question"}])

        self.assertEqual(result, "completion")
        self.assertFalse(client.chat.completions.kwargs["include_reasoning"])
        self.assertEqual(client.chat.completions.kwargs["reasoning_effort"], "low")

    def test_qwen_hides_and_disables_reasoning(self):
        client = _Client()
        settings = types.SimpleNamespace(
            groq_model="openai/gpt-oss-20b",
            llm_temperature=0.1,
        )

        with (
            patch.object(llm, "_get_client", return_value=client),
            patch.object(llm, "get_settings", return_value=settings),
        ):
            llm.chat(
                [{"role": "user", "content": "question"}],
                model="qwen/qwen3.6-27b",
            )

        self.assertEqual(client.chat.completions.kwargs["reasoning_format"], "hidden")
        self.assertEqual(client.chat.completions.kwargs["reasoning_effort"], "none")


if __name__ == "__main__":
    unittest.main()

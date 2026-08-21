"""Tests for web-search provider handling and agent summaries."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import agent, tools


class _Settings:
    enable_web_search = True


class _FakeDDGS:
    results = []
    error: Exception | None = None
    init_kwargs = None
    text_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def text(self, query, **kwargs):
        type(self).text_kwargs = {"query": query, **kwargs}
        if type(self).error is not None:
            raise type(self).error
        return type(self).results


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        _FakeDDGS.results = []
        _FakeDDGS.error = None
        _FakeDDGS.init_kwargs = None
        _FakeDDGS.text_kwargs = None
        self.ddgs_module = types.SimpleNamespace(DDGS=_FakeDDGS)

    def _search(self):
        with (
            patch.object(tools, "get_settings", return_value=_Settings()),
            patch.dict(sys.modules, {"ddgs": self.ddgs_module}),
        ):
            return tools.web_search("Who is president of Poland?", max_results=4)

    def test_maps_provider_results(self):
        _FakeDDGS.results = [
            {
                "title": "President of Poland",
                "body": "Official biography",
                "href": "https://www.president.pl/en/president",
            }
        ]

        result = self._search()

        self.assertEqual(
            result["results"],
            [
                {
                    "title": "President of Poland",
                    "snippet": "Official biography",
                    "url": "https://www.president.pl/en/president",
                }
            ],
        )
        self.assertEqual(_FakeDDGS.init_kwargs, {"timeout": 10})
        self.assertEqual(_FakeDDGS.text_kwargs["backend"], "auto")

    def test_preserves_empty_results(self):
        result = self._search()

        self.assertEqual(result, {"query": "Who is president of Poland?", "results": []})

    def test_returns_provider_error(self):
        _FakeDDGS.error = RuntimeError("provider unavailable")

        result = self._search()

        self.assertEqual(
            result,
            {
                "query": "Who is president of Poland?",
                "error": "web search failed: provider unavailable",
            },
        )

    def test_agent_summary_distinguishes_empty_results_from_errors(self):
        with patch.object(
            agent.tools,
            "web_search",
            side_effect=[
                {"query": "q", "results": []},
                {"query": "q", "error": "web search failed: rate limited"},
                {"query": "q", "results": [{"title": "result"}]},
            ],
        ):
            _, empty_summary = agent._execute_tool("web_search", {"query": "q"}, {}, None)
            _, error_summary = agent._execute_tool("web_search", {"query": "q"}, {}, None)
            _, success_summary = agent._execute_tool("web_search", {"query": "q"}, {}, None)

        self.assertEqual(empty_summary, "web search returned no results")
        self.assertEqual(error_summary, "web search error: web search failed: rate limited")
        self.assertEqual(success_summary, "web: 1 results")


if __name__ == "__main__":
    unittest.main()

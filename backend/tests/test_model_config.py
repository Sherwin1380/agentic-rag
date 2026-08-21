"""Tests for supported default model routing and cost estimation."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.llmops import estimate_cost


class ModelConfigTests(unittest.TestCase):
    def test_defaults_use_current_primary_and_independent_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.groq_model, "openai/gpt-oss-20b")
        self.assertEqual(settings.fallback_model, "qwen/qwen3.6-27b")
        self.assertNotEqual(settings.groq_model, settings.fallback_model)

    def test_current_model_costs_are_tracked(self):
        primary_cost = estimate_cost("openai/gpt-oss-20b", 1_000_000, 1_000_000)
        fallback_cost = estimate_cost("qwen/qwen3.6-27b", 1_000_000, 1_000_000)

        self.assertAlmostEqual(primary_cost, 0.375)
        self.assertAlmostEqual(fallback_cost, 3.60)


if __name__ == "__main__":
    unittest.main()

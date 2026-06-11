"""Agent tools: a safe calculator and a free web search.

These are the non-retrieval tools the agent can choose to call. Retrieval
itself is exposed as a tool in agent.py (search_documentation).
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, List

from .config import get_settings

# ---------------------------------------------------------------------------
# Calculator — safe arithmetic via AST whitelisting (no eval of arbitrary code)
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs,
    "round": round, "floor": math.floor, "ceil": math.ceil, "min": min, "max": max,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"function not allowed: {node.func.id}")
        return fn(*[_eval_node(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression: str) -> Dict[str, Any]:
    """Evaluate an arithmetic expression safely."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        return {"expression": expression, "result": result}
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the LLM
        return {"expression": expression, "error": str(exc)}


# ---------------------------------------------------------------------------
# Web search — DuckDuckGo, no API key required
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 4) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.enable_web_search:
        return {"query": query, "error": "web search is disabled"}
    try:
        from duckduckgo_search import DDGS

        results: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                )
        return {"query": query, "results": results}
    except Exception as exc:  # noqa: BLE001
        return {"query": query, "error": f"web search failed: {exc}"}

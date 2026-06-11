"""The agent layer.

A tool-using loop over Groq's function-calling API. On each turn the model may:
  - call search_documentation  -> hybrid RAG over the Claude-docs corpus
  - call calculator            -> safe arithmetic
  - call web_search            -> DuckDuckGo, for things outside the corpus
  - or answer directly         -> for greetings / general knowledge

This "decide whether to retrieve" behaviour is the difference between an agent
and a naive embed-and-retrieve pipeline. Citations are tracked across every
retrieval call so the final answer can reference [1], [2], ... stably.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from . import llm, tools, retriever
from .config import get_settings
from .models import AgentStep, ChatMessage, Source
from .observability import Trace

SYSTEM_PROMPT = """You are a precise assistant for U.S. banking regulations \
(Title 12 of the Code of Federal Regulations — Banks and Banking), covering the \
OCC, Federal Reserve, FDIC, NCUA, and CFPB rules.

You have tools available. Decide for each question:
- For anything about banking regulations, requirements, thresholds, or definitions, \
CALL search_documentation first. Do not answer from memory; ground every claim in \
retrieved regulation text.
- For arithmetic (e.g. computing a reserve, fee, or threshold), CALL calculator. \
Never do mental math.
- For current events or topics clearly outside the regulations, CALL web_search.
- For greetings or trivial chit-chat, answer directly without tools.

Rules for the final answer:
- Cite the regulation chunks you used inline as [1], [2], etc., matching the \
numbers returned by search_documentation.
- Only state facts supported by retrieved chunks or tool results. If the \
regulations do not contain the answer, say so plainly instead of guessing. This is \
not legal advice.
- Be concise and reference the specific CFR section when possible.
"""


def _tool_schemas() -> List[Dict[str, Any]]:
    settings = get_settings()
    schemas: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search_documentation",
                "description": (
                    "Hybrid semantic + keyword search over U.S. banking regulations "
                    "(Title 12 CFR: OCC, Federal Reserve, FDIC, NCUA, CFPB). Use for "
                    "any question about banking rules, requirements, thresholds, or "
                    "defined terms."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A focused search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Evaluate an arithmetic expression. Supports + - * / ** %, "
                    "parentheses, and sqrt/log/exp/min/max/round. Use for any math, "
                    "e.g. estimating API cost = tokens / 1e6 * price."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "e.g. '1500000 / 1000000 * 0.59'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    ]
    if settings.enable_web_search:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the public web (DuckDuckGo). Use ONLY for topics "
                        "outside the Claude documentation, e.g. current events."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "search query"}
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    return schemas


def _do_search(
    query: str, registry: Dict[str, Source], trace: Trace
) -> Dict[str, Any]:
    """Run hybrid retrieval and register sources with stable citation numbers."""
    span = trace.span(name="retrieval", input={"query": query})
    chunks = retriever.hybrid_search(query)
    payload = []
    for ch in chunks:
        _id = ch["id"]
        if _id not in registry:
            n = len(registry) + 1
            meta = ch.get("metadata") or {}
            registry[_id] = Source(
                n=n,
                title=meta.get("title", meta.get("source", "doc")),
                source=meta.get("source", _id),
                url=meta.get("url"),
                snippet=ch["text"][:300].strip(),
                dense_score=ch.get("dense_score"),
                sparse_score=ch.get("sparse_score"),
            )
        src = registry[_id]
        payload.append(
            {"citation": f"[{src.n}]", "title": src.title, "text": ch["text"]}
        )
    span.end(output={"num_chunks": len(payload)})
    return {"query": query, "chunks": payload}


def _execute_tool(
    name: str, args: Dict[str, Any], registry: Dict[str, Source], trace: Trace
) -> Tuple[Dict[str, Any], str]:
    """Dispatch a tool call. Returns (result_dict, short_summary)."""
    if name == "search_documentation":
        result = _do_search(args.get("query", ""), registry, trace)
        return result, f"retrieved {len(result['chunks'])} chunks"
    if name == "calculator":
        result = tools.calculator(args.get("expression", ""))
        summary = (
            f"{result.get('expression')} = {result.get('result')}"
            if "result" in result
            else f"error: {result.get('error')}"
        )
        return result, summary
    if name == "web_search":
        result = tools.web_search(args.get("query", ""))
        n = len(result.get("results", []))
        return result, f"web: {n} results"
    return {"error": f"unknown tool {name}"}, "unknown tool"


def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """Convert a Groq assistant message (with tool_calls) into a plain dict."""
    d: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


def _recover_without_tools(
    messages: List[Dict[str, Any]],
    user_message: str,
    registry: Dict[str, Source],
    steps: List[AgentStep],
    trace: Trace,
) -> str:
    """Fallback when Groq fails to emit a valid tool call (400 tool_use_failed).

    Llama tool-calling on Groq occasionally returns a malformed function call.
    Rather than crash the request, we guarantee grounding by retrieving on the
    raw user message (if nothing was retrieved yet), inject the chunks, and ask
    for a final answer with tools disabled.
    """
    if not registry:
        payload = _do_search(user_message, registry, trace)
        steps.append(
            AgentStep(
                tool="search_documentation",
                arguments={"query": user_message},
                summary=f"retrieved {len(payload['chunks'])} chunks (recovery)",
            )
        )
        context = "\n\n".join(
            f"{c['citation']} {c['title']}\n{c['text']}" for c in payload["chunks"]
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Answer the question using ONLY the following documentation, "
                    "citing sources inline as [n]:\n\n" + context
                ),
            }
        )
    try:
        completion = llm.chat(messages, tools=None)
        return completion.choices[0].message.content or ""
    except Exception:
        return "Sorry — I hit an error generating a response. Please try again."


def run_agent(
    message: str, history: List[ChatMessage]
) -> Dict[str, Any]:
    """Run the agentic RAG loop and return answer + sources + steps + trace id."""
    settings = get_settings()
    trace = Trace(name="agentic_rag_chat", user_input=message)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:  # keep the last few turns for context
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": message})

    registry: Dict[str, Source] = {}
    steps: List[AgentStep] = []
    tool_schemas = _tool_schemas()

    answer = ""
    for _ in range(settings.max_agent_steps):
        gen = trace.generation(
            name="llm_decision", model=settings.groq_model, input=messages
        )
        try:
            completion = llm.chat(messages, tools=tool_schemas, tool_choice="auto")
        except llm.LLMNotConfigured:
            raise
        except Exception as exc:  # noqa: BLE001 - recover from tool_use_failed etc.
            gen.end(output=f"[recovered: {exc}]")
            answer = _recover_without_tools(messages, message, registry, steps, trace)
            break
        msg = completion.choices[0].message
        gen.end(output=msg.content or "[tool_calls]")

        if getattr(msg, "tool_calls", None):
            messages.append(_message_to_dict(msg))
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result, summary = _execute_tool(
                    tc.function.name, args, registry, trace
                )
                steps.append(
                    AgentStep(tool=tc.function.name, arguments=args, summary=summary)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(result)[:8000],
                    }
                )
            continue

        # No tool calls -> this is the final answer.
        answer = msg.content or ""
        break
    else:
        # Loop exhausted: force a final answer with no further tool use.
        gen = trace.generation(name="llm_final", model=settings.groq_model, input=messages)
        completion = llm.chat(messages, tools=None)
        answer = completion.choices[0].message.content or ""
        gen.end(output=answer)

    sources = sorted(registry.values(), key=lambda s: s.n)
    trace.update(output=answer)
    trace.flush()

    return {
        "answer": answer,
        "sources": sources,
        "steps": steps,
        "trace_id": trace.id,
    }

"""The agent layer.

A tool-using loop over Groq's function-calling API. On each turn the model may:
  - call search_documentation  -> hybrid RAG over the banking regulations corpus
  - call calculator            -> safe arithmetic
  - call web_search            -> DDGS metasearch, for things outside the corpus
  - or answer directly         -> for greetings / general knowledge

This "decide whether to retrieve" behaviour is the difference between an agent
and a naive embed-and-retrieve pipeline. Citations are tracked across every
retrieval call so the final answer can reference [1], [2], ... stably.

LLMOps:
  - Every system prompt is version-tagged (PROMPT_VERSION).
  - Token counts and cost estimates are logged for each request.
  - A fallback model is activated automatically on tool_use_failed errors.
  - Responses are appended to a structured JSONL log for rollback-aware ops.

Governance:
  - Grounding score: fraction of answer sentences that carry a citation.
  - Hallucination risk: classified low/medium/high from grounding + source count.
  - Citation validation: orphaned [n] references are flagged.
  - Retrieval traceability: full audit trail of queries, chunks, and scores.
  - An append-only compliance audit log records every request.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import governance, llm, llmops, retriever, tools
from .config import get_settings
from .models import AgentStep, ChatMessage, Source
from .observability import Trace

_log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise assistant for U.S. banking regulations \
(Title 12 of the Code of Federal Regulations — Banks and Banking), covering the \
OCC, Federal Reserve, FDIC, NCUA, and CFPB rules.

You have tools available. Decide for each question:
- For anything about banking regulations, requirements, thresholds, or definitions, \
CALL search_documentation first. Do not answer from memory; ground every claim in \
retrieved regulation text.
- Call search_documentation at most once per user question. After documentation is \
returned, use those results to answer instead of searching again.
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

# Register this prompt with the LLMOps version registry so every deployed
# instance can be traced back to the exact prompt text that produced it.
llmops.register_prompt(llmops.PROMPT_VERSION, SYSTEM_PROMPT)


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
                        "Search the public web (DDGS metasearch). Use ONLY for topics "
                        "outside the banking regulations corpus, e.g. current events."
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
        query = args.get("query", "")
        if registry:
            return {
                "query": query,
                "chunks": [],
                "message": (
                    "Documentation was already retrieved for this question. "
                    "Use the previous search results and answer now."
                ),
            }, "reused previous retrieval"
        result = _do_search(query, registry, trace)
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
        if result.get("error"):
            error = str(result["error"])
            return result, f"web search error: {error[:160]}"
        n = len(result.get("results", []))
        if n == 0:
            return result, "web search returned no results"
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


_XML_TOOL_RE = re.compile(
    r"<function=(?P<name>\w+)\s*(?P<args>\{.*?\})\s*</function>", re.DOTALL
)


def _parse_xml_tool_call(exc: Exception) -> Tuple[Optional[str], Optional[Dict]]:
    """Groq's llama-3.3-70b-versatile sometimes outputs tool calls in XML format
    (<function=NAME{...}</function>) instead of JSON, triggering a 400
    tool_use_failed.  The intent is correct — extract name + args so the caller
    can execute the tool without a second LLM round-trip.
    """
    try:
        # Prefer the structured body if the SDK exposes it
        body = getattr(exc, "body", None) or {}
        fg = (body.get("error") or {}).get("failed_generation", "") if isinstance(body, dict) else ""
        if not fg:
            fg = str(exc)
        m = _XML_TOOL_RE.search(fg)
        if m:
            return m.group("name"), json.loads(m.group("args"))
    except Exception:
        pass
    return None, None


def _recover_without_tools(
    messages: List[Dict[str, Any]],
    user_message: str,
    registry: Dict[str, Source],
    steps: List[AgentStep],
    trace: Trace,
    model: str | None = None,
) -> str:
    """Fallback when Groq fails to emit a valid tool call (400 tool_use_failed).

    Guarantees grounding by retrieving on the raw user message if nothing was
    retrieved yet, then asks for a final answer with tools disabled.  The
    `model` parameter allows routing to the fallback model (e.g. llama3-8b-8192)
    so a smaller, more reliable model handles recovery instead of retrying the
    same primary model that just failed.
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
        completion = llm.chat(messages, tools=None, model=model)
        return completion.choices[0].message.content or ""
    except Exception as exc:
        trace.span(name="recovery_error", input={"model": model, "error": str(exc)}).end(
            output=str(exc)
        )
        _log.error("Recovery LLM call failed: %s", exc)
        return "Sorry — I hit an error generating a response. Please try again."


def run_agent(
    message: str, history: List[ChatMessage]
) -> Dict[str, Any]:
    """Run the agentic RAG loop and return answer + sources + steps + metadata."""
    settings = get_settings()
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    trace = Trace(
        name="agentic_rag_chat",
        user_input=message,
        metadata={"prompt_version": llmops.PROMPT_VERSION, "request_id": request_id},
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": message})

    registry: Dict[str, Source] = {}
    steps: List[AgentStep] = []
    tool_schemas = _tool_schemas()

    answer = ""
    fallback_used = False
    model_used = settings.groq_model
    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(settings.max_agent_steps):
        # Once retrieval has returned sources, switch to a tools-disabled final
        # answer turn. Some reasoning models keep attempting tool calls after
        # they have the required context, which can trigger provider-side
        # tool_use_failed errors instead of producing the grounded answer.
        active_tool_schemas = [] if registry else tool_schemas
        gen = trace.generation(
            name="llm_decision", model=model_used, input=messages
        )
        try:
            completion = llm.chat(
                messages, tools=active_tool_schemas, tool_choice="auto"
            )
        except llm.LLMNotConfigured:
            raise
        except Exception as exc:  # noqa: BLE001 - recover from tool_use_failed etc.
            _log.warning("LLM decision call failed (%s): %s", model_used, exc)
            # llama-3.3-70b-versatile sometimes outputs <function=NAME{...}</function>
            # instead of JSON tool calls.  Parse and execute directly — no second
            # LLM call needed, just inject the tool result and continue the loop.
            xml_name, xml_args = _parse_xml_tool_call(exc)
            valid_tools = {s["function"]["name"] for s in active_tool_schemas}
            if xml_name and xml_name in valid_tools:
                gen.end(output=f"[xml-tool-recovered: {exc}]")
                already_retrieved = (
                    xml_name == "search_documentation" and bool(registry)
                )
                result, summary = _execute_tool(
                    xml_name, xml_args or {}, registry, trace
                )
                if not already_retrieved:
                    steps.append(
                        AgentStep(
                            tool=xml_name,
                            arguments=xml_args or {},
                            summary=summary,
                        )
                    )
                tc_id = f"call_{uuid.uuid4().hex[:8]}"
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": xml_name, "arguments": json.dumps(xml_args or {})},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": xml_name,
                    "content": json.dumps(result)[:8000],
                })
                continue  # next iteration: model sees tool result and writes the answer
            # Unparseable error — fall back to the smaller model without tools
            gen.end(output=f"[fallback: {exc}]")
            fallback_used = True
            model_used = settings.fallback_model
            answer = _recover_without_tools(
                messages, message, registry, steps, trace, model=settings.fallback_model
            )
            break

        if hasattr(completion, "usage") and completion.usage:
            total_input_tokens += getattr(completion.usage, "prompt_tokens", 0) or 0
            total_output_tokens += getattr(completion.usage, "completion_tokens", 0) or 0

        msg = completion.choices[0].message
        gen.end(output=msg.content or "[tool_calls]")

        if getattr(msg, "tool_calls", None):
            messages.append(_message_to_dict(msg))
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                already_retrieved = (
                    tc.function.name == "search_documentation" and bool(registry)
                )
                result, summary = _execute_tool(
                    tc.function.name, args, registry, trace
                )
                if not already_retrieved:
                    steps.append(
                        AgentStep(
                            tool=tc.function.name,
                            arguments=args,
                            summary=summary,
                        )
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

        answer = msg.content or ""
        break
    else:
        gen = trace.generation(name="llm_final", model=model_used, input=messages)
        completion = llm.chat(messages, tools=None)
        answer = completion.choices[0].message.content or ""
        gen.end(output=answer)
        if hasattr(completion, "usage") and completion.usage:
            total_input_tokens += getattr(completion.usage, "prompt_tokens", 0) or 0
            total_output_tokens += getattr(completion.usage, "completion_tokens", 0) or 0

    sources = sorted(registry.values(), key=lambda s: s.n)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # --- Governance checks ---
    grounding_score = governance.compute_grounding_score(answer, sources)
    hallucination_risk = governance.assess_hallucination_risk(
        answer, sources, grounding_score
    )
    citation_check = governance.validate_citations(answer, sources)
    retrieval_trace = governance.build_retrieval_trace(steps, sources)

    # --- Cost estimation ---
    cost_usd = llmops.estimate_cost(model_used, total_input_tokens, total_output_tokens)

    trace.update(output=answer)
    trace.flush()

    # --- LLMOps response log ---
    try:
        llmops.log_response(
            llmops.ResponseLogEntry(
                request_id=request_id,
                timestamp=time.time(),
                prompt_version=llmops.PROMPT_VERSION,
                model_used=model_used,
                fallback_used=fallback_used,
                user_message=message,
                answer_preview=answer[:300],
                sources_cited=len(sources),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                trace_id=trace.id,
                grounding_score=grounding_score,
                hallucination_risk=hallucination_risk,
            )
        )
    except Exception:  # noqa: BLE001 - never let logging break the response
        pass

    # --- Compliance audit log (append-only) ---
    try:
        governance.audit_log(
            request_id=request_id,
            user_message=message,
            answer=answer,
            grounding_score=grounding_score,
            hallucination_risk=hallucination_risk,
            citation_check=citation_check,
            retrieval_trace=retrieval_trace,
            model=model_used,
            prompt_version=llmops.PROMPT_VERSION,
            trace_id=trace.id,
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "answer": answer,
        "sources": sources,
        "steps": steps,
        "trace_id": trace.id,
        "model_used": model_used,
        "fallback_used": fallback_used,
        "latency_ms": latency_ms,
        "grounding_score": grounding_score,
        "hallucination_risk": hallucination_risk,
    }

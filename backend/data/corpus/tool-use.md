---
title: Tool use and agents
url: https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
---

# Tool use

Tool use lets Claude call functions you define. You describe each tool with a name,
a description, and a JSON Schema for its inputs; Claude decides when to call them and
with what arguments. Your code executes the tool and returns the result, and Claude
continues. This is the foundation for building agents.

## Defining a tool

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
    },
    "required": ["location"]
  }
}
```

Write **prescriptive descriptions** that say *when* to call a tool, not just what it
does (e.g. "Call this when the user asks about current prices or recent events").
Recent Opus models reach for tools more conservatively, so trigger conditions in the
description measurably improve the should-call rate.

## Tool choice

Control whether Claude uses tools with `tool_choice`:

| Value | Behavior |
| --- | --- |
| `{"type": "auto"}` | Claude decides (default) |
| `{"type": "any"}` | Claude must use at least one tool |
| `{"type": "tool", "name": "..."}` | Claude must use the named tool |
| `{"type": "none"}` | Claude cannot use tools |

Add `"disable_parallel_tool_use": true` to force at most one tool call per response.

## The agentic loop

When `stop_reason == "tool_use"`, the response contains one or more `tool_use`
blocks. Execute each tool, then send the results back as `tool_result` blocks in a
new `user` message. Loop until `stop_reason == "end_turn"`.

```python
while True:
    response = client.messages.create(
        model="claude-opus-4-8", max_tokens=16000, tools=tools, messages=messages,
    )
    if response.stop_reason == "end_turn":
        break
    messages.append({"role": "assistant", "content": response.content})
    results = []
    for block in response.content:
        if block.type == "tool_use":
            out = execute_tool(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,   # must match the tool_use block id
                "content": out,
            })
    messages.append({"role": "user", "content": results})
```

Each `tool_result` must carry the matching `tool_use_id`. To signal failure, set
`"is_error": true` on the result with an informative message; Claude will adapt.

The official SDKs also provide a **tool runner** (beta) that runs this loop for you
and generates schemas from typed function signatures.

## Should you build an agent?

Reach for an agent only when the task is genuinely multi-step and hard to specify up
front, the outcome justifies the extra cost and latency, and errors are recoverable.
For single-shot classification, extraction, or Q&A, a single Messages API call is
the right tier.

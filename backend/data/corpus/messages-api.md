---
title: Messages API basics
url: https://docs.claude.com/en/api/messages
---

# The Messages API

Everything in the Claude API goes through a single endpoint: `POST /v1/messages`.
Tools, structured outputs, vision, and caching are all features of this one
endpoint rather than separate APIs. The API is **stateless** — you send the full
conversation history on every request.

## A basic request

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
for block in response.content:
    if block.type == "text":
        print(block.text)
```

`response.content` is a list of content blocks (`TextBlock`, `ThinkingBlock`,
`ToolUseBlock`, ...). Always check `block.type` before reading `block.text`.

## System prompts

Pass a `system` string (or a list of text blocks) to set behavior:

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    system="You are a helpful coding assistant. Always provide examples in Python.",
    messages=[{"role": "user", "content": "How do I read a JSON file?"}],
)
```

## Multi-turn conversations

Because the API is stateless, send the entire history each turn. Rules:

- The first message must have role `user`.
- Roles normally alternate `user` / `assistant`; consecutive same-role messages
  are combined into one turn.
- A 400 `invalid_request_error` is returned if the first message is `assistant`
  or messages do not alternate correctly.

## max_tokens guidance

`max_tokens` is an enforced ceiling on output. Default to ~16000 for non-streaming
requests (keeps responses under SDK HTTP timeouts) and ~64000 for streaming. Hitting
the cap yields `stop_reason: "max_tokens"` and truncates output mid-thought.

## Stop reasons

The `stop_reason` field explains why generation stopped:

| Value | Meaning |
| --- | --- |
| `end_turn` | Claude finished naturally |
| `max_tokens` | Hit the `max_tokens` limit — increase it or stream |
| `stop_sequence` | Hit a custom stop sequence |
| `tool_use` | Claude wants to call a tool — execute it and continue |
| `pause_turn` | Model paused and can be resumed (agentic/server-tool flows) |
| `refusal` | Claude refused for safety reasons — inspect `stop_details` |

When `stop_reason == "refusal"`, `response.stop_details` carries a structured
`category` and `explanation`.

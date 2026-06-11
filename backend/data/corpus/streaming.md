---
title: Streaming responses
url: https://docs.claude.com/en/docs/build-with-claude/streaming
---

# Streaming

Streaming returns the response incrementally as Server-Sent Events (SSE), which lets
you display tokens as they are generated and avoids HTTP timeouts on long outputs.
Default to streaming for any request with large input, long output, or high
`max_tokens`.

## Quick start (Python)

```python
with client.messages.stream(
    model="claude-opus-4-8",
    max_tokens=64000,
    messages=[{"role": "user", "content": "Write a story"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    final_message = stream.get_final_message()
    print(final_message.usage.output_tokens)
```

The `messages.stream()` helper accumulates state for you and exposes `text_stream`
and `get_final_message()`. Use `get_final_message()` / `finalMessage()` to obtain the
complete response even while streaming — this gives timeout protection without
handling individual events.

## Stream event types

| Event | When it fires |
| --- | --- |
| `message_start` | Once at the beginning; carries message metadata |
| `content_block_start` | When a text or tool_use block starts |
| `content_block_delta` | For each incremental token/chunk |
| `content_block_stop` | When a block finishes |
| `message_delta` | Message-level updates including `stop_reason` and usage |
| `message_stop` | Once at the end |

A `content_block_delta` carries a `text_delta` (with `.text`) or a `thinking_delta`
(with `.thinking`). Branch on `event.delta.type` to handle each.

## Best practices

- Flush output immediately so tokens appear as they arrive.
- The `message_delta` event contains output token usage.
- Requests with `max_tokens` above ~16000 should stream — the SDK raises a
  `ValueError` for non-streaming requests it estimates will exceed ~10 minutes,
  because idle connections drop. Opus models support up to 128K output tokens, but
  only via streaming.

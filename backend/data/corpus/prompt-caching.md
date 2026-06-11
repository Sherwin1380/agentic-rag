---
title: Prompt caching
url: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
---

# Prompt caching

Prompt caching lets you reuse large, stable portions of a prompt across requests,
cutting cost by up to ~90% and reducing latency on the cached prefix.

## The one invariant

**Prompt caching is a prefix match. Any byte change anywhere in the prefix
invalidates everything after it.** The cache key is derived from the exact bytes of
the rendered prompt up to each `cache_control` breakpoint. The render order is
`tools` → `system` → `messages`, so a breakpoint on the last system block caches
tools and system together.

## Using cache_control

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    system=[{
        "type": "text",
        "text": "<large shared document or instructions>",
        "cache_control": {"type": "ephemeral"},   # default 5-minute TTL
    }],
    messages=[{"role": "user", "content": "Summarize the key points"}],
)
```

Top-level `cache_control={"type": "ephemeral"}` on the request auto-caches the last
cacheable block — the simplest option when you do not need fine-grained placement.
You may place up to **4** breakpoints per request.

## TTLs and economics

The default TTL is **5 minutes**; you can request a **1-hour** TTL with
`{"type": "ephemeral", "ttl": "1h"}`.

- **Cache reads** cost ~0.1× the base input price.
- **Cache writes** cost **1.25× for the 5-minute TTL** and **2× for the 1-hour TTL**.

With the 5-minute TTL, two requests break even (1.25× write + 0.1× read = 1.35×
versus 2× uncached). The 1-hour TTL needs at least three requests to pay off because
of the doubled write cost, but keeps entries alive across gaps in bursty traffic.

## Minimum cacheable prefix

A prefix shorter than the model minimum silently will not cache (no error, just
`cache_creation_input_tokens: 0`). Minimums: 4096 tokens for Opus 4.8 / 4.7 / 4.6 /
4.5 and Haiku 4.5; 2048 tokens for Fable 5, Sonnet 4.6, and Haiku 3.5.

## Verifying cache hits

The response `usage` object reports cache activity:

- `cache_creation_input_tokens` — tokens written to cache (you paid the write premium)
- `cache_read_input_tokens` — tokens served from cache (you paid ~0.1×)
- `input_tokens` — the uncached remainder, processed at full price

If `cache_read_input_tokens` is zero across repeated identical-prefix requests, a
silent invalidator is at work — a `datetime.now()` or UUID in the system prompt,
unsorted `json.dumps()`, or a varying tool set. Keep the system prompt frozen and put
volatile content after the last breakpoint.

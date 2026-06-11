---
title: Extended thinking and the effort parameter
url: https://docs.claude.com/en/docs/build-with-claude/extended-thinking
---

# Thinking and effort

Recent Claude models can reason internally before answering. On Fable 5, Opus 4.8,
Opus 4.7, Opus 4.6, and Sonnet 4.6 the recommended mode is **adaptive thinking**,
where Claude decides when and how much to think.

## Adaptive thinking

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    output_config={"effort": "high"},   # low | medium | high | xhigh | max
    messages=[{"role": "user", "content": "Solve this step by step..."}],
)

for block in response.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
    elif block.type == "text":
        print("Answer:", block.text)
```

`thinking={"type": "enabled", "budget_tokens": N}` (manual extended thinking with a
fixed token budget) is **removed** on Fable 5, Opus 4.8, and Opus 4.7 — sending it
returns a 400 — and deprecated on Opus 4.6 and Sonnet 4.6. The concept of a fixed
thinking budget is replaced by adaptive thinking plus the effort parameter.

Note that adaptive thinking is **off by default** on Opus 4.7 and 4.8: a request with
no `thinking` field runs without thinking. Set `thinking={"type": "adaptive"}`
explicitly to enable it.

## The effort parameter

Effort controls thinking depth and overall token spend. It lives **inside**
`output_config`, not at the top level:

```python
output_config={"effort": "high"}
```

Values are `low`, `medium`, `high`, `xhigh`, and `max`. The default is `high`. Lower
effort means fewer, more-consolidated tool calls, less preamble, and terser output;
`high` is often the sweet spot, and `xhigh` is recommended for coding and agentic use
cases. `max` is for cases where correctness matters more than cost.

## Sampling parameters removed

On Fable 5, Opus 4.8, and Opus 4.7 the `temperature`, `top_p`, and `top_k` sampling
parameters are removed and return a 400 if sent. Guide behavior through prompting and
the effort parameter instead.

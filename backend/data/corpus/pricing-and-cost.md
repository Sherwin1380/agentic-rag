---
title: Pricing and cost optimization
url: https://docs.claude.com/en/docs/about-claude/pricing
---

# Pricing and cost optimization

Claude is billed per token, separately for input and output, at per-million-token
rates that depend on the model.

## Standard per-million-token rates

| Model | Input $/1M | Output $/1M |
| --- | --- | --- |
| Claude Fable 5 | $10.00 | $50.00 |
| Claude Opus 4.8 | $5.00 | $25.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |

To estimate a request's input cost, multiply input tokens by the per-token rate. For
Opus 4.8 at $5.00 per 1M input tokens, 1.5M input tokens cost
`1500000 / 1000000 * 5.00 = $7.50`.

## Ways to cut cost

1. **Prompt caching.** Cache reads cost ~0.1× the base input price, so reusing a
   large stable prefix across requests can save up to ~90% on that portion.
2. **The Message Batches API.** Submitting requests as a batch
   (`POST /v1/messages/batches`) processes them asynchronously at **50% of standard
   prices**. A batch can hold up to 100,000 requests or 256 MB, and most complete
   within an hour (maximum 24 hours).
3. **Choose the right model.** Default to Opus 4.8 for hard tasks, use Sonnet 4.6 for
   high-volume production, and reserve Haiku 4.5 for simple, speed-critical work.
4. **Count tokens before sending.** `client.messages.count_tokens(...)` returns the
   exact input token count for a given model so you can estimate cost up front. Token
   counts are model-specific — do not use third-party tokenizers like `tiktoken`,
   which undercount Claude tokens.

## Token counting example

```python
count = client.messages.count_tokens(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": large_text}],
)
estimated_input_cost = count.input_tokens / 1_000_000 * 5.00
```

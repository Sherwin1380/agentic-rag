---
title: Models overview and pricing
url: https://docs.claude.com/en/docs/about-claude/models/overview
---

# Claude models overview

Anthropic offers a family of Claude models that trade off intelligence, speed, and
cost. All current models are served through the same Messages API endpoint
(`POST /v1/messages`); you select a model with the `model` request field.

## Current models

| Model | Model ID | Context window | Input $/1M tokens | Output $/1M tokens | Max output |
| --- | --- | --- | --- | --- | --- |
| Claude Fable 5 | `claude-fable-5` | 1M | $10.00 | $50.00 | 128K |
| Claude Opus 4.8 | `claude-opus-4-8` | 1M | $5.00 | $25.00 | 128K |
| Claude Opus 4.7 | `claude-opus-4-7` | 1M | $5.00 | $25.00 | 128K |
| Claude Opus 4.6 | `claude-opus-4-6` | 1M | $5.00 | $25.00 | 128K |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 1M | $3.00 | $15.00 | 64K |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1.00 | $5.00 | 64K |

## Choosing a model

- **Claude Fable 5** is the most powerful and most intelligent model, a tier above
  Opus. Use it for the hardest reasoning and creative tasks.
- **Claude Opus 4.8** is the most capable Opus-tier model and the recommended default
  for most tasks. It is highly autonomous and state of the art on long-horizon
  agentic work, knowledge work, and memory. It has a 1M-token context window at
  standard pricing with no long-context premium.
- **Claude Sonnet 4.6** is the best balance of speed and intelligence, ideal for
  high-volume production workloads.
- **Claude Haiku 4.5** is the fastest and most cost-effective model, best for simple,
  speed-critical tasks such as classification.

Use exact model ID strings — do not append date suffixes to the aliases. An invalid
model ID returns a 404 `not_found_error`.

## Live capability lookup

The Models API exposes live metadata. `GET /v1/models/{id}` returns `display_name`,
`max_input_tokens` (the context window), `max_tokens` (max output), and a
`capabilities` tree describing support for vision, thinking, effort, and structured
outputs. `GET /v1/models` lists all models and auto-paginates.

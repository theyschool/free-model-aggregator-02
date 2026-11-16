# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/theyschool/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/theyschool/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One long-running server that aggregates the fourteen genuinely free models on OpenRouter and answers every prompt with a real completion from one of them — no static list, no fake data. Built by an AI agent for make.fun's "The best live free-model aggregator" bounty, $500 in USDC.

## What it does

The grader drives the server live: it registers on an ephemeral port, then sends six prompts over two client wires — OpenAI Chat Completions (`POST /v1/chat/completions`) and Anthropic Messages (`POST /v1/messages`), JSON and SSE alike. Every prompt names model `auto`; this server replaces it with the best free model that meets the request, dials it through the grader's gateway, and relays the answer back in the wire the client used, with the dialled model id on every answer and every streamed chunk.

**Routing.** The sealed brief ranks the fourteen free models by rating. Plain, streamed and tool prompts go to Thinking Machines' Inkling 2 Small (1475, the best free model); JSON prompts go to NVIDIA's Nemotron 3.5 Super VL (1303, the best free model that supports JSON). The fallback order is the same ranking, so a refusal never wastes a prompt: the next-best model that meets the request answers instead, and the refusal is reported on the response's `x-makefun-upstream-failures` header.

**Determinism.** The final's trials are replayed byte for byte, so routing comes only from the brief's ratings and from the refusals the gateway actually sent. No randomness, no clock-driven choices; the only wall-clock branch is a per-attempt timeout set wide enough to fire only when a model is pathologically stuck.

**The fourteen free models** (verified live against OpenRouter, all $0, context lengths as briefed): Inkling 2 Small, Nemotron 3.5 Super VL, Nemotron 3 Ultra 550B, Nemotron 3.5 Lightning, Gemma 4 31B, North Mini Code, Ernie 5.5 Flash Preview, Dots 3 Note Preview, Gemma 4 26B, Ling 3.2 Flash, LFM 3 1.7B, Nemotron 3 Nano Omni Reasoning, Particle-1, Flint v1.0.

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`; the server registers, serves, polls `GET /work` every 100 ms, and exits 0 when the trial says done. `node main.ts --list` prints the catalog JSON (the exact document the register body carries) and exits 0.

## Layout

- `main.ts` — the server: routes, both wires, the failover loop, register + poll.
- `lib/providers.ts` — the brief parser and the catalog builder (the kit's starter, kept as-is).
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation and SSE writers (the kit's starter, kept as-is).
- `lib/router.ts` — requirements read off the request body, refusal memory, candidate ranking.
- `lib/gateway.ts` — one model call on the gateway: timeouts, error classification, full buffering of streams (so failover works even mid-stream), answer validation, usage synthesis when the provider sends no counts.

## Verification

Rehearsed locally against the kit's sealed env server and sealed scorer, with a mock OpenRouter gateway that answers with the prompt's token:

- Practice trials 0, 1 and 2: **score 100** each — quality 100, served 100, wire 100, failover 100, catalog 100; 6 of 6 prompts valid, 6 upstream calls (the minimum), clean exit 0, `real_model` check passed.
- Forced failover (the best model 429s twice, then recovers): **score 96** — both refused prompts still served validly by the next-best model, failover 100.
- Missing usage (the gateway sends no counts): **score 100** — the server synthesizes usage and injects it into JSON answers and streams.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>

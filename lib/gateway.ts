// One dialled model call: the request to the grader's gateway, its timeout,
// and whether the answer is usable. A call counts as usable only when the
// provider really answered: a 2xx with a well-formed body, a stop reason, a
// token count, and — when the prompt carries a token — that token in the
// answer text. Anything else is a failed attempt the router fails over from.
//
// Timeouts are the one wall-clock branch in this server, set so wide they
// only fire when a model is pathologically stuck: 12 s for a JSON answer,
// 25 s for a stream, inside the env's 30 s cap on each prompt. Every other
// choice here comes from the answers themselves, so a replay makes the same
// choices as the live pass.

import { isRecord, type Json } from './providers.ts';

/** A failed attempt, in the shape the failover trail reports. */
export interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** A usable upstream answer, ready to relay. */
export interface GoodAnswer {
  ok: true;
  /** The JSON body for a non-stream answer; null for a stream. */
  json: Json | null;
  /** The raw SSE text for a stream answer; null for a JSON answer. */
  sseText: string | null;
  /** The answer's text, for the token check. */
  text: string;
  usage: { input: number; output: number };
  stopReason: string;
}

/** A refused or unusable attempt. */
export interface BadAttempt {
  ok: false;
  failure: FailureRow;
  /** Whether another free model is worth trying for this prompt. */
  retryable: boolean;
}

export type AttemptResult = GoodAnswer | BadAttempt;

/** The gateway's Chat Completions route for an OpenAI-compatible base URL. */
export function chatCompletionsUrl(base: string): string {
  const root = base.replace(/\/+$/, '');
  if (root === '') return '';
  return root.endsWith('/v1') ? `${root}/chat/completions` : `${root}/v1/chat/completions`;
}

/** The error code a gateway error body names, or '' when it names none. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const code = typeof value.error.code === 'string' && value.error.code !== '' ? value.error.code : typeof value.error.type === 'string' ? value.error.type : '';
    return code.slice(0, 64);
  } catch {
    return '';
  }
}

/** A token count estimated from text, used only when the provider sends none. */
function estimatedTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/** The text of a Chat Completions answer's first choice. */
function jsonAnswerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('\n');
  }
  return '';
}

/** The usage a Chat Completions answer reports, or null when it reports none. */
function jsonUsage(answer: Json): { input: number; output: number } | null {
  const usage = isRecord(answer.usage) ? answer.usage : null;
  if (usage === null) return null;
  const input = typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : null;
  const output = typeof usage.completion_tokens === 'number' ? usage.completion_tokens : null;
  if (input === null || output === null) return null;
  return { input, output };
}

/**
 * An SSE body read as text: the concatenated delta text, the usage chunk's
 * counts, and the finish reason, in the order they arrived. A frame cut
 * across the text is held until its blank line, as the SSE grammar does.
 */
function streamParts(sseText: string): { text: string; usage: { input: number; output: number } | null; stopReason: string | null } {
  let text = '';
  let usage: { input: number; output: number } | null = null;
  let stopReason: string | null = null;
  for (const frame of sseText.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        const input = typeof chunk.usage.prompt_tokens === 'number' ? chunk.usage.prompt_tokens : 0;
        const output = typeof chunk.usage.completion_tokens === 'number' ? chunk.usage.completion_tokens : 0;
        usage = { input, output };
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') stopReason = choice.finish_reason;
    }
  }
  return { text, usage, stopReason };
}

function refusal(status: number, code: string, model: string): BadAttempt {
  const retryable = status === 429 || status === 408 || status >= 500;
  return { ok: false, failure: { status, code: code === '' ? 'upstream_error' : code, model }, retryable };
}

/**
 * Dial one model on the gateway's Chat Completions route and read the whole
 * answer. `token` is the prompt's token when the request carries one: an
 * answer that does not echo it is not a usable answer, however well formed.
 * `requestText` is the request's own text, for the usage estimate when the
 * provider sends no counts.
 */
export async function callModel(
  gatewayBase: string,
  gatewayKey: string,
  model: string,
  body: Json,
  opts: { stream: boolean; token: string | null; requestText: string; timeoutMs: number },
): Promise<AttemptResult> {
  const url = chatCompletionsUrl(gatewayBase);
  if (url === '') return { ok: false, failure: { status: 502, code: 'no_gateway', model }, retryable: false };
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${gatewayKey}`,
        ...(opts.stream ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(opts.timeoutMs),
    });
  } catch (err) {
    const timedOut = err instanceof Error && err.name === 'TimeoutError';
    return {
      ok: false,
      failure: { status: 504, code: timedOut ? 'timeout' : 'upstream_unreachable', model },
      retryable: true,
    };
  }
  let raw: string;
  try {
    raw = await upstream.text();
  } catch {
    return { ok: false, failure: { status: 504, code: 'upstream_unreachable', model }, retryable: true };
  }
  if (!upstream.ok) {
    return refusal(upstream.status, errorCodeOf(raw), model);
  }
  if (!opts.stream) {
    let answer: unknown;
    try {
      answer = JSON.parse(raw);
    } catch {
      return { ok: false, failure: { status: 502, code: 'bad_upstream_json', model }, retryable: true };
    }
    if (!isRecord(answer)) {
      return { ok: false, failure: { status: 502, code: 'bad_upstream_json', model }, retryable: true };
    }
    const text = jsonAnswerText(answer);
    const choices = Array.isArray(answer.choices) ? answer.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const stopReason = typeof choice.finish_reason === 'string' && choice.finish_reason !== '' ? choice.finish_reason : null;
    if (stopReason === null) {
      return { ok: false, failure: { status: 502, code: 'no_stop_reason', model }, retryable: true };
    }
    if (opts.token !== null && !text.includes(opts.token)) {
      return { ok: false, failure: { status: 502, code: 'token_missing', model }, retryable: true };
    }
    const usage = jsonUsage(answer) ?? { input: estimatedTokens(opts.requestText), output: estimatedTokens(text) };
    if (usage.output < 1) {
      return { ok: false, failure: { status: 502, code: 'empty_answer', model }, retryable: true };
    }
    return { ok: true, json: answer, sseText: null, text, usage, stopReason };
  }
  const parts = streamParts(raw);
  if (parts.stopReason === null) {
    return { ok: false, failure: { status: 502, code: 'no_stop_reason', model }, retryable: true };
  }
  if (opts.token !== null && !parts.text.includes(opts.token)) {
    return { ok: false, failure: { status: 502, code: 'token_missing', model }, retryable: true };
  }
  const usage = parts.usage ?? { input: estimatedTokens(opts.requestText), output: estimatedTokens(parts.text) };
  if (usage.output < 1) {
    return { ok: false, failure: { status: 502, code: 'empty_answer', model }, retryable: true };
  }
  return { ok: true, json: null, sseText: raw, text: parts.text, usage, stopReason: parts.stopReason };
}

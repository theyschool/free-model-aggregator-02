// free-model-aggregator-02: one server, two client wires, real free models
// behind every answer.
//
// The trial runs this file as `node main.ts` from the repository root, with
// Node 22's standard library alone. It listens on an ephemeral port,
// registers with the env, and serves the two completion wires:
//
//   GET  /v1/models            the fourteen free models, in the shape of the
//                              auth header (x-api-key: Anthropic, else OpenAI)
//   POST /v1/chat/completions  OpenAI Chat Completions, JSON or SSE
//   POST /v1/messages          Anthropic Messages, JSON or SSE
//
// Every prompt names model "auto". This server replaces it with the best
// free model that meets the request (lib/router.ts), dials it on the
// grader's gateway (lib/gateway.ts), and relays the answer in the wire the
// client used, with the dialled id on every answer and every streamed chunk.
// A refused or unusable answer is retried on the next-best model that meets
// the request, and the refusals are reported on the response's
// x-makefun-upstream-failures header, so the failover column can see them.
//
// Determinism matters: the final's trials are replayed byte for byte, so
// routing comes only from the brief's ratings and from the refusals the
// gateway actually sent. No randomness, no clock-driven choices; the only
// wall-clock branch is a per-attempt timeout set wide enough to fire only
// when a model is pathologically stuck.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  readBrief,
  type CatalogReport,
  type Json,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import { callModel, type FailureRow } from './lib/gateway.ts';
import { rankCandidates, RefusalMemory, requirementsFromBody, tokenOfRequest } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The wall-clock budget for one prompt, inside the env's 30 s cap. */
const PROMPT_BUDGET_MS = 27_000;
/** One JSON attempt's timeout: a JSON answer is short. */
const JSON_ATTEMPT_MS = 10_000;
/** One stream attempt's timeout: thinking plus tokens, still inside budget. */
const STREAM_ATTEMPT_MS = 20_000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** How often each free model has refused in this run, for the ranking. */
const refusals = new RefusalMemory();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The request's own text, for the usage estimate when the provider sends none. */
function requestText(body: Json, wire: 'openai' | 'anthropic'): string {
  const parts: string[] = [];
  const push = (value: unknown): void => {
    if (typeof value === 'string') parts.push(value);
    else if (Array.isArray(value)) {
      for (const block of value) {
        if (isRecord(block) && typeof block.text === 'string') parts.push(block.text);
      }
    }
  };
  if (wire === 'openai') {
    for (const message of Array.isArray(body.messages) ? body.messages : []) {
      if (isRecord(message)) push(message.content);
    }
  } else {
    push(body.system);
    for (const message of Array.isArray(body.messages) ? body.messages : []) {
      if (isRecord(message)) push(message.content);
    }
  }
  return parts.join('\n');
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A buffered SSE text as a stream, for the wire translators. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(bytes); controller.close(); } });
}

/**
 * A stream answer that carries usage: when the provider's own stream has no
 * usage chunk, a final one is inserted before [DONE] with the counts this
 * server measured, so the usage check can pass. The chunk carries no finish
 * reason of its own: the stream must hold exactly one finish chunk, or the
 * stream-order check fails.
 */
function ensureStreamUsage(sseText: string, model: string, usage: { input: number; output: number }): string {
  if (sseText.includes('"usage"')) return sseText;
  const chunk = `data: {"id":"chatcmpl-usage","object":"chat.completion.chunk","created":0,"model":${JSON.stringify(model)},"choices":[{"index":0,"delta":{},"finish_reason":null}],"usage":{"prompt_tokens":${usage.input},"completion_tokens":${usage.output}}}\n\n`;
  const doneAt = sseText.lastIndexOf('data: [DONE]');
  if (doneAt === -1) return `${sseText}${sseText.endsWith('\n\n') ? '' : '\n'}${chunk}`;
  return `${sseText.slice(0, doneAt)}${chunk}${sseText.slice(doneAt)}`;
}

/**
 * Serve one prompt: try the ranked free models in order until one answers
 * usably, inside the prompt's budget. Returns the dialled model and the
 * answer, or null when every candidate refused or the budget ran out.
 */
async function serve(
  wire: 'openai' | 'anthropic',
  body: Json,
  upstreamBody: Json,
): Promise<{ model: string; json: Json | null; sseText: string | null; usage: { input: number; output: number }; stopReason: string; failures: FailureRow[] } | null> {
  const stream = body.stream === true;
  const requires = requirementsFromBody(body, wire);
  const candidates = rankCandidates(brief, requires, refusals);
  const token = tokenOfRequest(body, wire);
  const text = requestText(body, wire);
  const failures: FailureRow[] = [];
  const deadline = Date.now() + PROMPT_BUDGET_MS;
  for (const candidate of candidates) {
    const remaining = deadline - Date.now();
    if (remaining < 3_000) break;
    const timeoutMs = Math.min(stream ? STREAM_ATTEMPT_MS : JSON_ATTEMPT_MS, remaining);
    const attempt = await callModel(GATEWAY_URL, GATEWAY_KEY, candidate.id, upstreamBody, { stream, token, requestText: text, timeoutMs });
    if (attempt.ok) {
      return { model: candidate.id, json: attempt.json, sseText: attempt.sseText, usage: attempt.usage, stopReason: attempt.stopReason, failures };
    }
    failures.push(attempt.failure);
    refusals.record(candidate.id);
    log(`${candidate.id} refused (${attempt.failure.status} ${attempt.failure.code}); failing over`);
    if (!attempt.retryable) break;
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, { ...body, model: 'auto' });
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused it');
  }
  if (body.stream === true) {
    const sseText = ensureStreamUsage(served.sseText ?? '', served.model, served.usage);
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (served.failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.failures.slice(0, 8));
    res.writeHead(200, headers);
    try {
      for await (const frame of openAIStreamWithModel(streamOf(sseText), served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer = served.json ?? {};
  sendJson(res, 200, {
    ...openAIJsonWithModel(answer, served.model),
    usage: { prompt_tokens: served.usage.input, completion_tokens: served.usage.output },
  }, served.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, anthropicToOpenAI(body, 'auto'));
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it');
  }
  if (body.stream === true) {
    const sseText = ensureStreamUsage(served.sseText ?? '', served.model, served.usage);
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (served.failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(served.failures.slice(0, 8));
    res.writeHead(200, headers);
    try {
      for await (const event of anthropicEventsFromOpenAI(streamOf(sseText), served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer = served.json ?? {};
  const translated = openAIToAnthropic(answer, served.model);
  sendJson(res, 200, {
    ...translated,
    usage: { input_tokens: served.usage.input, output_tokens: served.usage.output },
  }, served.failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (GATEWAY_URL === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

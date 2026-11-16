// The routing core: which free model serves a request, and in what order the
// rest are tried when it refuses. Everything here is deterministic — no clock
// and no randomness — so a trial's replay makes the same choices as its live
// pass: the order comes from the brief's ratings and from the refusals the
// gateway actually sent, never from wall-clock time.
//
// Quality is 60% of the score and is the rating of the model that really
// answered, over the best rating for a model meeting the prompt's
// requirements. So the ranking is the rating, highest first, among the models
// that meet the requirements. The only tie-breaker is how often a model has
// refused in this run (fewer first), then the brief's own order.

import {
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  type Brief,
  type Json,
  type ModelRow,
  type Requirement,
} from './providers.ts';

/** One refused gateway call, as the failover trail reports it. */
export interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The token shape every trial prompt carries, e.g. SPARROW-7435. */
const TOKEN_SHAPE = /(^|[^A-Z0-9])([A-Z]{3,12}-[0-9]{4})(?![0-9A-Z])/;

/**
 * The requirements a request body carries, read off the body itself rather
 * than the brief's arrival order: a tools array means tools, an image block
 * means vision, a JSON response_format means JSON mode. A prompt the brief
 * lists and a body the client sends always agree on these.
 */
export function requirementsFromBody(body: Json, wire: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  let tools = false;
  if (wire === 'openai') {
    tools = Array.isArray(body.tools) && body.tools.length > 0;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
      }
    }
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return { tools, vision, json: format === 'json_object' || format === 'json_schema' };
  }
  tools = Array.isArray(body.tools) && body.tools.length > 0;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && block.type === 'image') vision = true;
    }
  }
  return { tools, vision, json: false };
}

/** Every text the request carries, in order: system prompt, then each message. */
function requestTexts(body: Json, wire: 'openai' | 'anthropic'): string[] {
  const out: string[] = [];
  const push = (value: unknown): void => {
    if (typeof value === 'string') out.push(value);
    else if (Array.isArray(value)) {
      for (const block of value) {
        if (isRecord(block) && typeof block.text === 'string') out.push(block.text);
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
  return out;
}

/**
 * The prompt's unique token, when one is written in the request: the
 * instruction the answer must echo. Null when the request carries none, in
 * which case an answer is judged on shape alone.
 */
export function tokenOfRequest(body: Json, wire: 'openai' | 'anthropic'): string | null {
  for (const text of requestTexts(body, wire)) {
    const found = TOKEN_SHAPE.exec(text);
    if (found !== null) return found[2];
  }
  return null;
}

/**
 * How often each model has refused in this run. Counts come only from
 * gateway answers, so two runs that see the same answers make the same
 * choices; a model that refused twice still outranks a weaker model that
 * never did, because rating decides first.
 */
export class RefusalMemory {
  private counts = new Map<string, number>();

  record(model: string): void {
    this.counts.set(model, (this.counts.get(model) ?? 0) + 1);
  }

  count(model: string): number {
    return this.counts.get(model) ?? 0;
  }
}

/**
 * The free models that can serve these requirements, best first: the
 * brief's rating, highest first; among equals, the model that has refused
 * least in this run; then the brief's order. A model the brief does not
 * list as free never appears.
 */
export function rankCandidates(brief: Brief, requires: Requirement, memory: RefusalMemory): ModelRow[] {
  const indexed = freeModels(brief)
    .map((model, index) => ({ model, index }))
    .filter(({ model }) => meetsRequirements(requires, capabilitiesOf(model)));
  indexed.sort((a, b) => {
    if (b.model.rating_milli !== a.model.rating_milli) return b.model.rating_milli - a.model.rating_milli;
    const refusals = memory.count(a.model.id) - memory.count(b.model.id);
    if (refusals !== 0) return refusals;
    return a.index - b.index;
  });
  return indexed.map(({ model }) => model);
}

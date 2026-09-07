/**
 * Thin client for abliteration.ai's OpenAI-compatible chat endpoint.
 *
 * Used for the story pipeline only: turning a one-line premise into an arc, the arc
 * into scenes, and the scenes into H3-shaped video prompts. The model is uncensored,
 * which matters because a 10Eros-Max pipeline whose *planner* refuses is useless even
 * when the video model itself would have complied.
 */

const ENDPOINT = process.env.LLM_BASE_URL || 'https://api.abliteration.ai/v1';
const MODEL = process.env.LLM_MODEL || 'abliterated-model-large-v2';
const API_KEY = process.env.ABLITERATION_API_KEY || '';

export const llmConfigured = () => !!API_KEY;

/**
 * One non-streaming completion.
 *
 * We deliberately do not stream. Every call here produces a JSON document that is
 * useless until complete, and the progress the UI actually cares about is
 * "planning / rendering segment 3 of 6", which the pipeline reports itself.
 */
export async function chat(messages, {
  maxTokens = 8000,
  temperature = 1,
  reasoningEffort = 'high',
  timeoutMs = 300000,
  signal,
} = {}) {
  if (!API_KEY) throw new Error('ABLITERATION_API_KEY is not configured');

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const onAbort = () => ctrl.abort();
  if (signal) signal.addEventListener('abort', onAbort, { once: true });

  try {
    const r = await fetch(`${ENDPOINT}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: MODEL,
        messages,
        temperature,
        top_p: 1,
        max_tokens: maxTokens,
        reasoning_effort: reasoningEffort,
        stream: false,
      }),
      signal: ctrl.signal,
    });
    if (!r.ok) {
      const body = await r.text().catch(() => '');
      throw new Error(`LLM ${r.status}: ${body.slice(0, 300)}`);
    }
    const j = await r.json();
    const text = j?.choices?.[0]?.message?.content;
    if (typeof text !== 'string' || !text.trim()) {
      throw new Error('LLM returned an empty completion');
    }
    return text;
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

/**
 * Pull a JSON value out of a model response.
 *
 * Reasoning models wrap JSON in prose and fenced blocks even when told not to, so
 * parsing the raw string outright fails often enough to be worth handling. Strategy,
 * cheapest first: parse as-is, then the contents of a fenced block, then the widest
 * brace/bracket span in the text.
 */
export function extractJson(text) {
  const attempts = [];

  attempts.push(text.trim());

  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) attempts.push(fence[1].trim());

  for (const [open, close] of [['{', '}'], ['[', ']']]) {
    const first = text.indexOf(open);
    const last = text.lastIndexOf(close);
    if (first !== -1 && last > first) attempts.push(text.slice(first, last + 1));
  }

  for (const a of attempts) {
    try { return JSON.parse(a); } catch { /* try the next shape */ }
  }
  throw new Error(`could not parse JSON from model output: ${text.slice(0, 200)}…`);
}

/**
 * chat() + extractJson(), with one retry that shows the model its own bad output.
 * Cheaper and far more reliable than failing the whole story plan on a stray comma.
 */
export async function chatJson(messages, opts = {}) {
  let last;
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = await chat(attempt === 0 ? messages : [
      ...messages,
      { role: 'assistant', content: last.text.slice(0, 4000) },
      {
        role: 'user',
        content: `That did not parse as JSON (${last.err}). Reply with the JSON value ONLY — `
               + 'no prose, no explanation, no markdown fence.',
      },
    ], opts);
    try {
      return extractJson(text);
    } catch (e) {
      last = { text, err: e.message };
    }
  }
  throw new Error(last.err);
}

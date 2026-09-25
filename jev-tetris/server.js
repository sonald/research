import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { TypeSafeClient, choice } from '@typesafe-ai/sdk';
import { ALL_ACTIONS as ACTIONS } from './engine.js';
import { DEFAULT_CONFIG, buildDecision, validateGame } from './experiment.js';

export const DEFAULT_QUESTION = 'Which single action should the player take next to clear rows and survive? Only cleared rows earn points. Prefer fewer holes, a low stack and a flat surface. Each Choice key is the action to execute now. If provided, immediate_gain is its actual change; expected_gain is the total gain after the additional followup actions, selected as the best available preview. An empty followup means no additional action is needed. Forecasts are conditional, not guaranteed, and followup actions are NOT automatically executed; a forecast hard_drop remains hypothetical if excluded from actual choices. Avoid pausing or restarting an active game.';
const files = { '/': ['index.html', 'text/html'], '/index.html': ['index.html', 'text/html'], '/app.js': ['app.js', 'text/javascript'], '/style.css': ['style.css', 'text/css'], '/engine.js': ['engine.js', 'text/javascript'], '/experiment.js': ['experiment.js', 'text/javascript'], '/row-choices.js': ['row-choices.js','text/javascript'] };

export function createServer({ apiKey = process.env.TYPESAFE_API_KEY, model = process.env.TYPESAFE_DEFAULT_MODEL || 'jev-latest', client, maxRequests = 120 } = {}) {
  let inFlight = false;
  let windowStart = Date.now();
  let requests = 0;
  const configured = Boolean(apiKey);
  const sdk = client || (configured ? new TypeSafeClient({ apiKey, baseURL: 'https://api.typesafe.ai', timeout: 20_000, retry: { maxRetries: 0 }, logLevel: 'off' }) : null);
  return http.createServer({ requestTimeout: 10_000, headersTimeout: 10_000 }, async (req, res) => {
    const json = (status, body) => {
      res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
      res.end(JSON.stringify(body));
    };
    const fail = (status, error) => json(status, { error });
    const pathname = req.url.split('?')[0];
    // Reject DNS-rebinding hosts before reaching the credential-backed API.
    try {
      const hostname = new URL(`http://${req.headers.host}`).hostname;
      if (!['localhost', '127.0.0.1', '[::1]'].includes(hostname)) throw new Error();
    } catch { fail(403, 'Only localhost hosts are allowed.'); return; }
    if (req.method === 'GET' && pathname === '/api/config') {
      json(200, { configured, model, question: DEFAULT_QUESTION, actions: ACTIONS, experiment: DEFAULT_CONFIG });
      return;
    }
    if (req.method === 'GET' && Object.hasOwn(files, pathname)) {
      try {
        const [name, type] = files[pathname];
        const body = await readFile(new URL(name, import.meta.url));
        res.writeHead(200, { 'Content-Type': `${type}; charset=utf-8`, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
        res.end(body);
      } catch { fail(500, 'Could not load application file.'); }
      return;
    }
    if (pathname !== '/api/decide' || req.method !== 'POST') { fail(404, 'Not found.'); return; }
    // Browser requests must originate from this app. CLI requests have no Origin.
    if (req.headers.origin) {
      try {
        const origin = new URL(req.headers.origin);
        if (!['http:', 'https:'].includes(origin.protocol) || origin.host !== req.headers.host) throw new Error();
      } catch { fail(403, 'Cross-origin requests are not allowed.'); return; }
    }
    if (req.headers['sec-fetch-site'] === 'cross-site') { fail(403, 'Cross-site requests are not allowed.'); return; }
    if (req.headers['content-type']?.split(';')[0].trim() !== 'application/json') { fail(415, 'Content-Type must be application/json.'); return; }
    if (!sdk) { fail(503, 'Set TYPESAFE_API_KEY in the server .env file and restart the server to enable Jev.'); return; }
    if (inFlight) { fail(429, 'A Jev request is already running.'); return; }
    if (Date.now() - windowStart >= 60_000) { windowStart = Date.now(); requests = 0; }
    if (requests >= maxRequests) { fail(429, 'Jev request limit reached. Please wait one minute.'); return; }
    inFlight = true;
    try {
      const chunks = [];
      let size = 0;
      for await (const chunk of req) {
        size += chunk.length;
        if (size > 120_000) { fail(413, 'Request body is too large.'); return; }
        chunks.push(chunk);
      }
      let body;
      try { body = JSON.parse(Buffer.concat(chunks).toString('utf8')); }
      catch { fail(400, 'Invalid JSON body.'); return; }
      const { game, config, question, model: requestedModel } = body || {};
      if (typeof question !== 'string' || !question.trim() || question.length > 4_000 || typeof requestedModel !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,99}$/.test(requestedModel)) {
        fail(400, 'Provide question (1–4000 characters) and a valid model name (1–100 characters).'); return;
      }
      let decision;
      try { decision = buildDecision(validateGame(game), config); }
      catch { fail(400, 'Invalid game snapshot or experiment configuration.'); return; }
      const { state, actions } = decision;
      if (!Object.keys(actions).length) { fail(422, 'No candidate actions remain. Enable at least one action available in this state.'); return; }
      if ((typeof state === 'string' ? state : JSON.stringify(state)).length > 48_000) { fail(400, 'Encoded state is too large.'); return; }
      requests++;
      const started = performance.now();
      const response = await sdk.systemOne({ state, questions: { next_action: choice(question, actions) }, model: requestedModel });
      const answer = response.answers?.next_action;
      if (!answer || !Object.hasOwn(actions, answer.choice)) { fail(502, 'Jev returned an invalid action.'); return; }
      json(200, { answer, model: response.model, usage: response.usage, latencyMs: Math.round(performance.now() - started), question, state, actions, config: decision.config });
    } catch (error) {
      const status = error?.status ?? error?.statusCode;
      if (status === 401 || status === 403) fail(502, 'TypeSafe rejected the server API key. Check TYPESAFE_API_KEY.');
      else if (status === 429) fail(429, 'TypeSafe rate limit reached. Please wait before retrying.');
      else fail(502, 'Jev request failed or timed out. Check the server connection and TypeSafe account, then retry.');
    } finally { inFlight = false; }
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 4173);
  const host = process.env.HOST || '127.0.0.1';
  createServer().listen(port, host, () => console.log(`Jev Tetris: http://${host}:${port}`));
}

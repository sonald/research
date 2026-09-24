import test from 'node:test';
import assert from 'node:assert/strict';
import { TypeSafeClient } from '@typesafe-ai/sdk';
import { createServer } from './server.js';
import { ACTIONS } from './engine.js';

async function serve(t, options) {
  const server = createServer(options);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}
const body = { state: 'board:\n..........\n....TT....', question: 'What action next?', model: 'jev-latest' };
const post = (base, value = body, headers = {}) => fetch(`${base}/api/decide`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(value) });

test('server forwards exact text through the SDK, preserves answer, and rejects bad requests', async t => {
  let calls = 0;
  const answer = { type: 'choice', choice: Object.keys(ACTIONS)[0], probabilities: { [Object.keys(ACTIONS)[0]]: 1 }, confidence: 1 };
  const client = new TypeSafeClient({ apiKey: 'secret-for-test', baseURL: 'https://api.typesafe.ai', retry: { maxRetries: 0 }, logLevel: 'off', async fetch(url, options) {
    calls++;
    assert.equal(new URL(url).origin, 'https://api.typesafe.ai');
    const request = JSON.parse(options.body);
    assert.equal(request.state, body.state);
    assert.equal(request.model, body.model);
    assert.deepEqual(request.questions.next_action, { type: 'choice', instructions: body.question, criteria: ACTIONS });
    return Response.json({ answers: { next_action: answer }, model: 'jev-test', usage: { input_tokens: 10, output_tokens: 2 } });
  } });
  const base = await serve(t, { apiKey: 'secret-for-test', maxRequests: 1, client });
  for (const path of ['/.env', '/server.js', '/package.json', '/node_modules/@typesafe-ai/sdk/package.json']) {
    assert.equal((await fetch(base + path)).status, 404);
  }
  const config = await (await fetch(`${base}/api/config`)).json();
  assert.equal(config.configured, true);
  assert.equal(JSON.stringify(config).includes('secret-for-test'), false);
  assert.equal((await post(base, body, { Origin: 'https://attacker.invalid' })).status, 403);
  assert.equal((await post(base, body, { Host: 'attacker.invalid', Origin: 'http://attacker.invalid' })).status, 403);
  assert.equal((await post(base, body, { 'Content-Type': 'text/plain' })).status, 415);
  assert.equal((await post(base, { ...body, state: '' })).status, 400);
  assert.equal((await post(base, { ...body, model: 'https://attacker.invalid' })).status, 400);
  assert.equal((await post(base, { ...body, state: 'a'.repeat(24001) })).status, 400);
  const response = await post(base, body, { Origin: base });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.deepEqual(result.answer, answer);
  assert.equal(result.state, body.state);
  assert.equal(result.question, body.question);
  assert.equal(result.model, 'jev-test');
  assert.ok(result.latencyMs >= 0);
  assert.equal((await post(base)).status, 429);
  assert.equal(calls, 1);
});

test('missing keys and upstream failures never expose secrets or produce fallback moves', async t => {
  const empty = await serve(t, { apiKey: '' });
  assert.equal((await post(empty)).status, 503);
  const base = await serve(t, { apiKey: 'top-secret', client: { async systemOne() { throw new Error('top-secret upstream response'); } } });
  const response = await post(base);
  assert.equal(response.status, 502);
  assert.equal((await response.text()).includes('top-secret'), false);
});

test('only one in-flight request can spend API credits', async t => {
  let resolveRequest;
  let entered;
  const called = new Promise(resolve => { entered = resolve; });
  const base = await serve(t, { apiKey: 'test', client: { systemOne() {
    entered();
    return new Promise(resolve => { resolveRequest = resolve; });
  } } });
  const first = post(base);
  await called;
  assert.equal((await post(base)).status, 429);
  resolveRequest({ answers: { next_action: { choice: Object.keys(ACTIONS)[0] } } });
  assert.equal((await first).status, 200);
});

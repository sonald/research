import test from 'node:test';
import assert from 'node:assert/strict';
import { TypeSafeClient } from '@typesafe-ai/sdk';
import { createServer, DEFAULT_QUESTION, PLACEMENT_QUESTION } from './server.js';
import { createGame, step } from './engine.js';
import { buildDecision, normalizeConfig } from './experiment.js';
import { stateVersion, validatePlan, executePlanStep } from './placement-plan.js';

const config = { choiceMode: 'reachable_placements', format: 'json_object',
  search: { maxStates: 100, maxPathLength: 64, maxChoices: 20 }, sections: { actions: false } };
const requestFor = (game, overrides = {}) => ({ game, state_version: stateVersion(game), config,
  question: PLACEMENT_QUESTION, model: 'jev-test', ...overrides });
const post = (base, body) => fetch(`${base}/api/decide`, { method: 'POST',
  headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
async function serve(t, choose) {
  const wires = [];
  const client = new TypeSafeClient({ apiKey: 'integration-test', retry: { maxRetries: 0 }, logLevel: 'off',
    async fetch(url, options) {
      const wire = JSON.parse(options.body);
      wires.push(wire);
      const id = choose(wire.questions.next_action.criteria);
      return Response.json({ answers: { next_action: { type: 'choice', choice: id,
        probabilities: { [id]: 1 }, confidence: 1 } }, model: 'jev-test' });
    } });
  const server = createServer({ apiKey: 'integration-test', client });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return { base: `http://127.0.0.1:${server.address().port}`, wires };
}

test('reachable placements cross the actual SDK wire as compact choices and execute only one beat', async t => {
  const game = createGame('reachable-integration');
  const snapshot = structuredClone(game);
  const { base, wires } = await serve(t, choices => Object.keys(choices).find(id => choices[id].is_current_target)
    ?? Object.keys(choices).find(id => choices[id].steps_to_lock > 1));
  const response = await post(base, requestFor(game));
  assert.equal(response.status, 200);
  const result = await response.json();
  const wire = wires[0];
  assert.equal(wire.questions.next_action.instructions, PLACEMENT_QUESTION);
  assert.deepEqual(wire.state, result.state);
  assert.deepEqual(wire.questions.next_action.criteria, result.actions);
  assert.ok(Object.keys(result.actions).length > 1);
  assert.ok(Object.keys(result.actions).length <= 20);
  for (const choice of Object.values(result.actions)) {
    assert.deepEqual(Object.keys(choice).sort(), ['execute_now', 'is_current_target', 'steps_to_lock', 'summary']);
    assert.equal(typeof choice.summary, 'string');
    assert.ok(choice.summary.length > 0);
  }
  assert.equal(result.execute_now, result.plan.remainingPath[0]);
  assert.equal(result.choice_id, result.plan.targetId);
  assert.equal(result.state_version, stateVersion(game));
  assert.ok(validatePlan(game, result.plan, config.search).ok);
  assert.ok(result.placement_details.some(detail => detail.id === result.choice_id));
  // Observing a decision leaves the submitted snapshot untouched; execution is explicit.
  assert.deepEqual(game, snapshot);
  const oneBeat = structuredClone(game);
  step(oneBeat, result.execute_now);
  const executed = executePlanStep(game, result.plan, config.search);
  assert.deepEqual(game, oneBeat);
  assert.equal(game.tick, snapshot.tick + 1);
  assert.equal(game.pieces, snapshot.pieces);
  assert.equal(executed.plan.remainingPath.length, result.plan.remainingPath.length - 1);
  const tinyConfig = { ...config, search: { ...config.search, maxStates: 1 } };
  const continued = await post(base, requestFor(game, { config: tinyConfig, current_plan: executed.plan }));
  assert.equal(continued.status, 200);
  const next = await continued.json();
  assert.equal(next.choice_id, result.choice_id);
  assert.equal(next.actions[result.choice_id].is_current_target, true);
  assert.equal(next.search.current_target_valid, true);
  assert.equal(next.search.search_complete, false);
  assert.ok(next.search.truncated_by.includes('maxStates'));
  assert.ok(validatePlan(game, next.plan, tinyConfig.search).ok);
});

test('stale versions, invalid budgets and absent candidates stop before SDK; invalid placement IDs fail closed', async t => {
  const game = createGame('reachable-validation');
  const { base, wires } = await serve(t, () => 'placement_not_in_candidates');
  for (const state_version of [undefined, 'stale']) {
    assert.equal((await post(base, requestFor(game, { state_version }))).status, 409);
  }
  for (const search of [{ maxStates: 0 }, { maxStates: 10001 }, { maxPathLength: 0 },
    { maxPathLength: 129 }, { maxChoices: 0 }, { maxChoices: 21 }, { maxChoices: '20' }]) {
    assert.throws(() => normalizeConfig({ ...config, search }));
    assert.equal((await post(base, requestFor(game, { config: { ...config, search } }))).status, 400);
  }
  const paused = { ...game, paused: true };
  assert.equal((await post(base, requestFor(paused))).status, 422);
  assert.equal(wires.length, 0);
  assert.equal((await post(base, requestFor(game))).status, 502);
  assert.equal(wires.length, 1);
});

test('tampered or stale current plans are ignored and never become a trusted target', async t => {
  const game = createGame('reachable-invalid-plan');
  const decision = buildDecision(game, config);
  const plan = Object.values(decision.plans).find(value => value.remainingPath.length > 1);
  assert.ok(plan);
  const invalidPlans = [{ ...plan, stateVersion: 'stale' }, { ...plan, remainingPath: ['restart'] },
    { ...plan, terminalKey: 'forged' }, { ...plan, pieceId: 'different-piece' }];
  const { base, wires } = await serve(t, choices => Object.keys(choices)[0]);
  for (const current_plan of invalidPlans) {
    const response = await post(base, requestFor(game, { current_plan }));
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(result.search.current_target_valid, false);
    assert.ok(Object.values(result.actions).every(value => !value.is_current_target));
    assert.ok(validatePlan(game, result.plan, config.search).ok);
  }
  assert.equal(wires.length, invalidPlans.length);
});

test('legacy action mode keeps its question and immediate action contract', async t => {
  const game = createGame('reachable-legacy');
  const { base } = await serve(t, () => 'left');
  const response = await post(base, { game, config: { choiceMode: 'description' },
    question: DEFAULT_QUESTION, model: 'jev-test' });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.execute_now, 'left');
  assert.equal(result.choice_id, 'left');
  assert.equal(result.question, DEFAULT_QUESTION);
  assert.equal(result.plan, undefined);
  assert.equal(result.search, undefined);
});

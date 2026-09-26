import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, step } from './engine.js';
import { pieceId, terminalKey, stateVersion, targetId, allowedSearchActions, replayPlacement, validatePlan, executePlanStep } from './placement-plan.js';

function planFor(game, path) {
  const replay = replayPlacement(game, path);
  assert.equal(replay.ok, true, replay.error);
  return { pieceId: pieceId(game), targetId: targetId(game, replay.game), remainingPath: path,
    terminalKey: replay.terminalKey, stateVersion: stateVersion(game) };
}

test('placement plan executes a macro in one beat and keeps the same target until lock', () => {
  const game = createGame('plan');
  const original = structuredClone(game);
  const plan = planFor(game, ['right_2', 'rotate_cw', 'hard_drop']);
  assert.deepEqual(game, original);
  const first = executePlanStep(game, plan);
  assert.equal(first.action, 'right_2');
  assert.equal(game.tick, 1);
  assert.equal(game.active.x, original.active.x + 2);
  assert.equal(game.pieces, original.pieces);
  assert.equal(first.plan.targetId, plan.targetId);
  assert.equal(validatePlan(game, first.plan).ok, true);
  const second = executePlanStep(game, first.plan);
  const last = executePlanStep(game, second.plan);
  assert.equal(last.plan, null);
  assert.equal(terminalKey(game), plan.terminalKey);
});

test('stale, excluded, corrupted and after-lock plans cannot mutate the game', () => {
  const game = createGame('plan');
  const plan = planFor(game, ['right_2', 'hard_drop']);
  const original = structuredClone(game);
  for (const invalid of [
    { ...plan, stateVersion: 'stale' }, { ...plan, pieceId: 'other' },
    { ...plan, targetId: 'placement_wrong' }, { ...plan, remainingPath: ['hard_drop', 'left'] },
  ]) assert.throws(() => executePlanStep(game, invalid));
  assert.throws(() => executePlanStep(game, plan, { excludedActions: ['right'] }));
  assert.deepEqual(game, original);
  step(game, 'wait');
  assert.equal(validatePlan(game, plan).ok, false);
});

test('canonical identity ignores object order but versions detect time and history changes', () => {
  const game = createGame('canonical');
  const reordered = Object.fromEntries(Object.entries(game).reverse());
  reordered.active = Object.fromEntries(Object.entries(game.active).reverse());
  assert.equal(stateVersion(game), stateVersion(reordered));
  const later = structuredClone(game);
  later.tick++;
  later.history.push('wait');
  assert.equal(terminalKey(game), terminalKey(later));
  assert.equal(pieceId(game), pieceId(later));
  assert.notEqual(stateVersion(game), stateVersion(later));
});

test('search actions include blocked inputs and honor direction and exact exclusions', () => {
  const actions = allowedSearchActions(['left', 'rotate_cw', 'right_3']);
  assert.ok(!actions.some(action => action === 'left' || action.startsWith('left_')));
  assert.ok(!actions.includes('rotate_cw') && !actions.includes('right_3'));
  assert.ok(actions.includes('right_2') && actions.includes('wait'));
  assert.ok(!actions.includes('hold'));
  const game = createGame('blocked');
  game.active = { type: 'O', x: -1, y: 0, rotation: 0 };
  const result = replayPlacement(game, ['left', 'hard_drop']);
  assert.equal(result.ok, true);
  assert.equal(result.game.tick, game.tick + 2);
  assert.equal(replayPlacement(game, ['wait']).ok, false);
  assert.equal(replayPlacement(game, ['hard_drop'], {}, 'wrong').ok, false);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {createGame, step} from './engine.js';
import {buildRowChoices} from './row-choices.js';
import {searchPlacements, selectPlacements} from './placement-search.js';
import {replayPlacement, terminalKey, executePlanStep} from './placement-plan.js';

function tuck(type = 'O') {
  const game = createGame('tuck');
  game.active = {type, x: 7, y: 14, rotation: 0};
  game.board[16] = [...'JJJJJJJJ..'];
  game.board[19] = [...'JJJJJJJJ..'];
  return game;
}
const boardKey = board => JSON.stringify(board.map(row => typeof row === 'string' ? row : row.join('')));

test('BFS finds descending then sideways tuck that row previews cannot reach; every result replays', () => {
  const game = tuck(), original = structuredClone(game), result = searchPlacements(game);
  const previous = new Set(Object.values(buildRowChoices(game)).flatMap(choice => choice.variants.map(v => boardKey(v.board_after))));
  const tucked = result.candidates.find(c => c.path.slice(0,3).every(a => a === 'soft_drop') && c.path[3] === 'left_2');
  assert.ok(tucked);
  assert.ok(tucked.gain.holes_delta < 0);
  assert.ok(!previous.has(boardKey(tucked.terminal.board)));
  assert.equal(result.diagnostics.search_complete, true);
  assert.equal(result.diagnostics.unique_outcomes, result.diagnostics.verified_outcomes);
  for (const c of result.candidates) {
    assert.equal(replayPlacement(game, c.path, {}, c.terminalKey).ok, true);
    assert.equal(c.terminal.pieces, game.pieces + 1);
  }
  for (const [id, action] of Object.entries(result.actions)) {
    assert.deepEqual(Object.keys(action).sort(), ['execute_now','is_current_target','steps_to_lock','summary']);
    assert.equal(action.execute_now, result.plans[id].remainingPath[0]);
  }
  assert.deepEqual(game, original);
});

test('low rotation after descent is found and follows actual timing', () => {
  const game = tuck('T'), result = searchPlacements(game);
  const desired = ['rotate_cw','soft_drop','soft_drop','rotate_cw','hard_drop'];
  const replay = replayPlacement(game, desired);
  assert.equal(replay.ok, true);
  const candidate = result.candidates.find(c => c.terminalKey === replay.terminalKey);
  assert.ok(candidate);
  const previous = new Set(Object.values(buildRowChoices(game)).flatMap(choice => choice.variants.map(v => boardKey(v.board_after))));
  assert.ok(!previous.has(boardKey(candidate.terminal.board)));
  const pose = structuredClone(game);
  for (const action of candidate.path) {
    if (action.startsWith('rotate') && pose.active.y > game.active.y) break;
    step(pose, action);
  }
  assert.ok(pose.active.y > game.active.y);
});

test('closed cavity is absent from all outcomes', () => {
  const game = tuck();
  game.board[16] = [...'JJJJJJJJJ.']; // one-cell entrance cannot fit O
  const result = searchPlacements(game);
  for (const c of result.candidates) for (const y of [17,18]) {
    assert.ok(c.terminal.board[y].slice(0,8).every(cell => cell === '.'));
  }
});

test('budgets bound search independently of 20 choices; deterministic and honest truncation', () => {
  const game = createGame('budget');
  const options = {maxStates: 5, maxPathLength: 2, maxChoices: 3};
  const a = searchPlacements(game, options), b = searchPlacements(game, options);
  assert.deepEqual(a.actions, b.actions);
  assert.deepEqual(a.plans, b.plans);
  assert.ok(a.diagnostics.visited_states <= 5);
  assert.ok(a.diagnostics.shown_choices <= 3);
  assert.equal(a.diagnostics.search_complete, false);
  assert.ok(a.diagnostics.truncated_by.includes('maxStates'));
  assert.ok(a.diagnostics.truncated_by.includes('maxPathLength'));
  assert.ok(a.candidates.every(c => c.path.length <= 2));
  for (const key of ['maxStates','maxPathLength','maxChoices']) {
    assert.throws(() => searchPlacements(game, {[key]: 0}));
    assert.throws(() => searchPlacements(game, {[key]: 1.5}));
  }
});

test('valid current target survives tight subsequent budget and maintains stable ID after one step', () => {
  const game = tuck(), initial = searchPlacements(game);
  const c = initial.candidates.find(c => c.path[3] === 'left_2');
  const plan = initial.plans[c.id];
  assert.ok(plan);
  const next = executePlanStep(game, plan).plan;
  const result = searchPlacements(game, {maxStates: 1, maxChoices: 1}, next);
  assert.equal(result.diagnostics.current_target_valid, true);
  assert.deepEqual(Object.keys(result.actions), [c.id]);
  assert.equal(result.actions[c.id].is_current_target, true);
  assert.equal(replayPlacement(game, result.plans[c.id].remainingPath, {}, c.terminalKey).ok, true);
  const denied = searchPlacements(game, {maxStates: 1, excludedActions: ['left']}, next);
  assert.equal(denied.diagnostics.current_target_valid, false);
});

test('macro actions remain one beat and disabled action families never appear in paths', () => {
  const game = createGame('macro'); game.active = {type:'O',x:3,y:0,rotation:0}; game.gravityBeat = 3;
  const result = searchPlacements(game, {maxStates:200,maxPathLength:2});
  const c = result.candidates.find(c => c.path[0] === 'left_2');
  assert.ok(c);
  const macro = structuredClone(game); step(macro, 'left_2');
  const singles = structuredClone(game); step(singles,'left'); step(singles,'left');
  assert.equal(macro.tick, game.tick + 1);
  assert.notEqual(macro.active.y, singles.active.y);
  assert.equal(terminalKey(step(macro,'hard_drop')), c.terminalKey);
  const disabled = searchPlacements(game, {maxStates:50,excludedActions:['left','hard_drop']});
  assert.ok(disabled.candidates.every(c => c.path.every(a => !a.startsWith('left') && a !== 'hard_drop')));
});

test('selection reserves representatives and current target, excludes death only when survival exists', () => {
  const make = (id,rankScore,gain={},current=false) => ({id,rankScore,path:['hard_drop'],is_current_target:current,
    gain:{cleared_lines:0,holes_after:10,height_after:10,roughness_after:10,game_over:false,...gain}});
  const values = [make('ordinary',100),make('target',-100,{},true),make('lines',-10,{cleared_lines:4}),
    make('holes',-20,{holes_after:0}),make('height',-30,{height_after:0}),make('roughness',-40,{roughness_after:0}),
    make('death',1000,{game_over:true})];
  assert.deepEqual(new Set(selectPlacements(values,5).map(c=>c.id)),new Set(['target','lines','holes','height','roughness']));
  assert.equal(selectPlacements(values,1)[0].id,'target');
  assert.equal(selectPlacements([values.at(-1)],20)[0].id,'death');
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, applyPoseInput, cells, ghostCells, step } from './engine.js';
import { buildRowChoices } from './row-choices.js';

function fixture(type = 'L') {
  const game = createGame('row-preview');
  game.active = {type, x: 3, y: 5, rotation: 0};
  return game;
}

test('row choices enumerate all reachable L columns without mutation or hidden randomness', () => {
  const game = fixture(), before = structuredClone(game);
  const actions = buildRowChoices(game);
  assert.deepEqual(Object.keys(actions), ['left', 'left_2', 'left_3', 'right', 'right_2', 'right_3', 'right_4', 'rotate_cw', 'rotate_ccw', 'rotate_180', 'hard_drop']);
  for (const choice of Object.values(actions)) {
    assert.equal(choice.reference_y, 5);
    assert.equal(choice.execute_now in actions, true);
    assert.ok(choice.variants.length >= 1);
    for (const variant of choice.variants) {
      assert.equal(variant.pose_before_drop.y, 5);
      assert.equal(variant.board_after.length, 20);
      assert.ok(variant.board_after.every(row => row.length === 10));
    }
  }
  assert.deepEqual(game, before);
  assert.deepEqual(buildRowChoices(game), actions);
});

test('multi-column paths cannot jump across an obstacle at the source height', () => {
  const game = fixture();
  game.board[6][6] = 'J';
  const actions = buildRowChoices(game);
  assert.ok(actions.left_3);
  assert.ok(!Object.keys(actions).some(key => key.startsWith('right')));
});

test('O has one unique orientation and distant-column preview can clear two lines', () => {
  const game = fixture('O');
  game.board[18] = [...'..JJJJJJJJ']; game.board[19] = [...'..JJJJJJJJ'];
  const actions = buildRowChoices(game);
  assert.ok(!actions.rotate_cw);
  assert.equal(actions.left_4.variants.length, 1);
  assert.equal(actions.left_4.variants[0].newly_cleared_lines, 2);
  assert.equal(actions.left_4.variants[0].score_delta, 300);
  assert.equal(actions.hard_drop.variants[0].newly_cleared_lines, 0);
});

test('executing macro moves only its columns, while immediate snapshot includes real gravity', () => {
  const game = fixture(); game.gravityBeat = 4;
  const choice = buildRowChoices(game).right_2;
  const actual = step(structuredClone(game), choice.execute_now);
  assert.equal(actual.active.x, 5); assert.equal(actual.active.y, 6);
  assert.equal(actual.active.rotation, 0); assert.equal(actual.pieces, 0);
  assert.deepEqual(choice.immediate_after.active, actual.active);
  assert.deepEqual(choice.immediate_after.cells, cells(actual));
  assert.ok(choice.variants.every(value => value.pose_before_drop.y === 5));
});

test('preview and direct rotation candidates reuse engine SRS including wall kicks', () => {
  const game = fixture('T'); game.active = {type: 'T', x: -1, y: 5, rotation: 1};
  const choices = buildRowChoices(game);
  for (const action of ['rotate_cw', 'rotate_ccw', 'rotate_180']) {
    const pose = structuredClone(game);
    const legal = applyPoseInput(pose, action);
    assert.equal(Boolean(choices[action]), legal);
    if (legal) {
      assert.equal(choices[action].variants.length, 1);
      assert.deepEqual(choices[action].variants[0].pose_before_drop, pose.active);
      assert.deepEqual(choices[action].variants[0].landing_cells, ghostCells(pose));
      assert.notEqual(pose.active.x, game.active.x);
    }
  }
});

test('excluded direction removes macros, individual macros and rotations stay independently configurable', () => {
  const choices = buildRowChoices(fixture(), ['left', 'right_2', 'rotate_cw', 'hard_drop']);
  assert.ok(!Object.keys(choices).some(key => key.startsWith('left')));
  assert.ok(!choices.right_2); assert.ok(choices.right_3);
  assert.ok(!choices.rotate_cw); assert.ok(!choices.hard_drop);
  for (const choice of Object.values(choices)) {
    assert.equal(choice.hard_drop_excluded_from_choices, true);
    assert.ok(choice.variants.every(value => value.rotation_action !== 'rotate_cw'));
  }
});

test('actual forced lock is distinguished from conditional frozen-row forecasts', () => {
  const game = fixture(); game.active.y = 18; game.groundedBeats = 1; game.lockResets = 15;
  const choice = buildRowChoices(game).left;
  assert.equal(choice.immediate_after.pieces, 1);
  assert.equal(choice.actual_beat_ends_piece, true);
  assert.equal(choice.continuation_possible_after_actual_beat, false);
  assert.equal(choice.variants.length, 1);
  assert.equal(choice.variants[0].actual_first_beat, true);
  assert.deepEqual(choice.variants[0].board_after, step(structuredClone(game), 'left').board.map(row => row.join('')));
  for (const value of choice.variants) assert.ok(value.board_after.join('').replaceAll('.', '').length <= 4);
});

test('paused and ended games only have applicable controls with no placement forecast', () => {
  const game = fixture(); game.paused = true;
  const paused = buildRowChoices(game);
  assert.deepEqual(Object.keys(paused), ['resume', 'restart']);
  assert.equal(paused.resume.variants, null);
  game.over = true;
  assert.deepEqual(Object.keys(buildRowChoices(game)), ['restart']);
  assert.deepEqual(buildRowChoices(game, ['restart']), {});
});

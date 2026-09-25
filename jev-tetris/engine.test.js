import test from 'node:test';
import assert from 'node:assert/strict';
import { ACTIONS, ALL_ACTIONS, applyPoseInput, createGame, step, cells, ghostCells, encodeState, legalActions } from './engine.js';

test('multi-column input moves only horizontally and consumes one normal beat', () => {
  for (const [action, x] of [['left_2', 1], ['right_2', 5]]) {
    const game = createGame();
    game.active = {type:'L', x:3, y:4, rotation:0};
    step(game, action);
    assert.equal(game.active.x, x); assert.equal(game.active.y, 4);
    assert.equal(game.tick, 1); assert.equal(game.gravityBeat, 1); assert.equal(game.pieces, 0);
    game.gravityBeat = 4;
    step(game, action === 'left_2' ? 'right_2' : 'left_2');
    assert.equal(game.active.x, 3); assert.equal(game.active.y, 5);
    assert.equal(game.tick, 2); assert.equal(game.gravityBeat, 0);
  }
  assert.equal(Object.keys(ACTIONS).length, 12);
  assert.equal(Object.keys(ALL_ACTIONS).length, 28);
  assert.equal(legalActions(createGame()).left_2, undefined);
});

test('multi-column input checks intermediate cells and rolls back the whole shift', () => {
  const game = createGame();
  game.active = {type:'I', x:-2, y:4, rotation:1};
  game.board[5][1] = 'Z'; // Destination column 2 is clear, but the path crosses column 1.
  const before = structuredClone(game);
  assert.equal(applyPoseInput(game, 'right_2'), false);
  assert.deepEqual(game, before);
  step(game, 'right_2');
  assert.deepEqual(game.active, before.active); assert.equal(game.tick, 1);
  game.board[5][1] = '.';
  game.active.x = 6; // column 8; first shift fits, second would cross the wall.
  const pose = {...game.active};
  assert.equal(applyPoseInput(game, 'right_2'), false);
  assert.deepEqual(game.active, pose);
  assert.throws(() => step(game, 'right_10'), /Unknown action/);
});

test('pose previews reuse SRS while leaving timing and other game state untouched', () => {
  const game = createGame();
  game.active = {type:'I', x:-2, y:5, rotation:1};
  game.gravityBeat = 4; game.groundedBeats = 1; game.lockResets = 3;
  const before = structuredClone(game);
  assert.equal(applyPoseInput(game, 'rotate_ccw'), true);
  assert.deepEqual(game.active, {type:'I', x:0, y:5, rotation:0});
  assert.deepEqual({...game, active: before.active}, before);
  const after = structuredClone(game);
  assert.equal(applyPoseInput(game, 'hard_drop'), false);
  assert.deepEqual(game, after);
  for (const flag of ['paused', 'over']) {
    const stopped = {...structuredClone(game), [flag]: true};
    const snapshot = structuredClone(stopped);
    assert.equal(applyPoseInput(stopped, 'right_2'), false);
    assert.deepEqual(stopped, snapshot);
  }
});

test('same seed and actions reproduce board, bag, timing and score; bags contain all seven pieces', () => {
  const a = createGame('repeat'), b = createGame('repeat');
  for (const action of ['left','rotate_cw','hard_drop','hold','right','hard_drop','wait']) { step(a, action); step(b, action); }
  assert.deepEqual(a,b);
  const c = createGame('bag');
  const first = [c.active.type, ...c.next, c.bag.at(-1)];
  assert.equal(new Set(first).size, 7);
});

test('hard drop clears a row, counts points, spawns without gravity', () => {
  const g = createGame();
  g.board[19] = ['J','J','J','.','.','.','.','J','J','J'];
  g.active = { type:'I', x:3, y:0, rotation:0 };
  g.gravityBeat = 4;
  step(g, 'hard_drop');
  assert.equal(g.lines,1); assert.equal(g.pieces,1); assert.equal(g.score,100);
  assert.ok(g.board.flat().every(c => c === '.'));
  assert.equal(g.active.y,0); assert.equal(g.gravityBeat,0);
});

test('only clearing rows earns points, independent of drop distance and lock method', () => {
  for (const action of ['soft_drop', 'hard_drop', 'left', 'right', 'rotate_cw', 'rotate_ccw', 'rotate_180', 'hold', 'wait']) {
    const game = createGame();
    step(game, action);
    assert.equal(game.score, 0, action);
  }
  const falling = createGame();
  for (let i = 0; i < 110; i++) step(falling, 'wait');
  assert.ok(falling.pieces > 0);
  assert.equal(falling.score, 0);
  for (const count of [1, 2, 3, 4]) for (const y of [0, 15, 16]) for (const action of ['hard_drop', 'soft_drop', 'wait']) {
    const game = createGame();
    game.active = {type:'I', x:2, y, rotation:1};
    for (let row = 20 - count; row < 20; row++) game.board[row] = [...'JJJJ.JJJJJ'];
    for (let i = 0; i < 110 && game.pieces === 0; i++) step(game, action);
    assert.equal(game.pieces, 1);
    assert.equal(game.lines, count);
    assert.equal(game.score, [0, 100, 300, 500, 800][count], `${count} rows, y=${y}, ${action}`);
  }
});

test('hold is available once per piece and restored by lock; held piece resets rotation', () => {
  const g = createGame(), initial = g.active.type, next = g.next[0];
  step(g,'hold'); assert.equal(g.hold,initial); assert.equal(g.active.type,next); assert.equal(g.canHold,false);
  step(g,'hold'); assert.equal(g.hold,initial); assert.equal(g.active.type,next); assert.equal(g.tick,2);
  step(g,'hard_drop'); assert.equal(g.canHold,true);
  step(g,'hold'); assert.equal(g.active.type,initial); assert.equal(g.active.rotation,0); assert.equal(g.canHold,false);
});

test('SRS kicks I off left wall; moves cannot cross walls or locked cells', () => {
  const g = createGame(); g.active = {type:'I',x:-2,y:5,rotation:1};
  assert.ok(cells(g).every(c=>c.x===0));
  step(g,'left'); assert.equal(g.active.x,-2);
  step(g,'rotate_ccw'); assert.equal(g.active.rotation,0); assert.equal(g.active.x,0);
  assert.ok(cells(g).every(c=>c.x>=0 && c.x<10));
  g.board[6][4]='Z'; step(g,'right'); assert.equal(g.active.x,0);
});

test('gravity advances every five beats and two grounded beats lock', () => {
  const g=createGame();
  for(let i=0;i<4;i++) step(g,'wait'); assert.equal(g.active.y,0);
  step(g,'wait'); assert.equal(g.active.y,1);
  g.active = {type:'O',x:3,y:18,rotation:0}; g.gravityBeat=0; g.groundedBeats=0;
  step(g,'wait'); assert.equal(g.pieces,0); assert.equal(g.groundedBeats,1);
  step(g,'left'); assert.equal(g.pieces,0); assert.equal(g.lockResets,1); assert.equal(g.groundedBeats,1);
  step(g,'wait'); assert.equal(g.pieces,1);
});

test('grounded movement reset cap prevents indefinite floor stalling', () => {
  const g=createGame(); g.active={type:'O',x:3,y:18,rotation:0};
  step(g,'wait');
  for(let i=0;i<15;i++) step(g,i%2?'right':'left');
  assert.equal(g.pieces,0); assert.equal(g.lockResets,15);
  step(g,'right'); assert.equal(g.pieces,1);
});

test('blocked spawn ends game; pause freezes gameplay; restart reproduces initial state', () => {
  const g=createGame('end'), initial=structuredClone(g);
  step(g,'pause'); step(g,'hard_drop'); assert.equal(g.tick,0); assert.equal(g.pieces,0);
  step(g,'resume');
  g.board[0].fill('Z'); g.board[1].fill('Z'); step(g,'hold'); assert.equal(g.over,true);
  const tick=g.tick; step(g,'left'); assert.equal(g.tick,tick);
  step(g,'restart'); assert.deepEqual(g,initial);
  assert.throws(()=>step(g,'invalid'),/Unknown action/);
});

test('state text exposes active, locked, queue, legal action vocabulary and timing', () => {
  const g=createGame(); g.board[19][0]='J';
  assert.equal(ghostCells(g).length,4); assert.ok(ghostCells(g).every(c=>c.y<20));
  const text=encodeState(g);
  assert.match(text,/19  J/); assert.match(text,/active_cells=/); assert.match(text,/Next \(first plays next\)/);
  assert.match(text,/gravity_beat=0\/5/); assert.match(text,/rotate_180:/); assert.match(text,/can_hold=true/);
});

test('saved real Jev run replays gameplay facts despite state wording changes', async () => {
  const { readFile } = await import('node:fs/promises');
  const { records } = JSON.parse(await readFile(new URL('./live-run.json', import.meta.url), 'utf8'));
  const game = createGame('42');
  // Historical records used drop bonuses; preserve them and compare all other gameplay facts.
  const facts = text => ({
    board: text.match(/^\d{2}  [A-Za-z.]{10}$/gm),
    active: JSON.parse(text.match(/Active: (\{[^\n]+?\});/)[1]),
    activeCells: JSON.parse(text.match(/active_cells=(\[[^\n;]*\])/)[1]),
    landing: JSON.parse(text.match(/landing_cells=(\[[^\n;]*\])/)[1]),
    next: text.match(/^Next .*$/m)[0],
    counters: ['lines', 'locked_pieces', 'tick', 'paused', 'game_over', 'seed'].map(key =>
      text.match(new RegExp('(?:^|[ ;])' + (key === 'lines' ? '(?:cleared_)?lines' : key) + '=([^;\\n]+)', 'm'))[1]),
    timing: text.match(/^Timing:.*$/m)[0],
    history: text.match(/^Recent actions.*$/m)[0],
  });
  assert.equal(records.length, 9);
  let expectedScore = 0;
  for (const entry of records) {
    assert.deepEqual(facts(entry.state), facts(encodeState(game)));
    assert.equal(entry.action, entry.answer.choice);
    const previousLines = game.lines;
    step(game, entry.action);
    expectedScore += [0, 100, 300, 500, 800][game.lines - previousLines];
    assert.equal(game.score, expectedScore);
    assert.deepEqual(facts(entry.result), facts(encodeState(game)));
  }
  assert.equal(game.pieces, 9);
  assert.equal(game.lines, 0);
  assert.equal(game.score, 0);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame, step, cells, ghostCells, encodeState } from './engine.js';

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
  assert.equal(g.lines,1); assert.equal(g.pieces,1); assert.equal(g.score,136);
  assert.ok(g.board.flat().every(c => c === '.'));
  assert.equal(g.active.y,0); assert.equal(g.gravityBeat,0);
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

test('saved real Jev run replays exact request and result states', async () => {
  const { readFile } = await import('node:fs/promises');
  const { records } = JSON.parse(await readFile(new URL('./live-run.json', import.meta.url), 'utf8'));
  const game = createGame('42');
  assert.equal(records.length, 9);
  for (const entry of records) {
    assert.equal(entry.state, encodeState(game));
    assert.equal(entry.action, entry.answer.choice);
    step(game, entry.action);
    assert.equal(entry.result, encodeState(game));
  }
  assert.equal(game.pieces, 9);
  assert.equal(game.lines, 0);
});

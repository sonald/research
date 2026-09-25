export const ACTIONS = Object.freeze({
  left: 'Move one column left', right: 'Move one column right',
  soft_drop: 'Move down one row; earns no drop points',
  hard_drop: 'Drop to the landing position and lock immediately; earns points only if rows clear',
  rotate_cw: 'Rotate 90 degrees clockwise using SRS wall kicks',
  rotate_ccw: 'Rotate 90 degrees counterclockwise using SRS wall kicks',
  rotate_180: 'Attempt two consecutive clockwise SRS rotations in one beat',
  hold: 'Swap with hold, once per active piece; empty hold draws from queue',
  wait: 'Take no input for one beat', pause: 'Pause the game without advancing time',
  resume: 'Resume the game without advancing time', restart: 'Restart the same seed without advancing time',
});

export const ALL_ACTIONS = Object.freeze({
  ...ACTIONS,
  ...Object.fromEntries(['left', 'right'].flatMap(direction => Array.from({length: 8}, (_, i) =>
    [`${direction}_${i + 2}`, `Move ${i + 2} columns ${direction} at the current row, only if the entire path is clear; takes one beat`]))),
});

const SHAPES = {
  I: ['....', 'IIII', '....', '....'], O: ['.OO.', '.OO.', '....', '....'],
  T: ['.T.', 'TTT', '...'], J: ['J..', 'JJJ', '...'], L: ['..L', 'LLL', '...'],
  S: ['.SS', 'SS.', '...'], Z: ['ZZ.', '.ZZ', '...'],
};
// SRS offsets use upward-positive y; converted to board coordinates on use.
const KICKS = {
  '0>1': [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],
  '1>0': [[0,0],[1,0],[1,-1],[0,2],[1,2]],
  '1>2': [[0,0],[1,0],[1,-1],[0,2],[1,2]],
  '2>1': [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],
  '2>3': [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
  '3>2': [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
  '3>0': [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
  '0>3': [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
};
const I_KICKS = {
  '0>1': [[0,0],[-2,0],[1,0],[-2,-1],[1,2]],
  '1>0': [[0,0],[2,0],[-1,0],[2,1],[-1,-2]],
  '1>2': [[0,0],[-1,0],[2,0],[-1,2],[2,-1]],
  '2>1': [[0,0],[1,0],[-2,0],[1,-2],[-2,1]],
  '2>3': [[0,0],[2,0],[-1,0],[2,1],[-1,-2]],
  '3>2': [[0,0],[-2,0],[1,0],[-2,-1],[1,2]],
  '3>0': [[0,0],[1,0],[-2,0],[1,-2],[-2,1]],
  '0>3': [[0,0],[-1,0],[2,0],[-1,2],[2,-1]],
};

function pieceCells(piece) {
  let matrix = SHAPES[piece.type].map(row => [...row]);
  if (piece.type !== 'O') for (let i = 0; i < piece.rotation; i++) {
    matrix = matrix[0].map((_, x) => matrix.map(row => row[x]).reverse());
  }
  return matrix.flatMap((row, y) => row.flatMap((type, x) => type === '.' ? [] : [{ x: x + piece.x, y: y + piece.y, type }]));
}
export function cells(game) { return game.active ? pieceCells(game.active) : []; }
function fits(game, piece) {
  return pieceCells(piece).every(({x,y}) => x >= 0 && x < 10 && y >= -4 && y < 20 && (y < 0 || game.board[y][x] === '.'));
}
function random(game) {
  let t = game.rng = (game.rng + 0x6D2B79F5) >>> 0;
  t = Math.imul(t ^ t >>> 15, t | 1);
  t ^= t + Math.imul(t ^ t >>> 7, t | 61);
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function draw(game) {
  if (!game.bag.length) {
    game.bag = Object.keys(SHAPES);
    for (let i = 6; i > 0; i--) {
      const j = Math.floor(random(game) * (i + 1));
      [game.bag[i], game.bag[j]] = [game.bag[j], game.bag[i]];
    }
  }
  return game.bag.pop();
}
function spawn(game, type) {
  if (!type) { type = game.next.shift(); game.next.push(draw(game)); }
  game.active = { type, x: 3, y: 0, rotation: 0 };
  game.gravityBeat = 0; game.groundedBeats = 0; game.lockResets = 0;
  if (!fits(game, game.active)) game.over = true;
}
export function createGame(seed = 'jev') {
  seed = String(seed);
  let rng = 2166136261;
  for (const char of seed) rng = Math.imul(rng ^ char.charCodeAt(0), 16777619) >>> 0;
  const game = { seed, rng, bag: [], board: Array.from({length: 20}, () => Array(10).fill('.')),
    active: null, next: [], hold: null, canHold: true, score: 0, lines: 0, pieces: 0,
    tick: 0, paused: false, over: false, gravityBeat: 0, groundedBeats: 0, lockResets: 0, history: [] };
  game.next = Array.from({length: 5}, () => draw(game));
  spawn(game);
  return game;
}
function move(game, dx, dy) {
  const candidate = {...game.active, x: game.active.x + dx, y: game.active.y + dy};
  if (!fits(game, candidate)) return false;
  game.active = candidate; return true;
}
function rotate(game, delta) {
  if (game.active.type === 'O') return false;
  const from = game.active.rotation, to = (from + delta + 4) % 4;
  for (const [dx, dy] of (game.active.type === 'I' ? I_KICKS : KICKS)[`${from}>${to}`]) {
    const candidate = {...game.active, rotation: to, x: game.active.x + dx, y: game.active.y - dy};
    if (fits(game, candidate)) { game.active = candidate; return true; }
  }
  return false;
}
// Pose-only input for same-row placement previews. No gravity, timers, or locking.
export function applyPoseInput(game, action) {
  if (game.paused || game.over || !game.active) return false;
  const shift = /^(left|right)(?:_([2-9]))?$/.exec(action);
  if (shift) {
    const original = game.active;
    for (let i = 0; i < Number(shift[2] ?? 1); i++) {
      if (!move(game, shift[1] === 'left' ? -1 : 1, 0)) { game.active = original; return false; }
    }
    return true;
  }
  if (action === 'rotate_cw' || action === 'rotate_ccw') return rotate(game, action === 'rotate_cw' ? 1 : -1);
  if (action === 'rotate_180') {
    const first = rotate(game, 1);
    return rotate(game, 1) || first;
  }
  return false;
}
function lock(game) {
  if (cells(game).some(({y}) => y < 0)) { game.over = true; return; }
  for (const {x,y,type} of cells(game)) game.board[y][x] = type;
  const remaining = game.board.filter(row => row.some(cell => cell === '.'));
  const cleared = 20 - remaining.length;
  game.board = [...Array.from({length: cleared}, () => Array(10).fill('.')), ...remaining];
  game.lines += cleared; game.score += [0,100,300,500,800][cleared];
  game.pieces++; game.canHold = true; spawn(game);
}
export function ghostCells(game) {
  if (!game.active || game.over) return [];
  let ghost = {...game.active};
  while (fits(game, {...ghost, y: ghost.y + 1})) ghost.y++;
  return pieceCells(ghost);
}
export function boardMetrics(board) {
  const columnHeights = Array.from({length: 10}, (_, x) => {
    const top = board.findIndex(row => row[x] !== '.');
    return top < 0 ? 0 : 20 - top;
  });
  return {
    definitions: 'Locked blocks only. Column height = 20 minus top occupied y (empty column = 0). Holes = empty cells below an occupied cell in the same column. Bumpiness = sum of absolute adjacent column-height differences.',
    columnHeights,
    maxHeight: Math.max(...columnHeights),
    holes: columnHeights.reduce((sum, height, x) => sum + board.slice(20 - height).filter(row => row[x] === '.').length, 0),
    bumpiness: columnHeights.slice(1).reduce((sum, height, i) => sum + Math.abs(height - columnHeights[i]), 0),
  };
}
export function step(game, action) {
  if (!Object.hasOwn(ALL_ACTIONS, action)) throw new Error(`Unknown action: ${action}`);
  if (action === 'restart') { Object.assign(game, createGame(game.seed)); return game; }
  if (action === 'pause') { game.paused = true; return game; }
  if (action === 'resume') { game.paused = false; return game; }
  if (game.paused || game.over) return game;
  game.tick++;
  game.history.push(action); game.history = game.history.slice(-8);
  const wasGrounded = !fits(game, {...game.active, y: game.active.y + 1});
  const reset = applyPoseInput(game, action);
  if (action === 'soft_drop') move(game, 0, 1);
  if (action === 'hard_drop') {
    while (move(game, 0, 1)) {}
    lock(game); return game;
  }
  if (action === 'hold' && game.canHold) {
    const old = game.hold; game.hold = game.active.type;
    spawn(game, old); game.canHold = false; return game;
  }
  if (wasGrounded && reset && game.lockResets < 15) {
    game.groundedBeats = 0; game.lockResets++;
  }
  game.gravityBeat++;
  if (game.gravityBeat >= 5) { move(game, 0, 1); game.gravityBeat = 0; }
  if (!fits(game, {...game.active, y: game.active.y + 1})) {
    game.groundedBeats++;
    if (game.groundedBeats >= 2) lock(game);
  } else game.groundedBeats = 0;
  return game;
}
export function stateSections(game) {
  const active = cells(game), ghost = ghostCells(game);
  const visible = game.board.map(row => [...row]);
  for (const {x,y,type} of active) if (y >= 0 && y < 20 && x >= 0 && x < 10) visible[y][x] = type.toLowerCase();
  return {
    goal: 'TETRIS STATE. Goal: survive as long as possible and maximize cleared lines and score.',
    board: ['Board: width=10 height=20. Coordinates x=0..9 left to right, y=0..19 top to bottom. Negative y is above the board.',
    'Legend: .=empty, uppercase I/O/T/J/L/S/Z=locked block, lowercase=active block. Ghost is NOT a locked obstacle.',
    '    0123456789', ...visible.map((row,y) => `${String(y).padStart(2,'0')}  ${row.join('')}`)].join('\n'),
    active: `Active: ${JSON.stringify(game.active)}; active_cells=${JSON.stringify(active)}`,
    landing: `landing_cells=${JSON.stringify(ghost)}`,
    next: `Next (first plays next): ${game.next.join(',')}; hold=${game.hold ?? 'empty'}; can_hold=${game.canHold}`,
    stats: `Score: score=${game.score}; cleared_lines=${game.lines}; locked_pieces=${game.pieces}`,
    status: `tick=${game.tick}; paused=${game.paused}; game_over=${game.over}; seed=${JSON.stringify(game.seed)}`,
    timing: `Timing: gravity_beat=${game.gravityBeat}/5; grounded_beats=${game.groundedBeats}/2; lock_resets=${game.lockResets}/15.`,
    rules: ['Rules: Each gameplay action takes one beat, including blocked moves and unavailable hold. After input gravity moves down one row on every fifth beat. Grounded means down is blocked; lock occurs after two grounded beats. Successful grounded lateral moves or rotations reset the lock timer up to 15 times; the current beat then counts as grounded beat 1 if still grounded. Airborne resets grounded count.',
    'Multi-column shifts (left_2..left_9/right_2..right_9) check every intermediate cell at the current row; a blocked path cancels the entire shift. A shift consumes one beat regardless of distance.',
    'Hard drop locks immediately; successful hold spawns immediately; both reset piece timers and do not apply gravity to the new piece. Pause/resume/restart use no beat. While paused or over only these control actions work. Restart resets the same seed and loses the current run.',
    '7-bag pieces; five previews. Spawn origin=(3,0), rotation=0. SRS 90-degree rotations; 180 is two independent clockwise attempts (a partial 90-degree rotation can result). O rotation is a no-op. Collision above the board is allowed to y=-4; locking any cell above the board or blocked spawn ends the game.',
    'Full rows clear simultaneously. Only cleared rows earn points: 1=100, 2=300, 3=500, 4=800. Soft drop and hard drop earn no movement points. Fixed gravity; no combo or T-spin bonuses.'].join('\n'),
    shapes: 'Spawn matrices (rows separated by /; rotate clockwise around matrix center): ' + Object.entries(SHAPES).map(([type, rows]) => `${type}=${rows.join('/')}`).join(' '),
    history: `Recent actions (oldest first): ${game.history.join(',') || 'none'}`,
    actions: 'Available actions: ' + Object.entries(ACTIONS).map(([id, description]) => `${id}: ${description}`).join('; '),
  };
}

export function encodeState(game) {
  return Object.values(stateSections(game)).join('\n');
}

// Test input feasibility before gravity: blocked inputs can still advance time in step().
export function legalActions(game) {
  if (game.over) return { restart: ACTIONS.restart };
  if (game.paused) return { resume: ACTIONS.resume, restart: ACTIONS.restart };
  return Object.fromEntries(Object.entries(ACTIONS).filter(([action]) => {
    const probe = { ...game, active: { ...game.active } };
    if (action === 'left' || action === 'right' || action.startsWith('rotate_')) return applyPoseInput(probe, action);
    if (action === 'soft_drop') return move(probe, 0, 1);
    if (action === 'hold') return game.canHold;
    return action !== 'resume';
  }));
}

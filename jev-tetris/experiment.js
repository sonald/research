import { ACTIONS, cells, stateSections, legalActions, step } from './engine.js';

export const STATE_SECTION_LABELS = Object.freeze({
  goal: '目标 Goal', active: '当前方块 Active', landing: '落点 Landing', next: '预告与暂存 Next / Hold',
  stats: '得分与消行 Score', status: '游戏状态 Status', timing: '计时 Timing', rules: '规则 Rules',
  shapes: '方块形状 Spawn matrices', history: '近期动作 Recent actions', actions: '动作说明 Available actions',
});

export const DEFAULT_CONFIG = Object.freeze({format: 'original', choiceMode: 'description', legalOnly: true, includeMetrics: false, excludedActions: Object.freeze([]), sections: Object.freeze(Object.fromEntries(Object.keys(STATE_SECTION_LABELS).map(key => [key, true])))});
const FORMATS = ['original', 'grid', 'coordinates', 'json', 'json_object'];
const TYPES = 'IOTJLSZ';
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const pieceType = value => typeof value === 'string' && value.length === 1 && TYPES.includes(value);
const integer = (value, min = 0, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= min && value <= max;

export function normalizeConfig(config = {}) {
  if (!record(config) || Object.keys(config).some(key => !Object.hasOwn(DEFAULT_CONFIG, key))) throw new Error('Invalid experiment config');
  const result = {...DEFAULT_CONFIG, ...config};
  if (!FORMATS.includes(result.format) || !['description', 'outcome'].includes(result.choiceMode) || typeof result.legalOnly !== 'boolean' || typeof result.includeMetrics !== 'boolean') throw new Error('Invalid experiment config');
  if (!Array.isArray(result.excludedActions) || result.excludedActions.length > Object.keys(ACTIONS).length || result.excludedActions.some(action => typeof action !== 'string' || !Object.hasOwn(ACTIONS, action))) throw new Error('Invalid excluded actions');
  result.excludedActions = Object.keys(ACTIONS).filter(action => result.excludedActions.includes(action));
  if (!record(result.sections) || Object.entries(result.sections).some(([key, value]) => !Object.hasOwn(STATE_SECTION_LABELS, key) || typeof value !== 'boolean')) throw new Error('Invalid state sections');
  result.sections = {...DEFAULT_CONFIG.sections, ...result.sections};
  return result;
}

export function validateGame(input) {
  const fail = () => { throw new Error('Invalid game snapshot'); };
  if (!record(input)) fail();
  if (typeof input.seed !== 'string' || input.seed.length > 80) fail();
  if (!Array.isArray(input.board) || input.board.length !== 20 || input.board.some(row => !Array.isArray(row) || row.length !== 10 || row.some(c => c !== '.' && !pieceType(c)))) fail();
  if (!record(input.active) || !pieceType(input.active.type) || !integer(input.active.rotation, 0, 3) || !integer(input.active.x, -4, 9) || !integer(input.active.y, -4, 19)) fail();
  if (!Array.isArray(input.next) || input.next.length !== 5 || !input.next.every(pieceType)) fail();
  if (input.hold !== null && !pieceType(input.hold)) fail();
  if (['canHold', 'paused', 'over'].some(key => typeof input[key] !== 'boolean')) fail();
  if (['score', 'lines', 'pieces', 'tick'].some(key => !integer(input[key]))) fail();
  if (!integer(input.rng, 0, 0xffffffff) || !integer(input.gravityBeat, 0, 4) || !integer(input.groundedBeats, 0, 2) || !integer(input.lockResets, 0, 15)) fail();
  if (!Array.isArray(input.bag) || input.bag.length > 7 || !input.bag.every(pieceType) || new Set(input.bag).size !== input.bag.length) fail();
  if (!Array.isArray(input.history) || input.history.length > 8 || input.history.some(action => typeof action !== 'string' || !Object.hasOwn(ACTIONS, action))) fail();
  const game = Object.fromEntries(['seed','rng','hold','canHold','score','lines','pieces','tick','paused','over','gravityBeat','groundedBeats','lockResets'].map(key => [key,input[key]]));
  Object.assign(game, {board: input.board.map(row => [...row]), active: {type: input.active.type, x: input.active.x, y: input.active.y, rotation: input.active.rotation}, next: [...input.next], bag: [...input.bag], history: [...input.history]});
  if (cells(game).some(({x,y}) => x < 0 || x >= 10 || y < -4 || y >= 20 || (!game.over && y >= 0 && game.board[y][x] !== '.'))) fail();
  return game;
}

function metrics(board) {
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

export function buildDecision(game, config) {
  config = normalizeConfig(config);
  const actions = Object.fromEntries(Object.entries(config.legalOnly ? legalActions(game) : ACTIONS)
    .filter(([action]) => !config.excludedActions.includes(action))
    .map(([action, operation]) => {
      if (config.choiceMode === 'description') return [action, operation];
      const after = step(structuredClone(game), action);
      return [action, {
        operation,
        horizon: 'One action including this beat’s gravity and locking; no further actions.',
        score_after: after.score, score_delta: after.score - game.score,
        cleared_lines_after: after.lines, newly_cleared_lines: Math.max(0, after.lines - game.lines),
        locked_pieces_delta: after.pieces - game.pieces,
        active_after: {...after.active, cells: cells(after)},
        spawned_new_piece: after.pieces > game.pieces || (action === 'hold' && !game.paused && !game.over && game.canHold) || action === 'restart',
        next_piece_type: after.next[0], hold_after: after.hold, can_hold_after: after.canHold,
        tick_after: after.tick, gravity_beat_after: after.gravityBeat,
        grounded_beats_after: after.groundedBeats, lock_resets_after: after.lockResets,
        game_over_after: after.over, paused_after: after.paused, reset_run: action === 'restart',
      }];
    }));
  const sections = stateSections(game);
  sections.actions = 'Available actions: ' + Object.entries(actions).map(([id, description]) => `${id}: ${typeof description === 'string' ? description : JSON.stringify(description)}`).join('; ');
  const visible = game.board.map(row => [...row]);
  for (const {x,y,type} of cells(game)) if (y >= 0 && y < 20 && x >= 0 && x < 10) visible[y][x] = type.toLowerCase();
  const rows = visible.map(row => row.join(''));
  const features = config.includeMetrics ? metrics(game.board) : undefined;
  let state;
  if (config.format === 'json_object') {
    state = Object.fromEntries(Object.entries(sections).filter(([key]) => key === 'board' || config.sections[key]));
    if (config.sections.actions && config.choiceMode === 'outcome') state.actions = actions;
    if (features) state.metrics = features;
  } else if (config.format === 'json') {
    const context = ['Board: width=10 height=20; x=0..9 left to right, y=0..19 top to bottom; negative y is above board. .=empty, uppercase=locked, lowercase=active.', ...Object.entries(sections).filter(([key]) => key !== 'board' && key !== 'actions' && config.sections[key]).map(([, text]) => text)];
    state = JSON.stringify({context, board: rows, ...(config.sections.actions ? {available_actions: actions} : {}), ...(features ? {metrics: features} : {})});
  } else {
    if (config.format === 'grid') sections.board = 'Board: width=10 height=20; x=0..9 left to right, y=0..19 top to bottom; .=empty, uppercase=locked, lowercase=active. Rows:\n' + rows.join('\n');
    if (config.format === 'coordinates') sections.board = 'Board: width=10 height=20; x=0..9 left to right, y=0..19 top to bottom. Occupied board cells (all unlisted cells are empty; uppercase=locked, lowercase=active): ' + JSON.stringify(visible.flatMap((row,y) => row.flatMap((type,x) => type === '.' ? [] : [{x,y,type}])));
    const parts = Object.entries(sections).filter(([key]) => key === 'board' || config.sections[key]).map(([, text]) => text);
    if (features) parts.push('Board metrics: ' + JSON.stringify(features));
    state = parts.join('\n');
  }
  return {state, actions, config};
}

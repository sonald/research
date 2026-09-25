import { ACTIONS, ALL_ACTIONS, boardMetrics as metrics, cells, stateSections, legalActions, step, ghostCells } from './engine.js';
import { buildRowChoices } from './row-choices.js';

export const STATE_SECTION_LABELS = Object.freeze({
  goal: '目标 Goal', active: '当前方块 Active', landing: '落点 Landing', next: '预告与暂存 Next / Hold',
  stats: '得分与消行 Score', status: '游戏状态 Status', timing: '计时 Timing', rules: '规则 Rules',
  shapes: '方块形状 Spawn matrices', history: '近期动作 Recent actions', actions: '动作说明 Available actions',
});

export const DEFAULT_CONFIG = Object.freeze({format: 'original', choiceMode: 'description', lookaheadDepth: 0, legalOnly: true, includeMetrics: false, excludedActions: Object.freeze([]), sections: Object.freeze(Object.fromEntries(Object.keys(STATE_SECTION_LABELS).map(key => [key, true])))});
const FORMATS = ['original', 'grid', 'coordinates', 'json', 'json_object'];
const TYPES = 'IOTJLSZ';
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const pieceType = value => typeof value === 'string' && value.length === 1 && TYPES.includes(value);
const integer = (value, min = 0, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= min && value <= max;

export function normalizeConfig(config = {}) {
  if (!record(config) || Object.keys(config).some(key => !Object.hasOwn(DEFAULT_CONFIG, key))) throw new Error('Invalid experiment config');
  const result = {...DEFAULT_CONFIG, ...config};
  if (!FORMATS.includes(result.format) || !['description', 'outcome', 'row_placements'].includes(result.choiceMode) || typeof result.legalOnly !== 'boolean' || typeof result.includeMetrics !== 'boolean') throw new Error('Invalid experiment config');
  if (!integer(result.lookaheadDepth, 0, 2)) throw new Error('Invalid lookahead depth');
  if (!Array.isArray(result.excludedActions) || result.excludedActions.length > Object.keys(ALL_ACTIONS).length || result.excludedActions.some(action => typeof action !== 'string' || !Object.hasOwn(ALL_ACTIONS, action))) throw new Error('Invalid excluded actions');
  result.excludedActions = Object.keys(ALL_ACTIONS).filter(action => result.excludedActions.includes(action));
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
  if (!Array.isArray(input.history) || input.history.length > 8 || input.history.some(action => typeof action !== 'string' || !Object.hasOwn(ALL_ACTIONS, action))) fail();
  const game = Object.fromEntries(['seed','rng','hold','canHold','score','lines','pieces','tick','paused','over','gravityBeat','groundedBeats','lockResets'].map(key => [key,input[key]]));
  Object.assign(game, {board: input.board.map(row => [...row]), active: {type: input.active.type, x: input.active.x, y: input.active.y, rotation: input.active.rotation}, next: [...input.next], bag: [...input.bag], history: [...input.history]});
  if (cells(game).some(({x,y}) => x < 0 || x >= 10 || y < -4 || y >= 20 || (!game.over && y >= 0 && game.board[y][x] !== '.'))) fail();
  return game;
}

// Compare only locked cells: an airborne piece has not changed the stack yet.
function boardOutcome(before, board) {
  const {definitions, ...after} = metrics(board);
  return {
    ...after,
    holes_created: Math.max(0,after.holes-before.holes),
    holes_removed: Math.max(0,before.holes-after.holes),
    height_change: after.maxHeight-before.maxHeight,
    surface_change: after.bumpiness-before.bumpiness,
    summary: `${after.holes} holes; stack height ${after.maxHeight}/20; surface unevenness ${after.bumpiness} (lower is flatter).`,
  };
}

function lookaheadPreview(after, config) {
  const beforeBoard = metrics(after.board);
  const controls = ['left','right','soft_drop','rotate_cw','rotate_ccw','rotate_180'].filter(action => !config.excludedActions.includes(action));
  let best, bestRank, candidateCount = 0;
  const visit = (node, followup) => {
    const terminal = node.over || node.pieces > after.pieces;
    const projected = terminal ? node : step(structuredClone(node), 'hard_drop');
    const quality = boardOutcome(beforeBoard, projected.board);
    const cleared = projected.lines - after.lines;
    const rank = [Number(projected.over), -cleared, quality.holes, quality.maxHeight, quality.bumpiness, followup.length];
    candidateCount++;
    const difference = bestRank && rank.findIndex((value, i) => value !== bestRank[i]);
    if (!bestRank || (difference >= 0 && rank[difference] < bestRank[difference])) {
      bestRank = rank;
      best = {
        followup_actions: followup,
        requires_additional_hard_drop: !terminal,
        hard_drop_excluded_from_choices: config.excludedActions.includes('hard_drop'),
        piece_type: after.active.type,
        additional_score: projected.score - after.score,
        additional_cleared_lines: cleared,
        board_after: quality,
        game_over: projected.over,
      };
    }
    if (terminal || followup.length === config.lookaheadDepth) return;
    const legal = legalActions(node);
    for (const action of controls) if (Object.hasOwn(legal, action)) visit(step(structuredClone(node), action), [...followup, action]);
  };
  visit(after, []);
  return {
    depth: config.lookaheadDepth, forecast_only: true,
    assumption: 'Forecast only; this choice executes only its first action. Explore up to depth additional legal moves/rotations, including gravity and locking, then hypothetically hard-drop if not locked. Hard drop remains hypothetical even if excluded from actual choices. Stop at the first lock or game over; never play the next piece.',
    selection_rule: 'Within this first action only: prefer survival, then more cleared rows, fewer total holes, lower maximum height, lower bumpiness, then fewer followup actions. This is an optimistic forecast, not a guaranteed future policy.',
    candidate_count: candidateCount,
    ...best,
  };
}

export function evaluateActions(game, config) {
  config = normalizeConfig(config);
  const beforeBoard = metrics(game.board);
  const actions = config.choiceMode === 'row_placements' ? buildRowChoices(game,config.excludedActions) : Object.fromEntries(Object.entries(config.legalOnly ? legalActions(game) : ACTIONS)
    .filter(([action]) => !config.excludedActions.includes(action))
    .map(([action, operation]) => {
      if (config.choiceMode === 'description') return [action, operation];
      const after = step(structuredClone(game), action);
      const newlyCleared = Math.max(0,after.lines-game.lines);
      const locked = after.pieces > game.pieces;
      const boardAfter = boardOutcome(beforeBoard,after.board);
      const control = ['pause','resume','restart'].includes(action);
      let landingPreview = null;
      if (!control && !game.paused && !game.over) {
        if (locked) {
          landingPreview = {requires_additional_hard_drop:false,summary:'This action already locked the piece; use the actual result above. No next-piece projection.'};
        } else if (!after.over && !after.paused) {
          const projected = step(structuredClone(after),'hard_drop');
          const lines = projected.lines-after.lines;
          const quality = boardOutcome(beforeBoard,projected.board);
          landingPreview = {
            requires_additional_hard_drop:true,
            assumption:'If the active piece is hard-dropped immediately from its resulting position, with no further movement or rotation. Forecast only; NOT executed by this choice.',
            piece_type:after.active.type,landing_cells:ghostCells(after),
            additional_score:projected.score-after.score,additional_cleared_lines:lines,
            board_after:quality,game_over:projected.over,
            summary:`A subsequent hard drop would clear ${lines} rows, earn ${projected.score-after.score} points, create ${quality.holes_created} holes and remove ${quality.holes_removed} holes.`,
          };
        }
      }
      return [action, {
        operation,
        scoring:'Points come only from cleared rows. Soft drop and hard drop have no movement bonus.',
        benefit: action==='restart'?'Resets this run and loses its accumulated score.':
          `${newlyCleared} rows cleared this beat; ${after.score-game.score} points. ${locked?'Piece locked; placement committed.':after.over?'Game ended.':after.paused?'Game paused.':'No piece locked yet; the active piece can still be adjusted.'} ${boardAfter.holes_created} holes created, ${boardAfter.holes_removed} removed.`,
        board_after:boardAfter,
        landing_preview:landingPreview,
        ...(config.lookaheadDepth > 0 ? {lookahead_preview: control || game.paused || game.over || after.over || after.paused || locked ? null : lookaheadPreview(after, config)} : {}),
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
  return actions;
}

function summarizeAction(game, beforeBoard, value) {
  const isRow = Object.hasOwn(value, 'variants');
  const immediate = isRow
    ? {score:value.immediate_after.score-game.score, cleared_lines:Math.max(0,value.immediate_after.cleared_lines-game.lines), game_over:value.immediate_after.game_over}
    : {score:value.score_delta, cleared_lines:value.newly_cleared_lines, game_over:value.game_over_after};
  let score=immediate.score, cleared=immediate.cleared_lines, quality=beforeBoard, over=immediate.game_over, followup=[];
  if (isRow) {
    const best=value.variants?.[value.best_variant_index];
    if (best) {
      score=best.score_delta;cleared=best.newly_cleared_lines;over=best.game_over;
      quality={holes:best.holes,maxHeight:best.max_height,bumpiness:best.bumpiness};
      if (!best.actual_first_beat) followup=[...(best.rotation_action?[best.rotation_action]:[]),'hard_drop'];
    } else {
      quality=metrics(step(structuredClone(game),value.execute_now).board);
    }
  } else {
    quality=value.board_after;
    const preview=value.lookahead_preview ?? value.landing_preview;
    if (preview?.board_after) {
      score+=preview.additional_score;cleared+=preview.additional_cleared_lines;
      quality=preview.board_after;over=preview.game_over;
      followup=[...(preview.followup_actions??[]),...(preview.requires_additional_hard_drop?['hard_drop']:[])];
    }
  }
  return {
    immediate_gain:immediate,
    expected_gain:{score,cleared_lines:cleared,holes_delta:quality.holes-beforeBoard.holes,height_delta:quality.maxHeight-beforeBoard.maxHeight,surface_roughness:quality.bumpiness,game_over:over},
    followup,
  };
}

export function buildDecision(game, config) {
  config=normalizeConfig(config);
  const beforeBoard=metrics(game.board);
  const actions=Object.fromEntries(Object.entries(evaluateActions(game,config)).map(([action,value])=>
    [action, typeof value==='string'?value:summarizeAction(game,beforeBoard,value)]));
  const sections = stateSections(game);
  sections.actions = 'Available actions: ' + Object.entries(actions).map(([id, description]) => `${id}: ${typeof description === 'string' ? description : JSON.stringify(description)}`).join('; ');
  const visible = game.board.map(row => [...row]);
  for (const {x,y,type} of cells(game)) if (y >= 0 && y < 20 && x >= 0 && x < 10) visible[y][x] = type.toLowerCase();
  const rows = visible.map(row => row.join(''));
  const features = config.includeMetrics ? metrics(game.board) : undefined;
  let state;
  if (config.format === 'json_object') {
    state = Object.fromEntries(Object.entries(sections).filter(([key]) => key === 'board' || config.sections[key]));
    if (config.sections.actions && config.choiceMode !== 'description') state.actions = actions;
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

import { ACTIONS, ALL_ACTIONS, boardMetrics as metrics, cells, stateSections, legalActions, step, ghostCells } from './engine.js';
import { buildRowChoices } from './row-choices.js';
import { searchPlacements } from './placement-search.js';

export const STATE_SECTION_LABELS = Object.freeze({
  decision_guide: '决策与收益说明 Decision guide', current_target: '当前目标 Current target', goal: '目标 Goal', active: '当前方块 Active', landing: '落点 Landing', next: '预告与暂存 Next / Hold',
  stats: '得分与消行 Score', status: '游戏状态 Status', timing: '计时 Timing', rules: '规则 Rules',
  shapes: '方块形状 Spawn matrices', history: '近期动作 Recent actions', actions: '动作说明 Available actions',
});

export const DEFAULT_CONFIG = Object.freeze({format: 'original', choiceMode: 'description', lookaheadDepth: 0, search: Object.freeze({maxStates:5000,maxPathLength:64,maxChoices:20}), legalOnly: true, includeMetrics: false, excludedActions: Object.freeze([]), sections: Object.freeze(Object.fromEntries(Object.keys(STATE_SECTION_LABELS).map(key => [key, true])))});
const FORMATS = ['original', 'grid', 'coordinates', 'json', 'json_object'];
const TYPES = 'IOTJLSZ';
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const pieceType = value => typeof value === 'string' && value.length === 1 && TYPES.includes(value);
const integer = (value, min = 0, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= min && value <= max;

export function normalizeConfig(config = {}) {
  if (!record(config) || Object.keys(config).some(key => !Object.hasOwn(DEFAULT_CONFIG, key))) throw new Error('Invalid experiment config');
  const result = {...DEFAULT_CONFIG, ...config};
  if (!FORMATS.includes(result.format) || !['description', 'outcome', 'row_placements', 'reachable_placements'].includes(result.choiceMode) || typeof result.legalOnly !== 'boolean' || typeof result.includeMetrics !== 'boolean') throw new Error('Invalid experiment config');
  if (!integer(result.lookaheadDepth, 0, 2)) throw new Error('Invalid lookahead depth');
  if (!Array.isArray(result.excludedActions) || result.excludedActions.length > Object.keys(ALL_ACTIONS).length || result.excludedActions.some(action => typeof action !== 'string' || !Object.hasOwn(ALL_ACTIONS, action))) throw new Error('Invalid excluded actions');
  result.excludedActions = Object.keys(ALL_ACTIONS).filter(action => result.excludedActions.includes(action));
  if (!record(result.sections) || Object.entries(result.sections).some(([key, value]) => !Object.hasOwn(STATE_SECTION_LABELS, key) || typeof value !== 'boolean')) throw new Error('Invalid state sections');
  result.sections = {...DEFAULT_CONFIG.sections, ...result.sections};
  if (!record(result.search) || Object.keys(result.search).some(key=>!Object.hasOwn(DEFAULT_CONFIG.search,key))) throw new Error('Invalid search settings');
  result.search={...DEFAULT_CONFIG.search,...result.search};
  if(!integer(result.search.maxStates,1,10000)||!integer(result.search.maxPathLength,1,128)||!integer(result.search.maxChoices,1,20))throw new Error('Invalid search budget');
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
  const change = (amount, up, down, same) => amount > 0 ? `${up} ${amount}` : amount < 0 ? `${down} ${-amount}` : same;
  const points = amount => change(amount,'gain','lose','no change in');
  immediate.summary = `This action clears ${immediate.cleared_lines} rows; ${points(immediate.score)} points. ${immediate.game_over?'The game ends.':'The game does not end immediately.'}`;
  const gain = (score, cleared, quality, over, scenario, conditional) => {
    const result={scenario,score,cleared_lines:cleared,holes_delta:quality.holes-beforeBoard.holes,height_delta:quality.maxHeight-beforeBoard.maxHeight,surface_roughness:quality.bumpiness,game_over:over};
    const surface=quality.bumpiness<=4?'flat':quality.bumpiness<=9?'slightly uneven':quality.bumpiness<=16?'bumpy':'very jagged';
    result.summary = `${conditional?'Conditional possibility, not immediate:':'Result after this action:'} ${cleared} rows cleared; ${points(score)} points. ${change(result.holes_delta,'Holes increase by','Holes decrease by','Hole count is unchanged')}. ${change(result.height_delta,'Stack grows by','Stack lowers by','Stack height is unchanged')}${result.height_delta?' rows':''}. Surface is ${surface}. ${over?'Game ends in this result.':'No game over in this result.'}`;
    return result;
  };
  let expected;
  if (isRow && value.variants?.length) {
    expected=value.variants.map(v=>gain(v.score_delta,v.newly_cleared_lines,
      {holes:v.holes,maxHeight:v.max_height,bumpiness:v.bumpiness},v.game_over,
      v.actual_first_beat?'Immediate result':v.rotation_action?`After ${v.rotation_action} and hard_drop`:'After hard_drop',!v.actual_first_beat));
  } else {
    let score=immediate.score, cleared=immediate.cleared_lines, over=immediate.game_over;
    let quality=isRow?metrics(step(structuredClone(game),value.execute_now).board):value.board_after;
    const preview=isRow?null:value.lookahead_preview ?? value.landing_preview;
    if (preview?.board_after) {
      score+=preview.additional_score;cleared+=preview.additional_cleared_lines;
      quality=preview.board_after;over=preview.game_over;
    }
    const conditional=Boolean(preview?.board_after);
    expected=[gain(score,cleared,quality,over,conditional?'After the conditional placement preview':'Immediate result',conditional)];
  }
  return {immediate_gain:immediate,expected_gains:expected};
}

export function buildDecision(game, config, currentPlan = null) {
  config=normalizeConfig(config);
  const beforeBoard=metrics(game.board);
  const placement=config.choiceMode==='reachable_placements'?searchPlacements(game,{...config.search,excludedActions:config.excludedActions},currentPlan):null;
  const actions=placement?placement.actions:Object.fromEntries(Object.entries(evaluateActions(game,config)).map(([action,value])=>
    [action, typeof value==='string'?value:summarizeAction(game,beforeBoard,value)]));
  const sections = {
    decision_guide: [
      'Decision guide: The goal is to clear rows and keep playing. Only line clears earn points; movement and dropping alone earn none. Each Choice key is the only action executed now. left_N/right_N moves N columns in one beat. Avoid pause/restart while actively playing.',
      'immediate_gain describes the result of this one action. score is the change in points, not total score; cleared_lines is the number of newly cleared rows, not the accumulated count; game_over=true means this action ends the game. No immediate gain can still be useful preparation for a later placement.',
      `Current evaluation mode: ${config.choiceMode}${config.choiceMode==='outcome'?`, additional lookahead depth=${config.lookaheadDepth}`:''}.`,
      'expected_gains is a list of mutually exclusive possible outcomes, NOT a sequence to execute or add together. In same-row mode it contains EVERY distinct feasible optional-rotation-then-hard-drop outcome for this action, with no best-only filtering. scenario identifies the condition, not an instruction. Other preview modes keep their existing evaluation horizon. Each outcome is conditional, not guaranteed; its changes are relative to the current state BEFORE this action and already include immediate_gain. Do NOT sum outcomes or add immediate_gain again.',
      'For each expected_gains entry: prefer more cleared_lines and score; holes_delta<0 means fewer covered empty cells (better), >0 means more holes (worse), 0 means unchanged, NOT necessarily hole-free. height_delta<0 lowers the highest stack (usually better), >0 raises it. surface_roughness is the final sum of adjacent column-height differences, NOT a change; smaller is flatter. Labels: 0-4 flat, 5-9 slightly uneven, 10-16 bumpy, 17+ very jagged. game_over=true is a losing result.',
      'Select a current action that offers at least one favorable feasible outcome, rather than averaging its good and bad rotation alternatives. Fewer holes is a benefit even with zero score and no line clear; lower height and a flatter surface are also useful. Compare those opportunities with other actions while respecting immediate game-over risk. Do not assume the favorable outcome is automatic. Only the current action runs; the next decision is recalculated with no committed plan. Preview hard drops may remain hypothetical when excluded.',
    ].join('\n'),
    ...stateSections(game),
  };
  if (placement) {
    sections.decision_guide = [
      'Decision guide: Each Choice is one distinct placement of the CURRENT piece, found with bounded search and independently replay-verified against the real engine. Choose a placement ID. Only its execute_now action runs this turn, NOT the whole path. Hold is outside this search.',
      'The summary describes the completed placement, not the immediate action: score gain, newly cleared rows, holes before->after, maximum height before->after, roughness before->after and survival. Fewer holes is useful even with no line clear; prefer survival, clears, a lower stack and flatter surface. Only cleared rows earn points.',
      'steps_to_lock is the number of remaining engine inputs, including the final lock. left_N/right_N is ONE beat, while N separate inputs take N beats. is_current_target marks the previous target whose remaining path has just replayed successfully. Prefer continuing a still-valid target when benefits are similar, rather than repeatedly switching.',
      'Search is bounded and may be incomplete. Candidates are shortlisted after verification, preserving line/holes/height/roughness representatives and a valid current target, then using a heuristic to fill at most 20 options. The shortlist is not proof of globally optimal play. All outcomes terminating the game are omitted when surviving outcomes were found.',
      `Search coverage: ${placement.diagnostics.search_complete?'complete for the configured action set':'INCOMPLETE'}; truncated_by=${JSON.stringify(placement.diagnostics.truncated_by)}.`,
    ].join('\n');
    const target=Object.entries(placement.actions).find(([,choice])=>choice.is_current_target);
    sections.current_target=target?`Current target: ${target[0]}; remaining steps=${target[1].steps_to_lock}; replay verified=true. Prefer retaining it when benefits are similar.`:'Current target: none (or previous plan invalidated).';
  }
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
  return {state, actions, config, ...(placement?{search:placement.diagnostics,plans:placement.plans,placements:placement.candidates.filter(candidate=>Object.hasOwn(actions,candidate.id))}:{})};
}

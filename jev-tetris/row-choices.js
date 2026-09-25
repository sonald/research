import { ALL_ACTIONS, applyPoseInput, boardMetrics, cells, ghostCells, step } from './engine.js';

const rotations = ['rotate_cw', 'rotate_ccw', 'rotate_180'];
const snapshot = game => ({active: {...game.active}, cells: cells(game), score: game.score,
  cleared_lines: game.lines, tick: game.tick, game_over: game.over, pieces: game.pieces});

function variant(source, pose, rotationAction, actualAfter) {
  const landing = ghostCells(pose);
  const after = actualAfter ?? step(structuredClone(pose), 'hard_drop');
  const quality = boardMetrics(after.board);
  return {rotation_action: rotationAction, pose_before_drop: {...pose.active}, landing_cells: landing,
    board_after: after.board.map(row => row.join('')), score_delta: after.score - source.score,
    newly_cleared_lines: after.lines - source.lines,
    holes: quality.holes, max_height: quality.maxHeight, bumpiness: quality.bumpiness,
    game_over: after.over};
}

const rank = value => [Number(value.game_over), -value.newly_cleared_lines, value.holes, value.max_height, value.bumpiness];
function better(a, b) {
  const left = rank(a), right = rank(b);
  const index = left.findIndex((value, i) => value !== right[i]);
  return index >= 0 && left[index] < right[index];
}

export function buildRowChoices(game, excludedActions = []) {
  const excluded = action => excludedActions.includes(action) ||
    (/^(left|right)_/.test(action) && excludedActions.includes(action.split('_')[0]));
  if (game.over || game.paused) return Object.fromEntries((game.over ? ['restart'] : ['resume', 'restart'])
    .filter(action => !excluded(action)).map(action => [action, {operation: ALL_ACTIONS[action], execute_now: action,
      immediate_after: snapshot(step(structuredClone(game), action)), forecast_only: true, variants: null, best_variant_index: null}]));
  const actions = ['left', ...Array.from({length: 8}, (_, i) => `left_${i + 2}`),
    'right', ...Array.from({length: 8}, (_, i) => `right_${i + 2}`), ...rotations, 'hard_drop'];
  return Object.fromEntries(actions.filter(action => !excluded(action)).flatMap(action => {
    const pose = structuredClone(game);
    if (action !== 'hard_drop' && !applyPoseInput(pose, action)) return [];
    const immediate = step(structuredClone(game), action);
    const ended = immediate.over || immediate.pieces > game.pieces;
    const variants = [variant(game, pose, null, ended ? immediate : undefined)];
    if (ended) variants[0].actual_first_beat = true;
    if (!ended && /^(left|right)(_|$)/.test(action)) for (const rotation of rotations.filter(rotation => !excluded(rotation))) {
      const rotated = structuredClone(pose);
      if (applyPoseInput(rotated, rotation)) variants.push(variant(game, rotated, rotation));
    }
    const seen = new Set();
    const unique = variants.filter(value => {
      const key = JSON.stringify([value.landing_cells.map(({x, y}) => [x, y]).sort((a, b) => a[1] - b[1] || a[0] - b[0]), value.board_after]);
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
    let best = 0;
    for (let i = 1; i < unique.length; i++) if (better(unique[i], unique[best])) best = i;
    return [[action, {operation: ALL_ACTIONS[action], execute_now: action,
      immediate_after: snapshot(immediate), forecast_only: true,
      actual_beat_ends_piece: ended,
      continuation_possible_after_actual_beat: !ended,
      reference_y: game.active.y,
      assumption: 'Only execute_now is executed, including its normal one-beat gravity and locking. Variants geometrically apply that input at the source row BEFORE gravity, then optionally one listed rotation using SRS kicks, then hypothetically hard-drop the current piece once. No next-piece play; no promise that this frozen-row continuation remains reachable after the actual beat. Multi-column inputs check every intermediate column. If the actual first beat locks or ends the game, variants contain only that actual terminal result and no continuation.',
      hard_drop_excluded_from_choices: excluded('hard_drop'),
      selection_rule: 'Best variant within this action: survival, more cleared lines, fewer holes, lower maximum height, then lower bumpiness. All variants remain available; no action is chosen by this ranking.',
      variants: unique, best_variant_index: best}]];
  }));
}

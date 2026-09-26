import { ALL_ACTIONS, step } from './engine.js';

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}
function serialize(value) { return JSON.stringify(canonical(value)); }
// Deterministic browser/server identifier; callers reject any same-ID/different-terminal collision.
function hash(text) {
  let a = 2166136261, b = 2246822507;
  for (let i = 0; i < text.length; i++) {
    a = Math.imul(a ^ text.charCodeAt(i), 16777619);
    b = Math.imul(b ^ text.charCodeAt(i), 3266489909);
  }
  return (a >>> 0).toString(16).padStart(8, '0') + (b >>> 0).toString(16).padStart(8, '0');
}
export function pieceId(game) {
  return 'piece_' + hash(serialize({ seed: game.seed, pieces: game.pieces, type: game.active?.type,
    hold: game.hold, canHold: game.canHold, next: game.next, rng: game.rng, bag: game.bag }));
}
export function terminalKey(game) {
  const { board, active, score, lines, pieces, next, hold, canHold, rng, bag, seed, paused, over,
    gravityBeat, groundedBeats, lockResets } = game;
  return serialize({ board, active, score, lines, pieces, next, hold, canHold, rng, bag, seed, paused, over,
    gravityBeat, groundedBeats, lockResets });
}
export function stateVersion(game) { return 'state_' + hash(serialize([terminalKey(game), game.tick, game.history])); }
export function targetId(game, terminal) {
  return 'placement_' + hash(pieceId(game) + '|' + (typeof terminal === 'string' ? terminal : terminalKey(terminal)));
}
export function allowedSearchActions(excluded = []) {
  const denied = new Set(excluded);
  const ordered = ['hard_drop', 'soft_drop', 'left', 'right', 'rotate_cw', 'rotate_ccw', 'rotate_180', 'wait',
    ...Object.keys(ALL_ACTIONS).filter(action => /^(left|right)_[2-9]$/.test(action))];
  return ordered.filter(action => !denied.has(action) && !denied.has(action.split('_')[0]));
}
export function replayPlacement(game, path, { excludedActions = [], maxPathLength = 64 } = {}, expectedKey) {
  const fail = error => ({ ok: false, error });
  if (game.paused || game.over || !game.active) return fail('Current piece is not playable');
  if (!Number.isInteger(maxPathLength) || maxPathLength < 1 || maxPathLength > 128 ||
      !Array.isArray(path) || path.length < 1 || path.length > maxPathLength) return fail('Invalid placement path length');
  const allowed = new Set(allowedSearchActions(excludedActions));
  if (path.some(action => typeof action !== 'string' || !allowed.has(action))) return fail('Placement path contains a disabled or unsupported action');
  const replay = structuredClone(game);
  for (let i = 0; i < path.length; i++) {
    step(replay, path[i]);
    const terminal = replay.over || replay.pieces > game.pieces;
    if (terminal && i !== path.length - 1) return fail('Placement path continues after the current piece ends');
    if (!terminal && i === path.length - 1) return fail('Placement path does not finish the current piece');
  }
  const key = terminalKey(replay);
  if (expectedKey !== undefined && key !== expectedKey) return fail('Placement terminal state does not match');
  return { ok: true, game: replay, terminalKey: key, pieceId: pieceId(game) };
}
export function validatePlan(game, plan, config = {}) {
  if (!plan || typeof plan !== 'object' || Array.isArray(plan) ||
      typeof plan.pieceId !== 'string' || typeof plan.targetId !== 'string' ||
      typeof plan.terminalKey !== 'string' || typeof plan.stateVersion !== 'string' || !Array.isArray(plan.remainingPath)) {
    return { ok: false, error: 'Invalid placement plan' };
  }
  if (plan.pieceId !== pieceId(game)) return { ok: false, error: 'Placement plan belongs to a different piece' };
  if (plan.stateVersion !== stateVersion(game)) return { ok: false, error: 'Placement plan is stale' };
  if (plan.targetId !== targetId(game, plan.terminalKey)) return { ok: false, error: 'Placement target ID does not match' };
  return replayPlacement(game, plan.remainingPath, config, plan.terminalKey);
}
export function executePlanStep(game, plan, config = {}) {
  const validation = validatePlan(game, plan, config);
  if (!validation.ok) throw new Error(validation.error);
  const action = plan.remainingPath[0], originalPieces = game.pieces;
  step(game, action);
  if (game.over || game.pieces > originalPieces) return { action, plan: null };
  const remaining = { ...plan, remainingPath: plan.remainingPath.slice(1), stateVersion: stateVersion(game) };
  const checked = validatePlan(game, remaining, config);
  if (!checked.ok) throw new Error(checked.error);
  return { action, plan: remaining };
}

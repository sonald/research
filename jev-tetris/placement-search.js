import {boardMetrics, step} from './engine.js';
import {pieceId, terminalKey, stateVersion, targetId, allowedSearchActions, replayPlacement, validatePlan} from './placement-plan.js';

const clone = game => ({...game, active: game.active && {...game.active}, board: game.board.map(row => [...row]),
  next: [...game.next], bag: [...game.bag], history: [...game.history]});
const poseKey = g => JSON.stringify([g.active, g.gravityBeat, g.groundedBeats, g.lockResets, g.canHold, g.score, g.lines]);
const compare = (a, b) => b.rankScore - a.rankScore || a.path.length - b.path.length || a.id.localeCompare(b.id);

export function selectPlacements(candidates, maxChoices = 20) {
  const surviving = candidates.filter(c => !c.gain.game_over);
  const pool = (surviving.length ? surviving : candidates).slice().sort(compare);
  const selected = [];
  const add = c => { if (c && selected.length < maxChoices && !selected.includes(c)) selected.push(c); };
  add(pool.find(c => c.is_current_target));
  for (const metric of [c => -c.gain.cleared_lines, c => c.gain.holes_after, c => c.gain.height_after, c => c.gain.roughness_after]) {
    add(pool.slice().sort((a,b) => metric(a) - metric(b) || compare(a,b))[0]);
  }
  pool.forEach(add);
  return selected.sort(compare);
}

function candidate(source, terminal, path, key, currentKey) {
  const before = boardMetrics(source.board), after = boardMetrics(terminal.board);
  const gain = {score: terminal.score - source.score, cleared_lines: terminal.lines - source.lines,
    holes_delta: after.holes - before.holes, height_delta: after.maxHeight - before.maxHeight,
    roughness_delta: after.bumpiness - before.bumpiness,
    holes_before: before.holes, holes_after: after.holes, height_before: before.maxHeight, height_after: after.maxHeight,
    roughness_before: before.bumpiness, roughness_after: after.bumpiness, game_over: terminal.over};
  return {id: targetId(source, key), pieceId: pieceId(source), path, terminalKey: key, terminal, gain,
    rankScore: 100 * gain.cleared_lines - 40 * gain.holes_delta - 4 * gain.height_delta - gain.roughness_delta,
    is_current_target: key === currentKey};
}

function pathTo(nodes, index, lastAction) {
  const path = [lastAction];
  while (nodes[index].parent !== -1) {
    path.push(nodes[index].action); index = nodes[index].parent;
  }
  return path.reverse();
}

export function searchPlacements(game, options = {}, currentPlan = null) {
  const config = {maxStates: 5000, maxPathLength: 64, maxChoices: 20, excludedActions: [], ...options};
  for (const [key, maximum] of [['maxStates',10000], ['maxPathLength',128], ['maxChoices',20]]) {
    if (!Number.isInteger(config[key]) || config[key] < 1 || config[key] > maximum) throw new Error(`Invalid ${key}`);
  }
  const start = performance.now(), truncated = new Set(), outcomes = new Map();
  const current = currentPlan ? validatePlan(game, currentPlan, config) : null;
  const currentKey = current?.ok ? current.terminalKey : null;
  const nodes = game.over || game.paused || !game.active ? [] : [{game: clone(game), parent: -1, depth: 0}];
  const visited = new Set(nodes.length ? [poseKey(game)] : []);
  const actions = allowedSearchActions(config.excludedActions);
  for (let cursor = 0; cursor < nodes.length; cursor++) {
    const node = nodes[cursor];
    for (const action of actions) {
      const after = step(clone(node.game), action);
      if (after.pieces !== game.pieces || after.over) {
        const key = terminalKey(after);
        if (!outcomes.has(key)) outcomes.set(key, {index: cursor, action});
        continue;
      }
      const key = poseKey(after);
      if (visited.has(key)) continue;
      if (node.depth + 1 >= config.maxPathLength) { truncated.add('maxPathLength'); continue; }
      if (nodes.length >= config.maxStates) { truncated.add('maxStates'); continue; }
      visited.add(key);
      nodes.push({game: after, parent: cursor, action, depth: node.depth + 1});
    }
  }
  const searchEnd = performance.now(), candidates = [];
  for (const [key, end] of outcomes) {
    const path = pathTo(nodes, end.index, end.action);
    const replay = replayPlacement(game, path, config, key);
    if (replay.ok) candidates.push(candidate(game, replay.game, path, key, currentKey));
  }
  if (current?.ok && !candidates.some(c => c.terminalKey === currentKey)) {
    candidates.push(candidate(game, current.game, [...currentPlan.remainingPath], currentKey, currentKey));
  }
  const replayEnd = performance.now();
  const selected = selectPlacements(candidates, config.maxChoices);
  const publicActions = {}, plans = {};
  for (const c of selected) {
    if (plans[c.id] && plans[c.id].terminalKey !== c.terminalKey) throw new Error('Placement ID collision');
    const g = c.gain;
    publicActions[c.id] = {execute_now: c.path[0],
      summary: `After this piece locks: clear ${g.cleared_lines} lines, gain ${g.score} points; holes ${g.holes_before} -> ${g.holes_after}, height ${g.height_before} -> ${g.height_after}, roughness ${g.roughness_before} -> ${g.roughness_after} (lower is better); ${g.game_over ? 'GAME OVER' : 'survives'}.`,
      steps_to_lock: c.path.length, is_current_target: c.is_current_target};
    plans[c.id] = {pieceId: c.pieceId, targetId: c.id, remainingPath: [...c.path], terminalKey: c.terminalKey, stateVersion: stateVersion(game)};
  }
  return {actions: publicActions, plans, candidates,
    diagnostics: {visited_states: nodes.length, unique_outcomes: outcomes.size, verified_outcomes: candidates.length,
      shown_choices: selected.length, search_complete: truncated.size === 0, truncated_by: [...truncated].sort(),
      search_ms: searchEnd - start, replay_ms: replayEnd - searchEnd, current_target_valid: Boolean(current?.ok)}};
}

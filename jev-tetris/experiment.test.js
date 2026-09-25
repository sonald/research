import test from 'node:test';
import assert from 'node:assert/strict';
import { ACTIONS, cells, createGame, legalActions, step, encodeState } from './engine.js';
import { STATE_SECTION_LABELS, DEFAULT_CONFIG, normalizeConfig, buildDecision, evaluateActions, validateGame } from './experiment.js';

test('legal actions exclude blocked input before gravity and retain wall kicks', () => {
  const g = createGame(); g.active = {type:'I',x:-2,y:5,rotation:1}; g.gravityBeat=4;
  const before = structuredClone(g), legal = legalActions(g);
  assert.equal(legal.left, undefined); assert.equal(legal.rotate_ccw, ACTIONS.rotate_ccw);
  assert.deepEqual(g, before);
  step(g,'left'); assert.equal(g.active.y,6); // Gravity must not make left appear legal.
  g.board[7][1]='Z'; assert.equal(legalActions(g).right, undefined);
});

test('grounded hard drop locks, O rotations and unavailable hold are excluded', () => {
  const g=createGame(); g.active={type:'O',x:3,y:18,rotation:0}; g.canHold=false;
  const legal=legalActions(g);
  for (const action of ['soft_drop','rotate_cw','rotate_ccw','rotate_180','hold','resume']) assert.equal(legal[action],undefined);
  for (const action of ['hard_drop','wait','pause','restart']) assert.equal(legal[action],ACTIONS[action]);
  step(g,'hard_drop'); assert.equal(g.pieces,1);
  g.paused=true; assert.deepEqual(Object.keys(legalActions(g)),['resume','restart']);
  g.over=true; assert.deepEqual(Object.keys(legalActions(g)),['restart']);
});

test('180 legality follows two clockwise attempts including kicks', () => {
  const g=createGame(); g.active={type:'I',x:-2,y:5,rotation:1};
  assert.equal(legalActions(g).rotate_180,ACTIONS.rotate_180);
  step(g,'rotate_180'); assert.equal(g.active.rotation,3);
});

test('configuration rejects ambiguous values and unknown fields', () => {
  assert.deepEqual(normalizeConfig(),DEFAULT_CONFIG);
  assert.deepEqual(normalizeConfig({format:'json'}),{...DEFAULT_CONFIG,format:'json'});
  for(const config of [null,[],true,{format:'yaml'},{legalOnly:'true'},{includeMetrics:1},{unknown:true},{format:undefined}]) assert.throws(()=>normalizeConfig(config));
});

test('formats preserve board and common facts without exposing hidden random state', () => {
  const g=createGame('formats'); g.board[19][0]='J';
  const original=encodeState(g);
  assert.equal(buildDecision(g,{legalOnly:false}).state,original);
  const decisions=['original','grid','coordinates','json'].map(format=>buildDecision(g,{format}));
  for (const format of ['original','grid','coordinates','json']) assert.equal(buildDecision(validateGame(g),{format}).state,buildDecision(g,{format}).state);
  const json=JSON.parse(decisions[3].state);
  assert.equal(json.board[19][0],'J'); assert.equal(json.board.length,20);
  assert.match(decisions[2].state,/"x":0,"y":19,"type":"J"/);
  for (const decision of decisions) {
    assert.deepEqual(decision.actions,legalActions(g));
    assert.ok(decision.state.includes('gravity_beat=0/5'));
    assert.ok(decision.state.includes('7-bag pieces; five previews'));
    assert.ok(decision.state.includes('Recent actions (oldest first): none'));
    assert.ok(decision.state.includes('Next (first plays next):'));
    assert.ok(!decision.state.includes('"rng"'));
    assert.ok(!decision.state.includes('"bag"'));
  }
});

test('metrics count locked-board holes and heights only and never mutate snapshot', () => {
  const g=createGame(); g.board[17][0]='J'; g.board[19][0]='J'; g.board[19][1]='Z';
  const before=structuredClone(g);
  const {metrics}=JSON.parse(buildDecision(g,{format:'json',includeMetrics:true}).state);
  assert.deepEqual(metrics.columnHeights,[3,1,0,0,0,0,0,0,0,0]);
  assert.equal(metrics.maxHeight,3); assert.equal(metrics.holes,1); assert.equal(metrics.bumpiness,3);
  assert.deepEqual(g,before);
  assert.equal(JSON.parse(buildDecision(g,{format:'json'}).state).metrics,undefined);
});

test('snapshot validation copies valid state and rejects malformed boundaries', () => {
  const g=createGame(); assert.deepEqual(validateGame(g),g);
  const copy=validateGame(g); copy.board[19][0]='Z'; assert.equal(g.board[19][0],'.');
  const bad=[{board:[]},{active:{...g.active,type:'X'}},{active:{...g.active,x:100}},{next:['I']},{hold:'X'},{paused:'false'},{score:-1},{rng:2**32},{gravityBeat:5},{groundedBeats:3},{lockResets:16},{bag:['I','I']},{history:['bad']},{seed:'x'.repeat(81)}];
  for (const patch of bad) assert.throws(()=>validateGame({...g,...patch}),/Invalid game snapshot/);
  const blocked=structuredClone(g); blocked.board[0].fill('Z'); blocked.board[1].fill('Z');
  assert.throws(()=>validateGame(blocked)); blocked.over=true; assert.doesNotThrow(()=>validateGame(blocked));
});

test('blocked rotations are excluded while immediately losing hold remains a legal choice', () => {
  const g=createGame(); g.active={type:'T',x:3,y:10,rotation:0};
  g.board=Array.from({length:20},()=>Array(10).fill('Z'));
  for(const [x,y] of [[4,10],[3,11],[4,11],[5,11]]) g.board[y][x]='.';
  const legal=legalActions(g);
  for(const action of ['left','right','soft_drop','rotate_cw','rotate_ccw','rotate_180']) assert.equal(legal[action],undefined);
  assert.equal(legal.hold,ACTIONS.hold);
  step(g,'hold'); assert.equal(g.over,true);
});


test('excluded actions intersect legality and all-action mode in every encoding',()=>{
  const game=createGame('42');
  for(const format of ['original','grid','coordinates','json'])for(const legalOnly of [true,false]){
    const d=buildDecision(game,{format,legalOnly,excludedActions:['hard_drop','pause']});
    assert.equal(Object.hasOwn(d.actions,'hard_drop'),false);
    assert.equal(Object.hasOwn(d.actions,'pause'),false);
    assert.ok(d.actions.left);
    if(format==='json')assert.deepEqual(JSON.parse(d.state).available_actions,d.actions);
    else assert.ok(!d.state.split('Available actions: ')[1].includes('hard_drop:'));
  }
  game.paused=true;
  assert.deepEqual(buildDecision(game,{excludedActions:['resume','restart']}).actions,{});
  assert.deepEqual(buildDecision(game,{legalOnly:false,excludedActions:Object.keys(ACTIONS)}).actions,{});
  assert.deepEqual(normalizeConfig({}).excludedActions,[]);
  assert.deepEqual(normalizeConfig({excludedActions:['hard_drop','hard_drop']}).excludedActions,['hard_drop']);
  for(const excludedActions of ['hard_drop',[null],['unknown'],[1]])assert.throws(()=>normalizeConfig({excludedActions}));
});


test('each optional section can be removed independently in every encoding', () => {
  const g = createGame('sections');
  step(g, 'left'); g.score = 321; g.lines = 7;
  const markers = {
    goal: 'TETRIS STATE. Goal:', active: 'Active:', landing: 'landing_cells=',
    next: 'Next (first plays next):', stats: 'Score:', status: 'tick=', timing: 'Timing:',
    rules: 'Rules:', shapes: 'Spawn matrices', history: 'Recent actions', actions: 'Available actions:',
  };
  for (const format of ['original', 'grid', 'coordinates', 'json']) {
    const baseline = buildDecision(g, {format});
    assert.match(baseline.state, /score=321; cleared_lines=7/);
    assert.doesNotMatch(baseline.state, /[ ;]lines=/);
    for (const key of Object.keys(STATE_SECTION_LABELS)) {
      const d = buildDecision(g, {format, sections: {[key]: false}});
      const content = format === 'json' ? JSON.parse(d.state).context.join('\n') : d.state;
      assert.ok(!content.includes(markers[key]), `${format}: ${key} removed`);
      for (const other of Object.keys(markers).filter(other => other !== key && !(format === 'json' && other === 'actions'))) {
        assert.ok(content.includes(markers[other]), `${format}: ${key} retains ${other}`);
      }
      assert.deepEqual(d.actions, baseline.actions);
      if (key === 'rules') for (const rule of ['Hard drop locks immediately;', '7-bag pieces;', 'Full rows clear simultaneously.']) assert.ok(!content.includes(rule));
      if (format === 'json') {
        assert.equal(Object.hasOwn(JSON.parse(d.state), 'available_actions'), key !== 'actions');
        assert.deepEqual(JSON.parse(d.state).board, JSON.parse(baseline.state).board);
      }
    }
    const sections = Object.fromEntries(Object.keys(STATE_SECTION_LABELS).map(key => [key, false]));
    const d = buildDecision(g, {format, sections, includeMetrics: true});
    assert.deepEqual(d.actions, baseline.actions);
    if (format === 'json') {
      const json = JSON.parse(d.state); assert.equal(json.context.length, 1); assert.match(json.context[0], /x=0..9.*uppercase=locked, lowercase=active/);
      assert.equal(json.board.length, 20); assert.ok(json.metrics); assert.equal(json.available_actions, undefined);
    } else { assert.match(d.state, /Board:/); assert.match(d.state, /Board metrics:/); }
  }
});

test('section configuration merges defaults, copies inputs and rejects invalid fields', () => {
  const sections = {history:false};
  const normalized = normalizeConfig({sections});
  assert.equal(normalized.sections.history, false); assert.equal(normalized.sections.rules, true);
  normalized.sections.history = true; assert.equal(sections.history, false);
  normalized.sections.rules = false; assert.equal(DEFAULT_CONFIG.sections.rules, true);
  for (const sections of [null, [], true, undefined, {history:'false'}, {rules:0}, {history:undefined}, {board:false}, {unknown:true}]) assert.throws(() => normalizeConfig({sections}));
});


test('JSON dict uses a field per enabled paragraph and keeps original paragraph content',()=>{
  const game=createGame('dict');
  const config={format:'json_object',sections:{rules:false,history:false,actions:false},includeMetrics:true};
  const result=buildDecision(game,config);
  assert.equal(typeof result.state,'object');
  assert.ok(!Array.isArray(result.state));
  assert.ok(!Object.hasOwn(result.state,'rules'));
  assert.ok(!Object.hasOwn(result.state,'history'));
  assert.ok(!Object.hasOwn(result.state,'actions'));
  assert.ok(!Object.hasOwn(result.state,'context'));
  assert.match(result.state.board,/width=10 height=20/);
  assert.match(result.state.stats,/cleared_lines=0/);
  assert.ok(result.actions.left);
  assert.equal(result.state.metrics.holes,0);
  const all=buildDecision(game,{format:'json_object'}).state;
  for(const [key,value] of Object.entries(result.state))if(key!=='metrics')assert.equal(value,all[key]);
  assert.deepEqual(Object.keys(buildDecision(game,{format:'json_object',sections:Object.fromEntries(Object.keys(DEFAULT_CONFIG.sections).map(key=>[key,false]))}).state),['board']);
  assert.equal(typeof buildDecision(game,{format:'json'}).state,'string');
});

test('outcome choices simulate exactly one engine step without mutating or exposing hidden state', () => {
  const compare = (game, action, expected = {}) => {
    const before = structuredClone(game);
    const choice = evaluateActions(game, {choiceMode:'outcome', legalOnly:false})[action];
    assert.deepEqual(game, before);
    const after = step(structuredClone(game), action);
    assert.deepEqual(choice.active_after, {...after.active, cells:cells(after)});
    for (const [key, field] of Object.entries({score_after:'score', cleared_lines_after:'lines', next_piece_type:null, hold_after:'hold', can_hold_after:'canHold', tick_after:'tick', gravity_beat_after:'gravityBeat', grounded_beats_after:'groundedBeats', lock_resets_after:'lockResets', game_over_after:'over', paused_after:'paused'})) {
      assert.equal(choice[key], field ? after[field] : after.next[0], `${action}: ${key}`);
    }
    assert.equal(choice.score_delta, after.score - game.score);
    assert.equal(choice.newly_cleared_lines, Math.max(0, after.lines - game.lines));
    assert.equal(choice.locked_pieces_delta, after.pieces - game.pieces);
    assert.equal(choice.operation, ACTIONS[action]);
    for (const [key, value] of Object.entries(expected)) assert.deepEqual(choice[key], value, `${action}: ${key}`);
    for (const key of ['rng','bag','next','board']) assert.ok(!Object.hasOwn(choice,key));
    return choice;
  };
  const g = createGame('outcomes'); g.active = {type:'O', x:3, y:0, rotation:0};
  compare(g, 'soft_drop', {score_delta:0, spawned_new_piece:false});
  compare(g, 'hard_drop', {score_delta:0, locked_pieces_delta:1, spawned_new_piece:true});
  g.active.y = 18;
  g.board[18].fill('J'); g.board[19].fill('J');
  for (const y of [18,19]) for (const x of [4,5]) g.board[y][x] = '.';
  const clear = compare(g, 'hard_drop', {newly_cleared_lines:2, score_delta:300, spawned_new_piece:true});
  assert.equal(clear.active_after.type,g.next[0]); assert.equal(clear.active_after.x,3); assert.equal(clear.active_after.y,0);

  const timed = createGame('timed'); timed.active = {type:'O',x:3,y:18,rotation:0};
  timed.gravityBeat=4; timed.groundedBeats=1; timed.lockResets=15;
  compare(timed,'left',{locked_pieces_delta:1, spawned_new_piece:true});
  const wall=createGame('wall'); wall.active={type:'I',x:-2,y:5,rotation:1}; wall.gravityBeat=4;
  const blocked=compare(wall,'left',{spawned_new_piece:false}); assert.equal(blocked.active_after.x,-2); assert.equal(blocked.active_after.y,6);
  compare(wall,'rotate_cw'); compare(wall,'rotate_ccw'); compare(wall,'rotate_180');
  const held=compare(wall,'hold',{spawned_new_piece:true,can_hold_after:false,hold_after:'I'}); assert.equal(held.active_after.type,wall.next[0]);
  wall.hold='T'; compare(wall,'hold',{spawned_new_piece:true,next_piece_type:wall.next[0]});
  wall.canHold=false; compare(wall,'hold',{spawned_new_piece:false});
  compare(wall,'pause',{paused_after:true,tick_after:0});
  wall.paused=true; compare(wall,'hold',{spawned_new_piece:false}); compare(wall,'resume',{paused_after:false});
  wall.score=400; wall.lines=3; wall.pieces=5;
  compare(wall,'restart',{reset_run:true,score_after:0,score_delta:-400,cleared_lines_after:0,newly_cleared_lines:0,locked_pieces_delta:-5,spawned_new_piece:true});
});

test('outcome mode is independent from state encoding, sections and candidate filtering', () => {
  const g=createGame('outcome-formats');
  assert.equal(normalizeConfig({format:'json_object'}).choiceMode,'description');
  for (const choiceMode of [undefined,null,true,'bad',{}]) assert.throws(()=>normalizeConfig({choiceMode}));
  for (const format of ['original','grid','coordinates','json','json_object']) {
    const config={format,choiceMode:'outcome',excludedActions:['hard_drop','restart']};
    const result=buildDecision(g,config);
    assert.deepEqual(Object.keys(result.actions),Object.keys(legalActions(g)).filter(id=>!config.excludedActions.includes(id)));
    assert.equal(typeof result.actions.left,'object');
    if (format==='json_object') assert.deepEqual(result.state.actions,result.actions);
    else if (format==='json') assert.deepEqual(JSON.parse(result.state).available_actions,result.actions);
    else { assert.match(result.state,/left: \{"immediate_gain":/); assert.doesNotMatch(result.state,/\[object Object\]/); }
    const hidden=buildDecision(g,{...config,sections:{actions:false}});
    assert.deepEqual(hidden.actions,result.actions);
    if(format==='json_object') assert.equal(hidden.state.actions,undefined);
    else if(format==='json') assert.equal(JSON.parse(hidden.state).available_actions,undefined);
    else assert.ok(!hidden.state.includes('Available actions:'));
  }
});


test('enhanced single-action choices distinguish actual effects from conditional landing benefits',()=>{
  const game=createGame('gap');game.active={type:'O',x:0,y:0,rotation:0};game.canHold=false;
  for(const y of [18,19])game.board[y]=['.','.',...Array(8).fill('J')];
  const before=structuredClone(game);
  const actions=evaluateActions(game,{choiceMode:'outcome'});
  const left=actions.left, drop=actions.hard_drop;
  assert.equal(left.score_delta,0);assert.equal(left.newly_cleared_lines,0);
  assert.equal(left.board_after.height_change,0);
  assert.equal(left.landing_preview.requires_additional_hard_drop,true);
  assert.equal(left.landing_preview.additional_cleared_lines,2);
  assert.equal(left.landing_preview.additional_score,300);
  assert.equal(left.landing_preview.board_after.maxHeight,0);
  assert.equal(drop.score_delta,0);assert.equal(drop.newly_cleared_lines,0);
  assert.equal(drop.board_after.holes_created,2);
  assert.equal(drop.landing_preview.requires_additional_hard_drop,false);
  assert.match(left.benefit,/No piece locked yet/);assert.match(drop.benefit,/placement committed/);
  assert.equal(actions.pause.landing_preview,null);assert.equal(actions.restart.landing_preview,null);
  assert.deepEqual(game,before);
  step(game,'left');assert.equal(game.lines,0);assert.equal(game.score,0);assert.equal(game.pieces,0);
  step(game,'hard_drop');assert.equal(game.lines,2);assert.equal(game.score,300);
});

test('two-step lookahead reveals a distant gap and exactly replays its conditional outcome', () => {
  const game=createGame('distant-gap'); game.active={type:'O',x:2,y:0,rotation:0}; game.canHold=false;
  for (const y of [18,19]) game.board[y]=['.','.',...Array(8).fill('J')];
  const before=structuredClone(game);
  const config={choiceMode:'outcome',lookaheadDepth:2};
  const choices=evaluateActions(game,config);
  const preview=choices.left.lookahead_preview;
  assert.equal(evaluateActions(game,{choiceMode:'outcome'}).left.lookahead_preview,undefined);
  assert.equal(evaluateActions(game,{...config,lookaheadDepth:1}).left.lookahead_preview.additional_cleared_lines,0);
  assert.deepEqual(preview.followup_actions,['left','left']);
  assert.equal(preview.additional_cleared_lines,2); assert.equal(preview.additional_score,300);
  assert.equal(preview.forecast_only,true); assert.equal(preview.requires_additional_hard_drop,true);
  const replay=step(structuredClone(game),'left'); const initial=structuredClone(replay);
  for (const action of preview.followup_actions) {
    assert.ok(legalActions(replay)[action]); step(replay,action);
  }
  if (preview.requires_additional_hard_drop) step(replay,'hard_drop');
  assert.equal(preview.additional_score,replay.score-initial.score);
  assert.equal(preview.additional_cleared_lines,replay.lines-initial.lines);
  assert.equal(preview.game_over,replay.over);
  assert.equal(replay.pieces,initial.pieces+1);
  const actual=buildDecision(replay,{format:'json_object',includeMetrics:true}).state.metrics;
  for (const key of ['columnHeights','holes','maxHeight','bumpiness']) assert.deepEqual(preview.board_after[key],actual[key]);
  assert.equal(preview.board_after.holes_created,0); assert.equal(preview.board_after.height_change,-2);
  assert.deepEqual(game,before);
  const excluded=evaluateActions(game,{...config,excludedActions:['left','hard_drop']});
  assert.equal(excluded.left,undefined);
  assert.equal(excluded.soft_drop.lookahead_preview.additional_cleared_lines,0);
  for (const choice of Object.values(excluded)) if (choice.lookahead_preview) {
    assert.ok(!choice.lookahead_preview.followup_actions.includes('left'));
    assert.equal(choice.lookahead_preview.hard_drop_excluded_from_choices,true);
  }
  assert.equal(choices.hard_drop.lookahead_preview,null);
  assert.equal(choices.pause.lookahead_preview,null); assert.equal(choices.restart.lookahead_preview,null);
  assert.deepEqual(evaluateActions(game,{lookaheadDepth:2}),evaluateActions(game,{}));
});

test('lookahead respects gravity locks, paused controls, hold replacement and strict depth bounds', () => {
  for (const lookaheadDepth of [-1,3,1.5,'2',null,true,undefined]) assert.throws(()=>normalizeConfig({lookaheadDepth}));
  for (const lookaheadDepth of [0,1,2]) assert.equal(normalizeConfig({lookaheadDepth}).lookaheadDepth,lookaheadDepth);
  const game=createGame('lock-stop'); game.active={type:'O',x:3,y:18,rotation:0}; game.lockResets=15;
  const config={choiceMode:'outcome',lookaheadDepth:2,legalOnly:false};
  const preview=evaluateActions(game,config).wait.lookahead_preview;
  // After wait the piece has one grounded beat; either lateral input locks it.
  assert.equal(preview.candidate_count,3);
  assert.ok(preview.followup_actions.length <= 1);
  game.active.x=1; // One more left reaches the edge and reduces surface bumpiness.
  const noDrop=evaluateActions(game,{...config,excludedActions:['right']}).left.lookahead_preview;
  assert.deepEqual(noDrop.followup_actions,['left']);
  assert.equal(noDrop.requires_additional_hard_drop,false);
  const after=step(structuredClone(game),'left');
  for (const action of noDrop.followup_actions) step(after,action);
  assert.equal(after.pieces,1);
  assert.equal(noDrop.piece_type,'O');
  game.groundedBeats=1;
  assert.equal(evaluateActions(game,config).left.lookahead_preview,null);
  game.paused=true;
  for (const choice of Object.values(evaluateActions(game,config))) assert.equal(choice.lookahead_preview,null);
  game.paused=false; game.over=true;
  for (const choice of Object.values(evaluateActions(game,config))) assert.equal(choice.lookahead_preview,null);
  const held=createGame('held'); held.hold='O';
  assert.equal(evaluateActions(held,config).hold.lookahead_preview.piece_type,'O');
});

test('row-placement mode is independent and validates macro history/exclusions',()=>{
  const game=createGame('42');game.history=['left_2'];
  assert.doesNotThrow(()=>validateGame(game));
  for(const format of ['original','grid','coordinates','json','json_object']){
    const d=buildDecision(game,{format,choiceMode:'row_placements',excludedActions:['right_2']});
    assert.ok(d.actions.left_2);assert.equal(d.actions.right_2,undefined);
    assert.deepEqual(d.actions.left_2.immediate_gain,{score:0,cleared_lines:0,game_over:false});
    if(format==='json_object')assert.deepEqual(d.state.actions,d.actions);
    else if(format==='json')assert.deepEqual(JSON.parse(d.state).available_actions,d.actions);
    else assert.ok(!d.state.includes('[object Object]'));
    const hidden=buildDecision(game,{format,choiceMode:'row_placements',sections:{actions:false}});
    assert.ok(hidden.actions.left_2);
  }
  assert.ok(!Object.hasOwn(buildDecision(game).actions,'left_2'));
});

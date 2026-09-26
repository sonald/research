import test from 'node:test';
import assert from 'node:assert/strict';
import { TypeSafeClient } from '@typesafe-ai/sdk';
import { createServer, DEFAULT_QUESTION } from './server.js';
import { ACTIONS, createGame, stateSections } from './engine.js';
import { buildDecision, evaluateActions, normalizeConfig } from './experiment.js';

async function serve(t, options) {
  const server = createServer(options);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}
function assertGain(value, expected) {
  const {summary,scenario,...numbers}=value;
  assert.deepEqual(numbers,expected);
  assert.equal(typeof summary,'string'); assert.ok(summary.length > 0);
}

const body = { game: createGame('42'), config: { format:'original', legalOnly:true, includeMetrics:false }, question: 'What action next?', model: 'jev-latest' };
const expected = buildDecision(body.game,body.config);
const post = (base, value = body, headers = {}) => fetch(`${base}/api/decide`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(value) });

test('server forwards exact text through the SDK, preserves answer, and rejects bad requests', async t => {
  let calls = 0;
  const answer = { type: 'choice', choice: Object.keys(ACTIONS)[0], probabilities: { [Object.keys(ACTIONS)[0]]: 1 }, confidence: 1 };
  const client = new TypeSafeClient({ apiKey: 'secret-for-test', baseURL: 'https://api.typesafe.ai', retry: { maxRetries: 0 }, logLevel: 'off', async fetch(url, options) {
    calls++;
    assert.equal(new URL(url).origin, 'https://api.typesafe.ai');
    const request = JSON.parse(options.body);
    assert.equal(request.state, expected.state);
    assert.equal(request.model, body.model);
    assert.deepEqual(request.questions.next_action, { type: 'choice', instructions: body.question, criteria: expected.actions });
    return Response.json({ answers: { next_action: answer }, model: 'jev-test', usage: { input_tokens: 10, output_tokens: 2 } });
  } });
  const base = await serve(t, { apiKey: 'secret-for-test', maxRequests: 1, client });
  for (const path of ['/.env', '/server.js', '/package.json', '/node_modules/@typesafe-ai/sdk/package.json']) {
    assert.equal((await fetch(base + path)).status, 404);
  }
  const config = await (await fetch(`${base}/api/config`)).json();
  assert.equal(config.configured, true);
  assert.equal(JSON.stringify(config).includes('secret-for-test'), false);
  assert.equal((await post(base, body, { Origin: 'https://attacker.invalid' })).status, 403);
  assert.equal((await post(base, body, { Host: 'attacker.invalid', Origin: 'http://attacker.invalid' })).status, 403);
  assert.equal((await post(base, body, { 'Content-Type': 'text/plain' })).status, 415);
  assert.equal((await post(base, { ...body, game: {} })).status, 400);
  assert.equal((await post(base, { ...body, model: 'https://attacker.invalid' })).status, 400);
  assert.equal((await post(base, { ...body, config: { ...body.config, legalOnly:'yes' } })).status, 400);
  const response = await post(base, body, { Origin: base });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.deepEqual(result.answer, answer);
  assert.equal(result.state, expected.state);
  assert.equal(result.question, body.question);
  assert.equal(result.model, 'jev-test');
  assert.deepEqual(result.actions, expected.actions);
  assert.deepEqual(result.config, normalizeConfig(body.config));
  assert.equal(Object.hasOwn(result.actions, 'resume'),false);
  assert.ok(result.latencyMs >= 0);
  assert.equal((await post(base)).status, 429);
  assert.equal(calls, 1);
});

test('missing keys and upstream failures never expose secrets or produce fallback moves', async t => {
  const empty = await serve(t, { apiKey: '' });
  assert.equal((await post(empty)).status, 503);
  const base = await serve(t, { apiKey: 'top-secret', client: { async systemOne() { throw new Error('top-secret upstream response'); } } });
  const response = await post(base);
  assert.equal(response.status, 502);
  assert.equal((await response.text()).includes('top-secret'), false);
});

test('only one in-flight request can spend API credits', async t => {
  let resolveRequest;
  let entered;
  const called = new Promise(resolve => { entered = resolve; });
  const base = await serve(t, { apiKey: 'test', client: { systemOne() {
    entered();
    return new Promise(resolve => { resolveRequest = resolve; });
  } } });
  const first = post(base);
  await called;
  assert.equal((await post(base)).status, 429);
  resolveRequest({ answers: { next_action: { choice: Object.keys(ACTIONS)[0] } } });
  assert.equal((await first).status, 200);
});


test('server calculates candidates itself and rejects out-of-set model choices', async t => {
  const game = createGame('42');
  game.active = {type:'O',x:3,y:0,rotation:0};
  game.canHold = false;
  let seen;
  const base = await serve(t,{apiKey:'test',client:{async systemOne(request){
    seen=request;
    return {answers:{next_action:{choice:'rotate_cw'}}};
  }}});
  const response=await post(base,{...body,game,config:{format:'json',legalOnly:true,includeMetrics:true},actions:ACTIONS,state:'ignore the real board'});
  assert.equal(response.status,502);
  assert.equal(Object.hasOwn(seen.questions.next_action.criteria,'rotate_cw'),false);
  assert.equal(Object.hasOwn(seen.questions.next_action.criteria,'hold'),false);
  assert.equal(Object.hasOwn(seen.questions.next_action.criteria,'resume'),false);
  assert.equal(typeof seen.state,'string');
  assert.ok(JSON.parse(seen.state));
  assert.notEqual(seen.state,'ignore the real board');
});

test('all-action baseline and every configured format reach SDK unchanged', async t => {
  let seen;
  const base=await serve(t,{apiKey:'test',client:{async systemOne(request){
    seen=request;return {answers:{next_action:{choice:'wait'}},model:'test'};
  }}});
  for(const format of ['original','grid','coordinates','json']) {
    for(const legalOnly of [true,false]) {
      const config={format,legalOnly,includeMetrics:true};
      const response=await post(base,{...body,config});
      assert.equal(response.status,200);
      const result=await response.json();
      const expected=buildDecision(body.game,config);
      assert.equal(seen.state,expected.state);
      assert.deepEqual(seen.questions.next_action.criteria,expected.actions);
      assert.deepEqual(result.config,normalizeConfig(config));
      if(!legalOnly)assert.deepEqual(result.actions,ACTIONS);
    }
  }
});

test('excluded actions never reach the SDK criteria or pass response validation', async t => {
  let seen;
  let answer = 'wait';
  const base = await serve(t, { apiKey: 'test', client: { async systemOne(request) {
    seen = request;
    return { answers: { next_action: { choice: answer } } };
  } } });
  for (const legalOnly of [true, false]) {
    const config = { ...body.config, legalOnly, excludedActions: ['hard_drop'] };
    const response = await post(base, { ...body, config });
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(Object.hasOwn(seen.questions.next_action.criteria, 'hard_drop'), false);
    assert.equal(Object.hasOwn(seen.questions.next_action.criteria, 'resume'), !legalOnly);
    assert.deepEqual(result.actions, seen.questions.next_action.criteria);
    assert.deepEqual(result.config, normalizeConfig(config));
    answer = 'hard_drop';
    assert.equal((await post(base, { ...body, config })).status, 502);
    answer = 'wait';
  }
});

test('empty candidate sets and invalid exclusions stop before the SDK', async t => {
  let calls = 0;
  const base = await serve(t, { apiKey: 'test', client: { async systemOne() {
    calls++;
    return { answers: { next_action: { choice: 'wait' } } };
  } } });
  for (const legalOnly of [true, false]) {
    const config = { ...body.config, legalOnly, excludedActions: Object.keys(ACTIONS) };
    assert.equal((await post(base, { ...body, config })).status, 422);
  }
  for (const excludedActions of ['hard_drop', null, {}, [null], [1], ['unknown'], ['__proto__']]) {
    assert.equal((await post(base, { ...body, config: { ...body.config, excludedActions } })).status, 400);
  }
  assert.equal(calls, 0);
});


test('disabled state paragraphs stay absent through the actual SDK request',async t=>{
  let seen;
  const base=await serve(t,{apiKey:'test',client:{async systemOne(request){seen=request;return {answers:{next_action:{choice:'left'}}};}}});
  for(const format of ['original','grid','coordinates','json']){
    const config={...body.config,format,sections:{history:false,rules:false,shapes:false,landing:false,actions:false}};
    const response=await post(base,{...body,config});assert.equal(response.status,200);
    const result=await response.json();
    assert.equal(seen.state,result.state);
    for(const marker of ['Recent actions','Rules:','7-bag pieces','Spawn matrices','landing_cells','Available actions:'])assert.ok(!seen.state.includes(marker));
    assert.match(seen.state,/cleared_lines=0/);assert.match(seen.state,/score=0/);
    assert.ok(seen.questions.next_action.criteria.left);
    assert.equal(result.config.sections.history,false);
  }
  const response=await post(base,{...body,config:{sections:{history:'false'}}});assert.equal(response.status,400);
});

test('JSON object state preserves enabled paragraphs through SDK serialization', async t => {
  let request;
  const client = new TypeSafeClient({ apiKey: 'test', retry: { maxRetries: 0 }, logLevel: 'off', async fetch(url, options) {
    request = JSON.parse(options.body);
    return Response.json({ answers: { next_action: { type: 'choice', choice: 'left', probabilities: { left: 1 }, confidence: 1 } }, model: 'jev-test' });
  } });
  const base = await serve(t, { apiKey: 'test', client });
  for (const includeMetrics of [false, true]) {
    const config = { ...body.config, format: 'json_object', includeMetrics, excludedActions: ['hard_drop'], sections: { history: false, rules: false, actions: false } };
    const response = await post(base, { ...body, config });
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(typeof request.state, 'object');
    assert.equal(Array.isArray(request.state), false);
    const paragraphs = stateSections(body.game);
    for (const [key, text] of Object.entries(paragraphs)) {
      if (['history', 'rules', 'actions'].includes(key)) assert.equal(Object.hasOwn(request.state, key), false);
      else assert.equal(request.state[key], text);
    }
    assert.match(request.state.stats, /score=0; cleared_lines=0; locked_pieces=0/);
    assert.equal(Object.hasOwn(request.state, 'metrics'), includeMetrics);
    if (includeMetrics) assert.deepEqual(request.state.metrics.columnHeights, Array(10).fill(0));
    assert.ok(request.questions.next_action.criteria.left);
    assert.equal(Object.hasOwn(request.questions.next_action.criteria, 'hard_drop'), false);
    assert.deepEqual(result.state, request.state);
    assert.deepEqual(result.actions, request.questions.next_action.criteria);
    assert.deepEqual(result.config, normalizeConfig(config));
  }
});


test('outcome objects are actual SDK criteria and independent of state paragraph visibility',async t=>{
  let wire;
  const client=new TypeSafeClient({apiKey:'test',baseURL:'https://api.typesafe.ai',retry:{maxRetries:0},logLevel:'off',async fetch(url,options){
    wire=JSON.parse(options.body);
    return Response.json({answers:{next_action:{type:'choice',choice:'left',confidence:1,probabilities:{left:1}}},model:'test'});
  }});
  const base=await serve(t,{apiKey:'test',client});
  const config={...body.config,format:'json_object',choiceMode:'outcome',excludedActions:['hard_drop'],sections:{actions:false}};
  const response=await post(base,{...body,config});assert.equal(response.status,200);
  const result=await response.json();
  const criteria=wire.questions.next_action.criteria;
  assert.equal(typeof criteria.left,'object');
  assertGain(criteria.left.immediate_gain,{score:0,cleared_lines:0,game_over:false});
  assert.equal(criteria.left.expected_gains[0].score,0);
  assert.equal(criteria.left.expected_gains[0].cleared_lines,0);
  assert.equal(typeof criteria.left.expected_gains[0].holes_delta,'number');
  assert.equal(Object.hasOwn(criteria.left,'followup'),false);
  assert.ok(!Object.hasOwn(criteria,'hard_drop'));
  assert.ok(!Object.hasOwn(wire.state,'actions'));
  assert.deepEqual(result.actions,criteria);
  assert.equal(result.config.choiceMode,'outcome');
  assert.equal((await post(base,{...body,config:{choiceMode:'unknown'}})).status,400);
});


test('bounded lookahead reaches SDK but does not change first-action choice contract',async t=>{
  let seen;
  const base=await serve(t,{apiKey:'test',client:{async systemOne(request){seen=request;return {answers:{next_action:{choice:'left'}}};}}});
  const config={format:'json_object',choiceMode:'outcome',lookaheadDepth:2,sections:{actions:false}};
  const response=await post(base,{...body,config});assert.equal(response.status,200);
  const result=await response.json();
  assert.equal(result.answer.choice,'left');assert.equal(result.config.lookaheadDepth,2);
  assert.ok(seen.questions.next_action.criteria.left.expected_gains[0]);
  assert.equal(Object.hasOwn(seen.questions.next_action.criteria.left,'followup'),false);
  assert.deepEqual(result.actions,seen.questions.next_action.criteria);
  for(const lookaheadDepth of [-1,3,1.5,'2',null])assert.equal((await post(base,{...body,config:{...config,lookaheadDepth}})).status,400);
});

test('row placement choices roundtrip through SDK as macro actions, never execute forecasts',async t=>{
  let wire;
  const client=new TypeSafeClient({apiKey:'test',baseURL:'https://api.typesafe.ai',retry:{maxRetries:0},logLevel:'off',async fetch(url,options){
    wire=JSON.parse(options.body);return Response.json({answers:{next_action:{type:'choice',choice:'left_2',probabilities:{left_2:1},confidence:1}},model:'test'});
  }});
  const base=await serve(t,{apiKey:'test',client});
  const config={format:'json_object',choiceMode:'row_placements',sections:{actions:false}};
  const response=await post(base,{...body,config});assert.equal(response.status,200);
  const result=await response.json();
  assert.equal(result.answer.choice,'left_2');
  assert.deepEqual(result.actions,wire.questions.next_action.criteria);
  const choice=result.actions.left_2;
  assertGain(choice.immediate_gain,{score:0,cleared_lines:0,game_over:false});
  assert.equal(choice.expected_gains[0].score,0);
  const raw=evaluateActions(body.game,config);
  for(const [action,value] of Object.entries(result.actions)) {
    assert.equal(value.expected_gains.length,raw[action].variants.length, action);
    for(const [index,gain] of value.expected_gains.entries()) {
      assert.equal(gain.score,raw[action].variants[index].score_delta);
      assert.equal(gain.cleared_lines,raw[action].variants[index].newly_cleared_lines);
      assert.equal(typeof gain.scenario,'string');
    }
  }
  assert.equal(Object.hasOwn(choice,'followup'),false);
  const excluded=await post(base,{...body,config:{...config,excludedActions:['left']}});
  assert.equal(excluded.status,502);
  assert.ok(!Object.keys(wire.questions.next_action.criteria).some(a=>a==='left'||a.startsWith('left_')));
  const macroGame=structuredClone(body.game);macroGame.history=['left_2'];
  assert.equal((await post(base,{...body,game:macroGame,config})).status,200);
});


test('compact expected gains reach the raw SDK wire without detailed simulation fields',async t=>{
  let wire;
  const client=new TypeSafeClient({apiKey:'test',retry:{maxRetries:0},logLevel:'off',async fetch(url,options){
    wire=JSON.parse(options.body);
    return Response.json({answers:{next_action:{type:'choice',choice:'left',probabilities:{left:1},confidence:1}},model:'test'});
  }});
  const base=await serve(t,{apiKey:'test',client});
  const game=createGame('compact-gap');game.active={type:'O',x:0,y:0,rotation:0};game.canHold=false;
  for(const y of [18,19])game.board[y]=['.','.',...Array(8).fill('J')];
  const forbidden=new Set(['board_after','landing_cells','active_after','variants','immediate_after','assumption','followup','followup_actions']);
  const check=value=>{
    if(value&&typeof value==='object')for(const [key,child] of Object.entries(value)){
      assert.ok(!forbidden.has(key),`unexpected detailed field: ${key}`);check(child);
    }
  };
  for(const choiceMode of ['outcome','row_placements'])for(const lookaheadDepth of [0,2]){
    const config={format:'json_object',choiceMode,lookaheadDepth,excludedActions:['hard_drop','rotate_cw'],sections:{actions:true}};
    const response=await post(base,{...body,game,config});assert.equal(response.status,200);
    const result=await response.json();
    const criteria=wire.questions.next_action.criteria;
    check(criteria);check(wire.state.actions);
    for(const [action,value] of Object.entries(criteria)) {
      assert.ok(Array.isArray(value.expected_gains), action);
      assert.ok(value.expected_gains.length > 0, action);
      assert.equal(Object.hasOwn(value,'expected_gain'),false);
      for(const gain of value.expected_gains) {
        assert.equal(typeof gain.scenario,'string'); assert.ok(gain.scenario.length > 0);
        assert.equal(typeof gain.summary,'string');
      }
    }
    assertGain(criteria.left.immediate_gain,{score:0,cleared_lines:0,game_over:false});
    assertGain(criteria.left.expected_gains[0],{score:300,cleared_lines:2,holes_delta:0,height_delta:-2,surface_roughness:0,game_over:false});
    assert.equal(Object.hasOwn(criteria.left,'followup'),false);
    assert.equal(criteria.hard_drop,undefined);assert.equal(criteria.rotate_cw,undefined);
    assert.deepEqual(result.actions,criteria);assert.deepEqual(wire.state.actions,criteria);
    assert.deepEqual(Object.keys(criteria.left).sort(),['expected_gains','immediate_gain']);
  }
});


test('default question asks only for the next action and guidance lives in state', async t => {
  assert.equal(DEFAULT_QUESTION,'Which action should the player take next?');
  let seen;
  const base=await serve(t,{apiKey:'test',client:{async systemOne(request){
    seen=request;return {answers:{next_action:{choice:'left'}}};
  }}});
  assert.equal((await (await fetch(`${base}/api/config`)).json()).question,DEFAULT_QUESTION);
  const response=await post(base,{...body,question:DEFAULT_QUESTION,config:{format:'json_object',choiceMode:'outcome',sections:{actions:false}}});
  assert.equal(response.status,200);
  assert.equal(seen.questions.next_action.instructions,DEFAULT_QUESTION);
  assert.match(seen.state.decision_guide,/immediate_gain/);
  assert.match(seen.state.decision_guide,/expected_gains/);
});

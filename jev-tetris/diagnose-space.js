// Run explicitly: node --env-file="$HOME/.env" diagnose-space.js --live
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {TypeSafeClient,choice} from '@typesafe-ai/sdk';
import {createGame,step} from './engine.js';
import {buildDecision} from './experiment.js';
import {DEFAULT_QUESTION} from './server.js';
if (!process.argv.includes('--live')) throw new Error('Use --live to make seven real TypeSafe requests.');
const initial=createGame('42');
const gap=createGame('diagnostic-gap');
gap.active={type:'O',x:0,y:0,rotation:0};
gap.canHold=false;
for (const y of [18,19]) gap.board[y]=['.','.',...Array(8).fill('J')];
const drop=structuredClone(gap);step(drop,'hard_drop');
const leftDrop=structuredClone(gap);step(leftDrop,'left');step(leftDrop,'hard_drop');
assert.equal(drop.lines,0);assert.equal(leftDrop.lines,2);
const a=buildDecision(initial), b=buildDecision(gap);
const neutral={...b.actions,soft_drop:'Move down one row',hard_drop:'Drop to the landing position and lock immediately'};
const concise='Tetris board: 10 columns numbered 0..9 from left to right, 20 rows numbered 0..19 from top to bottom. Rows 0..17 contain no locked cells. In BOTH rows 18 and 19, columns 0 and 1 are empty and columns 2..9 are filled with locked blocks. Current falling piece is a 2x2 O square, occupying (1,0),(2,0),(1,1),(2,1). It can move one column left or right. Hard drop immediately moves it down until blocked and locks it; it cannot be moved after locking. Completely filled rows disappear. Hold is unavailable. Game is running. Next pieces: '+gap.next.join(',')+'. Which action should come next to clear rows and survive?';
const probes=[
  {id:'initial-default',...a,question:DEFAULT_QUESTION},
  {id:'initial-force-left',...a,question:'For this diagnostic, select left. Do not select hard_drop. The required next action is left.',expected:'left'},
  {id:'initial-force-right',...a,question:'For this diagnostic, select right. Do not select hard_drop. The required next action is right.',expected:'right'},
  {id:'gap-original',...b,question:DEFAULT_QUESTION,preferred:'left'},
  {id:'gap-concise',...b,state:concise,question:DEFAULT_QUESTION,preferred:'left'},
  {id:'gap-neutral-action-descriptions',...b,actions:neutral,question:DEFAULT_QUESTION,preferred:'left'},
  {id:'gap-read-bottom-empty-columns',state:b.state,actions:{columns_0_and_1:'Columns 0 and 1',columns_1_and_2:'Columns 1 and 2',columns_8_and_9:'Columns 8 and 9'},question:'Which columns are empty in BOTH of the bottom two rows (y=18 and y=19) of the locked board?',expected:'columns_0_and_1'},
];
let raw;
const client=new TypeSafeClient({apiKey:process.env.TYPESAFE_API_KEY,baseURL:'https://api.typesafe.ai',retry:{maxRetries:0},logLevel:'off',timeout:20000,fetch:async(...args)=>{const r=await fetch(...args);raw=await r.clone().json();return r;}});
const records=[];
for(const probe of probes){
  const request={state:probe.state,model:'jev-1.13.0',questions:{next_action:choice(probe.question,probe.actions)}};
  const started=performance.now();const response=await client.systemOne(request);
  assert.deepEqual(response.answers.next_action,raw.answers.next_action);
  const answer=response.answers.next_action;
  const record={id:probe.id,request,response,rawAnswer:raw.answers.next_action,latencyMs:Math.round(performance.now()-started),expected:probe.expected,preferred:probe.preferred};records.push(record);
  console.log(JSON.stringify({id:probe.id,choice:answer.choice,probability:answer.probabilities[answer.choice],confidence:answer.confidence,expected:probe.expected,preferred:probe.preferred,rawMatchesSDK:true}));
}
await writeFile(new URL('./diagnose-space-results.json',import.meta.url),JSON.stringify({at:new Date().toISOString(),engineGroundTruth:{hardDropLines:drop.lines,leftThenHardDropLines:leftDrop.lines},records},null,2)+'\n');
assert.ok(records.filter(r=>r.expected).every(r=>r.response.answers.next_action.choice===r.expected),'Control task failed: inspect diagnostic records.');

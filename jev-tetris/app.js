import { ACTIONS, createGame, step, cells, ghostCells } from './engine.js';
import { DEFAULT_CONFIG, STATE_SECTION_LABELS, normalizeConfig, buildDecision } from './experiment.js';

const $ = id => document.getElementById(id);
const labels = {left:'左移',right:'右移',soft_drop:'软降',hard_drop:'硬降',rotate_cw:'顺时针',rotate_ccw:'逆时针',rotate_180:'旋转 180°',hold:'暂存 / 交换',wait:'等待',pause:'暂停',resume:'继续',restart:'重新开始'};
const shortcuts = {ArrowLeft:'left',a:'left',ArrowRight:'right',d:'right',ArrowDown:'soft_drop',s:'soft_drop',ArrowUp:'rotate_cw',w:'rotate_cw',x:'rotate_cw',z:'rotate_ccw',q:'rotate_180',' ':'hard_drop',c:'hold','.':'wait',r:'restart'};
const controls = [['left','←'],['right','→'],['soft_drop','↓'],['hard_drop','Space'],['rotate_cw','↑ / X'],['rotate_ccw','Z'],['rotate_180','Q'],['hold','C'],['wait','.']];
let game = createGame('42'), revision = 0, mode = 'manual', busy = false, running = false, runToken = 0;
let configured = false, records = [], decisions = 0;
let experiment = {...DEFAULT_CONFIG};
try { const saved=localStorage.getItem('jev-tetris-experiment'); if(saved)experiment=normalizeConfig(JSON.parse(saved)); } catch {}
$('state-format').value=experiment.format;
$('action-filter').value=experiment.legalOnly?'legal':'all';
$('choice-mode').value=experiment.choiceMode;
$('include-metrics').checked=experiment.includeMetrics;
for(const [action,label] of Object.entries(labels)) {
  const row=document.createElement('label');row.className='experiment-check';
  const input=document.createElement('input');input.type='checkbox';input.value=action;input.checked=experiment.excludedActions.includes(action);
  row.append(input,document.createTextNode(`${label} (${action})`));$('excluded-actions').append(row);
}
for(const [section,label] of Object.entries(STATE_SECTION_LABELS)) {
  const row=document.createElement('label');row.className='experiment-check';
  const input=document.createElement('input');input.type='checkbox';input.value=section;input.checked=experiment.sections[section];
  row.append(input,document.createTextNode(label));$('state-sections').append(row);
}
function stateText(state) { return typeof state==='string'?state:JSON.stringify(state,null,2); }
function currentDecision() { return buildDecision(game,experiment); }
function clearDistribution() {
  $('distribution').className='empty';$('distribution').textContent='配置已更新，请重新请求动作分布。';
  $('confidence').textContent='';$('latency').textContent='';
}
const boardCells = Array.from({length:200}, () => { const el=document.createElement('div'); el.className='cell'; $('board').append(el); return el; });
for (const [action,key] of controls) {
  const button=document.createElement('button'); button.dataset.action=action; button.title=ACTIONS[action];
  const kbd=document.createElement('kbd'); kbd.textContent=key; button.append(kbd,document.createTextNode(labels[action]));
  button.addEventListener('click',()=>human(action)); $('game-controls').append(button);
}
function preview(type) {
  const box=document.createElement('div'); box.className='mini';
  if (!type) {box.textContent='空'; return box;}
  const grid=document.createElement('div');grid.className='mini-grid'; grid.setAttribute('aria-label',type+' 方块');grid.role='img';
  for(const cell of cells({active:{type,x:0,y:0,rotation:0}})) {
    const el=document.createElement('span');el.className='mini-cell';el.style.gridColumn=cell.x+1;el.style.gridRow=cell.y+1;el.style.background=`var(--${type})`;grid.append(el);
  }
  box.append(grid);return box;
}
function status(text,notice) { $('status').textContent=text;if(notice!==undefined)$('notice').textContent=notice; }
function updateButtons() {
  const noCandidates=!Object.keys(currentDecision().actions).length;
  $('ask').disabled=busy||!configured||noCandidates;
  $('ask').querySelector('small').textContent=$('observe-only').checked?'仅询问，保留当前棋盘':'让模型选择并执行下一个动作';
  $('auto').disabled=!configured||busy&&!running||$('observe-only').checked||noCandidates;
  $('auto').firstChild.textContent=running?'■ 停止运行':'▶▶ 连续运行';
  $('auto').querySelector('small').textContent=running?'立即停止，丢弃尚未返回的动作':'让模型持续进行游戏';
  $('manual-mode').setAttribute('aria-pressed',String(mode==='manual'));
  $('jev-mode').setAttribute('aria-pressed',String(mode==='jev'));
  $('realtime').disabled=mode==='jev';
}
function render() {
  boardCells.forEach((el,i)=>{el.className='cell';el.style.removeProperty('--color'); const type=game.board[Math.floor(i/10)][i%10]; if(type!=='.'){el.classList.add('block');el.style.setProperty('--color',`var(--${type})`);}});
  for(const {x,y} of ghostCells(game))if(y>=0&&y<20)boardCells[y*10+x].classList.add('ghost');
  for(const {x,y,type} of cells(game))if(y>=0&&y<20&&x>=0&&x<10){const el=boardCells[y*10+x];el.className='cell block';el.style.setProperty('--color',`var(--${type})`);}
  $('board').setAttribute('aria-label',`俄罗斯方块棋盘，当前 ${game.active.type} 方块，位置 ${game.active.x},${game.active.y}，得分 ${game.score}，消行 ${game.lines}`);
  $('score').textContent=String(game.score).padStart(4,'0');$('lines').textContent=game.lines;$('pieces').textContent=game.pieces;$('beat').textContent=`BEAT ${game.tick}`;
  $('hold').replaceWith(Object.assign(preview(game.hold),{id:'hold'}));$('hold').title=game.canHold?'可以暂存':'本方块已暂存';
  $('next').replaceChildren(...game.next.map(preview));
  $('overlay').hidden=!game.paused&&!game.over;$('overlay').textContent=game.over?'GAME OVER\n按 R 重新开始':'已暂停\n按 P 继续';
  $('pause').textContent=game.paused?'继续 P':'暂停 P';const input=currentDecision();const renderedState=stateText(input.state);$('state').textContent=renderedState;
  $('candidate-list').textContent=Object.keys(input.actions).length?`当前候选 ${Object.keys(input.actions).length} 个：${Object.keys(input.actions).map(a=>labels[a]).join('、')}`:'没有可选动作：请取消排除至少一个当前可用的动作。';
  $('choices').textContent=JSON.stringify(input.actions,null,2);
  $('state-label').textContent=`下一次请求的 state · ${typeof input.state==='string'?'文本':'JSON 对象'} · ${renderedState.length} 字符`;
  $('budget').textContent=`${decisions} 次 Jev 决策`;updateButtons();
}
function record(item) {
  records.push({at:new Date().toISOString(),...item});$('log-count').textContent=records.length;
  if(records.length===1)$('logs').replaceChildren();
  const row=document.createElement('details');row.className='log';const summary=document.createElement('summary');
  summary.textContent=`${String(records.length).padStart(3,'0')} · ${item.source} · ${labels[item.action]||item.action||'请求失败'}${item.discarded?' · 已丢弃':item.error?' · 失败':item.executed===false?' · 仅观察':''}${item.config?' · '+item.config.format:''}`;
  const detail=document.createElement('pre');detail.textContent=JSON.stringify(records.at(-1),null,2);row.append(summary,detail);$('logs').prepend(row);
  // ponytail: keep 100 rendered entries; complete JSON stays in memory for export.
  while($('logs').children.length>100)$('logs').lastChild.remove();
}
function stop(message) {
  running=false;runToken++;revision++;
  if(message)status(message,'尚未返回的模型动作不会再执行。可以继续手动操作。');updateButtons();
}
function setMode(value) {
  stop();mode=value;$('realtime').checked=false;updateButtons();
  status(value==='jev'?'Jev 就绪':'手动控制','一次请求，一个动作。等待模型时棋盘冻结。');
}
function apply(action,source,extra={}) {
  const input=currentDecision(), gameBefore=structuredClone(game);step(game,action);revision++;
  record({source,action,...input,gameBefore,gameAfter:structuredClone(game),executed:true,result:currentDecision().state,...extra});render();
}
function human(action) {
  stop();apply(action,'Human');
  status(game.over?'游戏结束':game.paused?'已暂停':'手动操作',`已执行：${labels[action]}。${busy?'先前的 Jev 请求会被丢弃。':'可以让 Jev 从当前状态接手。'}`);
}
function drawDistribution(answer,actions) {
  $('distribution').className='';$('distribution').replaceChildren();
  const sorted=Object.keys(actions).sort((a,b)=>(answer.probabilities?.[b]||0)-(answer.probabilities?.[a]||0));
  for(const action of sorted) {
    const value=answer.probabilities?.[action];const row=document.createElement('div');row.className='probability'+(action===answer.choice?' winner':'');
    const label=document.createElement('span');label.textContent=labels[action];const track=document.createElement('div');track.className='track';
    const fill=document.createElement('div');fill.className='fill';fill.style.width=`${Number.isFinite(value)?Math.max(0,Math.min(100,value*100)):0}%`;track.append(fill);
    const percent=document.createElement('span');percent.className='value';percent.textContent=Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'—';row.append(label,track,percent);$('distribution').append(row);
  }
  $('confidence').textContent=`选择：${labels[answer.choice]} · 置信度 ${Number.isFinite(answer.confidence)?(answer.confidence*100).toFixed(1)+'%':'未返回'} · 对应最近一次请求的棋盘`;
}
async function decide() {
  if(busy||!configured)return false;
  const {state,actions,config}=currentDecision();
  if(!Object.keys(actions).length){running=false;status('没有可选动作','请取消排除至少一个当前可用的动作。');updateButtons();return false;}
  busy=true;const version=revision;const gameBefore=structuredClone(game);const execute=!$('observe-only').checked;const question=$('question').value;const model=$('model').value.trim();
  status('Jev 正在选择…','棋盘已冻结。模型只收到右侧预览的 state 和动作空间。');updateButtons();
  let success=false;
  try {
    const response=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({game:gameBefore,config,question,model}),signal:AbortSignal.timeout(30000)});
    const result=await response.json();if(!response.ok)throw new Error(result.error||'模型请求失败');
    if(!result.actions||!Object.hasOwn(actions,result.answer?.choice)||!Object.hasOwn(result.actions,result.answer.choice))throw new Error('服务端返回了候选之外的动作');
    decisions++;
    const evidence={state:result.state,actions:result.actions,config:result.config,gameBefore,question,requestedModel:model,model:result.model,answer:result.answer,usage:result.usage,latencyMs:result.latencyMs};
    if(version!==revision){record({source:'Jev',action:result.answer.choice,...evidence,executed:false,discarded:true});status('已丢弃过期决策','棋盘或运行状态已改变，该动作没有执行。');}
    else {
      if(execute)apply(result.answer.choice,'Jev',evidence);
      else record({source:'Jev',action:result.answer.choice,...evidence,executed:false,gameAfter:structuredClone(game),result:state});
      drawDistribution(result.answer,result.actions);$('latency').textContent=`${(result.latencyMs/1000).toFixed(2)}s`;
      $('model-footer').textContent=`TypeSafe SDK / ${result.model}`;
      status(`${execute?'已执行':'仅观察'}：${labels[result.answer.choice]}`,`模型 ${result.model} · 输入 ${result.usage?.input_tokens??'—'} tokens · 输出 ${result.usage?.output_tokens??'—'} tokens`);
      success=execute&&!game.over&&!game.paused&&result.answer.choice!=='restart';
      if(!success&&running){running=false;status('连续运行已停止',`模型选择 ${labels[result.answer.choice]}，或对局已结束。记录已保留。`);}
    }
  } catch(error) {
    record({source:'Jev',state,actions,config,gameBefore,question,model,executed:false,error:error.message});running=false;status('请求失败',error.message);
  } finally {busy=false;render();}
  return success;
}
$('ask').addEventListener('click',async()=>{setMode('jev');await decide();});
$('auto').addEventListener('click',async()=>{
  if(running){stop('已停止连续运行');return;}
  setMode('jev');const limit=Number($('limit').value);
  if(!Number.isInteger(limit)||limit<1||limit>500){status('步数无效','请输入 1–500 的整数。');return;}
  running=true;const token=++runToken;updateButtons();
  let completed=0;
  for(let i=0;i<limit&&running&&token===runToken;i++) {
    if(!await decide())break;
    completed++;
    if(i+1<limit&&running&&token===runToken)await new Promise(resolve=>setTimeout(resolve,400));
  }
  if(token===runToken){running=false;if(completed===limit)status('本轮运行结束','已达到步数上限或运行被中止。可导出完整实验记录。');updateButtons();}
});
$('manual-mode').addEventListener('click',()=>setMode('manual'));$('jev-mode').addEventListener('click',()=>setMode('jev'));
$('pause').addEventListener('click',()=>human(game.paused?'resume':'pause'));$('restart').addEventListener('click',()=>human('restart'));
$('new-seed').addEventListener('click',()=>{stop();const old=currentDecision();const gameBefore=structuredClone(game);game=createGame($('seed').value||'42');revision++;record({source:'Human',action:'restart',...old,gameBefore,gameAfter:structuredClone(game),executed:true,result:currentDecision().state,newSeed:game.seed});render();status('新对局已开始',`种子 ${game.seed}。相同种子与相同动作序列可重现对局。`);});
$('realtime').addEventListener('change',()=>{stop();status($('realtime').checked?'自动时钟运行中':'逐拍模式','手动游玩使用同一套引擎。Jev 接手时会关闭自动时钟。');});
for(const id of ['state-format','action-filter','choice-mode','include-metrics','observe-only','excluded-actions','state-sections'])$(id).addEventListener('change',()=>{
  stop();$('realtime').checked=false;
  experiment=normalizeConfig({format:$('state-format').value,choiceMode:$('choice-mode').value,legalOnly:$('action-filter').value==='legal',includeMetrics:$('include-metrics').checked,sections:Object.fromEntries([...$('state-sections').querySelectorAll('input')].map(input=>[input.value,input.checked])),excludedActions:[...$('excluded-actions').querySelectorAll('input:checked')].map(input=>input.value)});
  try {localStorage.setItem('jev-tetris-experiment',JSON.stringify(experiment));} catch {}
  clearDistribution();render();status('实验配置已更新','棋盘保持不变；下一次请求使用新配置。正在等待的旧决策将被丢弃。');
});
for(const id of ['model','question'])$(id).addEventListener('input',()=>stop());
setInterval(()=>{if($('realtime').checked&&mode==='manual'&&!busy&&!running&&!game.paused&&!game.over)apply('wait','Clock');},150);
document.addEventListener('keydown',event=>{
  if(event.ctrlKey||event.metaKey||event.altKey||/^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)||event.target.isContentEditable)return;
  if(event.target.tagName==='BUTTON'&&[' ','Enter'].includes(event.key))return;
  const key=event.key.length===1?event.key.toLowerCase():event.key;
  const action=key==='p'?(game.paused?'resume':'pause'):shortcuts[key];if(!action)return;event.preventDefault();
  if(event.repeat&&!['left','right','soft_drop'].includes(action))return;human(action);
});
for(const tab of ['state','log'])$(tab+'-tab').addEventListener('click',()=>{for(const other of ['state','log']){$(other+'-tab').setAttribute('aria-selected',String(other===tab));$(other+'-panel').hidden=other!==tab;}});
$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(stateText(currentDecision().state));$('copy').textContent='已复制';setTimeout(()=>$('copy').textContent='复制',1200);}catch{status('复制不可用','请在 STATE 面板中选中文本复制。');}});
$('export').addEventListener('click',()=>{
  const payload={format:'jev-tetris-v2',exportedAt:new Date().toISOString(),seed:game.seed,actions:ACTIONS,question:$('question').value,model:$('model').value,decisions,config:experiment,observeOnly:$('observe-only').checked,finalState:currentDecision().state,finalGame:game,records};
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`jev-tetris-${new Date().toISOString().replaceAll(':','-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
try {const response=await fetch('/api/config');if(!response.ok)throw new Error('配置加载失败');const config=await response.json();configured=config.configured;$('model').value=config.model;$('question').value=config.question;$('model-footer').textContent=`TypeSafe SDK / ${config.model}`;status(configured?'准备就绪':'尚未配置密钥',configured?'手动操作，或点击「Jev 下一步」让模型接手。':'服务端设置 TYPESAFE_API_KEY 后重启。手动游戏仍然可用。');updateButtons();}
catch(error){status('服务不可用',error.message);}

const NS = "http://www.w3.org/2000/svg";

const palette = {
  ink: "#15171d",
  muted: "#626977",
  line: "#d9dee9",
  blue: "#2457e6",
  green: "#00a676",
  amber: "#f2a51a",
  coral: "#e45062",
  violet: "#6d4aff",
  panel: "#ffffff",
  softBlue: "#eef3ff",
  softGreen: "#e9fbf3",
  softAmber: "#fff7e3",
  softCoral: "#fff0f2",
};

function svg(tag, attrs = {}, children = []) {
  const el = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") {
      el.textContent = value;
    } else if (key === "className") {
      el.setAttribute("class", value);
    } else {
      el.setAttribute(key, value);
    }
  }
  for (const child of children) el.appendChild(child);
  return el;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function text(x, y, value, className = "svg-label", anchor = "start") {
  return svg("text", { x, y, className, "text-anchor": anchor, text: value });
}

function rect(x, y, width, height, fill, stroke = palette.line, rx = 8) {
  return svg("rect", { x, y, width, height, rx, fill, stroke, "stroke-width": 2 });
}

function line(x1, y1, x2, y2, stroke = palette.line, width = 2) {
  return svg("line", {
    x1,
    y1,
    x2,
    y2,
    stroke,
    "stroke-width": width,
    "stroke-linecap": "round",
  });
}

function arrow(x1, y1, x2, y2, stroke = palette.ink) {
  return svg("path", {
    d: `M${x1},${y1} L${x2},${y2}`,
    stroke,
    "stroke-width": 3,
    "stroke-linecap": "round",
    markerEnd: "url(#arrow)",
    fill: "none",
  });
}

function addDefs(root) {
  const defs = svg("defs");
  const marker = svg("marker", {
    id: "arrow",
    markerWidth: 10,
    markerHeight: 10,
    refX: 8,
    refY: 3,
    orient: "auto",
    markerUnits: "strokeWidth",
  });
  marker.appendChild(svg("path", { d: "M0,0 L0,6 L9,3 z", fill: palette.ink }));
  defs.appendChild(marker);
  root.appendChild(defs);
}

const state = {
  stage: "normal",
  layers: 8,
  spanEnd: 6,
  preserveRng: true,
  code: "flatten",
};

function layerFill(index, stage, spanEnd) {
  const inside = index >= 2 && index <= spanEnd;
  if (stage === "normal") return palette.softAmber;
  if (stage === "checkpoint" && inside) return palette.softBlue;
  if (stage === "backward" && inside) return palette.softGreen;
  return "#f7f8fb";
}

function renderFlow() {
  const root = document.querySelector("#flowSvg");
  clear(root);
  addDefs(root);
  const layers = state.layers;
  const spanEnd = Math.min(state.spanEnd, layers);
  const stage = state.stage;
  const startX = 70;
  const gap = 20;
  const w = Math.min(72, (760 - gap * (layers - 1)) / layers);
  const y = 168;

  root.appendChild(text(54, 58, "训练 step 中的一段网络", "svg-label"));
  root.appendChild(
    text(
      54,
      86,
      stage === "normal"
        ? "普通 forward：每层把 backward 要用的激活留下"
        : stage === "checkpoint"
          ? "checkpoint forward：区间内部不存激活，只存边界输入"
          : "backward：恢复输入与 RNG，重跑 checkpoint 区间生成局部 graph",
      "svg-small",
    ),
  );

  root.appendChild(rect(52, 118, 810, 238, "#fff", palette.line, 8));
  root.appendChild(text(82, 142, "x", "svg-label"));
  root.appendChild(text(820, 142, "loss", "svg-label", "end"));

  for (let i = 1; i <= layers; i += 1) {
    const x = startX + (i - 1) * (w + gap);
    const inside = i >= 2 && i <= spanEnd;
    const fill = layerFill(i, stage, spanEnd);
    root.appendChild(rect(x, y, w, 82, fill, inside ? palette.blue : palette.line, 8));
    root.appendChild(text(x + w / 2, y + 34, `L${i}`, "svg-label", "middle"));
    root.appendChild(text(x + w / 2, y + 58, "op", "svg-small", "middle"));
    if (i < layers) root.appendChild(arrow(x + w + 4, y + 41, x + w + gap - 4, y + 41));

    const shouldSave =
      stage === "normal" || (stage === "checkpoint" && !inside) || (stage === "backward" && inside);
    const saveColor =
      stage === "normal" ? palette.amber : stage === "checkpoint" ? palette.blue : palette.green;
    if (shouldSave) {
      root.appendChild(rect(x + 10, y + 102, w - 20, 40, saveColor, saveColor, 5));
      root.appendChild(text(x + w / 2, y + 127, "save", "svg-small", "middle"));
    } else {
      root.appendChild(
        svg("path", {
          d: `M${x + 14},${y + 122} L${x + w - 14},${y + 122}`,
          stroke: palette.coral,
          "stroke-width": 4,
          "stroke-linecap": "round",
        }),
      );
    }
  }

  const bracketX1 = startX + (w + gap);
  const bracketX2 = startX + (spanEnd - 1) * (w + gap) + w;
  root.appendChild(
    svg("path", {
      d: `M${bracketX1},${y - 28} L${bracketX2},${y - 28}`,
      stroke: palette.blue,
      "stroke-width": 4,
      "stroke-linecap": "round",
    }),
  );
  root.appendChild(text((bracketX1 + bracketX2) / 2, y - 42, "checkpoint 区间", "svg-small", "middle"));

  const memorySaved =
    stage === "normal" ? layers : stage === "checkpoint" ? layers - (spanEnd - 1) + 1 : spanEnd - 1;
  const compute = stage === "backward" ? `${(1 + (spanEnd - 1) / layers).toFixed(2)}x` : "1.0x";
  document.querySelector("#savedCount").textContent = `${memorySaved} tensors`;
  document.querySelector("#computeCost").textContent = compute;
  document.querySelector("#stageAction").textContent =
    stage === "normal" ? "保存所有激活" : stage === "checkpoint" ? "只留边界" : "重算区间";

  root.appendChild(text(54, 410, "forward saved tensor 数量示意", "svg-small"));
  for (let i = 0; i < layers; i += 1) {
    const on = i < memorySaved;
    root.appendChild(
      rect(54 + i * 34, 430, 24, 42, on ? palette.green : "#edf0f6", "none", 4),
    );
  }
  root.appendChild(text(54, 494, "绿色越少，前向峰值激活越低；但 backward 可能多跑一段 forward。", "svg-small"));
}

function renderTape() {
  const root = document.querySelector("#tapeSvg");
  clear(root);
  addDefs(root);
  root.appendChild(text(42, 50, "默认 eager autograd", "svg-label"));
  const ops = [
    ["linear", palette.blue],
    ["gelu", palette.green],
    ["dropout", palette.amber],
    ["linear", palette.violet],
    ["loss", palette.coral],
  ];
  ops.forEach(([name, color], i) => {
    const x = 64 + i * 120;
    root.appendChild(rect(x, 156, 88, 68, "#fff", color, 8));
    root.appendChild(text(x + 44, 197, name, "svg-small", "middle"));
    if (i < ops.length - 1) root.appendChild(arrow(x + 92, 190, x + 116, 190));
    if (i < ops.length - 1) {
      root.appendChild(rect(x + 24, 256, 40, 88 + i * 10, color, color, 4));
      root.appendChild(line(x + 44, 224, x + 44, 256, color, 2));
    }
  });
  root.appendChild(text(60, 110, "forward 越往后，保存的激活越多", "svg-small"));
  root.appendChild(text(60, 386, "这些保存项会等 backward 反向走到对应算子时才释放", "svg-small"));
}

function renderRecompute() {
  const root = document.querySelector("#recomputeSvg");
  clear(root);
  addDefs(root);
  root.appendChild(text(42, 48, "checkpoint 的时间线", "svg-label"));
  const rows = [
    ["1 forward", "no_grad 跑区间", palette.blue, 94],
    ["2 save", "只保存 x / RNG", palette.amber, 172],
    ["3 backward", "收到 grad_y", palette.coral, 250],
    ["4 recompute", "enable_grad 重跑区间", palette.green, 328],
  ];
  rows.forEach(([label, desc, color, y], i) => {
    root.appendChild(rect(54, y, 156, 48, color, color, 6));
    root.appendChild(text(132, y + 30, label, "svg-small", "middle"));
    root.appendChild(rect(250, y, 340, 48, i === 3 ? palette.softGreen : "#fff", palette.line, 6));
    root.appendChild(text(270, y + 30, desc, "svg-small"));
    if (i < rows.length - 1) root.appendChild(arrow(132, y + 52, 132, y + 74));
  });
  root.appendChild(
    svg("path", {
      d: "M590,352 C674,318 674,112 220,118",
      fill: "none",
      stroke: palette.green,
      "stroke-width": 4,
      "stroke-linecap": "round",
      markerEnd: "url(#arrow)",
    }),
  );
  root.appendChild(text(430, 112, "同一个 run_fn 再跑一次", "svg-small"));
}

function seededMask(seed, n) {
  let x = seed;
  const values = [];
  for (let i = 0; i < n; i += 1) {
    x = (1103515245 * x + 12345) % 2147483648;
    values.push(x / 2147483648 > 0.5);
  }
  return values;
}

function renderRng() {
  const root = document.querySelector("#rngSvg");
  clear(root);
  const maskA = seededMask(2024, 36);
  const maskB = state.preserveRng ? seededMask(2024, 36) : seededMask(3031, 36);
  const drawGrid = (x0, label, mask) => {
    root.appendChild(text(x0, 56, label, "svg-label"));
    mask.forEach((on, i) => {
      const x = x0 + (i % 6) * 34;
      const y = 94 + Math.floor(i / 6) * 34;
      root.appendChild(rect(x, y, 24, 24, on ? palette.green : "#edf0f6", "none", 4));
    });
  };
  drawGrid(80, "forward mask", maskA);
  drawGrid(430, "recompute mask", maskB);
  const same = maskA.every((v, i) => v === maskB[i]);
  root.appendChild(
    rect(
      288,
      142,
      126,
      54,
      same ? palette.softGreen : palette.softCoral,
      same ? palette.green : palette.coral,
      8,
    ),
  );
  root.appendChild(text(351, 175, same ? "一致" : "漂移", "svg-label", "middle"));
  document.querySelector("#rngCaption").textContent = same
    ? "保存 RNG 时，两次 mask 对齐，梯度对应同一次 forward。"
    : "关闭 RNG 保存后，重算会消耗新的随机数；dropout mask 漂移，梯度不再对应原 forward。";
}

const snippets = {
  flatten: `input_spec, tensor_inputs = _flatten_tensors((args, kwargs))

# autograd.Function 只真正关心 tensor leaves。
# dict/list/tuple 和 Python 常量留在 spec 里，
# backward 重算前再按同样结构拼回去。`,
  forward: `ctx.save_for_backward(*tensor_inputs)
ctx.forward_rng_state = _capture_rng_state(device_types)

with torch.no_grad():
    outputs = run_fn(*args, **kwargs)

# checkpoint 区域内部没有 autograd graph，
# 所以内部激活不会被保存到 forward 峰值里。`,
  backward: `detached = x.detach().requires_grad_(True)

with torch.enable_grad():
    recomputed = run_fn(*args, **kwargs)

torch.autograd.backward(recomputed, grad_outputs)

# detached.grad 就是要返回给 checkpoint 边界的输入梯度；
# 参数梯度会由重算 graph 自然累积到 module.parameters()。`,
  rng: `caller_state = capture_rng()
set_rng_state(forward_state)

try:
    recompute()
finally:
    set_rng_state(caller_state)

# 重算要复现 forward 内部随机算子，
# 但不能让 backward 额外消耗全局 RNG。`,
};

function renderCodeWindow() {
  document.querySelector("#codeWindow").textContent = snippets[state.code];
}

function renderPytree() {
  const root = document.querySelector("#pytreeSvg");
  clear(root);
  addDefs(root);
  root.appendChild(text(42, 48, "Python 结构和 tensor leaves 分离", "svg-label"));
  const left = [
    ["args", 70, 96, palette.blue],
    ["payload['x']", 94, 178, palette.green],
    ["kwargs.bias", 260, 178, palette.green],
    ["scale=1.25", 426, 178, palette.amber],
  ];
  left.forEach(([label, x, y, color]) => {
    root.appendChild(rect(x, y, 128, 46, "#fff", color, 6));
    root.appendChild(text(x + 64, y + 29, label, "svg-small", "middle"));
  });
  root.appendChild(arrow(134, 142, 134, 176));
  root.appendChild(arrow(134, 142, 324, 176));
  root.appendChild(arrow(134, 142, 490, 176));
  root.appendChild(rect(110, 292, 202, 64, palette.softGreen, palette.green, 8));
  root.appendChild(text(211, 320, "tensor_inputs", "svg-label", "middle"));
  root.appendChild(text(211, 344, "[x, bias]", "svg-small", "middle"));
  root.appendChild(rect(390, 292, 202, 64, palette.softAmber, palette.amber, 8));
  root.appendChild(text(491, 320, "tree spec", "svg-label", "middle"));
  root.appendChild(text(491, 344, "dict/list/const 形状", "svg-small", "middle"));
  root.appendChild(arrow(158, 226, 202, 288, palette.green));
  root.appendChild(arrow(324, 226, 234, 288, palette.green));
  root.appendChild(arrow(490, 226, 490, 288, palette.amber));
}

function renderHooks() {
  const root = document.querySelector("#hooksSvg");
  clear(root);
  addDefs(root);
  root.appendChild(text(46, 48, "non-reentrant checkpoint 的 holder 协议", "svg-label"));
  const nodes = [
    ["forward op", 64, 112, palette.blue],
    ["pack hook", 246, 112, palette.amber],
    ["holder", 428, 112, palette.violet],
    ["backward unpack", 246, 256, palette.coral],
    ["recompute + early stop", 428, 256, palette.green],
  ];
  nodes.forEach(([label, x, y, color]) => {
    root.appendChild(rect(x, y, 148, 62, "#fff", color, 8));
    root.appendChild(text(x + 74, y + 37, label, "svg-small", "middle"));
  });
  root.appendChild(arrow(214, 143, 244, 143));
  root.appendChild(arrow(396, 143, 426, 143));
  root.appendChild(arrow(502, 178, 322, 252, palette.coral));
  root.appendChild(arrow(396, 287, 426, 287, palette.green));
  root.appendChild(text(66, 214, "大激活不常驻", "svg-small"));
  root.appendChild(text(428, 214, "需要时才重算", "svg-small"));
  root.appendChild(
    svg("path", {
      d: "M574,287 C660,256 660,138 578,138",
      fill: "none",
      stroke: palette.green,
      "stroke-width": 4,
      "stroke-linecap": "round",
      markerEnd: "url(#arrow)",
    }),
  );
}

function renderTradeoff() {
  const root = document.querySelector("#tradeoffSvg");
  clear(root);
  addDefs(root);
  root.appendChild(text(48, 48, "显存 / 计算取舍图", "svg-label"));
  root.appendChild(line(98, 338, 648, 338, palette.ink, 3));
  root.appendChild(line(98, 338, 98, 82, palette.ink, 3));
  root.appendChild(text(650, 360, "峰值激活内存", "svg-small", "end"));
  root.appendChild(text(104, 78, "额外计算", "svg-small"));
  const points = [
    ["普通 eager", 580, 304, palette.amber],
    ["checkpoint", 298, 194, palette.blue],
    ["更细粒度 checkpoint", 196, 134, palette.green],
    ["重算过多", 150, 92, palette.coral],
  ];
  points.forEach(([label, x, y, color]) => {
    root.appendChild(svg("circle", { cx: x, cy: y, r: 11, fill: color }));
    root.appendChild(text(x + 18, y + 5, label, "svg-small"));
  });
  root.appendChild(
    svg("path", {
      d: "M580,304 C454,260 362,210 298,194 C246,178 214,154 196,134",
      fill: "none",
      stroke: palette.blue,
      "stroke-width": 4,
      "stroke-linecap": "round",
    }),
  );
}

function attachControls() {
  document.querySelectorAll(".stage-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".stage-button").forEach((b) => b.classList.remove("is-active"));
      button.classList.add("is-active");
      state.stage = button.dataset.stage;
      renderFlow();
    });
  });

  document.querySelector("#layerCount").addEventListener("input", (event) => {
    state.layers = Number(event.target.value);
    const span = document.querySelector("#spanEnd");
    span.max = String(state.layers);
    state.spanEnd = Math.min(Number(span.value), state.layers);
    span.value = String(state.spanEnd);
    renderFlow();
  });

  document.querySelector("#spanEnd").addEventListener("input", (event) => {
    state.spanEnd = Math.max(2, Number(event.target.value));
    renderFlow();
  });

  document.querySelector("#rngToggle").addEventListener("change", (event) => {
    state.preserveRng = event.target.checked;
    renderRng();
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      state.code = tab.dataset.code;
      renderCodeWindow();
    });
  });

  const explanations = {
    true:
      "判断正确。关键是这段 forward 不记录内部 graph，所以内部 saved tensors 不会推高前向峰值。",
    false:
      "判断正确。detach 切开的是重算 graph 与旧 graph 的直接连接；custom Function.backward 会把 detached.grad 返回给原输入边界。",
  };

  document.querySelectorAll(".judge-item").forEach((item) => {
    item.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        item.querySelectorAll("button").forEach((b) => b.classList.remove("is-picked"));
        button.classList.add("is-picked");
        const ok = button.dataset.choice === item.dataset.answer;
        const feedback = item.querySelector(".judge-feedback");
        feedback.classList.toggle("is-correct", ok);
        feedback.classList.toggle("is-wrong", !ok);
        if (ok) {
          feedback.textContent =
            item.dataset.answer === "true" ? explanations.true : explanations.false;
        } else {
          feedback.textContent =
            item.dataset.answer === "true"
              ? "这里要选“对”：no_grad 正是为了让 checkpoint 区域内部不保存激活。"
              : "这里要选“错”：detach 是重算时建立新叶子输入的必要步骤，梯度会由 custom backward 返回。";
        }
      });
    });
  });
}

function boot() {
  attachControls();
  renderFlow();
  renderTape();
  renderRecompute();
  renderRng();
  renderCodeWindow();
  renderPytree();
  renderHooks();
  renderTradeoff();
}

boot();

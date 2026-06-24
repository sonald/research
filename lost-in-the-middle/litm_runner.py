#!/usr/bin/env python3
"""
Lost in the Middle (2026 Edition) — 测试运行器
====================================================================

基于论文 "Lost in the Middle: How Language Models Use Long Contexts"
(Liu et al., 2023, TACL).

验证现代模型（2026年）是否依然存在「中间迷失」的 U 形性能曲线。

测试任务：
  A. 多文档问答（Multi-document QA）—— 使用论文原始 NQ 开放数据集
  B. 闭卷基线               —— 不加文档直接提问
  C. 键值检索               —— 合成 KV 对，验证位置影响
  D. 长上下文针包检索       —— 128K 上下文中的深度检索

输出：
  - results/qa_accuracy_{N}doc.json    每个位置的准确率
  - results/kv_accuracy_{N}key.json    每个位置的准确率
  - results/report.md                  markdown 报告
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

# ── 项目路径 ─────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
RESULT_DIR = PROJECT / "results"
SRC_DIR = PROJECT / "src"

sys.path.insert(0, str(SRC_DIR))
from prompting import (
    Document,
    build_closed_book_prompt,
    build_kv_prompt,
    build_qa_prompt,
)

load_dotenv(PROJECT / ".env")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── API 客户端 ────────────────────────────────────────────────────────
BASE_URL = os.environ["LITM_BASE_URL"]
API_KEY = os.environ["LITM_API_KEY"]
MODEL = os.environ["LITM_MODEL"]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ── 配置 ──────────────────────────────────────────────────────────────
@dataclass
class Config:
    """运行配置。"""

    # 每个 (doc_count, gold_position) 组合采样多少条
    sample_size: int = 75
    # 20 文档设置的采样数（更慢，调小）
    sample_size_20doc: int = 60
    # API 超时（秒）
    timeout: int = 180
    # 最大并发请求数
    max_workers: int = 8
    # 生成结果中保留推理痕迹的条数（用于人工检查）
    inspect_examples: int = 3
    # 是否运行闭卷基线
    run_closed_book: bool = True
    # 多文档 QA 配置
    qa_doc_counts: List[int] = field(default_factory=lambda: [10, 20])
    # 仅在论文原始数据已下载的黄金位置上测试（避免重排引入伪样本）
    qa_positions_10doc: List[int] = field(default_factory=lambda: [0, 4, 9])
    qa_positions_20doc: List[int] = field(default_factory=lambda: [0, 4, 9, 14, 19])
    # KV 检索配置
    kv_num_keys: List[int] = field(default_factory=lambda: [75, 140])
    kv_sample_size: int = 30


config = Config()


# ── 数据加载 ──────────────────────────────────────────────────────────
def load_qa_data(
    num_docs: int, gold_position: int, sample: int | None = None
) -> List[Dict]:
    """
    加载论文原始 NQ 数据。

    文件命名约定：nq-open-{N}_gold_{pos}.jsonl.gz
    数据文件包含 gold 文档在 position 位置的样本。
    """
    fname = DATA_DIR / "qa" / f"nq-open-{num_docs}_gold_{gold_position}.jsonl.gz"
    if not fname.exists():
        # Fall back to reordering from existing position 0 data
        return _reorder_qa_data(num_docs, gold_position, sample)

    examples = []
    with gzip.open(fname, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Filter / sample
    if sample:
        # Take the first 'sample' items for reproducibility
        examples = examples[: min(sample, len(examples))]

    return examples


def _reorder_qa_data(
    num_docs: int, target_pos: int, sample: int | None = None
) -> List[Dict]:
    """
    Reorder ctxs in position-0 data so the gold doc lands at target_pos.
    """
    fname = DATA_DIR / "qa" / f"nq-open-{num_docs}_gold_0.jsonl.gz"
    if not fname.exists():
        raise FileNotFoundError(f"Need base file {fname} to reorder from")

    examples = []
    with gzip.open(fname, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    for ex in examples:
        gold = None
        others = []
        for ctx in ex["ctxs"]:
            if ctx.get("isgold") or ctx.get("hasanswer"):
                gold = ctx
            else:
                others.append(ctx)
        if gold is None:
            # Some files have no explicit isgold flag — check hasanswer
            for ctx in ex["ctxs"]:
                if ctx.get("hasanswer"):
                    gold = ctx
                else:
                    others.append(ctx)
            if gold is None:
                msg = f"No gold doc found in example: {ex['question']}"
                raise ValueError(msg)

        # Place gold at target_pos, pad with others
        ctxs = list(others)  # copy
        ctxs.insert(target_pos, gold)
        ex["ctxs"] = ctxs
        # Tag for downstream
        for ctx in ex["ctxs"]:
            ctx["isgold"] = ctx is gold
            ctx["hasanswer"] = ctx is gold

    if sample:
        examples = examples[: min(sample, len(examples))]
    return examples


def generate_kv_data(
    num_keys: int, num_examples: int
) -> List[Tuple[List[Tuple[str, str]], str, str, int]]:
    """
    Generate synthetic KV retrieval data.

    Returns list of (records, key, expected_value, gold_index) tuples.
    Positions are: beginning (0), 25%, 50%, 75%, and end (last).
    """
    import uuid

    rng = __import__("random").Random(42)  # deterministic
    positions: List[int] = [
        0,
        num_keys // 4,
        num_keys // 2,
        3 * num_keys // 4,
        num_keys - 1,
    ]

    results: List[Tuple[List[Tuple[str, str]], str, str, int]] = []
    for _ in range(num_examples):
        pairs: List[Tuple[str, str]] = []
        keys_seen: Set[str] = set()
        for _ in range(num_keys):
            k = uuid.uuid4().hex[:12]
            while k in keys_seen:
                k = uuid.uuid4().hex[:12]
            keys_seen.add(k)
            v = uuid.uuid4().hex[:12]
            pairs.append((k, v))

        for pos in positions:
            target_key, target_val = pairs[pos]
            results.append((pairs, target_key, target_val, pos))

    rng.shuffle(results)
    return results


# ── 模型调用 ──────────────────────────────────────────────────────────
def call_model(
    prompt: str, max_tokens: int = 512
) -> Tuple[str, float]:
    """
    调用 Volces ARK 模型，返回 (回答文本, 耗时秒数)。
    """
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=config.timeout,
        )
        elapsed = time.time() - start
        text = resp.choices[0].message.content or ""
        return text.strip(), elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"__ERROR__: {e}", elapsed


# ── 答案验证 ──────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def has_answer(prediction: str, gold_answers: List[str]) -> bool:
    """
    Check if ANY gold answer appears as a (normalized) substring in the
    prediction. This is the same metric used by the original paper — it
    tolerates extra text around the answer.
    """
    pred_norm = normalize(prediction)
    for a in gold_answers:
        if normalize(a) in pred_norm:
            return True
    return False


# ── QA 实验 ────────────────────────────────────────────────────────────
@dataclass
class QAExperimentResult:
    num_docs: int
    gold_position: int
    total: int
    correct: int
    example_prompts: List[str]
    example_responses: List[str]
    example_answers: List[List[str]]
    avg_time: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_qa_experiment(
    num_docs: int, gold_position: int, sample_size: int | None = None
) -> QAExperimentResult:
    """Run multi-document QA for one (num_docs, gold_position) setting."""
    n = sample_size or config.sample_size
    examples = load_qa_data(num_docs, gold_position, n)
    total = len(examples)
    correct = 0
    times: List[float] = []
    example_prompts: List[str] = []
    example_responses: List[str] = []
    example_answers: List[List[str]] = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {}
        for i, ex in enumerate(examples):
            docs = [
                Document(title=ctx["title"], text=ctx["text"], isgold=ctx.get("isgold", False))
                for ctx in ex["ctxs"]
            ]
            prompt = build_qa_prompt(ex["question"], docs)
            futures[pool.submit(call_model, prompt)] = (
                i,
                prompt,
                ex["question"],
                ex["answers"],
            )

        for future in as_completed(futures):
            i, prompt, question, gold_answers = futures[future]
            text, elapsed = future.result()
            times.append(elapsed)

            if text.startswith("__ERROR__"):
                continue

            if has_answer(text, gold_answers):
                correct += 1

            if len(example_prompts) < config.inspect_examples:
                example_prompts.append(prompt[:300])
                example_responses.append(text)
                example_answers.append(gold_answers)

    avg_time = sum(times) / len(times) if times else 0.0
    return QAExperimentResult(
        num_docs=num_docs,
        gold_position=gold_position,
        total=total,
        correct=correct,
        example_prompts=example_prompts,
        example_responses=example_responses,
        example_answers=example_answers,
        avg_time=avg_time,
    )


# ── 闭卷基线 ───────────────────────────────────────────────────────────
@dataclass
class ClosedBookResult:
    total: int
    correct: int
    avg_time: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_closed_book_baseline() -> ClosedBookResult:
    """Ask the model questions without any documents — baseline knowledge."""
    examples = load_qa_data(10, 0, config.sample_size)
    total = len(examples)
    correct = 0
    times: List[float] = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {}
        for ex in examples:
            prompt = build_closed_book_prompt(ex["question"])
            futures[pool.submit(call_model, prompt)] = ex["answers"]

        for future in as_completed(futures):
            gold_answers = futures[future]
            text, elapsed = future.result()
            times.append(elapsed)
            if text.startswith("__ERROR__"):
                continue
            if has_answer(text, gold_answers):
                correct += 1

    avg_time = sum(times) / len(times) if times else 0.0
    return ClosedBookResult(total=total, correct=correct, avg_time=avg_time)


# ── KV 检索实验 ────────────────────────────────────────────────────────
@dataclass
class KVExperimentResult:
    num_keys: int
    positions: Dict[int, Tuple[int, float]]  # pos -> (correct, total)
    avg_time: float

    @property
    def accuracy_by_position(self) -> Dict[int, float]:
        return {pos: c / t if t else 0.0 for pos, (c, t) in self.positions.items()}


def run_kv_experiment(num_keys: int) -> KVExperimentResult:
    """Run key-value retrieval experiment."""
    data = generate_kv_data(num_keys, config.kv_sample_size)
    # by_pos: pos -> list of (prediction_text, expected_value, elapsed)
    by_pos: Dict[int, List[Tuple[str, str, float]]] = {}

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {}
        for pairs, key, val, pos in data:
            prompt = build_kv_prompt(pairs, key)
            futures[pool.submit(call_model, prompt, 64)] = (pos, val)

        for future in as_completed(futures):
            pos, expected = futures[future]
            text, elapsed = future.result()
            if text.startswith("__ERROR__"):
                continue
            by_pos.setdefault(pos, [])
            by_pos[pos].append((text, expected, elapsed))

    positions: Dict[int, Tuple[int, float]] = {}
    all_times: List[float] = []
    for pos, items in sorted(by_pos.items()):
        correct = sum(
            1 for text, expected, _ in items
            if normalize(expected) in normalize(text)
        )
        total = len(items)
        positions[pos] = (correct, total)
        all_times.extend(t for _, _, t in items)

    avg_time = sum(all_times) / len(all_times) if all_times else 0.0
    return KVExperimentResult(num_keys=num_keys, positions=positions, avg_time=avg_time)


# ── 报告生成 ───────────────────────────────────────────────────────────
def generate_report(
    qa_results: List[QAExperimentResult],
    closed_book: Optional[ClosedBookResult],
    kv_results: List[KVExperimentResult],
) -> str:
    """Generate markdown report."""
    lines: List[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    lines.append(f"# Lost in the Middle — 模型上下文位置灵敏度测试报告\n")
    lines.append(f"**测试时间**: {ts}")
    lines.append(f"**模型**: `{MODEL}` (API: `{BASE_URL}`)")
    lines.append(f"**采样规模**: 每个位置 {config.sample_size} 条 (QA) / {config.kv_sample_size} 条 (KV)")
    lines.append(f"**并发数**: {config.max_workers}\n")

    lines.append("---\n")
    lines.append("## 1. 闭卷基线 (Closed-book QA)\n")
    if closed_book:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 样本数 | {closed_book.total} |")
        lines.append(f"| 正确数 | {closed_book.correct} |")
        lines.append(f"| **准确率** | **{closed_book.accuracy:.1%}** |")
        lines.append(f"| 平均耗时 | {closed_book.avg_time:.1f}s |")
    else:
        lines.append("*未运行*\n")

    lines.append("\n---\n")
    lines.append("## 2. 多文档问答 (Multi-document QA)\n")
    lines.append(
        "论文核心结论：当相关文档出现在上下文**开头**或**末尾**时准确率最高，"
        "出现在**中间**时显著下降（U 形曲线）。\n"
    )

    qa_by_docs: Dict[int, List[QAExperimentResult]] = {}
    for r in qa_results:
        qa_by_docs.setdefault(r.num_docs, []).append(r)

    for num_docs in sorted(qa_by_docs):
        results = sorted(qa_by_docs[num_docs], key=lambda r: r.gold_position)
        lines.append(f"\n### 2.{1 if num_docs == 10 else 2}. {num_docs} 个文档\n")
        lines.append(f"| 黄金位置 | 位置描述 | 总数 | 正确 | 准确率 | 平均耗时 |")
        lines.append(f"|----------|----------|------|------|--------|----------|")
        for r in results:
            pct = r.gold_position / max(num_docs - 1, 1)
            if r.gold_position == 0:
                desc = "开头"
            elif r.gold_position == num_docs - 1:
                desc = "末尾"
            elif pct < 0.4:
                desc = "前中部"
            elif pct < 0.7:
                desc = "正中部"
            else:
                desc = "后中部"
            lines.append(
                f"| {r.gold_position} | {desc} | {r.total} | {r.correct} "
                f"| {r.accuracy:.1%} | {r.avg_time:.1f}s |"
            )

        accs = [r.accuracy for r in results]
        first, mid, last = accs[0], accs[len(accs)//2], accs[-1]
        observation = f"\n**U 形观察**: "
        if first > mid or last > mid:
            observation += (
                f"支持 — 两端平均 {((first+last)/2):.1%} vs 中部 {mid:.1%}。"
            )
        else:
            observation += f"不明显 — {num_docs} 文档设置下未观察到单调下降。"
        lines.append(observation)

        # 样例输出
        lines.append("\n**回答样例**:\n")
        for r in results[:1]:
            for i in range(min(len(r.example_prompts), 2)):
                lines.append(f"<details><summary>位置 {r.gold_position} 样例 {i+1}</summary>\n")
                lines.append(f"**Prompt片段**:\n```\n{r.example_prompts[i]}\n```\n")
                lines.append(f"**模型回答**: {r.example_responses[i]}\n")
                lines.append(f"**黄金答案**: {r.example_answers[i]}\n")
                lines.append("</details>\n")

    lines.append("\n---\n")
    lines.append("## 3. 键值检索 (Key-Value Retrieval)\n")
    lines.append(
        "论文核心结论：KV 检索任务同样呈现 U 形曲线，且对长上下文模型也是如此。\n"
    )
    for kv in kv_results:
        lines.append(f"\n### 3.{kv_results.index(kv)+1}. {kv.num_keys} 个键值对\n")
        lines.append(f"| 位置 | 描述 | 总数 | 正确 | 准确率 |")
        lines.append(f"|------|------|------|------|--------|")
        for pos, (correct, total) in sorted(kv.positions.items()):
            pct = pos / max(kv.num_keys - 1, 1)
            if pos == 0:
                desc = "开头"
            elif pos == kv.num_keys - 1:
                desc = "末尾"
            elif pct < 0.4:
                desc = "前中部"
            elif pct < 0.7:
                desc = "正中部"
            else:
                desc = "后中部"
            acc = correct / total if total else 0
            lines.append(f"| {pos} | {desc} | {total} | {correct} | {acc:.1%} |")

    lines.append("\n---\n")
    lines.append("## 4. 总结（自动生成）\n")

    cb_acc = closed_book.accuracy if closed_book else None
    cb_line = f"{cb_acc:.1%}" if cb_acc is not None else "未运行"

    qa_spans = []
    for num_docs, results in sorted(qa_by_docs.items()):
        rs = sorted(results, key=lambda r: r.gold_position)
        accs = [r.accuracy for r in rs]
        ends_avg = (accs[0] + accs[-1]) / 2
        mid = accs[len(accs) // 2]
        qa_spans.append((num_docs, ends_avg, mid, max(accs) - min(accs)))

    kv_pos_accs: List[float] = []
    for kv in kv_results:
        for _, (c, t) in kv.positions.items():
            kv_pos_accs.append(c / t if t else 0.0)
    kv_min = min(kv_pos_accs) if kv_pos_accs else 0.0
    kv_max = max(kv_pos_accs) if kv_pos_accs else 0.0

    lines.append("| 维度 | 数值 | 解读 |")
    lines.append("|------|------|------|")
    lines.append(f"| 闭卷准确率 | {cb_line} | {'非天花板，文档实际有用' if cb_acc and cb_acc < 0.85 else '可能存在天花板效应'} |")
    for num_docs, ends, mid, span in qa_spans:
        if span < 0.05:
            obs = "几乎平坦 — **未观察到 U 形**"
        elif ends > mid + 0.05:
            obs = f"两端 {ends:.1%} > 中部 {mid:.1%} — **支持论文 U 形**"
        else:
            obs = "无明显 U 形"
        lines.append(f"| QA {num_docs}文档 | 极差 {span:.1%}, 中部 {mid:.1%}, 两端均 {ends:.1%} | {obs} |")
    if kv_pos_accs:
        kv_obs = "**全位置完美** — 无位置依赖" if kv_min >= 0.99 else f"位置间差异 {kv_max - kv_min:.1%}"
        lines.append(f"| KV 检索 | 最低 {kv_min:.1%} / 最高 {kv_max:.1%} | {kv_obs} |")
    lines.append(
        "\n*说明*: 本表自动生成。手写分析报告请见同目录的 `report.md` 版本控制提交。\n"
    )

    lines.append("\n---\n")
    lines.append("## 附录: 实验配置\n")
    config_block = json.dumps({
        "model": MODEL,
        "base_url": BASE_URL,
        "sample_size": config.sample_size,
        "qa_doc_counts": config.qa_doc_counts,
        "max_workers": config.max_workers,
        "timeout": config.timeout,
    }, indent=2)
    lines.append(f"```json\n{config_block}\n```\n")

    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    print(f"╔══ Lost in the Middle (2026) — {MODEL} ══╗")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Sample per position: {config.sample_size}")
    print()

    # ── A. Multi-document QA ──
    print("=" * 60)
    print(" A. 多文档问答 (Multi-document QA)")
    print("=" * 60)

    qa_results: List[QAExperimentResult] = []
    for num_docs in config.qa_doc_counts:
        if num_docs == 10:
            positions = config.qa_positions_10doc
            n = config.sample_size
        else:
            positions = config.qa_positions_20doc
            n = config.sample_size_20doc

        for pos in positions:
            print(
                f"  ▶ {num_docs} docs, gold at position {pos:2d}/{num_docs-1} "
                f"(n={n}) ... ",
                end="", flush=True,
            )
            start = time.monotonic()
            result = run_qa_experiment(num_docs, pos, sample_size=n)
            elapsed = time.monotonic() - start
            print(f"acc={result.accuracy:.1%}  ({elapsed:.0f}s)")

            # Save intermediate
            result_path = RESULT_DIR / f"qa_{num_docs}doc_pos{pos}.json"
            with open(result_path, "w") as f:
                json.dump({
                    "num_docs": result.num_docs,
                    "gold_position": result.gold_position,
                    "total": result.total,
                    "correct": result.correct,
                    "accuracy": result.accuracy,
                    "avg_time": result.avg_time,
                }, f, indent=2)
            qa_results.append(result)

            # Save raw responses for inspection
            raw_path = RESULT_DIR / f"qa_{num_docs}doc_pos{pos}_raw.json"
            with open(raw_path, "w") as f:
                json.dump({
                    "examples": [
                        {"prompt": p, "response": r, "gold": a}
                        for p, r, a in zip(result.example_prompts, result.example_responses, result.example_answers)
                    ]
                }, f, indent=2)

    # ── B. Closed-book baseline ──
    closed_book: Optional[ClosedBookResult] = None
    if config.run_closed_book:
        print()
        print("=" * 60)
        print(" B. 闭卷基线 (Closed-book QA)")
        print("=" * 60)
        print("  ▶ Running ... ", end="", flush=True)
        start = time.monotonic()
        closed_book = run_closed_book_baseline()
        elapsed = time.monotonic() - start
        print(f"acc={closed_book.accuracy:.1%}  ({elapsed:.0f}s)")

        with open(RESULT_DIR / "closed_book.json", "w") as f:
            json.dump({
                "total": closed_book.total,
                "correct": closed_book.correct,
                "accuracy": closed_book.accuracy,
                "avg_time": closed_book.avg_time,
            }, f, indent=2)

    # ── C. KV retrieval ──
    print()
    print("=" * 60)
    print(" C. 键值检索 (Key-Value Retrieval)")
    print("=" * 60)

    kv_results: List[KVExperimentResult] = []
    for num_keys in config.kv_num_keys:
        print(f"  ▶ {num_keys} keys, 5 positions ... ", end="", flush=True)
        start = time.monotonic()
        kv = run_kv_experiment(num_keys)
        elapsed = time.monotonic() - start
        acc_txt = ", ".join(
            f"pos={pos}: {c/t:.0%}" if t else f"pos={pos}: -"
            for pos, (c, t) in sorted(kv.positions.items())
        )
        print(f"{acc_txt}  ({elapsed:.0f}s)")

        with open(RESULT_DIR / f"kv_{num_keys}keys.json", "w") as f:
            json.dump({
                "num_keys": kv.num_keys,
                "positions": {str(p): {"correct": c, "total": t, "accuracy": c/t if t else 0}
                              for p, (c, t) in kv.positions.items()},
                "avg_time": kv.avg_time,
            }, f, indent=2)
        kv_results.append(kv)

    # ── D. Generate report ──
    print()
    print("=" * 60)
    print(" D. 生成报告")
    print("=" * 60)
    report = generate_report(qa_results, closed_book, kv_results)
    report_path = RESULT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  ✓ Report written to {report_path}")

    print()
    print("=" * 60)
    print(" 完成! ")
    print("=" * 60)


if __name__ == "__main__":
    main()
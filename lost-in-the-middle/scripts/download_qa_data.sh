#!/usr/bin/env bash
# 下载论文原始 NQ-open 多文档 QA 数据。
# 仓库提供以下黄金位置（不是所有位置都有，因此本测试只用这些）：
#   10 文档: gold ∈ {0, 4, 9}        — 开头 / 中部 / 末尾
#   20 文档: gold ∈ {0, 4, 9, 14, 19} — 开头 / 1/4 / 中部 / 3/4 / 末尾
#
# Usage: bash scripts/download_qa_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data"
mkdir -p data/qa

for n in 10 20; do
  for i in $(seq 0 $((n-1))); do
    url="$BASE/${n}_total_documents/nq-open-${n}_total_documents_gold_at_${i}.jsonl.gz"
    out="data/qa/nq-open-${n}_gold_${i}.jsonl.gz"
    if [ -f "$out" ]; then
      echo "✓ $out (cached)"
      continue
    fi
    if curl -sfL "$url" -o "$out"; then
      echo "✓ $out ($(wc -c < "$out") bytes)"
    else
      rm -f "$out"
      echo "✗ $url (not available — paper only publishes specific positions)"
    fi
  done
done

echo
echo "Done. Files in data/qa/:"
ls -la data/qa/

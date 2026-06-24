"""Prompt construction utilities matching the original 'Lost in the Middle' repo.

The templates are reproduced verbatim from nelson-liu/lost-in-the-middle so that
results here are directly comparable with the 2023 paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


QA_TEMPLATE = (
    "Write a high-quality answer for the given question using only the provided "
    "search results (some of which might be irrelevant).\n"
    "\n"
    "{search_results}\n"
    "\n"
    "Question: {question}\n"
    "Answer:"
)

# Closed-book baseline (no documents). Useful to measure how much the docs help
# at all on the 2026 model — if closed-book is already near-perfect on NQ, the
# multi-doc score is a ceiling effect rather than evidence about position.
CLOSED_BOOK_TEMPLATE = (
    "Write a high-quality answer for the given question.\n"
    "\n"
    "Question: {question}\n"
    "Answer:"
)

KV_TEMPLATE = (
    'Extract the value corresponding to the specified key in the JSON object below.\n'
    "\n"
    "JSON data:\n"
    "{formatted_kv_records}\n"
    "\n"
    'Key: "{key}"\n'
    "Corresponding value:"
)


@dataclass
class Document:
    title: str
    text: str
    isgold: bool = False


def format_documents(docs: Sequence[Document]) -> str:
    return "\n".join(
        f"Document [{i + 1}](Title: {d.title}) {d.text}" for i, d in enumerate(docs)
    )


def build_qa_prompt(question: str, docs: Sequence[Document]) -> str:
    return QA_TEMPLATE.format(
        search_results=format_documents(docs), question=question
    )


def build_closed_book_prompt(question: str) -> str:
    return CLOSED_BOOK_TEMPLATE.format(question=question)


def build_kv_prompt(records: Iterable[tuple[str, str]], key: str) -> str:
    items: List[str] = []
    pairs = list(records)
    for i, (k, v) in enumerate(pairs):
        prefix = "{" if i == 0 else ""
        suffix = "}" if i == len(pairs) - 1 else ","
        items.append(f'{prefix}"{k}": "{v}"{suffix}')
    return KV_TEMPLATE.format(formatted_kv_records="\n".join(items), key=key)

from __future__ import annotations

import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._token_to_id = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}
        self._id_to_token = {self.pad_token_id: self.pad_token, self.eos_token_id: self.eos_token}

    def _encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        for char in text:
            if char not in self._token_to_id:
                token_id = len(self._token_to_id)
                self._token_to_id[char] = token_id
                self._id_to_token[token_id] = char
            encoded.append(self._token_to_id[char])
        return encoded

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            parts.append(f"<{message['role']}>{message['content']}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        else:
            parts.append(self.eos_token)
        rendered = "".join(parts)
        if tokenize:
            return self._encode(rendered)
        return rendered

    def __call__(self, texts, padding=False, return_tensors=None, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}

        encoded = [self._encode(text) for text in texts]
        if not padding:
            return {"input_ids": encoded}

        max_len = max(len(item) for item in encoded)
        padded = [item + [self.pad_token_id] * (max_len - len(item)) for item in encoded]
        attention = [[1] * len(item) + [0] * (max_len - len(item)) for item in encoded]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": attention}

    def decode(self, ids, skip_special_tokens=True) -> str:
        chars = []
        for token_id in ids:
            if isinstance(token_id, torch.Tensor):
                token_id = int(token_id.item())
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            chars.append(self._id_to_token.get(int(token_id), "?"))
        return "".join(chars)


class ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden_size: int = 24) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return types.SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target / "pytorch_model.bin")


"""Byte-Pair-Encoding-Tokenizer, komplett von Grund auf geschrieben (keine Tokenizer-Bibliothek).

Idee: Wir starten mit den 256 möglichen Bytes als "Buchstaben". Dann suchen wir immer wieder das
häufigste benachbarte Paar und verschmelzen es zu einem neuen Token ("merge"). Nach ``n`` Merges
hat das Vokabular 256 + n Einträge. Häufige Wörter werden so zu einem einzigen Token.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

EOT = "<|endoftext|>"

# Vor-Zerlegung in "Wörter" (ähnlich wie GPT-2): Merges passieren nie über Wortgrenzen hinweg.
_PRETOKEN = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = merges or []
        self._rebuild()

    # ------------------------------------------------------------------ Aufbau
    def _rebuild(self) -> None:
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.rank: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self.eot_id: int = len(self.vocab)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 1  # +1 für <|endoftext|>

    # ----------------------------------------------------------------- Training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int,
              progress: Callable[[int, int, bytes], None] | None = None) -> "BPETokenizer":
        n_merges = max(0, vocab_size - 256 - 1)
        word_counts: Counter[str] = Counter()
        for t in texts:
            word_counts.update(_PRETOKEN.findall(t))

        words: list[list[int]] = [list(w.encode("utf-8")) for w in word_counts]
        freqs: list[int] = list(word_counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[i]
                pair_words[p].add(i)

        merges: list[tuple[int, int]] = []
        vocab = [bytes([i]) for i in range(256)]
        for m in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            if pair_counts[best] < 2:
                break
            new_id = 256 + m
            merges.append(best)
            vocab.append(vocab[best[0]] + vocab[best[1]])
            for wi in list(pair_words[best]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):                      # alte Paare dieses Wortes entfernen
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                new_w: list[int] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        new_w.append(new_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[wi] = new_w
                for p in zip(new_w, new_w[1:]):              # neue Paare eintragen
                    pair_counts[p] += f
                    pair_words[p].add(wi)
            pair_words.pop(best, None)
            pair_counts.pop(best, None)
            if progress and (m % 25 == 0 or m == n_merges - 1):
                progress(m + 1, n_merges, vocab[-1])
        return cls(merges)

    # ---------------------------------------------------------------- Encoding
    def _encode_word(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word.encode("utf-8"))
        rank = self.rank
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids[best_i:best_i + 2] = [256 + best_rank]
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        parts = text.split(EOT)
        for k, part in enumerate(parts):
            for w in _PRETOKEN.findall(part):
                out.extend(self._encode_word(w))
            if k < len(parts) - 1:
                out.append(self.eot_id)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            if i == self.eot_id:
                buf.extend(EOT.encode())
            elif 0 <= i < len(self.vocab):
                buf.extend(self.vocab[i])
        return buf.decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Lesbare Darstellung eines einzelnen Tokens (für Diagramme)."""
        if i == self.eot_id:
            return EOT
        s = self.vocab[i].decode("utf-8", errors="replace")
        return s.replace("\n", "⏎").replace(" ", "·") if s.strip() or s else repr(s)

    def pieces(self, text: str) -> list[tuple[int, str]]:
        return [(i, self.token_str(i)) for i in self.encode(text)]

    # ----------------------------------------------------------------- Speichern
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "bpe", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])

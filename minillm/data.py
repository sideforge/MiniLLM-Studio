"""Datenpipeline: Hugging-Face-Texte -> Tokenizer trainieren -> Token-Dateien (train.bin / val.bin)."""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .catalog import BY_KEY, DATA_DIR, fetch_dataset, read_documents
from .tokenizer import BPETokenizer

PREPARED_DIR = DATA_DIR / "prepared"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_") or "daten"


def prepared_path(name: str) -> Path:
    return PREPARED_DIR / safe_name(name)


def list_prepared() -> list[dict]:
    out = []
    if PREPARED_DIR.exists():
        for d in sorted(PREPARED_DIR.iterdir()):
            meta = d / "meta.json"
            if meta.exists():
                out.append(json.loads(meta.read_text(encoding="utf-8")) | {"dir": str(d)})
    return out


def load_selection(selection: dict[str, int], progress: Callable[[str, int, int], None] | None = None,
                   ) -> dict[str, list[str]]:
    """Lädt (bzw. nutzt den Cache für) jeden gewählten Datensatz. selection = {key: Anzahl Dokumente}."""
    docs: dict[str, list[str]] = {}
    for key, n in selection.items():
        cb = (lambda i, total, k=key: progress(k, i, total)) if progress else None
        path = fetch_dataset(key, n, cb)
        docs[key] = read_documents(path)
    return docs


def prepare(name: str, docs_by_ds: dict[str, list[str]], vocab_size: int, val_fraction: float = 0.05,
            seed: int = 1337, tok_progress: Callable[[int, int, bytes], None] | None = None,
            enc_progress: Callable[[int, int], None] | None = None) -> dict:
    """Trainiert den Tokenizer, kodiert alle Texte und speichert alles unter data/prepared/<name>."""
    t0 = time.time()
    out = prepared_path(name)
    out.mkdir(parents=True, exist_ok=True)

    all_docs: list[tuple[str, str]] = [(k, d) for k, ds in docs_by_ds.items() for d in ds]
    rng = random.Random(seed)
    rng.shuffle(all_docs)
    n_val = max(1, int(len(all_docs) * val_fraction)) if len(all_docs) > 20 else max(1, len(all_docs) // 5)
    val_docs, train_docs = all_docs[:n_val], all_docs[n_val:]

    tok = BPETokenizer.train((d for _, d in train_docs), vocab_size, tok_progress)
    tok.save(out / "tokenizer.json")
    t_tok = time.time() - t0

    def encode_all(docs: list[tuple[str, str]], desc: str) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for i, (_, d) in enumerate(docs, 1):
            chunks.append(np.array(tok.encode(d) + [tok.eot_id], dtype=np.uint16))
            if enc_progress and (i % 200 == 0 or i == len(docs)):
                enc_progress(i, len(docs))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)

    train_ids = encode_all(train_docs, "train")
    val_ids = encode_all(val_docs, "val")
    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")

    chars = sum(len(d) for _, d in train_docs)
    counts = np.bincount(train_ids.astype(np.int64), minlength=tok.vocab_size)
    top = np.argsort(-counts)[:40]
    doc_tokens = [len(tok.encode(d)) for _, d in train_docs[:2000]]
    meta = {
        "name": safe_name(name),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {k: len(v) for k, v in docs_by_ds.items()},
        "vocab_size": tok.vocab_size,
        "n_docs_train": len(train_docs),
        "n_docs_val": len(val_docs),
        "tokens_train": int(len(train_ids)),
        "tokens_val": int(len(val_ids)),
        "chars_train": chars,
        "chars_per_token": chars / max(1, len(train_ids)),
        "top_tokens": [[tok.token_str(int(i)), int(counts[i])] for i in top],
        "doc_token_lengths": doc_tokens,
        "first_merges": [[tok.token_str(256 + i), int(counts[256 + i])] for i in range(min(40, len(tok.merges)))],
        "token_counts": counts.tolist(),
        "seconds_tokenizer": round(t_tok, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


class Batcher:
    """Zieht zufällige Textfenster aus den Token-Dateien (x = Tokens, y = dieselben, um 1 verschoben)."""

    def __init__(self, prepared_dir: str | Path, block_size: int):
        d = Path(prepared_dir)
        self.train = np.fromfile(d / "train.bin", dtype=np.uint16)
        self.val = np.fromfile(d / "val.bin", dtype=np.uint16)
        self.block_size = block_size
        if len(self.val) < block_size + 2:           # Validierung zu klein -> Ende des Trainings abzweigen
            cut = max(block_size + 2, len(self.train) // 20)
            self.val = np.concatenate([self.val, self.train[-cut:]])
            self.train = self.train[:-cut]
        if len(self.train) < block_size + 2:
            raise ValueError("Zu wenig Trainingsdaten für diese Kontextlänge – mehr Dokumente laden "
                             "oder die Kontextlänge verkleinern.")

    def to(self, device: torch.device) -> "Batcher":
        """Legt alle Tokens in den GPU-Speicher: Batches entstehen dann direkt dort, ohne Umweg über die CPU
        (sonst wartet die GPU auf die Daten). Bei sehr grossen Datensätzen bleibt es bei der CPU."""
        self.device = device
        if device.type != "cpu" and len(self.train) + len(self.val) <= 400_000_000:
            self._dev = {"train": torch.from_numpy(self.train.astype(np.int32)).to(device),
                         "val": torch.from_numpy(self.val.astype(np.int32)).to(device)}
            self._arange = torch.arange(self.block_size, device=device)
        return self

    device = torch.device("cpu")
    _dev: dict | None = None

    def get(self, split: str, batch_size: int, gen: torch.Generator | None = None):
        data = self.train if split == "train" else self.val
        if self._dev is not None:                                    # schneller Weg: alles auf der GPU
            d = self._dev[split]
            ix = torch.randint(len(data) - self.block_size - 1, (batch_size, 1), device=self.device)
            idx = ix + self._arange[None, :]
            return d[idx].long(), d[idx + 1].long()
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

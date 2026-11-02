"""Text erzeugen: Token für Token das nächste Wort aus der Wahrscheinlichkeitsverteilung ziehen."""
from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import BPETokenizer


def filter_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperatur, Top-k und Top-p (Nucleus) anwenden. Gibt Wahrscheinlichkeiten zurück."""
    logits = logits / max(temperature, 1e-4)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        remove = cum - sorted_p > top_p
        sorted_p = sorted_p.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_i, sorted_p)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


@torch.no_grad()
def stream_generate(model: GPT, tok: BPETokenizer, prompt: str, max_new_tokens: int = 100,
                    temperature: float = 0.8, top_k: int = 40, top_p: float = 1.0,
                    seed: int | None = None) -> Iterator[tuple[int, str, torch.Tensor, torch.Tensor]]:
    """Erzeugt Tokens einzeln. Liefert (token_id, text, probs, raw_probs) pro Schritt.

    ``probs`` = Verteilung nach Filter, ``raw_probs`` = Verteilung ohne Temperatur/Filter.
    """
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    device = next(model.parameters()).device
    ids = tok.encode(prompt) or [tok.eot_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    try:
        for _ in range(max_new_tokens):
            ctx = x[:, -model.cfg.block_size:]
            logits, _ = model(ctx)
            last = logits[0, -1].float().cpu()          # Würfeln passiert auf der CPU (nur ein Vokabular-Vektor)
            raw = F.softmax(last, dim=-1)
            probs = filter_logits(last, temperature, top_k, top_p)
            nxt = torch.multinomial(probs, 1, generator=gen)
            x = torch.cat([x, nxt.view(1, 1).to(device)], dim=1)
            yield int(nxt), tok.decode([int(nxt)]), probs, raw
    finally:
        model.train(was_training)


def generate_text(model: GPT, tok: BPETokenizer, prompt: str = "", **kw) -> str:
    out = [t for _, t, _, _ in stream_generate(model, tok, prompt, **kw)]
    return prompt + "".join(out)


@torch.no_grad()
def next_token_table(model: GPT, tok: BPETokenizer, prompt: str, k: int = 10) -> list[tuple[str, float]]:
    """Die k wahrscheinlichsten nächsten Tokens (für das Balkendiagramm)."""
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt) or [tok.eot_id]
    x = torch.tensor([ids[-model.cfg.block_size:]], device=next(model.parameters()).device)
    logits, _ = model(x)
    probs = F.softmax(logits[0, -1].float().cpu(), dim=-1)
    top = torch.topk(probs, k)
    model.train(was_training)
    return [(tok.token_str(int(i)), float(p)) for p, i in zip(top.values, top.indices)]


@torch.no_grad()
def next_logits(model: GPT, tok: BPETokenizer, prompt: str) -> torch.Tensor:
    """Rohe Logits für das nächste Token (vor Softmax)."""
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt) or [tok.eot_id]
    logits, _ = model(torch.tensor([ids[-model.cfg.block_size:]], device=next(model.parameters()).device))
    model.train(was_training)
    return logits[0, -1].float().cpu()


@torch.no_grad()
def attention_maps(model: GPT, tok: BPETokenizer, text: str):
    """Aufmerksamkeits-Matrizen für einen Text: Liste je Schicht mit Form (Köpfe, T, T)."""
    was_training = model.training
    model.eval()
    ids = tok.encode(text)[-model.cfg.block_size:]
    x = torch.tensor([ids], device=next(model.parameters()).device)
    _, _, atts = model(x, return_attn=True)
    model.train(was_training)
    return [tok.token_str(i) for i in ids], [a[0].float().cpu().numpy() for a in atts]

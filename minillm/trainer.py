"""Trainings-Schleife mit Live-Metriken. Läuft in einem Hintergrund-Thread, damit die App
währenddessen die Diagramme aktualisieren kann."""
from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import device as dev
from .data import Batcher
from .generate import attention_maps, generate_text
from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
PROBE_TEXT = "Once upon a time, there was a little girl named Lily. She loved to play."


@dataclass
class TrainConfig:
    batch_size: int = 16             # 0 = automatisch: auf der GPU so gross wie der Speicher erlaubt
    max_steps: int = 1000
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_iters: int = 10
    sample_interval: int = 200
    seed: int = 1337
    threads: int = 0                 # CPU-Threads, 0 = automatisch (nur relevant ohne GPU)
    device: str = "auto"             # "auto" = beste GPU, sonst CPU; "cpu" erzwingt die CPU
    amp: bool = True                 # Mixed Precision (bf16/fp16) auf der GPU

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Lernraten-Plan: linear hochfahren (Warmup), danach sanft per Kosinus absenken."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + coeff * (1.0 - cfg.min_lr_ratio))


class Trainer:
    """Hält den kompletten Zustand eines Trainingslaufs (Metriken, Beispiele, Status)."""

    def __init__(self, run_name: str, prepared_dir: str | Path, model_cfg: GPTConfig, train_cfg: TrainConfig):
        self.run_name = run_name
        self.prepared_dir = Path(prepared_dir)
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.run_dir = RUNS_DIR / run_name
        self.status = "bereit"           # bereit | läuft | fertig | gestoppt | fehler
        self.error = ""
        self.step = 0
        self.tokens_seen = 0
        self.start_time = 0.0
        self.elapsed = 0.0
        self.metrics: list[dict] = []     # pro Schritt: step, loss, lr, grad_norm, tok_s
        self.evals: list[dict] = []       # alle eval_interval: step, train_loss, val_loss
        self.samples: list[dict] = []     # erzeugte Beispieltexte
        self.weight_stats: list[dict] = []
        self.embedding_snaps: list[dict] = []
        self.attention_snap: dict | None = None
        self.n_params = 0
        self.best_val = float("inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.model: GPT | None = None
        self.device = torch.device("cpu")
        self.device_info: dict = {}
        self.tok: BPETokenizer | None = None

    # ------------------------------------------------------------------ Steuerung
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------- Helfer
    @torch.no_grad()
    def _estimate_loss(self, batcher: Batcher, gen: torch.Generator) -> dict[str, float]:
        assert self.model is not None
        self.model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(self.cfg.eval_iters):
                x, y = batcher.get(split, self.cfg.batch_size, gen)
                with dev.autocast(self.device, self.cfg.amp):
                    _, loss = self.model(x, y)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        self.model.train()
        return out

    def _snapshot_embeddings(self) -> None:
        """Die häufigsten Tokens als Punkte im 2D-Raum (PCA des Embeddings) – zeigt, was das Modell lernt."""
        assert self.model is not None and self.tok is not None
        meta = json.loads((self.prepared_dir / "meta.json").read_text(encoding="utf-8"))
        counts = np.array(meta["token_counts"])
        top = np.argsort(-counts)[:150]
        w = self.model.tok_emb.weight.detach().float().cpu()[torch.from_numpy(top)]
        w = w - w.mean(0, keepdim=True)
        try:
            u, s, v = torch.pca_lowrank(w, q=2, center=False)
            xy = (w @ v[:, :2]).numpy()
        except Exception:
            return
        self.embedding_snaps.append({"step": self.step,
                                     "tokens": [self.tok.token_str(int(i)) for i in top],
                                     "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist()})
        self.embedding_snaps = self.embedding_snaps[-25:]

    def _snapshot_attention(self) -> None:
        assert self.model is not None and self.tok is not None
        try:
            toks, atts = attention_maps(self.model, self.tok, PROBE_TEXT)
            self.attention_snap = {"step": self.step, "tokens": toks,
                                   "maps": [a.astype(np.float16) for a in atts]}
        except Exception:
            pass

    def save_checkpoint(self, name: str = "model.pt") -> None:
        assert self.model is not None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}, "model_cfg": self.model_cfg.to_dict(),
                    "train_cfg": self.cfg.to_dict(), "step": self.step}, self.run_dir / name)

    def _write_run_json(self) -> None:
        info = {
            "name": self.run_name, "status": self.status, "step": self.step, "max_steps": self.cfg.max_steps,
            "n_params": self.n_params, "tokens_seen": self.tokens_seen, "elapsed": round(self.elapsed, 1),
            "best_val": None if self.best_val == float("inf") else self.best_val,
            "final_train": self.evals[-1]["train"] if self.evals else None,
            "data": self.prepared_dir.name, "model_cfg": self.model_cfg.to_dict(),
            "train_cfg": self.cfg.to_dict(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device_info.get("name", "CPU"),
        }
        (self.run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    # --------------------------------------------------------------------- Lauf
    def _run(self) -> None:
        try:
            self.status = "läuft"
            cfg = self.cfg
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            self.device = dev.pick_device(cfg.device)
            self.device_info = dev.describe(self.device)
            dev.setup(self.device)
            if self.device.type == "cpu":
                torch.set_num_threads(cfg.threads or max(1, min(8, (os.cpu_count() or 4))))

            self.run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.prepared_dir / "tokenizer.json", self.run_dir / "tokenizer.json")
            self.tok = BPETokenizer.load(self.run_dir / "tokenizer.json")
            batcher = Batcher(self.prepared_dir, self.model_cfg.block_size).to(self.device)
            gen = torch.Generator().manual_seed(cfg.seed)

            if cfg.batch_size <= 0:        # automatisch: GPU voll auslasten
                cfg.batch_size = dev.find_max_batch(lambda: GPT(self.model_cfg), self.model_cfg.block_size,
                                                    self.model_cfg.vocab_size, self.device, cfg.amp)

            self.model = GPT(self.model_cfg).to(self.device)
            self.n_params = self.model.num_params()
            opt = self.model.configure_optimizer(cfg.lr, cfg.weight_decay)
            self.model.train()
            # fp16 braucht "Loss-Scaling"; bf16 und CPU nicht (dann ist der Scaler ein No-Op)
            scaler = torch.amp.GradScaler(self.device.type, enabled=cfg.amp and dev.amp_dtype(self.device) == torch.float16
                                          and self.device.type == "cuda")

            metrics_file = open(self.run_dir / "metrics.jsonl", "w", encoding="utf-8")
            self.start_time = time.time()
            self._on_eval(batcher, gen, metrics_file, first=True)

            log_every = 1 if self.device.type == "cpu" else 10
            pending: list[tuple] = []
            last_flush = [time.time()]

            def flush() -> None:
                dev.sync(self.device)
                now = time.time()
                tokens = sum(p[4] for p in pending)
                tok_s = tokens / max(1e-9, now - last_flush[0])
                self.elapsed = now - self.start_time
                for st, loss_t, gn_t, lr_, _ in pending:
                    rec = {"step": st, "loss": float(loss_t), "lr": lr_, "grad_norm": float(gn_t),
                           "tok_s": tok_s, "time": round(self.elapsed, 2)}
                    self.metrics.append(rec)
                    metrics_file.write(json.dumps({"type": "step", **rec}) + "\n")
                pending.clear()
                last_flush[0] = now

            for step in range(1, cfg.max_steps + 1):
                if self._stop.is_set():
                    self.status = "gestoppt"
                    break
                lr = lr_at(step - 1, cfg)
                for g in opt.param_groups:
                    g["lr"] = lr
                x, y = batcher.get("train", cfg.batch_size, gen)
                with dev.autocast(self.device, cfg.amp):
                    _, loss = self.model(x, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                # Werte erst alle paar Schritte von der GPU holen: jedes .item() zwingt die GPU zum Warten
                pending.append((step, loss.detach(), gnorm, lr, x.numel()))
                self.tokens_seen += x.numel()
                self.step = step
                due = (step % cfg.eval_interval == 0 or step % cfg.sample_interval == 0 or step == cfg.max_steps)
                if len(pending) >= log_every or due or self._stop.is_set():
                    flush()

                if step % cfg.eval_interval == 0 or step == cfg.max_steps:
                    self._on_eval(batcher, gen, metrics_file)
                if step % cfg.sample_interval == 0 or step == cfg.max_steps:
                    self._on_sample()
            if pending:
                flush()
            if self.status == "läuft":
                self.status = "fertig"
            self.elapsed = time.time() - self.start_time
            if not self.samples or self.samples[-1]["step"] != self.step:
                self._on_sample()
            self.save_checkpoint("model.pt")
            metrics_file.close()
            self._write_run_json()
        except Exception as e:  # pragma: no cover - Fehler sollen in der App sichtbar sein
            self.status = "fehler"
            self.error = f"{e}\n{traceback.format_exc()}"

    def _on_eval(self, batcher: Batcher, gen: torch.Generator, f, first: bool = False) -> None:
        losses = self._estimate_loss(batcher, gen)
        rec = {"step": self.step, "train": losses["train"], "val": losses["val"],
               "ppl": math.exp(min(20, losses["val"])), "tokens": self.tokens_seen}
        self.evals.append(rec)
        f.write(json.dumps({"type": "eval", **rec}) + "\n")
        f.flush()
        assert self.model is not None
        self.weight_stats.append({"step": self.step, "stats": self.model.layer_stats()})
        if losses["val"] < self.best_val:
            self.best_val = losses["val"]
        if first:
            self._snapshot_embeddings()
            self._snapshot_attention()

    def _on_sample(self) -> None:
        assert self.model is not None and self.tok is not None
        text = generate_text(self.model, self.tok, "", max_new_tokens=80, temperature=0.8, top_k=40, seed=self.step)
        self.samples.append({"step": self.step, "text": text})
        self._snapshot_embeddings()
        self._snapshot_attention()
        self.save_checkpoint("model.pt")
        self._write_run_json()


# ---------------------------------------------------------------- Laden alter Läufe
def list_runs() -> list[dict]:
    out = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (d / "run.json").exists() and (d / "model.pt").exists():
                out.append(json.loads((d / "run.json").read_text(encoding="utf-8")))
    return out


def load_run_metrics(run_name: str) -> tuple[list[dict], list[dict]]:
    steps, evals = [], []
    p = RUNS_DIR / run_name / "metrics.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            (steps if r.pop("type") == "step" else evals).append(r)
    return steps, evals


def load_model(run_name: str, device: "torch.device | None" = None) -> tuple[GPT, BPETokenizer, dict]:
    d = RUNS_DIR / run_name
    ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model = GPT(GPTConfig.from_dict(ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    model.to(device or dev.pick_device()).eval()      # zum Generieren ebenfalls die GPU nutzen
    return model, BPETokenizer.load(d / "tokenizer.json"), ckpt

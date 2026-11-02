"""Rechengerät wählen: GPU wenn irgendwie möglich, sonst CPU.

Reihenfolge bei "auto": NVIDIA/AMD (CUDA/ROCm) -> Intel (XPU) -> Apple (MPS) -> CPU.
"""
from __future__ import annotations

import contextlib
import shutil
import subprocess

import torch

CHOICES = ["auto", "cpu"]   # "auto" nimmt die beste GPU, die PyTorch sieht


def _has(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if kind == "mps":
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return False


def pick_device(pref: str = "auto") -> torch.device:
    """pref: "auto", "cpu", "cuda", "xpu" oder "mps"."""
    if pref == "cpu":
        return torch.device("cpu")
    for kind in ([pref] if pref != "auto" else ["cuda", "xpu", "mps"]):
        if _has(kind):
            return torch.device(kind)
    return torch.device("cpu")


def setup(device: torch.device) -> None:
    """Schaltet auf der GPU die schnellen Rechenwege ein (TF32, cuDNN-Autotuner)."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed Precision: halb so viel Speicher und deutlich schneller auf GPUs. Auf der CPU aus."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type in ("xpu", "mps"):
        return torch.bfloat16 if device.type == "xpu" else torch.float16
    return None


def autocast(device: torch.device, enabled: bool = True):
    dt = amp_dtype(device) if enabled else None
    if dt is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dt)


def sync(device: torch.device) -> None:
    """Wartet, bis die GPU fertig ist (nötig für ehrliche Zeitmessung)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def describe(device: torch.device | None = None) -> dict:
    """Infos für die App: Name, Speicher, Hinweise."""
    device = device or pick_device()
    info = {"type": device.type, "name": "CPU", "memory_gb": None, "amp": amp_dtype(device), "hint": ""}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(name=p.name, memory_gb=p.total_memory / 1e9)
    elif device.type == "xpu":
        p = torch.xpu.get_device_properties(0)
        info.update(name=p.name, memory_gb=getattr(p, "total_memory", 0) / 1e9 or None)
    elif device.type == "mps":
        info.update(name="Apple GPU (MPS)")
    else:
        info["hint"] = _cpu_hint()
    return info


def _cpu_hint() -> str:
    """Wenn eine NVIDIA-Karte da ist, PyTorch sie aber nicht benutzt, sagen wir warum."""
    if torch.version.cuda is None and shutil.which("nvidia-smi"):
        try:
            name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        except Exception:
            name = "NVIDIA-GPU"
        return (f"Es ist eine {name} eingebaut, aber PyTorch wurde ohne CUDA installiert. "
                "Installiere die GPU-Version von pytorch.org/get-started/locally (Auswahl: Pip + CUDA).")
    return "Keine GPU gefunden – das Training läuft auf der CPU (langsamer)."


def find_max_batch(make_model, block_size: int, vocab_size: int, device: torch.device, use_amp: bool = True,
                   start: int = 16, limit: int = 1024) -> int:
    """Probiert immer grössere Batches, bis der GPU-Speicher voll ist – so wird die GPU maximal ausgelastet.

    ``make_model`` ist eine Funktion, die ein frisches Modell liefert. Auf der CPU gibt es keinen Gewinn
    durch grosse Batches, dort wird ``start`` zurückgegeben.
    """
    if device.type == "cpu":
        return start
    best, bs = start, start
    while bs <= limit:
        model = None
        try:
            model = make_model().to(device)
            opt = model.configure_optimizer(1e-3, 0.0)
            x = torch.randint(0, vocab_size, (bs, block_size), device=device)
            with autocast(device, use_amp):
                _, loss = model(x, x)
            loss.backward()
            opt.step()
            sync(device)
            best, bs = bs, bs * 2
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:   # Speicher voll -> Vorgänger war das Maximum
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            break
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return max(start, int(best * 0.9) // 8 * 8)   # etwas Luft lassen (Evaluation, Snapshots)

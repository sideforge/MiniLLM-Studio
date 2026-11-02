"""Eigene Hugging-Face-Datensätze: im Hub suchen, Spalten erkennen, speichern und wie Katalog-Einträge benutzen."""
from __future__ import annotations

import json
import re
from typing import Callable

from .catalog import BY_KEY, CATALOG, DATA_DIR, DatasetInfo, is_schema_error, raw_rows, render_value

CUSTOM_FILE = DATA_DIR / "custom_datasets.json"

# Spalten, die fast nie Trainingstext sind
_SKIP = re.compile(r"(^|_)(id|ids|uuid|url|urls|link|hash|sha|sha1|timestamp|date|license|lang|language|path|label|labels|"
                   r"score|idx|index|source|dump|metadata|meta|tags|author|name|seed|fingerprint|size|length)(_|$)", re.I)
_PREFERRED = ["text", "content", "code", "body", "article", "document", "completion", "story", "abstract"]
_QUESTION = ["instruction", "question", "prompt", "query", "problem", "input", "context"]
_ANSWER = ["output", "response", "answer", "solution", "completion", "chosen", "target"]
_CHAT = ["messages", "conversations", "conversation", "dialogue", "chat"]


def detect_columns(row: dict) -> list[str]:
    """Rät, welche Spalten den Trainingstext enthalten."""
    low = {c.lower(): c for c in row}
    for name in _CHAT:
        if name in low and isinstance(row[low[name]], list):
            return [low[name]]
    for col, val in row.items():                      # einzelne Chat-Nachricht {"role": ..., "content": ...}
        if isinstance(val, dict) and "content" in val and ("role" in val or "from" in val):
            return [col]
    for name in _PREFERRED:
        if name in low and isinstance(row[low[name]], str) and row[low[name]].strip():
            return [low[name]]
    q = [low[n] for n in _QUESTION if n in low and isinstance(row[low[n]], str)]
    a = [low[n] for n in _ANSWER if n in low and isinstance(row[low[n]], str)]
    if q and a:
        return [q[0], a[0]]
    strs = {c: len(v) for c, v in row.items() if isinstance(v, str) and not _SKIP.search(c)}
    if strs:
        return [max(strs, key=strs.get)]
    strs = {c: len(v) for c, v in row.items() if isinstance(v, str)}
    return [max(strs, key=strs.get)] if strs else []


def make_formatter(columns: list[str], headings: bool) -> Callable[[dict], str]:
    def fmt(row: dict) -> str:
        parts = []
        for c in columns:
            text = render_value(row.get(c)).strip()
            if text:
                parts.append(f"### {c}:\n{text}" if headings and len(columns) > 1 else text)
        return "\n".join(parts)
    return fmt


def friendly_error(e: Exception) -> str:
    msg = f"{type(e).__name__}: {e}"
    low = msg.lower()
    if "scripts are no longer supported" in low:
        return ("Dieser Datensatz benutzt ein altes Lade-Skript und wird von der aktuellen `datasets`-Version nicht "
                "mehr unterstützt. Suche eine Variante im Parquet-Format.")
    if "gated" in low or "401" in low or "403" in low or "must be authenticated" in low:
        return ("Dieser Datensatz ist gesperrt oder privat. Beantrage auf huggingface.co Zugriff und melde dich dann mit "
                "`huggingface-cli login` an (oder setze die Umgebungsvariable `HF_TOKEN`).")
    if "doesn't exist" in low or "not found" in low or "404" in low:
        return "Datensatz nicht gefunden – prüfe die Schreibweise (Format: `besitzer/name`)."
    if "config name is missing" in low or "pick one among" in low:
        return "Dieser Datensatz hat mehrere Konfigurationen – wähle unten eine aus."
    if "unknown split" in low or "splitnotfound" in low:
        return "Diesen Split gibt es nicht – wähle einen der vorhandenen Splits."
    return msg[:400]


# ------------------------------------------------------------------------------- Hub
def search_hub(query: str, limit: int = 30, task_filter: str | None = None) -> list[dict]:
    from huggingface_hub import HfApi
    kw = {"filter": task_filter} if task_filter else {}
    out = []
    for d in HfApi().list_datasets(search=query or None, sort="downloads", limit=limit, **kw):
        out.append({"id": d.id, "downloads": d.downloads or 0, "likes": d.likes or 0,
                    "gated": bool(getattr(d, "gated", False)),
                    "tags": [t.split(":", 1)[1] for t in (d.tags or []) if t.startswith(("language:", "task_categories:"))][:4]})
    return out


def list_configs(repo: str) -> list[str]:
    from datasets import get_dataset_config_names
    return list(get_dataset_config_names(repo))


def list_splits(repo: str, config: str | None) -> list[str]:
    from datasets import get_dataset_split_names
    return list(get_dataset_split_names(repo, config))


def inspect_dataset(repo: str, config: str | None = None, split: str = "train", n: int = 3) -> dict:
    """Streamt die ersten ``n`` Zeilen. Gibt Spalten, Beispielwerte und die empfohlene Spaltenwahl zurück."""
    from datasets import load_dataset
    rows: list[dict] = []
    try:
        ds = load_dataset(repo, config, split=split, streaming=True)
        for row in ds:
            rows.append(row)
            if len(rows) >= n:
                break
    except Exception as e:
        if not is_schema_error(e):
            raise
        rows = []                                    # Zeilen mit verschiedenen Strukturen: Rückfall auf rohes JSON
        for row in raw_rows(repo, split):
            rows.append(row)
            if len(rows) >= n:
                break
    if not rows:
        raise ValueError("Der Datensatz ist leer.")
    first = rows[0]
    cols = {c: {"type": type(v).__name__, "example": render_value(v)[:160].replace("\n", " ")} for c, v in first.items()}
    return {"columns": cols, "recommended": detect_columns(first), "rows": rows}


# --------------------------------------------------------------------- Speichern
def _slug(repo: str, config: str | None) -> str:
    return "custom_" + re.sub(r"[^a-z0-9]+", "_", f"{repo}_{config or ''}".lower()).strip("_")[:50]


def _build(spec: dict) -> DatasetInfo:
    cols = spec["columns"]
    return DatasetInfo(
        spec["key"], spec["repo"], spec["repo"],
        f"Eigener Datensatz · Spalten: {', '.join(cols)}" + (f" · Konfiguration: {spec['config']}" if spec.get("config") else ""),
        "Eigene", "multi", make_formatter(cols, spec.get("headings", True)), config=spec.get("config"),
        split=spec.get("split", "train"), size_hint="", difficulty="unbekannt",
        min_chars=spec.get("min_chars", 20), max_chars=spec.get("max_chars"), tags=["eigener"],
    )


def _read() -> list[dict]:
    if CUSTOM_FILE.exists():
        try:
            return json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def add_custom(repo: str, columns: list[str], config: str | None = None, split: str = "train",
               headings: bool = True, min_chars: int = 20, max_chars: int | None = 8000) -> DatasetInfo:
    if not columns:
        raise ValueError("Wähle mindestens eine Spalte.")
    spec = {"key": _slug(repo, config), "repo": repo.strip(), "config": config or None, "split": split,
            "columns": columns, "headings": headings, "min_chars": min_chars, "max_chars": max_chars}
    specs = [s for s in _read() if s["key"] != spec["key"]] + [spec]
    CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_FILE.write_text(json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    return register(spec)


def register(spec: dict) -> DatasetInfo:
    info = _build(spec)
    if info.key in BY_KEY:                       # ersetzen, nicht doppelt anzeigen
        CATALOG[:] = [d for d in CATALOG if d.key != info.key]
    CATALOG.append(info)
    BY_KEY[info.key] = info
    return info


def remove_custom(key: str) -> None:
    CUSTOM_FILE.write_text(json.dumps([s for s in _read() if s["key"] != key], ensure_ascii=False, indent=1),
                           encoding="utf-8")
    CATALOG[:] = [d for d in CATALOG if d.key != key]
    BY_KEY.pop(key, None)


def register_saved() -> None:
    for spec in _read():
        try:
            register(spec)
        except Exception:
            pass


register_saved()   # beim Import: gespeicherte eigene Datensätze in den Katalog eintragen

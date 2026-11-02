"""Katalog der Hugging-Face-Datensätze, aus denen der Nutzer in der App auswählen kann.

Jeder Eintrag beschreibt, wo der Datensatz liegt und wie aus einer Zeile ein Trainingstext
wird. Alle Einträge wurden gegen den Hugging Face Hub getestet (streaming).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATA_DIR = Path(os.environ.get("MINILLM_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
RAW_DIR = DATA_DIR / "raw"


def _join(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p and p.strip())


def _instruct(instruction: str, context: str, answer: str) -> str:
    out = f"### Frage:\n{instruction.strip()}\n"
    if context and context.strip():
        out += f"### Kontext:\n{context.strip()}\n"
    return out + f"### Antwort:\n{answer.strip()}"



def _role_label(role: str) -> str:
    if role in ("user", "human", "prompter"):
        return "### Frage:"
    if role in ("assistant", "gpt", "bot", "model"):
        return "### Antwort:"
    return f"### {role}:" if role else ""


def _is_block_list(v: list) -> bool:
    """Liste von Inhalts-Blöcken wie [{"type": "text", "text": "..."}, {"type": "tool_use", ...}] (Claude/OpenAI-Format)."""
    return bool(v) and all(isinstance(m, dict) and "type" in m and "role" not in m and "from" not in m for m in v)


def _render_block(m: dict) -> str:
    t = m.get("type")
    if t == "text" or ("text" in m and t is None):
        return str(m.get("text", ""))
    if t == "tool_use":
        return f"[Werkzeug: {m.get('name', '?')}] " + json.dumps(m.get("input", {}), ensure_ascii=False)
    if t == "tool_result":
        return "[Ergebnis] " + render_value(m.get("content"))
    if t == "thinking":
        return str(m.get("thinking", ""))
    return render_value(m.get("text") or m.get("content") or "")


def render_value(v) -> str:
    """Macht aus beliebigen Zellen Text: Strings bleiben, Chat-Listen und Nachrichten werden zu Frage/Antwort,
    Inhalts-Blöcke werden zu Text, andere Listen werden verbunden."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, list):
        if _is_block_list(v):
            return "\n".join(x for x in (_render_block(m).strip() for m in v) if x)
        if v and all(isinstance(m, dict) for m in v):
            parts = []
            for m in v:
                role = str(m.get("role") or m.get("from") or "")
                content = m["content"] if "content" in m else m.get("value", m.get("text"))
                content = json.dumps(m, ensure_ascii=False) if content is None else render_value(content)
                label = _role_label(role)
                parts.append((label + "\n" if label else "") + content.strip())
            return "\n".join(parts)
        return "\n".join(render_value(x) for x in v)
    if isinstance(v, dict):
        if "content" in v and ("role" in v or "from" in v):          # einzelne Chat-Nachricht
            label = _role_label(str(v.get("role") or v.get("from") or ""))
            return (label + "\n" if label else "") + render_value(v["content"]).strip()
        return "\n".join(f"{k}: {render_value(x)}" for k, x in v.items())
    return str(v)


@dataclass
class DatasetInfo:
    key: str                       # kurzer eindeutiger Name
    title: str                     # Anzeigename
    hf_path: str                   # Hugging-Face-Repo
    description: str
    category: str                  # Geschichten | Wissen | Web | Anweisungen | Code | Nachrichten | Sonstiges
    language: str                  # en | de | multi
    fmt: Callable[[dict], str]     # Zeile -> Text
    config: str | None = None
    split: str = "train"
    size_hint: str = ""            # grobe Grösse auf dem Hub
    difficulty: str = "leicht"     # wie gut ein winziges Modell damit klarkommt
    min_chars: int = 20
    max_chars: int | None = None   # lange Dokumente (Bücher, Code-Dateien) werden hier abgeschnitten
    tags: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.hf_path}"


CATALOG: list[DatasetInfo] = [
    DatasetInfo(
        "tinystories", "TinyStories", "roneneldan/TinyStories",
        "Kurze, einfache Kindergeschichten (von GPT-4 geschrieben). Der Klassiker, um zu zeigen, "
        "dass schon winzige Modelle ordentliche Texte lernen.",
        "Geschichten", "en", lambda r: r["text"], size_hint="2.1 Mio. Geschichten", difficulty="sehr leicht",
        tags=["empfohlen"],
    ),
    DatasetInfo(
        "shakespeare", "Tiny Shakespeare", "Trelis/tiny-shakespeare",
        "Theaterstücke von Shakespeare (ca. 1 MB). Sehr klein, ideal für schnelle Tests.",
        "Geschichten", "en", lambda r: r["Text"], size_hint="~1 MB", difficulty="mittel", min_chars=1,
        tags=["empfohlen", "schnell"],
    ),
    DatasetInfo(
        "wikitext2", "WikiText-2", "Salesforce/wikitext",
        "Saubere englische Wikipedia-Artikel (Qualitäts-Artikel). Kleiner Standard-Benchmark der "
        "Sprachmodell-Forschung.", "Wissen", "en", lambda r: r["text"], config="wikitext-2-raw-v1",
        size_hint="~11 MB", difficulty="mittel", min_chars=40, tags=["benchmark"],
    ),
    DatasetInfo(
        "wikipedia_de", "Wikipedia (Deutsch)", "wikimedia/wikipedia",
        "Deutsche Wikipedia-Artikel (Stand November 2023). Schwierig für kleine Modelle, aber "
        "spannend für deutsche Texte.", "Wissen", "de", lambda r: _join(r["title"], r["text"]),
        config="20231101.de", size_hint="2.8 Mio. Artikel", difficulty="schwer", min_chars=200, max_chars=8000,
    ),
    DatasetInfo(
        "wikipedia_en", "Wikipedia (Englisch)", "wikimedia/wikipedia",
        "Englische Wikipedia-Artikel (Stand November 2023).", "Wissen", "en",
        lambda r: _join(r["title"], r["text"]), config="20231101.en", size_hint="6.4 Mio. Artikel",
        difficulty="schwer", min_chars=200, max_chars=8000,
    ),
    DatasetInfo(
        "tagesschau", "Tagesschau-Artikel (Deutsch)", "bjoernp/tagesschau-2018-2023",
        "Nachrichtenartikel der Tagesschau von 2018 bis 2023 – gut für deutschen Zeitungsstil.",
        "Nachrichten", "de", lambda r: _join(r["headline"], r["article"]), size_hint="~20k Artikel",
        difficulty="mittel", min_chars=200, tags=["deutsch"],
    ),
    DatasetInfo(
        "cosmopedia", "Cosmopedia (100k)", "HuggingFaceTB/cosmopedia-100k",
        "Synthetische Lehrbuch-Texte, Blogposts und Geschichten, von einem grossen Modell "
        "geschrieben. Sehr saubere Daten.", "Wissen", "en", lambda r: r["text"],
        size_hint="100k Texte", difficulty="mittel", min_chars=200,
    ),
    DatasetInfo(
        "fineweb_edu", "FineWeb-Edu", "HuggingFaceFW/fineweb-edu",
        "Gefilterte, bildungsrelevante Webseiten – wird für moderne Open-Source-Modelle benutzt.",
        "Web", "en", lambda r: r["text"], config="sample-10BT", size_hint="10 Mrd. Tokens (Sample)",
        difficulty="schwer", min_chars=200, tags=["streaming"],
    ),
    DatasetInfo(
        "c4", "C4 (Common Crawl, bereinigt)", "allenai/c4",
        "Riesiger bereinigter Web-Text, mit dem auch T5 trainiert wurde.", "Web", "en",
        lambda r: r["text"], config="en", size_hint="365 Mio. Dokumente", difficulty="schwer",
        min_chars=200, tags=["streaming"],
    ),
    DatasetInfo(
        "imdb", "IMDB Filmkritiken", "stanfordnlp/imdb",
        "50'000 Filmkritiken. Informelle Sprache mit Meinungen.", "Sonstiges", "en",
        lambda r: r["text"].replace("<br />", "\n"), size_hint="~65 MB", difficulty="mittel", min_chars=100,
    ),
    DatasetInfo(
        "rotten", "Rotten Tomatoes Sätze", "cornell-movie-review-data/rotten_tomatoes",
        "Kurze Filmkritik-Sätze – sehr klein.", "Sonstiges", "en", lambda r: r["text"],
        size_hint="~1 MB", difficulty="mittel", min_chars=10, tags=["schnell"],
    ),
    DatasetInfo(
        "ag_news", "AG News", "fancyzhx/ag_news",
        "Kurze Nachrichtenmeldungen aus vier Themenbereichen.", "Nachrichten", "en",
        lambda r: r["text"], size_hint="120k Meldungen", difficulty="mittel", min_chars=40,
    ),
    DatasetInfo(
        "cnn", "CNN / Daily Mail", "abisee/cnn_dailymail",
        "Lange Nachrichtenartikel mit Zusammenfassung.", "Nachrichten", "en",
        lambda r: _join(r["article"], "Zusammenfassung: " + r["highlights"]), config="3.0.0",
        size_hint="~300k Artikel", difficulty="schwer", min_chars=200,
    ),
    DatasetInfo(
        "quotes", "Englische Zitate", "Abirate/english_quotes",
        "Berühmte Zitate mit Autor – winzig und witzig.", "Sonstiges", "en",
        lambda r: f"{r['quote']} - {r['author']}", size_hint="~2.5k Zitate", difficulty="leicht",
        min_chars=10, tags=["schnell"],
    ),
    DatasetInfo(
        "alpaca", "Alpaca (Anweisungen)", "tatsu-lab/alpaca",
        "52k Frage-Antwort-Paare im Anweisungs-Format (von Stanford).", "Anweisungen", "en",
        lambda r: _instruct(r["instruction"], r["input"], r["output"]), size_hint="52k Beispiele",
        difficulty="mittel", min_chars=20, tags=["chat"],
    ),
    DatasetInfo(
        "dolly", "Dolly 15k (Anweisungen)", "databricks/databricks-dolly-15k",
        "15k von Menschen geschriebene Anweisungen und Antworten.", "Anweisungen", "en",
        lambda r: _instruct(r["instruction"], r["context"], r["response"]), size_hint="15k Beispiele",
        difficulty="mittel", min_chars=20, tags=["chat"],
    ),
    DatasetInfo(
        "oasst1", "OpenAssistant (mehrsprachig)", "OpenAssistant/oasst1",
        "Von Freiwilligen geschriebene Chat-Nachrichten in vielen Sprachen (auch Deutsch).",
        "Anweisungen", "multi", lambda r: ("### Frage:\n" if r["role"] == "prompter" else "### Antwort:\n") + r["text"],
        size_hint="~85k Nachrichten", difficulty="schwer", min_chars=20, tags=["chat"],
    ),
    DatasetInfo(
        "gsm8k", "GSM8K (Mathe)", "openai/gsm8k",
        "Mathe-Textaufgaben mit Lösungsweg. Für kleine Modelle fast unmöglich – aber lehrreich!",
        "Wissen", "en", lambda r: _instruct(r["question"], "", r["answer"]), config="main",
        size_hint="7.5k Aufgaben", difficulty="schwer", min_chars=20,
    ),
    DatasetInfo(
        "python_code", "Python-Code (18k)", "iamtarun/python_code_instructions_18k_alpaca",
        "Python-Aufgaben mit Lösungs-Code.", "Code", "en",
        lambda r: _instruct(r["instruction"], r["input"], r["output"]), size_hint="18k Beispiele",
        difficulty="schwer", min_chars=40,
    ),
    # ------------------------------------------------------------------ Code
    DatasetInfo(
        "codeparrot", "Python-Quellcode von GitHub (CodeParrot)", "codeparrot/codeparrot-clean",
        "Echte, bereinigte Python-Dateien aus GitHub-Projekten. Die beste Wahl, um ein Modell Code "
        "schreiben zu lassen (Dateien werden auf 6000 Zeichen gekürzt).", "Code", "en",
        lambda r: r["content"], size_hint="~5 Mio. Dateien", difficulty="mittel", min_chars=200,
        max_chars=6000, tags=["empfohlen", "streaming"],
    ),
    DatasetInfo(
        "magicoder", "Magicoder (Aufgabe + Lösung, viele Sprachen)", "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "75k Programmieraufgaben mit Lösungen in Python, C++, Java, Rust, TypeScript u. a. – "
        "von einem grossen Modell aus echtem Open-Source-Code erzeugt.", "Code", "en",
        lambda r: _instruct(r["problem"], "", r["solution"]), size_hint="75k Beispiele", difficulty="mittel",
        min_chars=100, tags=["empfohlen", "chat"],
    ),
    DatasetInfo(
        "evol_code", "Evol-Instruct Code (80k)", "nickrosh/Evol-Instruct-Code-80k-v1",
        "80k immer schwieriger gemachte Programmieraufgaben mit ausführlichen Antworten.", "Code", "en",
        lambda r: _instruct(r["instruction"], "", r["output"]), size_hint="80k Beispiele",
        difficulty="schwer", min_chars=100, max_chars=6000, tags=["chat"],
    ),
    DatasetInfo(
        "code_alpaca", "CodeAlpaca (20k)", "sahil2801/CodeAlpaca-20k",
        "20k kurze Code-Aufgaben – klein, sauber, ideal zum schnellen Ausprobieren.", "Code", "en",
        lambda r: _instruct(r["instruction"], r["input"], r["output"]), size_hint="20k Beispiele",
        difficulty="leicht", min_chars=30, tags=["schnell", "chat"],
    ),
    DatasetInfo(
        "python_25k", "Python-Codes (25k)", "flytech/python-codes-25k",
        "25k kurze Python-Skripte mit Aufgabenbeschreibung.", "Code", "en",
        lambda r: _instruct(r["instruction"], r["input"], r["output"]), size_hint="25k Beispiele",
        difficulty="leicht", min_chars=30, tags=["schnell", "chat"],
    ),
    DatasetInfo(
        "self_oss", "Self-OSS-Instruct (Python, geprüft)", "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "50k Python-Aufgaben, deren Lösungen automatisch ausgeführt und getestet wurden.", "Code", "en",
        lambda r: _instruct(r["instruction"], "", r["response"]), size_hint="50k Beispiele",
        difficulty="mittel", min_chars=100, max_chars=6000, tags=["chat"],
    ),
    # ------------------------------------------------------------ Text (mehr)
    DatasetInfo(
        "cosmopedia_v2", "Cosmopedia v2 (SmolLM-Korpus)", "HuggingFaceTB/smollm-corpus",
        "Hochwertige synthetische Lehrbuch-Texte, Blogposts und Erklärungen – damit wurde SmolLM trainiert. "
        "Sehr gut, um ein kleines Modell zusammenhängend schreiben zu lassen.", "Wissen", "en",
        lambda r: r["text"], config="cosmopedia-v2", size_hint="39 Mio. Texte", difficulty="mittel",
        min_chars=300, max_chars=8000, tags=["empfohlen", "streaming"],
    ),
    DatasetInfo(
        "cosmo_stories", "Cosmopedia Geschichten", "HuggingFaceTB/cosmopedia",
        "Längere, vielfältigere Geschichten als TinyStories (von Mixtral geschrieben).", "Geschichten", "en",
        lambda r: r["text"], config="stories", size_hint="~600k Geschichten", difficulty="mittel",
        min_chars=300, max_chars=8000, tags=["streaming"],
    ),
    DatasetInfo(
        "fineweb2_de", "FineWeb-2 Deutsch", "HuggingFaceFW/fineweb-2",
        "Gefilterter deutscher Web-Text (Nachrichten, Blogs, Foren, Shops). Das beste Material für ein "
        "deutsches Modell.", "Web", "de", lambda r: r["text"], config="deu_Latn",
        size_hint="Milliarden Tokens", difficulty="schwer", min_chars=300, max_chars=8000,
        tags=["empfohlen", "deutsch", "streaming"],
    ),
    DatasetInfo(
        "gutenberg", "Project Gutenberg (Bücher)", "manu/project_gutenberg",
        "Gemeinfreie Romane und Sachbücher. Lange, zusammenhängende Literatur-Sprache "
        "(jedes Buch wird auf 30'000 Zeichen gekürzt).", "Geschichten", "en", lambda r: r["text"],
        split="en", size_hint="~50k Bücher", difficulty="schwer", min_chars=2000, max_chars=30000,
        tags=["streaming"],
    ),
    DatasetInfo(
        "openwebtext", "OpenWebText (GPT-2-Daten)", "Skylion007/openwebtext",
        "Nachbau der Trainingsdaten von GPT-2: Webseiten, die auf Reddit verlinkt wurden.", "Web", "en",
        lambda r: r["text"], size_hint="8 Mio. Dokumente", difficulty="schwer", min_chars=300,
        max_chars=8000, tags=["streaming"],
    ),
    DatasetInfo(
        "pile", "The Pile (ohne Urheberrechtsmaterial)", "monology/pile-uncopyrighted",
        "Bunter Mix aus Web, Büchern, arXiv-Papern, GitHub-Code, Foren und Wikipedia – wie ein "
        "Mini-Internet.", "Web", "en", lambda r: r["text"], size_hint="~1 Bio. Zeichen",
        difficulty="schwer", min_chars=300, max_chars=8000, tags=["streaming"],
    ),
    DatasetInfo(
        "openwebmath", "OpenWebMath", "open-web-math/open-web-math",
        "Mathematik-Webseiten mit Formeln (LaTeX), Beweisen und Erklärungen.", "Wissen", "en",
        lambda r: r["text"], size_hint="6 Mio. Dokumente", difficulty="schwer", min_chars=300,
        max_chars=8000, tags=["streaming"],
    ),
    DatasetInfo(
        "wikitext103", "WikiText-103", "Salesforce/wikitext",
        "Die grosse Variante von WikiText: 100 Mio. Wörter geprüfter Wikipedia-Artikel.", "Wissen", "en",
        lambda r: r["text"], config="wikitext-103-raw-v1", size_hint="~540 MB", difficulty="mittel",
        min_chars=40,
    ),
    # ------------------------------------------------- Anspruchsvoll (für starke Modelle)
    DatasetInfo(
        "finemath", "FineMath (Mathematik-Webseiten)", "HuggingFaceTB/finemath",
        "Sorgfältig gefilterte Webseiten mit Mathematik, Formeln und Herleitungen – Trainingsdaten moderner "
        "Open-Source-Modelle.", "Wissen", "en", lambda r: r["text"], config="finemath-4plus",
        size_hint="~10 Mrd. Tokens", difficulty="sehr schwer", min_chars=300, max_chars=8000,
        tags=["streaming", "mathe"],
    ),
    DatasetInfo(
        "openr1_math", "OpenR1-Math (Aufgaben + Lösungsweg)", "open-r1/OpenR1-Math-220k",
        "220k anspruchsvolle Mathe-Aufgaben (Wettbewerbs-Niveau) mit ausführlicher Lösung.", "Wissen", "en",
        lambda r: _instruct(r["problem"], "", r["solution"]), config="default", size_hint="220k Aufgaben",
        difficulty="sehr schwer", min_chars=100, max_chars=6000, tags=["chat", "mathe"],
    ),
    DatasetInfo(
        "openmath2", "OpenMathInstruct-2", "nvidia/OpenMathInstruct-2",
        "Millionen Mathe-Aufgaben mit Lösungen (NVIDIA).", "Wissen", "en",
        lambda r: _instruct(r["problem"], "", r["generated_solution"]), size_hint="~14 Mio. Beispiele",
        difficulty="sehr schwer", min_chars=60, max_chars=6000, tags=["chat", "mathe", "streaming"],
    ),
    DatasetInfo(
        "codefeedback", "CodeFeedback (Code-Fragen, viele Sprachen)", "m-a-p/CodeFeedback-Filtered-Instruction",
        "Gefilterte Programmier-Fragen mit Antworten – Python, Java, C++, JavaScript, SQL u. a.", "Code", "en",
        lambda r: _instruct(r["query"], "", r["answer"]), size_hint="~156k Beispiele", difficulty="schwer",
        min_chars=60, max_chars=6000, tags=["empfohlen", "chat"],
    ),
    DatasetInfo(
        "smoltalk", "SmolTalk (Chat-Mix von Hugging Face)", "HuggingFaceTB/smoltalk",
        "Grosse Mischung aus Dialogen, Aufgaben, Code und Mathe – damit werden SmolLM-Chatmodelle feingetrimmt.",
        "Anweisungen", "en", lambda r: render_value(r["messages"]), config="all", size_hint="~1 Mio. Dialoge",
        difficulty="sehr schwer", min_chars=60, max_chars=6000, tags=["chat", "streaming"],
    ),
    DatasetInfo(
        "tulu3", "Tulu-3 SFT-Mischung", "allenai/tulu-3-sft-mixture",
        "Die Anweisungs-Mischung des Allen Institute (Wissen, Code, Mathe, Sicherheit).", "Anweisungen", "multi",
        lambda r: render_value(r["messages"]), size_hint="~940k Dialoge", difficulty="sehr schwer", min_chars=60,
        max_chars=6000, tags=["chat", "streaming"],
    ),
    DatasetInfo(
        "slimorca", "SlimOrca (GPT-4-Antworten)", "Open-Orca/SlimOrca",
        "~500k Aufgaben mit ausführlichen Antworten von GPT-4, mit Begründung Schritt für Schritt.", "Anweisungen",
        "en", lambda r: render_value(r["conversations"]), size_hint="~518k Dialoge", difficulty="schwer",
        min_chars=60, max_chars=6000, tags=["chat"],
    ),
    DatasetInfo(
        "guanaco", "Guanaco (mehrsprachiger Chat, 1k)", "mlabonne/guanaco-llama2-1k",
        "1000 Chat-Beispiele in mehreren Sprachen – winzig.", "Anweisungen", "multi",
        lambda r: r["text"].replace("<s>", "").replace("</s>", ""), size_hint="1k Beispiele",
        difficulty="mittel", min_chars=20, tags=["schnell", "chat"],
    ),
]

BY_KEY: dict[str, DatasetInfo] = {d.key: d for d in CATALOG}
CATEGORIES = sorted({d.category for d in CATALOG})


class SchemaChanged(Exception):
    """Das Schema der Datei wechselt mitten im Datensatz – der Aufrufer muss mit dem rohen Leser neu beginnen."""


def is_schema_error(e: Exception) -> bool:
    text = f"{type(e).__name__} {e}".lower()
    return any(k in text for k in ("couldn't cast", "casterror", "datasetgenerationerror", "arrowinvalid",
                                   "failed to parse", "schema", "column names don't match", "expected object"))


def raw_rows(repo: str, split: str | None = None) -> Iterator[dict]:
    """Liest JSON/JSONL-Dateien direkt (ohne festes Schema). Rückfall für Datensätze, deren Zeilen verschiedene
    Strukturen haben und an der ``datasets``-Bibliothek mit einem CastError scheitern."""
    import gzip
    import io
    from huggingface_hub import HfApi, HfFileSystem

    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.lower().endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz"))]
    if split:
        files = [f for f in files if split.lower() in f.lower()] or files
    if not files:
        raise ValueError("Keine JSON-Dateien im Datensatz gefunden.")
    fs = HfFileSystem()
    for f in sorted(files):
        with fs.open(f"datasets/{repo}/{f}", "rb") as raw:
            fh = gzip.open(raw) if f.lower().endswith(".gz") else raw
            if ".jsonl" in f.lower():
                for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
            else:
                data = json.load(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
                rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
                for row in rows:
                    if isinstance(row, dict):
                        yield row


def _rows(info: DatasetInfo, raw: bool) -> Iterator[dict]:
    if raw:
        yield from raw_rows(info.hf_path, info.split)
        return
    from datasets import load_dataset
    yield from load_dataset(info.hf_path, info.config, split=info.split, streaming=True)


def iter_texts(info: DatasetInfo, max_docs: int, raw: bool | None = None) -> Iterator[str]:
    """Liefert bis zu ``max_docs`` Texte aus dem Datensatz (per Streaming, ohne alles herunterzuladen).

    Scheitert das normale Laden am Schema (verschiedene Strukturen in einer Datei), wird automatisch auf den rohen
    JSON-Leser umgeschaltet. Passiert das erst nach einigen Dokumenten, wird ``SchemaChanged`` ausgelöst."""
    n = 0
    try:
        for row in _rows(info, bool(raw)):
            try:
                text = info.fmt(row)
            except Exception:
                continue
            text = text.strip()
            if len(text) < info.min_chars:
                continue
            if info.max_chars and len(text) > info.max_chars:
                cut = text[:info.max_chars]
                text = cut[:cut.rfind("\n")] if "\n" in cut[info.max_chars // 2:] else cut
            yield text
            n += 1
            if n >= max_docs:
                return
    except SchemaChanged:
        raise
    except Exception as e:
        if raw is None and is_schema_error(e):
            if n == 0:
                yield from iter_texts(info, max_docs, raw=True)
                return
            raise SchemaChanged(str(e)) from e
        raise


def raw_path(key: str, max_docs: int) -> Path:
    return RAW_DIR / f"{key}__{max_docs}.txt"


EOT = "<|endoftext|>"


def fetch_dataset(key: str, max_docs: int, progress: Callable[[int, int], None] | None = None,
                  force: bool = False) -> Path:
    """Lädt ``max_docs`` Dokumente und speichert sie lokal (ein Dokument pro Block, getrennt durch EOT)."""
    info = BY_KEY[key]
    path = raw_path(key, max_docs)
    if path.exists() and not force:
        return path
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    def write_all(raw: bool | None) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            for i, text in enumerate(iter_texts(info, max_docs, raw), 1):
                f.write(re.sub(re.escape(EOT), "", text) + "\n" + EOT + "\n")
                if progress and (i % 50 == 0 or i == max_docs):
                    progress(i, max_docs)

    try:
        write_all(None)
    except SchemaChanged:            # Schema wechselt mittendrin: sauber von vorn mit dem rohen JSON-Leser
        write_all(True)
    tmp.replace(path)
    return path


def read_documents(path: Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    return [d.strip() for d in raw.split(EOT) if d.strip()]


# Gespeicherte eigene Datensätze (data/custom_datasets.json) in den Katalog einhängen. Das Modul registriert sie am Ende
# seiner eigenen Datei – so klappt es unabhängig davon, ob zuerst catalog oder custom importiert wird.
try:
    from . import custom as _custom  # noqa: F401
except Exception:      # pragma: no cover - ein kaputter Eintrag darf die App nie am Starten hindern
    pass

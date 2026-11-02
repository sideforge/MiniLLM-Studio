"""Seite 1: Datensätze aus dem Hugging-Face-Katalog auswählen, laden und vorbereiten."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from minillm import custom
from minillm.catalog import BY_KEY, CATALOG, iter_texts
from minillm.data import list_prepared, load_selection, prepare, safe_name
from ui.common import fmt_int, hero, next_step, style_fig

DIFF_CLASS = {"sehr leicht": "good", "leicht": "good", "mittel": "", "schwer": "warn", "sehr schwer": "warn", "unbekannt": ""}
LANG_LABEL = {"en": "Englisch", "de": "Deutsch", "multi": "Mehrsprachig"}


def _card(info) -> None:
    key = info.key
    with st.container(border=True):
        c1, c2, c3 = st.columns([0.07, 0.68, 0.25], vertical_alignment="center")
        on = c1.checkbox("Auswählen", key=f"on_{key}", label_visibility="collapsed")
        tags = "".join(f'<span class="tag {"good" if t in ("empfohlen", "schnell") else ""}">{t}</span>'
                       for t in info.tags)
        c2.markdown(
            f"**{info.title}** &nbsp; <span class='tag'>{info.category}</span>"
            f"<span class='tag'>{LANG_LABEL[info.language]}</span>"
            f"<span class='tag {DIFF_CLASS.get(info.difficulty, '')}'>{info.difficulty}</span>{tags}<br>"
            f"<span class='muted'>{info.description}</span><br>"
            f"<span class='muted' style='font-size:.78rem'>{info.size_hint} · "
            f"<a href='{info.url}' target='_blank'>{info.hf_path}</a></span>",
            unsafe_allow_html=True,
        )
        c3.number_input("Dokumente", min_value=50, max_value=2_000_000, step=1000,
                        key=f"n_{key}", disabled=not on,
                        help="Wie viele Dokumente (Zeilen) geladen werden. Ab ca. 200 000 braucht das viel Arbeitsspeicher "
                             "und Zeit – mehr Daten machen das Modell aber deutlich stärker.")
        if key.startswith("custom_") and c3.button("🗑 entfernen", key=f"rm_{key}"):
            custom.remove_custom(key)
            st.session_state[f"on_{key}"] = False
            st.rerun()


def _restore() -> None:
    """Streamlit löscht den Zustand nicht gezeichneter Widgets (Filter / Seitenwechsel) – wir merken ihn uns selbst."""
    keep = st.session_state.setdefault("data_sel", {})
    for d in CATALOG:
        keep.setdefault(f"on_{d.key}", d.key == "tinystories")
        keep.setdefault(f"n_{d.key}", 20000 if d.key == "shakespeare" else 3000)
        for k in (f"on_{d.key}", f"n_{d.key}"):
            if k not in st.session_state:
                st.session_state[k] = keep[k]


def _save() -> None:
    keep = st.session_state["data_sel"]
    for d in CATALOG:
        for k in (f"on_{d.key}", f"n_{d.key}"):
            if k in st.session_state:
                keep[k] = st.session_state[k]


MIXES = {
    "📖 Geschichten (schnell)": {"tinystories": 5000},
    "💻 Code": {"codeparrot": 4000, "magicoder": 4000, "code_alpaca": 2000},
    "🇩🇪 Deutsch": {"fineweb2_de": 4000, "wikipedia_de": 2000, "tagesschau": 2000},
    "🌍 Wissen & Text": {"cosmopedia_v2": 3000, "fineweb_edu": 2000, "gutenberg": 300},
    "🧪 Text + Code": {"cosmopedia_v2": 3000, "codeparrot": 3000, "tinystories": 2000},
}


def _apply_mix(name: str) -> None:
    for d in CATALOG:
        st.session_state[f"on_{d.key}"] = d.key in MIXES[name]
        if d.key in MIXES[name]:
            st.session_state[f"n_{d.key}"] = MIXES[name][d.key]


# ----------------------------------------------------------- Eigene Datensätze vom Hub
@st.cache_data(ttl=900, show_spinner="Frage Hugging Face …")
def _hub_search(query: str, task: str | None) -> list[dict] | str:
    try:
        return custom.search_hub(query, 30, task)
    except Exception as e:
        return custom.friendly_error(e)


@st.cache_data(ttl=900, show_spinner="Prüfe Datensatz …")
def _hub_probe(repo: str, config: str | None, split: str | None) -> dict:
    """Konfigurationen, Splits und Spalten eines Datensatzes (alles gestreamt, nichts wird heruntergeladen)."""
    out: dict = {"configs": [], "splits": [], "error": None}
    try:
        try:
            out["configs"] = custom.list_configs(repo)
        except Exception:
            out["configs"] = []
        if out["configs"] and config not in out["configs"]:
            config = out["configs"][0]
        out["config"] = config
        try:
            out["splits"] = custom.list_splits(repo, config)
        except Exception:
            out["splits"] = ["train"]
        sp = split if split in out["splits"] else ("train" if "train" in out["splits"] else out["splits"][0])
        out["split"] = sp
        r = custom.inspect_dataset(repo, config, sp, 3)
        out["columns"], out["recommended"] = r["columns"], r["recommended"]
        out["rows"] = [{c: custom.render_value(v)[:3000] for c, v in row.items()} for row in r["rows"]]
    except Exception as e:
        out["error"] = custom.friendly_error(e)
    return out


TASKS = {"Alle": None, "Text erzeugen": "task_categories:text-generation",
         "Frage & Antwort": "task_categories:question-answering", "Zusammenfassung": "task_categories:summarization"}


def _custom_section() -> None:
    with st.expander("➕ Eigenen Hugging-Face-Datensatz hinzufügen – beliebige Texte, Code, Wissen …", expanded=False):
        st.caption("Es geht jeder öffentliche Datensatz von huggingface.co (im Parquet-Format). Du suchst, prüfst die Spalten "
                   "und fügst ihn dem Katalog hinzu. Er wird wie alle anderen geladen, gemischt und trainiert.")
        if pick := st.session_state.pop("_cd_pick", None):
            st.session_state["cd_repo"] = pick
        c1, c2, c3 = st.columns([0.5, 0.3, 0.2], vertical_alignment="bottom")
        query = c1.text_input("Im Hub suchen", key="cd_query", placeholder="z. B. python code, german news, math, wikipedia")
        task = c2.selectbox("Art", list(TASKS), key="cd_task")
        searching = c3.button("🔎 Suchen", width="stretch")
        if searching or st.session_state.get("cd_searched"):
            st.session_state["cd_searched"] = True
            res = _hub_search(query, TASKS[task])
            if isinstance(res, str):
                st.error(res)
            elif res:
                import pandas as pd
                df = pd.DataFrame([{"Datensatz": r["id"], "Downloads": r["downloads"], "Likes": r["likes"],
                                    "Gesperrt": "🔒" if r["gated"] else "", "Themen": ", ".join(r["tags"])} for r in res])
                ev = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                                  key="cd_table", height=260)
                rows = ev.selection.rows if ev and ev.selection else []
                if rows:
                    chosen_repo = df.iloc[rows[0]]["Datensatz"]
                    if chosen_repo != st.session_state.get("cd_repo"):
                        st.session_state["_cd_pick"] = chosen_repo
                        st.rerun()
                st.caption("Zeile anklicken, um den Datensatz unten zu übernehmen. 🔒 = gesperrt (Anmeldung nötig).")
            else:
                st.info("Nichts gefunden.")

        repo = st.text_input("Datensatz (Format `besitzer/name`)", key="cd_repo",
                             placeholder="z. B. codeparrot/codeparrot-clean").strip()
        if not repo:
            return
        a, b = st.columns(2)
        probe = _hub_probe(repo, st.session_state.get("cd_config"), st.session_state.get("cd_split"))
        if probe["error"]:
            st.error(probe["error"])
            return
        if len(probe["configs"]) > 1:
            a.selectbox("Konfiguration", probe["configs"], key="cd_config",
                        index=probe["configs"].index(probe["config"]) if probe["config"] in probe["configs"] else 0)
            probe = _hub_probe(repo, st.session_state["cd_config"], st.session_state.get("cd_split"))
            if probe["error"]:
                st.error(probe["error"])
                return
        b.selectbox("Split", probe["splits"], key="cd_split",
                    index=probe["splits"].index(probe["split"]) if probe["split"] in probe["splits"] else 0)
        if st.session_state.get("cd_split") != probe["split"]:
            probe = _hub_probe(repo, probe["config"], st.session_state["cd_split"])
            if probe["error"]:
                st.error(probe["error"])
                return

        cols = list(probe["columns"])
        st.write("**Spalten** (so sieht die erste Zeile aus):")
        st.dataframe([{"Spalte": c, "Typ": v["type"], "Beispiel": v["example"]} for c, v in probe["columns"].items()],
                     hide_index=True, width="stretch")
        key_cols = f"cd_cols_{repo}_{probe['config']}_{probe['split']}"
        chosen = st.multiselect("Welche Spalten sind der Trainingstext?", cols, default=probe["recommended"], key=key_cols,
                                help="Mehrere Spalten werden nacheinander zu einem Text zusammengefügt (z. B. Frage + Antwort).")
        d1, d2 = st.columns(2)
        headings = d1.checkbox("Spaltennamen als Überschriften (### Spalte:)", value=True, key="cd_headings")
        max_chars = d2.number_input("Dokumente kürzen auf … Zeichen (0 = nicht kürzen)", 0, 200_000, 8000, 1000, key="cd_max")
        if chosen:
            fmt = custom.make_formatter(chosen, headings)
            sample = fmt(probe["rows"][0])
            st.write("**Vorschau – so sieht ein Trainingsdokument aus:**")
            st.markdown(f'<div class="sample">{sample[:1200].replace("<", "&lt;")}</div>', unsafe_allow_html=True)
        if st.button("✅ Zum Katalog hinzufügen", type="primary", disabled=not chosen):
            info = custom.add_custom(repo, chosen, probe["config"] if probe["configs"] else None, probe["split"],
                                     headings, max_chars=int(max_chars) or None)
            st.session_state.setdefault("data_sel", {})[f"on_{info.key}"] = True
            st.session_state[f"on_{info.key}"] = True
            st.session_state["flash"] = f"**{repo}** wurde zum Katalog hinzugefügt und ist ausgewählt (ganz unten in der Liste)."
            st.rerun()


def render() -> None:
    hero("📚 Daten", "Wähle einen oder mehrere Datensätze von Hugging Face. Sie werden gemischt und zum Training vorbereitet.")
    if msg := st.session_state.pop("flash", None):
        st.success(msg)
        c1, c2, c3 = st.columns(3)
        with c1:
            next_step("train", "Direkt zum Training", "🏋️")
        with c2:
            next_step("model", "Modell einstellen", "🧠")
        with c3:
            next_step("tokenizer", "Tokenizer ansehen", "🔤")

    st.write("**Fertige Mischungen (ein Klick):**")
    mc = st.columns(len(MIXES))
    for col, name in zip(mc, MIXES):
        col.button(name, key=f"mix_{name}", on_click=_apply_mix, args=(name,), width="stretch")

    _custom_section()

    with st.expander("🔎 Filter", expanded=False):
        f1, f2, f3 = st.columns(3)
        cats = f1.multiselect("Kategorie", sorted({d.category for d in CATALOG}), default=[])
        langs = f2.multiselect("Sprache", list(LANG_LABEL.values()), default=[])
        diffs = f3.multiselect("Schwierigkeit für kleine Modelle", ["sehr leicht", "leicht", "mittel", "schwer", "sehr schwer", "unbekannt"])
    lang_rev = {v: k for k, v in LANG_LABEL.items()}
    shown = [d for d in CATALOG
             if (not cats or d.category in cats) and (not langs or d.language in [lang_rev[l] for l in langs])
             and (not diffs or d.difficulty in diffs)]

    _restore()

    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        st.subheader(f"Katalog ({len(shown)} von {len(CATALOG)})")
        for info in shown:
            _card(info)

    _save()
    selection = {d.key: int(st.session_state[f"n_{d.key}"]) for d in CATALOG if st.session_state[f"on_{d.key}"]}

    with right:
        st.subheader("Deine Mischung")
        if not selection:
            st.info("Hake links mindestens einen Datensatz an.")
        else:
            fig = px.pie(names=[BY_KEY[k].title for k in selection], values=list(selection.values()), hole=0.55)
            fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#0E1117", width=2)))
            st.plotly_chart(style_fig(fig, 260, legend=True), width="stretch")
            st.caption(f"**{sum(selection.values()):,}** Dokumente aus **{len(selection)}** Datensätzen")

            with st.expander("👀 Vorschau (je 2 Beispiele von Hugging Face)"):
                if st.button("Beispiele laden", key="preview"):
                    for k in selection:
                        st.markdown(f"**{BY_KEY[k].title}**")
                        try:
                            for t in iter_texts(BY_KEY[k], 2):
                                st.markdown(f'<div class="sample">{t[:500].replace("<", "&lt;")}</div>',
                                            unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Konnte nicht laden: {e}")

        st.divider()
        st.subheader("Vorbereiten")
        name = st.text_input("Name des Datensatzes", value="mein-datensatz", key="prep_name")
        vocab = st.select_slider("Vokabular-Grösse (Tokenizer)", options=[300, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768],
                                 value=1024, help="Anzahl verschiedener Tokens. Grösser = kürzere Sequenzen, aber "
                                                  "mehr Parameter im Embedding.")
        val_frac = st.slider("Anteil Validierungsdaten", 0.02, 0.20, 0.05, 0.01,
                             help="Diese Texte sieht das Modell beim Training nie – damit prüfen wir, ob es wirklich lernt.")
        go = st.button("🚀 Laden & vorbereiten", type="primary", disabled=not selection, width="stretch")

    if go:
        _prepare(name, selection, vocab, val_frac)

    prepared = list_prepared()
    if prepared:
        st.divider()
        st.subheader("Bereits vorbereitet")
        rows = [{"Name": m["name"], "Datensätze": ", ".join(f"{k} ({n})" for k, n in m["datasets"].items()),
                 "Vokabular": m["vocab_size"], "Tokens (Train)": fmt_int(m["tokens_train"]),
                 "Zeichen/Token": round(m["chars_per_token"], 2), "Erstellt": m["created"]} for m in prepared]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("Gewählter Datensatz = Auswahl links in der Seitenleiste.")
        next_step("train", "Weiter zum Training", "🏋️")


def _prepare(name: str, selection: dict[str, int], vocab: int, val_frac: float) -> None:
    name = safe_name(name)
    with st.status("Daten werden vorbereitet …", expanded=True) as status:
        st.write("**1/3 · Texte von Hugging Face laden**")
        bar = st.progress(0.0)
        keys = list(selection)

        def on_fetch(k: str, i: int, total: int) -> None:
            idx = keys.index(k)
            bar.progress(min(1.0, (idx + i / total) / len(keys)), text=f"{BY_KEY[k].title}: {i}/{total}")

        try:
            docs = load_selection(selection, on_fetch)
        except Exception as e:
            status.update(label="Fehler beim Laden", state="error")
            st.error(f"{type(e).__name__}: {e}")
            return
        bar.progress(1.0, text="Alle Datensätze geladen")

        st.write("**2/3 · Eigenen BPE-Tokenizer trainieren**")
        tbar = st.progress(0.0)
        last = st.empty()

        def on_tok(m: int, total: int, token: bytes) -> None:
            tbar.progress(m / total, text=f"Merge {m}/{total}")
            last.caption("Neuestes Token: `" + token.decode("utf-8", errors="replace").replace("\n", "⏎") + "`")

        st.write("**3/3 · Texte in Token-IDs umwandeln**")
        ebar = st.progress(0.0)
        meta = prepare(name, docs, vocab, val_fraction=val_frac, tok_progress=on_tok,
                       enc_progress=lambda i, n: ebar.progress(i / n, text=f"{i}/{n} Dokumente"))
        status.update(label=f"Fertig: {fmt_int(meta['tokens_train'])} Trainings-Tokens", state="complete")
    st.session_state["pending_active"] = meta["name"]
    st.session_state["flash"] = f"Datensatz **{meta['name']}** ist bereit. Weiter geht's bei 🔤 Tokenizer oder 🧠 Modell."
    st.rerun()

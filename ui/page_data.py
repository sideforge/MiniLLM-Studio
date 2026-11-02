"""Seite 1: Datensätze aus dem Hugging-Face-Katalog auswählen, laden und vorbereiten."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from minillm.catalog import BY_KEY, CATALOG, CATEGORIES, iter_texts
from minillm.data import list_prepared, load_selection, prepare, safe_name
from ui.common import fmt_int, hero, next_step, style_fig

DIFF_CLASS = {"sehr leicht": "good", "leicht": "good", "mittel": "", "schwer": "warn"}
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
            f"<span class='tag {DIFF_CLASS[info.difficulty]}'>{info.difficulty}</span>{tags}<br>"
            f"<span class='muted'>{info.description}</span><br>"
            f"<span class='muted' style='font-size:.78rem'>{info.size_hint} · "
            f"<a href='{info.url}' target='_blank'>{info.hf_path}</a></span>",
            unsafe_allow_html=True,
        )
        c3.number_input("Dokumente", min_value=50, max_value=200_000, step=500,
                        key=f"n_{key}", disabled=not on, help="Wie viele Dokumente (Zeilen) geladen werden.")


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

    with st.expander("🔎 Filter", expanded=False):
        f1, f2, f3 = st.columns(3)
        cats = f1.multiselect("Kategorie", CATEGORIES, default=[])
        langs = f2.multiselect("Sprache", list(LANG_LABEL.values()), default=[])
        diffs = f3.multiselect("Schwierigkeit für kleine Modelle", ["sehr leicht", "leicht", "mittel", "schwer"])
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
        vocab = st.select_slider("Vokabular-Grösse (Tokenizer)", options=[300, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192],
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

"""MiniLLM Studio – Hauptprogramm der App.   Start:  streamlit run app.py"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st

from ui import page_data, page_generate, page_model, page_runs, page_start, page_tokenizer, page_train
from ui.common import PAGES, fmt_time, prepared_names, setup_page, store

st.set_page_config(page_title="MiniLLM Studio", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
setup_page()

# Ein frisch vorbereiteter Datensatz wird automatisch ausgewählt (muss vor dem Widget passieren).
if pending := st.session_state.pop("pending_active", None):
    st.session_state["active_data"] = pending

pages = [
    st.Page(page_start.render, title="Start", icon="🏠", url_path="start", default=True),
    st.Page(page_data.render, title="Daten", icon="📚", url_path="daten"),
    st.Page(page_tokenizer.render, title="Tokenizer", icon="🔤", url_path="tokenizer"),
    st.Page(page_model.render, title="Modell", icon="🧠", url_path="modell"),
    st.Page(page_train.render, title="Training", icon="🏋️", url_path="training"),
    st.Page(page_generate.render, title="Generieren", icon="✨", url_path="generieren"),
    st.Page(page_runs.render, title="Läufe vergleichen", icon="📊", url_path="laeufe"),
]
PAGES.update(dict(zip(["start", "data", "tokenizer", "model", "train", "generate", "runs"], pages)))
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🧠 MiniLLM Studio")
    names = prepared_names()
    if names:
        if st.session_state.get("active_data") not in names:
            st.session_state["active_data"] = names[-1]
        st.selectbox("Aktiver Datensatz", names, key="active_data")
    else:
        st.caption("Noch kein Datensatz vorbereitet.")
    t = store()["trainer"]
    if t is not None:
        st.divider()
        icon = "🟢" if t.is_running else "⚪"
        st.markdown(f"**{icon} Lauf `{t.run_name}`**")
        st.progress(min(1.0, t.step / t.cfg.max_steps), text=f"{t.step}/{t.cfg.max_steps} · {t.status}")
        if t.evals:
            st.caption(f"Val-Loss {t.evals[-1]['val']:.3f} · {fmt_time(t.elapsed)}")
    st.divider()
    st.caption("Schulprojekt · GPT von Grund auf in PyTorch")

nav.run()

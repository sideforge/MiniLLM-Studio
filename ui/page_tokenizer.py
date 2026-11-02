"""Seite 2: Den Tokenizer und die Daten genauer anschauen."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from minillm.data import prepared_path
from minillm.tokenizer import BPETokenizer
from ui.common import PALETTE, fmt_int, hero, need_data, style_fig, token_chips


@st.cache_resource(show_spinner=False)
def _load_tok(path: str, stamp: float) -> BPETokenizer:
    return BPETokenizer.load(path)


@st.cache_data(show_spinner="Trainiere Test-Tokenizer …")
def _vocab_experiment(sample: str, sizes: tuple[int, ...]) -> list[dict]:
    rows = []
    for v in sizes:
        t = BPETokenizer.train([sample], v)
        n = len(t.encode(sample))
        rows.append({"Vokabular": v, "Zeichen pro Token": len(sample) / n, "Tokens": n})
    return rows


def render() -> None:
    hero("🔤 Tokenizer & Daten", "Bevor das Modell lernt, wird jeder Text in Tokens zerlegt. Hier siehst du, wie das aussieht.")
    meta = need_data()
    d = prepared_path(meta["name"])
    tok = _load_tok(str(d / "tokenizer.json"), (d / "tokenizer.json").stat().st_mtime)

    c = st.columns(5)
    c[0].metric("Vokabular", f"{meta['vocab_size']:,}")
    c[1].metric("Trainings-Tokens", fmt_int(meta["tokens_train"]))
    c[2].metric("Validierungs-Tokens", fmt_int(meta["tokens_val"]))
    c[3].metric("Zeichen pro Token", f"{meta['chars_per_token']:.2f}",
                help="Je höher, desto stärker komprimiert der Tokenizer den Text.")
    c[4].metric("Dokumente", f"{meta['n_docs_train']:,}")

    st.subheader("✍️ Ausprobieren")
    text = st.text_area("Text eingeben", "Once upon a time, a little dog named Max found a shiny red ball.",
                        height=90, key="tok_text")
    pieces = tok.pieces(text)
    st.markdown(token_chips(pieces), unsafe_allow_html=True)
    m = st.columns(3)
    m[0].metric("Zeichen", len(text))
    m[1].metric("Tokens", len(pieces))
    m[2].metric("Zeichen pro Token", f"{len(text) / max(1, len(pieces)):.2f}")
    with st.expander("Token-IDs ansehen"):
        st.code(str([i for i, _ in pieces]))
        st.caption("`·` steht für ein Leerzeichen, `⏎` für einen Zeilenumbruch. Jede Zahl ist ein Eintrag im Vokabular.")

    left, right = st.columns(2, gap="large")
    with left:
        top = pd.DataFrame(meta["top_tokens"][:30], columns=["Token", "Anzahl"])
        fig = px.bar(top[::-1], x="Anzahl", y="Token", orientation="h", title="Häufigste Tokens")
        fig.update_traces(marker_color=PALETTE["violet"])
        st.plotly_chart(style_fig(fig, 520, legend=False), width="stretch")
    with right:
        lens = meta["doc_token_lengths"]
        fig = px.histogram(x=lens, nbins=40, title="Länge der Dokumente (in Tokens)")
        fig.update_traces(marker_color=PALETTE["cyan"])
        fig.update_layout(xaxis_title="Tokens pro Dokument", yaxis_title="Anzahl Dokumente")
        st.plotly_chart(style_fig(fig, 250, legend=False), width="stretch")

        counts = sorted((c for c in meta["token_counts"] if c > 0), reverse=True)
        fig = go.Figure(go.Scatter(x=list(range(1, len(counts) + 1)), y=counts, mode="lines",
                                   line=dict(color=PALETTE["orange"], width=2)))
        fig.update_layout(title="Zipf-Verteilung: wenige Tokens sind sehr häufig", xaxis_type="log",
                          yaxis_type="log", xaxis_title="Rang des Tokens", yaxis_title="Häufigkeit")
        st.plotly_chart(style_fig(fig, 250, legend=False), width="stretch")

    st.subheader("🧩 So hat der Tokenizer gelernt")
    st.caption("BPE verschmilzt immer das häufigste Paar. Die ersten Merges sind Buchstaben-Paare, "
               "später entstehen ganze Wörter.")
    merges = meta["first_merges"]
    cols = st.columns(4)
    for i, (t, n) in enumerate(merges[:40]):
        cols[i % 4].markdown(f"`{i + 1:>2}.` **{t}** <span class='muted'>({n:,}×)</span>", unsafe_allow_html=True)

    later = [(256 + i, tok.token_str(256 + i)) for i in range(len(tok.merges) - 30, len(tok.merges))] \
        if len(tok.merges) > 70 else []
    if later:
        st.caption("Die letzten gelernten Tokens (seltener, dafür länger):")
        st.markdown(token_chips(later), unsafe_allow_html=True)

    with st.expander("📈 Experiment: Wie wirkt sich die Vokabular-Grösse aus?"):
        st.caption("Wir trainieren mehrere Tokenizer auf einem kleinen Textausschnitt und messen die Kompression.")
        if st.button("Experiment starten"):
            from minillm.catalog import read_documents, raw_path  # noqa: F401
            from minillm.catalog import RAW_DIR
            files = sorted(RAW_DIR.glob("*.txt"))
            if files:
                sample = "\n".join(read_documents(files[0]))[:300_000]
                rows = _vocab_experiment(sample, (300, 400, 600, 1000, 1500, 2500))
                fig = px.line(pd.DataFrame(rows), x="Vokabular", y="Zeichen pro Token", markers=True)
                fig.update_traces(line_color=PALETTE["green"])
                st.plotly_chart(style_fig(fig, 300, legend=False), width="stretch")
                st.caption("Mehr Tokens im Vokabular → jedes Token steht für mehr Zeichen → kürzere Sequenzen. "
                           "Dafür wird die Embedding-Tabelle grösser und seltene Tokens werden schlechter gelernt.")

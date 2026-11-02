"""Seite 3: Modell-Architektur einstellen und als Diagramm sehen."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from minillm import device as dev
from minillm.model import GPT, GPTConfig
from ui.common import PALETTE, fmt_int, hero, need_data, next_step, restore_state, save_state, style_fig

PRESETS = {
    "Winzig": dict(m_layer=2, m_embd=64, m_head=2, m_block=64),
    "Klein": dict(m_layer=4, m_embd=128, m_head=4, m_block=128),
    "Mittel": dict(m_layer=6, m_embd=256, m_head=8, m_block=128),
    "Gross": dict(m_layer=8, m_embd=384, m_head=6, m_block=256),
    "Riesig": dict(m_layer=12, m_embd=768, m_head=12, m_block=512),      # ca. 90 Mio. Parameter
    "Extrem": dict(m_layer=16, m_embd=1024, m_head=16, m_block=512),     # ca. 210 Mio. Parameter
}
DEFAULTS = dict(PRESETS["Klein"], m_drop=0.0)
EMBD_OPTIONS = [32, 48, 64, 96, 128, 192, 256, 288, 384, 512, 640, 768, 1024]
BLOCK_OPTIONS = [32, 64, 128, 192, 256, 384, 512, 1024]


def _apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def _apply_gpu_max() -> None:
    """Modell so gross wählen, dass die GPU wirklich zu tun hat (abhängig vom Grafikspeicher)."""
    for k, v in dev.suggest_model(dev.describe()["memory_gb"]).items():
        st.session_state[k] = v


def model_config(vocab_size: int) -> GPTConfig:
    s = st.session_state
    return GPTConfig(vocab_size=vocab_size, block_size=s["m_block"], n_layer=s["m_layer"], n_head=s["m_head"],
                     n_embd=s["m_embd"], dropout=s["m_drop"])


def architecture_dot(cfg: GPTConfig) -> str:
    d, H, L, V, T = cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.vocab_size, cfg.block_size
    return f"""
digraph G {{
  rankdir=BT; bgcolor="transparent"; nodesep=0.5; ranksep=0.32;
  node [shape=box, style="rounded,filled", fillcolor="#161B26", color="#8892A6", fontcolor="#E6E9F0",
        fontname="Helvetica", fontsize=12, margin="0.2,0.1", penwidth=1.4];
  edge [color="#8892A6", penwidth=1.4, arrowsize=0.8];

  inp  [label="Eingabe: Token-IDs\\n(höchstens {T} Tokens)", color="#22D3EE"];
  emb  [label="Token-Embedding  ({V} × {d})\\n+ Positions-Embedding  ({T} × {d})", color="#22D3EE"];
  inp -> emb;

  subgraph cluster_block {{
    label="{L} × Transformer-Block"; labeljust=l; fontcolor="#a99bff"; fontname="Helvetica-Bold"; fontsize=13;
    color="#7C5CFF"; style="rounded,dashed"; penwidth=1.6;
    ln1 [label="LayerNorm"];
    att [label="Causal Self-Attention\\n{H} Köpfe × {d // H} Dimensionen", color="#FB923C", penwidth=2];
    add1 [label="+", shape=circle, width=0.3, fixedsize=true, fillcolor="#252C3D"];
    ln2 [label="LayerNorm"];
    mlp [label="MLP\\n{d} → {4 * d} → {d}  (GELU)", color="#34D399", penwidth=2];
    add2 [label="+", shape=circle, width=0.3, fixedsize=true, fillcolor="#252C3D"];
    ln1 -> att -> add1 -> ln2 -> mlp -> add2;
  }}
  emb -> ln1;
  emb -> add1 [style=dashed, color="#7C5CFF", label=" Residual", fontcolor="#7C5CFF", fontsize=10, constraint=false];
  add1 -> add2 [style=dashed, color="#7C5CFF", constraint=false];

  lnf  [label="LayerNorm (final)"];
  head [label="Linear  {d} → {V}\\n(teilt Gewichte mit dem Embedding)", color="#F472B6"];
  out  [label="Softmax: Wahrscheinlichkeit für\\njedes mögliche nächste Token", color="#F472B6", penwidth=2];
  add2 -> lnf -> head -> out;
}}
"""


def render() -> None:
    hero("🧠 Modell", "Ein GPT-Transformer. Stelle die Grösse ein und sieh sofort, wie viele Parameter er hat.")
    meta = need_data()
    restore_state("model", DEFAULTS)

    info = dev.describe()
    on_gpu = info["type"] != "cpu"
    st.write("**Schnell-Vorlagen:**")
    pc = st.columns(len(PRESETS) + 1)
    for col, name in zip(pc, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=_apply_preset, args=(name,), width="stretch")
    pc[-1].button("🚀 GPU-Maximum", key="preset_gpumax", on_click=_apply_gpu_max, width="stretch", type="primary",
                  disabled=not on_gpu,
                  help=("Wählt die grösste sinnvolle Modellgrösse für deinen Grafikspeicher, damit die GPU voll "
                        "ausgelastet ist." if on_gpu else "Keine GPU gefunden."))

    left, right = st.columns([0.4, 0.6], gap="large")
    with left:
        st.subheader("Einstellungen")
        st.select_slider("Embedding-Grösse (Breite)", EMBD_OPTIONS, key="m_embd",
                         help="Mit wie vielen Zahlen wird jedes Token dargestellt?")
        heads = [h for h in (1, 2, 3, 4, 6, 8, 12, 16) if st.session_state["m_embd"] % h == 0]
        if st.session_state["m_head"] not in heads:
            st.session_state["m_head"] = min(heads, key=lambda h: abs(h - st.session_state["m_head"]))
        st.select_slider("Aufmerksamkeits-Köpfe", heads, key="m_head",
                         help="Jeder Kopf schaut auf andere Zusammenhänge im Text.")
        st.slider("Anzahl Schichten (Tiefe)", 1, 24, key="m_layer")
        st.select_slider("Kontextlänge (Tokens)", BLOCK_OPTIONS, key="m_block",
                         help="So viele Tokens kann das Modell gleichzeitig überblicken.")
        st.slider("Dropout", 0.0, 0.3, step=0.05, key="m_drop",
                  help="Zufälliges Abschalten von Neuronen gegen Überanpassung. Bei viel Daten: 0.")
        save_state("model")

    cfg = model_config(meta["vocab_size"])
    model = GPT(cfg)
    n = model.num_params()

    with right:
        st.subheader("Architektur")
        st.graphviz_chart(architecture_dot(cfg), width="stretch")

    c = st.columns(4)
    c[0].metric("Parameter", fmt_int(n), help=f"Genau: {n:,}")
    c[1].metric("Speicher fürs Training", f"{n * 16 / 1e6:.0f} MB",
                help="Gewichte + Gradienten + 2 Adam-Zustände, je 4 Byte")
    c[2].metric("Rechenaufwand / Token", f"{6 * n / 1e6:.1f} MFLOPs", help="Faustregel: 6 × Parameter")
    ratio = meta["tokens_train"] / n
    c[3].metric("Daten-Tokens pro Parameter", f"{ratio:.1f}",
                help="Faustregel (Chinchilla): ca. 20 Tokens pro Parameter sind optimal.")
    if not on_gpu and n > 15_000_000:
        st.warning(f"Ein Modell mit {fmt_int(n)} Parametern ist auf der **CPU sehr langsam** (Stunden bis Tage). "
                   "Dafür ist eine GPU gedacht – oder wähle eine kleinere Vorlage.")
    if ratio < 5:
        st.warning("Das Modell ist im Verhältnis zu den Daten gross → es wird die Daten eher **auswendig lernen** "
                   "(Overfitting). Nimm mehr Daten oder ein kleineres Modell.")
    elif ratio > 200:
        st.info("Du hast viel mehr Daten als das Modell aufnehmen kann → ein grösseres Modell könnte mehr lernen.")

    st.subheader("Wo stecken die Parameter?")
    bd = model.param_breakdown()
    df = pd.DataFrame({"Bauteil": list(bd), "Parameter": list(bd.values())})
    a, b = st.columns(2)
    fig = px.pie(df, names="Bauteil", values="Parameter", hole=0.5, title="Anteile")
    fig.update_traces(marker=dict(line=dict(color="#0E1117", width=2)))
    a.plotly_chart(style_fig(fig, 320), width="stretch")
    fig = px.bar(df[::-1], x="Parameter", y="Bauteil", orientation="h", title="Anzahl")
    fig.update_traces(marker_color=PALETTE["violet"])
    b.plotly_chart(style_fig(fig, 320, legend=False), width="stretch")
    st.caption("Bei kleinen Modellen mit grossem Vokabular steckt ein Grossteil der Parameter im Embedding. "
               "Bei grösseren Modellen dominieren Attention + MLP.")

    st.divider()
    next_step("train", "Weiter zum Training", "🏋️")

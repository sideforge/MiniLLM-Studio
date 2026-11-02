"""Seite 5: Mit dem trainierten Modell Text erzeugen und verstehen, wie es entscheidet."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from minillm.generate import attention_maps, filter_logits, next_logits, stream_generate
from minillm.trainer import RUNS_DIR, list_runs, load_model
from ui.common import PALETTE, fmt_int, hero, restore_state, save_state, store, style_fig

DEFAULTS = dict(g_prompt="Once upon a time", g_tokens=120, g_temp=0.8, g_topk=40, g_topp=1.0, g_seed=-1)


@st.cache_resource(show_spinner="Lade Modell …")
def _load(run: str, stamp: float):
    return load_model(run)


def render() -> None:
    hero("✨ Generieren", "Gib einen Textanfang ein – das Modell schreibt weiter. Und du siehst, wie es sich entscheidet.")
    restore_state("gen", DEFAULTS)
    runs = list_runs()
    if not runs:
        st.info("Es gibt noch kein trainiertes Modell. Starte zuerst ein Training unter **🏋️ Training**.")
        st.stop()

    live = store()["trainer"]
    names = [r["name"] for r in runs]
    default_idx = names.index(live.run_name) if live and live.run_name in names else 0
    c = st.columns([0.45, 0.55])
    run = c[0].selectbox("Modell", names, index=default_idx, key="g_run",
                         format_func=lambda n: f"{n}  ·  {fmt_int(next(r['n_params'] for r in runs if r['name'] == n))} Param.")
    info = next(r for r in runs if r["name"] == run)
    c[1].caption(f"Trainiert auf **{info['data']}** · Schritt {info['step']}/{info['max_steps']} · "
                 f"Val-Loss {info['best_val']:.3f}" if info.get("best_val") else f"Schritt {info['step']}")
    if live and live.is_running and live.run_name == run:
        st.warning("Dieses Modell wird gerade noch trainiert – du siehst den letzten gespeicherten Zwischenstand.")

    model, tok, _ = _load(run, (RUNS_DIR / run / "model.pt").stat().st_mtime)

    left, right = st.columns([0.55, 0.45], gap="large")
    with left:
        st.text_area("Textanfang (Prompt)", key="g_prompt", height=110)
        b1, b2 = st.columns([1, 3])
        gen_clicked = b1.button("✨ Erzeugen", type="primary", width="stretch")
        b2.caption("Tipp: Leer lassen, damit das Modell frei erfindet.")
    with right:
        st.slider("Maximale Länge (Tokens)", 10, 400, key="g_tokens")
        a, b = st.columns(2)
        a.slider("Temperatur", 0.1, 2.0, step=0.05, key="g_temp",
                 help="Niedrig = vorsichtig & wiederholend, hoch = kreativ & chaotisch.")
        b.slider("Top-k", 0, 200, key="g_topk", help="Nur die k wahrscheinlichsten Tokens kommen in Frage (0 = aus).")
        a.slider("Top-p", 0.1, 1.0, step=0.05, key="g_topp",
                 help="Nur die kleinste Token-Menge, die zusammen p Wahrscheinlichkeit hat (1.0 = aus).")
        b.number_input("Seed (-1 = Zufall)", -1, 10**6, key="g_seed")
    save_state("gen")
    s = st.session_state
    prompt = s["g_prompt"]

    if gen_clicked:
        box = st.empty()
        out, rows = "", []
        seed = None if s["g_seed"] < 0 else int(s["g_seed"])
        for tid, piece, probs, raw in stream_generate(model, tok, prompt, s["g_tokens"], s["g_temp"],
                                                      s["g_topk"], s["g_topp"], seed):
            if tid == tok.eot_id:
                rows.append((tok.token_str(tid), float(probs[tid]), float(raw[tid])))
                out += "\n\n"
                break
            out += piece
            rows.append((tok.token_str(tid), float(probs[tid]), float(raw[tid])))
            box.markdown(f'<div class="sample">{html.escape(prompt)}<span style="color:#a99bff">{html.escape(out)}</span></div>',
                         unsafe_allow_html=True)
        s["g_result"] = {"prompt": prompt, "text": out, "rows": rows, "run": run}

    res = s.get("g_result")
    if res and not gen_clicked:
        st.markdown(f'<div class="sample">{html.escape(res["prompt"])}<span style="color:#a99bff">{html.escape(res["text"])}</span></div>',
                    unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Wie sicher war das Modell?", "🎲 Nächstes Token", "🌡 Temperatur-Effekt", "👁 Aufmerksamkeit"])

    with tabs[0]:
        if res and res["run"] == run and res["rows"]:
            df = pd.DataFrame(res["rows"], columns=["Token", "p (gewählt)", "p (Modell)"])
            fig = go.Figure(go.Bar(x=list(range(len(df))), y=df["p (Modell)"], text=df["Token"],
                                   marker=dict(color=df["p (Modell)"], colorscale="Viridis", cmin=0, cmax=1),
                                   hovertemplate="%{text}<br>p = %{y:.1%}<extra></extra>"))
            fig.update_layout(title="Wahrscheinlichkeit, die das Modell dem gewählten Token gab",
                              xaxis_title="Position im erzeugten Text", yaxis_title="Wahrscheinlichkeit", yaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig, 320, legend=False), width="stretch", key="g_conf")
            st.caption(f"Mittlere Sicherheit: **{df['p (Modell)'].mean():.0%}**. Niedrige Balken = das Modell hat geraten; "
                       "oft an Satz-Anfängen oder bei Namen.")
        else:
            st.info("Erzeuge zuerst einen Text.")

    logits = next_logits(model, tok, prompt)
    with tabs[1]:
        probs = filter_logits(logits, s["g_temp"], s["g_topk"], s["g_topp"])
        raw = torch.softmax(logits, -1)
        top = torch.topk(raw, 15)
        labels = [tok.token_str(int(i)) for i in top.indices]
        fig = go.Figure()
        fig.add_bar(x=labels, y=top.values.numpy(), name="Modell (roh)", marker_color=PALETTE["violet"])
        fig.add_bar(x=labels, y=probs[top.indices].numpy(), name="nach Temperatur / Top-k / Top-p",
                    marker_color=PALETTE["orange"])
        fig.update_layout(barmode="group", title="Die 15 wahrscheinlichsten nächsten Tokens nach dem Prompt",
                          yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 340), width="stretch", key="g_next")
        ent = float(-(raw * torch.log(raw + 1e-12)).sum())
        st.caption(f"Entropie: **{ent:.2f} nats** (0 = sicher, {np.log(len(raw)):.1f} = völlig unsicher). "
                   "Tokens, die durch Top-k / Top-p herausgefiltert wurden, haben nach dem Filter 0 %.")

    with tabs[2]:
        fig = go.Figure()
        for t, col in zip((0.3, 0.8, 1.5), (PALETTE["cyan"], PALETTE["violet"], PALETTE["pink"])):
            p = torch.softmax(logits / t, -1)
            vals = torch.sort(p, descending=True).values[:25].numpy()
            fig.add_scatter(x=list(range(1, 26)), y=vals, mode="lines+markers", name=f"T = {t}", line=dict(color=col, width=3))
        fig.update_layout(title="Verteilung der Top-25 Tokens bei verschiedenen Temperaturen",
                          xaxis_title="Rang des Tokens", yaxis_title="Wahrscheinlichkeit", yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 340), width="stretch", key="g_temp_plot")
        st.caption("Niedrige Temperatur macht die Kurve spitz (das Top-Token gewinnt fast immer), hohe Temperatur flach "
                   "(auch seltsame Tokens bekommen eine Chance).")

    with tabs[3]:
        toks, maps = attention_maps(model, tok, prompt or " ")
        cc = st.columns(2)
        layer = cc[0].selectbox("Schicht", range(len(maps)), format_func=lambda i: f"Schicht {i + 1}", key="g_layer")
        head = cc[1].selectbox("Kopf", range(maps[0].shape[0]), format_func=lambda i: f"Kopf {i + 1}", key="g_head")
        labels = [f"{i}:{t}" for i, t in enumerate(toks)]
        fig = go.Figure(go.Heatmap(z=maps[layer][head], x=labels, y=labels, colorscale="Viridis", zmin=0, zmax=1))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 480, legend=False), width="stretch", key="g_att")

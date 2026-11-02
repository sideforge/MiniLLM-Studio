"""Seite 6: Trainings-Läufe vergleichen."""
from __future__ import annotations

import shutil

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from minillm.trainer import RUNS_DIR, list_runs, load_run_metrics
from ui.common import COLORS, fmt_int, fmt_time, hero, style_fig


def render() -> None:
    hero("📊 Läufe vergleichen", "Alle trainierten Modelle auf einen Blick – welche Einstellung war die beste?")
    runs = list_runs()
    if not runs:
        st.info("Noch keine abgeschlossenen oder laufenden Trainings gespeichert.")
        st.stop()

    df = pd.DataFrame([{
        "Lauf": r["name"], "Daten": r["data"], "Parameter": r["n_params"], "Schichten": r["model_cfg"]["n_layer"],
        "Breite": r["model_cfg"]["n_embd"], "Kontext": r["model_cfg"]["block_size"],
        "Schritte": f"{r['step']}/{r['max_steps']}", "Lernrate": r["train_cfg"]["lr"],
        "Batch": r["train_cfg"]["batch_size"], "Val-Loss": r.get("best_val"),
        "Perplexity": round(2.718281828 ** r["best_val"], 1) if r.get("best_val") else None,
        "Dauer": fmt_time(r["elapsed"]), "Status": r["status"], "Stand": r["finished"],
    } for r in runs])
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"Val-Loss": st.column_config.NumberColumn(format="%.3f"),
                                "Parameter": st.column_config.NumberColumn(format="%d")})

    names = [r["name"] for r in runs]
    pick = st.multiselect("Läufe im Diagramm vergleichen", names, default=names[:4])
    if pick:
        a, b = st.columns(2)
        fig_val, fig_tr = go.Figure(), go.Figure()
        for i, n in enumerate(pick):
            steps, evals = load_run_metrics(n)
            col = COLORS[i % len(COLORS)]
            if evals:
                fig_val.add_scatter(x=[e["step"] for e in evals], y=[e["val"] for e in evals], name=n,
                                    mode="lines+markers", line=dict(color=col, width=3))
            if steps:
                s = pd.Series([x["loss"] for x in steps]).ewm(alpha=0.05).mean()
                fig_tr.add_scatter(x=[x["step"] for x in steps], y=s, name=n, mode="lines", line=dict(color=col, width=2))
        fig_val.update_layout(title="Validierungs-Loss", xaxis_title="Schritt")
        fig_tr.update_layout(title="Trainings-Loss (geglättet)", xaxis_title="Schritt")
        a.plotly_chart(style_fig(fig_val, 340), width="stretch", key="r_val")
        b.plotly_chart(style_fig(fig_tr, 340), width="stretch", key="r_tr")

        scored = [r for r in runs if r["name"] in pick and r.get("best_val")]
        if scored:
            fig = go.Figure(go.Scatter(
                x=[r["n_params"] for r in scored], y=[r["best_val"] for r in scored], mode="markers+text",
                text=[r["name"] for r in scored], textposition="top center",
                marker=dict(size=[12 + r["elapsed"] ** 0.5 for r in scored], color=COLORS[1], opacity=0.8)))
            fig.update_layout(title="Parameter vs. Val-Loss (Punktgrösse = Trainingsdauer)",
                              xaxis_title="Parameter", yaxis_title="Val-Loss", xaxis_type="log")
            st.plotly_chart(style_fig(fig, 340, legend=False), width="stretch", key="r_scatter")

    st.subheader("Details & Aufräumen")
    sel = st.selectbox("Lauf", names)
    run_dir = RUNS_DIR / sel
    st.json(next(r for r in runs if r["name"] == sel), expanded=False)
    if st.button(f"🗑 Lauf „{sel}“ löschen"):
        shutil.rmtree(run_dir, ignore_errors=True)
        st.rerun()

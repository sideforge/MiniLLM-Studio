"""Seite 4: Training starten und live beobachten."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from minillm import device as dev
from minillm.data import Batcher, prepared_path
from minillm.model import GPT
from minillm.trainer import RUNS_DIR, TrainConfig, Trainer, lr_at
from ui.common import PALETTE, fmt_int, fmt_time, hero, need_data, restore_state, save_state, store, style_fig
from ui.page_model import DEFAULTS as MODEL_DEFAULTS
from ui.page_model import model_config

DEFAULTS = dict(t_name="lauf-1", t_steps=1500, t_batch="Auto", t_device="Auto (GPU wenn vorhanden)", t_amp=True, t_lr=3e-3, t_warm=50, t_eval=50, t_sample=250,
                t_wd=0.1, t_clip=1.0, t_threads=0)
LR_OPTIONS = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
VIEWS = ["📉 Kurven", "📝 Beispieltexte", "👁 Aufmerksamkeit", "🗺 Embeddings", "⚖️ Gewichte"]


def _train_config() -> TrainConfig:
    s = st.session_state
    return TrainConfig(batch_size=0 if s["t_batch"] == "Auto" else s["t_batch"],
                       device="cpu" if s["t_device"] == "CPU erzwingen" else "auto", amp=s["t_amp"], max_steps=s["t_steps"], lr=s["t_lr"], warmup_steps=s["t_warm"],
                       weight_decay=s["t_wd"], grad_clip=s["t_clip"], eval_interval=s["t_eval"],
                       sample_interval=s["t_sample"], threads=s["t_threads"])


def _benchmark(meta: dict) -> tuple[float, int, str]:
    """Misst Tokens/Sekunde dieses Rechners mit dem aktuellen Modell (wenige Trainingsschritte).

    Gibt (Tokens pro Sekunde, benutzte Batch-Grösse, Gerätename) zurück."""
    cfg = model_config(meta["vocab_size"])
    tcfg = _train_config()
    device = dev.pick_device(tcfg.device)
    dev.setup(device)
    if device.type == "cpu":
        torch.set_num_threads(tcfg.threads or max(1, min(8, torch.get_num_threads())))
    bs = tcfg.batch_size or dev.find_max_batch(lambda: GPT(cfg), cfg.block_size, cfg.vocab_size, device, tcfg.amp)
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(tcfg.lr, tcfg.weight_decay)
    b = Batcher(prepared_path(meta["name"]), cfg.block_size).to(device)
    speeds = []
    for i in range(5):
        x, y = b.get("train", bs)
        dev.sync(device)
        t0 = time.time()
        with dev.autocast(device, tcfg.amp):
            _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        dev.sync(device)
        if i > 0:
            speeds.append(x.numel() / (time.time() - t0))
    return float(np.mean(speeds)), bs, dev.describe(device)["name"]


def render() -> None:
    hero("🏋️ Training", "Das Modell rät das nächste Token, sieht seinen Fehler und verbessert sich – tausendfach. Schau live zu.")
    meta = need_data()
    restore_state("model", MODEL_DEFAULTS)
    restore_state("train", DEFAULTS)
    sto = store()
    trainer: Trainer | None = sto["trainer"]
    running = bool(trainer and trainer.is_running)

    with st.expander("⚙️ Trainings-Einstellungen", expanded=not running and (trainer is None)):
        a, b, c = st.columns(3)
        a.text_input("Name des Laufs", key="t_name", disabled=running)
        b.slider("Trainingsschritte", 100, 200000, step=100, key="t_steps", disabled=running,
                 help="Ein Schritt = ein Batch wird verarbeitet und die Gewichte werden einmal angepasst.")
        c.select_slider("Batch-Grösse", ["Auto", 4, 8, 16, 32, 64, 128, 256, 512], key="t_batch", disabled=running,
                        help="Wie viele Textfenster pro Schritt gleichzeitig gelernt werden. „Auto“ nimmt auf der GPU "
                             "so viel, wie in den Grafikspeicher passt (maximale Auslastung).")
        a, b, c = st.columns(3)
        a.select_slider("Lernrate (max.)", LR_OPTIONS, key="t_lr", format_func=lambda x: f"{x:g}", disabled=running,
                        help="Wie gross die Schritte beim Verbessern sind. Zu gross → instabil, zu klein → langsam.")
        b.slider("Warmup-Schritte", 0, 500, step=10, key="t_warm", disabled=running)
        c.slider("Bewertung alle … Schritte", 10, 500, step=10, key="t_eval", disabled=running)
        a, b, c = st.columns(3)
        a.slider("Beispieltext alle … Schritte", 50, 2000, step=50, key="t_sample", disabled=running)
        b.slider("Weight Decay", 0.0, 0.3, step=0.01, key="t_wd", disabled=running)
        c.slider("Gradient Clipping", 0.1, 5.0, step=0.1, key="t_clip", disabled=running)
        d1, d2, d3 = st.columns(3)
        d1.selectbox("Rechengerät", ["Auto (GPU wenn vorhanden)", "CPU erzwingen"], key="t_device", disabled=running)
        d2.checkbox("Mixed Precision (schneller auf GPU)", key="t_amp", disabled=running,
                    help="Rechnet mit 16-Bit-Zahlen: halb so viel Speicher, auf GPUs oft 2–3× schneller.")
        d3.slider("CPU-Threads (0 = automatisch)", 0, 16, key="t_threads", disabled=running)
        info = dev.describe(dev.pick_device("cpu" if st.session_state["t_device"] == "CPU erzwingen" else "auto"))
        if info["type"] == "cpu":
            st.warning(f"⚪ **CPU** – {info['hint']}")
        else:
            mem = f" · {info['memory_gb']:.0f} GB" if info["memory_gb"] else ""
            st.success(f"🟢 **GPU: {info['name']}**{mem} · {str(info['amp']).replace('torch.', '')} · "
                       "wird automatisch voll ausgelastet")
        save_state("train")
        save_state("model")

        cfg = model_config(meta["vocab_size"])
        n_params = GPT.estimate_params(cfg)
        st.caption(f"Modell: **{fmt_int(n_params)} Parameter** · {cfg.n_layer} Schichten · {cfg.n_head} Köpfe · "
                   f"Breite {cfg.n_embd} · Kontext {cfg.block_size} · Datensatz **{meta['name']}** "
                   f"({fmt_int(meta['tokens_train'])} Tokens)")

        bc1, bc2, bc3 = st.columns([1, 1, 2])
        if bc1.button("⏱ Dauer schätzen", disabled=running):
            with st.spinner("Messe Geschwindigkeit …"):
                speed, bs, devname = _benchmark(meta)
            total_tok = st.session_state["t_steps"] * bs * cfg.block_size
            st.session_state["_eta_msg"] = (f"**{devname}** schafft mit Batch {bs} ca. **{speed:,.0f} Tokens/s** → "
                                            f"{st.session_state['t_steps']} Schritte dauern etwa **{fmt_time(total_tok / speed)}** "
                                            f"({total_tok / meta['tokens_train']:.1f} Durchläufe durch die Daten).")
        if bc2.button("▶️ Training starten", type="primary", disabled=running):
            name = st.session_state["t_name"].strip() or "lauf"
            if (RUNS_DIR / name).exists():
                name = f"{name}-{time.strftime('%H%M%S')}"
            trainer = Trainer(name, prepared_path(meta["name"]), cfg, _train_config())
            sto["trainer"] = trainer
            trainer.start()
            st.session_state.pop("_eta_msg", None)
            st.rerun()
        if msg := st.session_state.get("_eta_msg"):
            st.info(msg)

    if trainer is None:
        st.info("Noch kein Training gestartet. Stelle oben alles ein und drücke **Training starten**.")
        return

    st.session_state["_was_running"] = running
    frag = st.fragment(_dashboard, run_every=1.0 if running else None)
    frag()


# ----------------------------------------------------------------------- Dashboard
def _dashboard() -> None:
    trainer: Trainer = store()["trainer"]
    if trainer is None:
        return
    running = trainer.is_running
    if st.session_state.get("_was_running") and not running:      # gerade fertig geworden -> alles neu zeichnen
        st.session_state["_was_running"] = False
        st.rerun()

    if trainer.status == "fehler":
        st.error("Das Training ist mit einem Fehler abgebrochen.")
        st.code(trainer.error)
        return

    cfg = trainer.cfg
    m = pd.DataFrame(trainer.metrics) if trainer.metrics else pd.DataFrame(columns=["step", "loss", "lr", "grad_norm", "tok_s"])
    ev = pd.DataFrame(trainer.evals) if trainer.evals else pd.DataFrame(columns=["step", "train", "val", "ppl"])

    recent_loss = m["loss"].tail(20).mean() if len(m) else float("nan")
    val = ev["val"].iloc[-1] if len(ev) else float("nan")
    val_delta = (ev["val"].iloc[-1] - ev["val"].iloc[-2]) if len(ev) > 1 else None
    tok_s = m["tok_s"].tail(20).mean() if len(m) else 0
    step_t = (m["time"].iloc[-1] - m["time"].iloc[max(0, len(m) - 31)]) / max(1, min(30, len(m) - 1)) if len(m) > 1 else 0
    eta = step_t * (cfg.max_steps - trainer.step)

    icon = {"läuft": "🟢", "fertig": "✅", "gestoppt": "⏹", "bereit": "⚪"}.get(trainer.status, "")
    r1 = st.columns(3)
    r1[0].metric("Status", f"{icon} {trainer.status}", help=f"Lauf: {trainer.run_name}")
    r1[1].metric("Schritt", f"{trainer.step} / {cfg.max_steps}")
    r1[2].metric("Tempo", f"{tok_s:,.0f} Tok/s", delta=f"noch {fmt_time(eta)}" if running else fmt_time(trainer.elapsed),
                 delta_color="off")
    r2 = st.columns(3)
    r2[0].metric("Train-Loss", f"{recent_loss:.3f}" if len(m) else "–", help="Mittel der letzten 20 Schritte")
    r2[1].metric("Val-Loss", f"{val:.3f}" if len(ev) else "–",
                 delta=f"{val_delta:+.3f}" if val_delta is not None else None, delta_color="inverse")
    r2[2].metric("Perplexity", f"{np.exp(min(val, 20)):.1f}" if len(ev) else "–",
                 help="e^Val-Loss: zwischen so vielen Tokens schwankt das Modell im Schnitt.")
    di = trainer.device_info
    if di:
        st.caption(f"Gerät: **{di['name']}**" + (f" ({di['memory_gb']:.0f} GB)" if di.get("memory_gb") else "")
                   + f" · Batch {cfg.batch_size} · Mixed Precision: {'ja' if cfg.amp and di.get('amp') else 'nein'}"
                   + (" · torch.compile: ja" if di.get("compiled") else ""))
    gs = dev.gpu_stats() if trainer.device.type == "cuda" else None
    if gs:
        g = st.columns(4)
        g[0].metric("GPU-Auslastung", f"{gs['util']:.0f} %", help="Wie viel der Zeit die GPU rechnet (nvidia-smi). Ziel: 90–100 %.")
        g[1].metric("Grafikspeicher", f"{gs['mem_used_gb']:.1f} / {gs['mem_total_gb']:.0f} GB")
        g[2].metric("Leistung", f"{gs['power_w']:.0f} / {gs['power_limit_w']:.0f} W")
        g[3].metric("Temperatur", f"{gs['temp_c']:.0f} °C")
        if running and trainer.step > 30 and gs["util"] < 70:
            st.info("💡 Die GPU wird noch nicht voll genutzt – das Modell ist zu klein für diese Karte. Auf der Seite "
                    "**🧠 Modell** gibt es den Knopf **„🚀 GPU-Maximum“**, und die Batch-Grösse sollte auf „Auto“ stehen.")
    st.progress(min(1.0, trainer.step / cfg.max_steps),
                text=f"{trainer.tokens_seen:,} Tokens gesehen · {trainer.n_params:,} Parameter · "
                     f"Laufzeit {fmt_time(trainer.elapsed)}")

    b1, b2 = st.columns([1, 5])
    if running and b1.button("⏹ Stoppen", key="stop_btn"):
        trainer.stop()
    if not running and trainer.status in ("fertig", "gestoppt"):
        b2.success(f"Lauf **{trainer.run_name}** gespeichert (Ordner `runs/{trainer.run_name}`). "
                   f"Beste Val-Loss: {trainer.best_val:.3f}. Probier ihn unter ✨ Generieren aus!")

    view = st.radio("Ansicht", VIEWS, horizontal=True, label_visibility="collapsed", key="train_view")
    if view == VIEWS[0]:
        _view_curves(trainer, m, ev)
    elif view == VIEWS[1]:
        _view_samples(trainer)
    elif view == VIEWS[2]:
        _view_attention(trainer)
    elif view == VIEWS[3]:
        _view_embeddings(trainer)
    else:
        _view_weights(trainer)


def _thin(df: pd.DataFrame, n: int = 1500) -> pd.DataFrame:
    return df if len(df) <= n else df.iloc[:: int(np.ceil(len(df) / n))]


def _view_curves(trainer: Trainer, m: pd.DataFrame, ev: pd.DataFrame) -> None:
    if m.empty:
        st.info("Warte auf den ersten Schritt …")
        return
    cfg = trainer.cfg
    alpha = max(0.02, 2 / (min(len(m), 100) + 1))
    m = m.assign(smooth=m["loss"].ewm(alpha=alpha).mean())
    mt = _thin(m)

    fig = go.Figure()
    fig.add_scatter(x=mt["step"], y=mt["loss"], mode="lines", name="Loss pro Schritt",
                    line=dict(color=PALETTE["violet"], width=1), opacity=0.25)
    fig.add_scatter(x=mt["step"], y=mt["smooth"], mode="lines", name="Loss (geglättet)",
                    line=dict(color=PALETTE["violet"], width=3))
    if len(ev):
        fig.add_scatter(x=ev["step"], y=ev["val"], mode="lines+markers", name="Validierung",
                        line=dict(color=PALETTE["orange"], width=3), marker=dict(size=7))
    fig.update_layout(title="Loss – je tiefer, desto besser", xaxis_title="Schritt", yaxis_title="Cross-Entropy")
    # Rate-Linie: ln(Vokabular) = Verlust eines Modells, das nur zufällig rät
    fig.add_hline(y=float(np.log(trainer.model_cfg.vocab_size)), line_dash="dot", line_color=PALETTE["grey"],
                  annotation_text="Zufall = ln(Vokabular)", annotation_position="top right")
    st.plotly_chart(style_fig(fig, 380), width="stretch", key="c_loss")

    a, b = st.columns(2)
    steps = np.arange(0, cfg.max_steps + 1, max(1, cfg.max_steps // 300))
    fig = go.Figure()
    fig.add_scatter(x=steps, y=[lr_at(int(s), cfg) for s in steps], mode="lines", name="Plan",
                    line=dict(color=PALETTE["cyan"], width=3))
    fig.add_scatter(x=[trainer.step], y=[m["lr"].iloc[-1]], mode="markers", name="jetzt",
                    marker=dict(size=12, color=PALETTE["pink"]))
    fig.update_layout(title="Lernrate (Warmup + Kosinus)", xaxis_title="Schritt")
    a.plotly_chart(style_fig(fig, 270, legend=False), width="stretch", key="c_lr")

    fig = go.Figure(go.Scatter(x=mt["step"], y=mt["grad_norm"], mode="lines", line=dict(color=PALETTE["green"], width=2)))
    fig.add_hline(y=cfg.grad_clip, line_dash="dot", line_color=PALETTE["grey"], annotation_text="Clipping")
    fig.update_layout(title="Gradienten-Norm (wie stark wird gerade korrigiert?)", xaxis_title="Schritt")
    b.plotly_chart(style_fig(fig, 270, legend=False), width="stretch", key="c_grad")

    a, b = st.columns(2)
    if len(ev):
        fig = go.Figure()
        fig.add_scatter(x=ev["step"], y=ev["ppl"], mode="lines+markers", line=dict(color=PALETTE["yellow"], width=3))
        fig.update_layout(title="Perplexity (Validierung)", xaxis_title="Schritt", yaxis_type="log")
        a.plotly_chart(style_fig(fig, 270, legend=False), width="stretch", key="c_ppl")
        gap = ev["val"] - ev["train"]
        fig = go.Figure(go.Bar(x=ev["step"], y=gap, marker_color=np.where(gap > 0.3, PALETTE["orange"], PALETTE["green"])))
        fig.update_layout(title="Lücke Val − Train (gross = Overfitting)", xaxis_title="Schritt")
        b.plotly_chart(style_fig(fig, 270, legend=False), width="stretch", key="c_gap")


def _view_samples(trainer: Trainer) -> None:
    if not trainer.samples:
        st.info(f"Der erste Beispieltext erscheint bei Schritt {trainer.cfg.sample_interval}.")
        return
    st.caption("So schreibt das Modell nach und nach besser. Jeder Text wurde ohne Vorgabe frei erzeugt.")
    for s in reversed(trainer.samples[-8:]):
        st.markdown(f"**Schritt {s['step']}**")
        text = s["text"].replace("<", "&lt;")
        st.markdown(f'<div class="sample">{text}</div>', unsafe_allow_html=True)


def _view_attention(trainer: Trainer) -> None:
    snap = trainer.attention_snap
    if snap is None:
        st.info("Noch keine Daten.")
        return
    maps = snap["maps"]
    n_layer, n_head = len(maps), maps[0].shape[0]
    c = st.columns(3)
    layer = c[0].selectbox("Schicht", range(n_layer), key="att_layer", format_func=lambda i: f"Schicht {i + 1}")
    head = c[1].selectbox("Kopf", range(n_head), key="att_head", format_func=lambda i: f"Kopf {i + 1}")
    c[2].caption(f"Stand: Schritt {snap['step']}")
    mat = maps[min(layer, n_layer - 1)][min(head, n_head - 1)].astype(np.float32)
    toks = [f"{i}:{t}" for i, t in enumerate(snap["tokens"])]
    fig = go.Figure(go.Heatmap(z=mat, x=toks, y=toks, colorscale="Viridis", zmin=0, zmax=1,
                               colorbar=dict(title="Aufmerksamkeit")))
    fig.update_yaxes(autorange="reversed", title="schaut von (Token)")
    fig.update_xaxes(title="auf (Token)")
    st.plotly_chart(style_fig(fig, 520, legend=False), width="stretch", key="c_att")
    st.caption("Jede Zeile zeigt, worauf ein Token beim Vorhersagen „schaut“. Das Dreieck oben rechts ist leer, weil "
               "ein Token nur Vergangenes sehen darf (Causal Mask). Zu Beginn ist die Aufmerksamkeit gleichmässig verteilt – "
               "mit dem Training entstehen Muster, z. B. der Blick aufs vorige Token.")


def _view_embeddings(trainer: Trainer) -> None:
    snaps = trainer.embedding_snaps
    if not snaps:
        st.info("Noch keine Daten.")
        return
    steps = [s["step"] for s in snaps]
    if len(steps) > 1:
        step = st.select_slider("Trainingsschritt", steps, value=steps[-1], key="emb_step")
    else:
        step = steps[0]
    s = snaps[steps.index(step)]
    fig = go.Figure(go.Scatter(x=s["x"], y=s["y"], mode="text", text=s["tokens"], textfont=dict(size=13, color=PALETTE["cyan"])))
    fig.update_layout(title=f"Die 150 häufigsten Tokens im Embedding-Raum (Schritt {step})",
                      xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(style_fig(fig, 560, legend=False), width="stretch", key="c_emb")
    st.caption("Jedes Token ist ein Punkt im hochdimensionalen Raum, hier auf 2 Dimensionen gequetscht (PCA). "
               "Am Anfang liegt alles zufällig; mit dem Training rücken ähnlich verwendete Tokens zusammen "
               "(z. B. Satzzeichen, Zahlen oder Namen).")


def _view_weights(trainer: Trainer) -> None:
    if not trainer.weight_stats:
        st.info("Noch keine Daten.")
        return
    last = trainer.weight_stats[-1]
    df = pd.DataFrame(last["stats"])
    df["name"] = df["name"].str.replace("blocks.", "B", regex=False)
    fig = go.Figure(go.Bar(x=df["name"], y=df["std"], marker_color=PALETTE["violet"]))
    fig.update_layout(title=f"Streuung (Std) der Gewichte je Matrix – Schritt {last['step']}", xaxis_tickangle=-45)
    st.plotly_chart(style_fig(fig, 340, legend=False), width="stretch", key="c_wstd")

    series = {}
    for snap in trainer.weight_stats:
        for r in snap["stats"]:
            series.setdefault(r["name"], []).append((snap["step"], r["norm"]))
    pick = [n for n in series if n.startswith(("tok_emb", "blocks.0.", f"blocks.{trainer.model_cfg.n_layer - 1}."))
            and n.endswith("weight")]
    fig = go.Figure()
    for n in pick:
        xs, ys = zip(*series[n])
        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=n.replace("blocks.", "B"))
    fig.update_layout(title="Wie sich die Gewichts-Grössen (Norm) verändern", xaxis_title="Schritt")
    st.plotly_chart(style_fig(fig, 340), width="stretch", key="c_wnorm")

    if trainer.model is not None:
        names = [n for n, p in trainer.model.named_parameters() if p.ndim >= 2]
        sel = st.selectbox("Verteilung der Gewichte einer Matrix", names, key="w_hist")
        w = dict(trainer.model.named_parameters())[sel].detach().flatten().numpy().copy()
        fig = go.Figure(go.Histogram(x=w, nbinsx=80, marker_color=PALETTE["cyan"]))
        fig.update_layout(title=f"Histogramm: {sel}", xaxis_title="Gewichtswert", yaxis_title="Anzahl")
        st.plotly_chart(style_fig(fig, 280, legend=False), width="stretch", key="c_whist")

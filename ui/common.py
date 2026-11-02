"""Gemeinsame Helfer für alle Seiten der App: Farben, CSS, Diagramm-Stil, geteilter Zustand."""
from __future__ import annotations

import colorsys
import html
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from minillm.data import PREPARED_DIR, list_prepared

PALETTE = {
    "violet": "#7C5CFF", "cyan": "#22D3EE", "orange": "#FB923C",
    "green": "#34D399", "pink": "#F472B6", "yellow": "#FACC15", "grey": "#8892A6",
}
COLORS = list(PALETTE.values())

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1280px;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
h1, h2, h3 {letter-spacing: -0.01em;}
[data-testid="stMetric"] {background: #161B26; border: 1px solid #252C3D; border-radius: 12px; padding: 12px 16px;}
[data-testid="stMetricLabel"] p {color: #8892A6; font-size: .8rem;}
.hero {background: linear-gradient(135deg,#2b1f6b 0%,#14304a 100%); border: 1px solid #2f3a57;
       border-radius: 16px; padding: 28px 32px; margin-bottom: 18px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2.1rem;}
.hero p {margin: 0; color: #c9d1e6; font-size: 1.02rem;}
.tag {display:inline-block; padding: 1px 9px; margin: 0 5px 0 0; border-radius: 99px; font-size: .72rem;
      background:#252C3D; color:#b8c2dc; border:1px solid #323b52;}
.tag.good {background:#12362b; color:#6ee7b7; border-color:#1d5a47;}
.tag.warn {background:#3b2a12; color:#fdba74; border-color:#5f4420;}
.tok {display:inline-block; padding: 2px 5px; margin: 2px 2px; border-radius: 6px; font-family: ui-monospace,Consolas,monospace;
      font-size: .9rem; color:#fff; white-space: pre;}
.step-card {background:#161B26; border:1px solid #252C3D; border-radius:12px; padding:14px 16px; height:100%;}
.step-card b {color:#a99bff;}
.sample {background:#10141d; border-left: 3px solid #7C5CFF; border-radius: 6px; padding: 10px 14px; margin: 6px 0;
         font-family: ui-monospace,Consolas,monospace; font-size: .88rem; white-space: pre-wrap; color:#dbe2f2;}
.muted {color:#8892A6;}
[data-testid="stPageLink"] a {background:#7C5CFF; color:#fff !important; border-radius:10px; padding:.55rem 1rem; font-weight:600;}
[data-testid="stPageLink"] a:hover {background:#6a49e6;}
</style>
"""


PAGES: dict = {}     # wird von app.py gefüllt: Schlüssel -> st.Page


def next_step(key: str, label: str, icon: str = "➡️") -> None:
    """Grosser Knopf, der zur nächsten Seite führt (damit man nie „stecken bleibt“)."""
    if key in PAGES:
        st.page_link(PAGES[key], label=label, icon=icon)


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=70 if legend else 45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORS,
        showlegend=legend, legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom"),
        font=dict(family="sans-serif", size=12), title_font=dict(size=15),
    )
    fig.update_xaxes(gridcolor="#222a3a", zerolinecolor="#222a3a")
    fig.update_yaxes(gridcolor="#222a3a", zerolinecolor="#222a3a")
    return fig


def fmt_int(n: float) -> str:
    n = float(n)
    for unit, div in (("Mrd", 1e9), ("Mio", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f} {unit}".replace(".00", "")
    return f"{int(n)}"


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def token_chips(pieces: list[tuple[int, str]]) -> str:
    """Farbige Token-Chips als HTML (jedes Token bekommt eine eigene Farbe)."""
    out = []
    for k, (tid, s) in enumerate(pieces):
        hue = (tid * 47 + k * 13) % 360
        r, g, b = colorsys.hls_to_rgb(hue / 360, 0.32, 0.55)
        out.append(f'<span class="tok" style="background:rgb({int(r*255)},{int(g*255)},{int(b*255)})" '
                   f'title="Token-ID {tid}">{html.escape(s)}</span>')
    return "".join(out)


def restore_state(name: str, defaults: dict) -> None:
    """Streamlit löscht den Zustand von Widgets, die auf einer anderen Seite waren – wir merken ihn uns selbst."""
    keep = st.session_state.setdefault(f"_keep_{name}", {})
    for k, v in defaults.items():
        keep.setdefault(k, v)
        if k not in st.session_state:
            st.session_state[k] = keep[k]


def save_state(name: str) -> None:
    keep = st.session_state.get(f"_keep_{name}", {})
    for k in keep:
        if k in st.session_state:
            keep[k] = st.session_state[k]


# ----------------------------------------------------------- geteilter Zustand
@st.cache_resource
def store() -> dict:
    """Wird von allen Browser-Sitzungen geteilt: hier lebt der laufende Trainer."""
    return {"trainer": None}


@st.cache_data(show_spinner=False)
def load_meta(name: str, _stamp: float) -> dict:
    return json.loads((PREPARED_DIR / name / "meta.json").read_text(encoding="utf-8"))


def active_data() -> dict | None:
    """Meta-Infos des in der Seitenleiste gewählten Datensatzes (oder None)."""
    name = st.session_state.get("active_data")
    if not name:
        return None
    p = PREPARED_DIR / name / "meta.json"
    return load_meta(name, p.stat().st_mtime) if p.exists() else None


def need_data() -> dict:
    meta = active_data()
    if meta is None:
        st.info("Du hast noch keinen vorbereiteten Datensatz. Gehe zuerst auf **📚 Daten** und lade Texte.")
        st.stop()
    return meta


def prepared_names() -> list[str]:
    return [m["name"] for m in sorted(list_prepared(), key=lambda m: m["created"])]

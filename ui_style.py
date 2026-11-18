"""Visual layer for the Streamlit app: CSS, hero banner, KPI cards, pipeline chips."""
from __future__ import annotations

import numpy as np

SPECTRUM = "linear-gradient(90deg,#7C3AED,#3B82F6,#22D3EE,#34D399,#FBBF24,#FB923C,#F43F5E)"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not([class*="material-icons"]):not(.katex):not(.katex *):not(code):not(pre) {{
       font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;}}
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {{
       font-family: 'Material Symbols Rounded', 'Material Icons' !important;}}
.stApp {{background: radial-gradient(1200px 500px at 85% -10%, rgba(124,58,237,.18), transparent 60%),
                    radial-gradient(900px 500px at -10% 10%, rgba(34,211,238,.10), transparent 55%), #0B1020;}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 3.2rem; max-width: 1450px;}}

/* ---------- hero ---------- */
.hero {{position: relative; overflow: hidden; border-radius: 20px; padding: 26px 34px 30px 34px; margin-bottom: 14px;
       background: linear-gradient(135deg, #121A3A 0%, #17214A 100%); border: 1px solid rgba(255,255,255,.08);
       box-shadow: 0 10px 40px rgba(0,0,0,.45);}}
.hero:before {{content:""; position:absolute; inset:0; opacity:.5;
       background-image: linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
       background-size: 36px 36px;}}
.hero-row {{position: relative; display:flex; align-items:center; justify-content:space-between; gap: 24px;}}
.hero .title {{margin:0; padding:0; font-size: 2.9rem; font-weight: 700; letter-spacing:-1px; line-height:1.05;
       background: {SPECTRUM}; background-size: 220% 100%; animation: slide 9s linear infinite;
       -webkit-background-clip: text; background-clip: text; color: transparent; filter: drop-shadow(0 0 18px rgba(124,58,237,.45));}}
.hero-left {{display:flex; align-items:center; gap:22px;}}
.hero .prism {{width:120px; height:84px; flex:0 0 120px;}}
.hero .badge {{display:inline-block; font-size:.7rem; letter-spacing:3px; color:#22D3EE; border:1px solid rgba(34,211,238,.4);
       padding:2px 10px; border-radius:999px; margin-bottom:8px; background:rgba(34,211,238,.08);}}
@media (max-width: 1100px) {{.hero svg:not(.prism) {{display:none;}} .hero .title {{font-size:2.1rem;}}}}
.hero .tag {{color:#9FB0E8; margin:8px 0 0 0; padding:0; font-size:.98rem; letter-spacing:.2px;}}
.hero svg {{flex: 0 0 360px; height: 84px;}}
.specbar {{position:absolute; left:0; right:0; bottom:0; height:5px; background:{SPECTRUM}; background-size: 200% 100%;
       animation: slide 8s linear infinite;}}
@keyframes slide {{from {{background-position: 0 0;}} to {{background-position: 200% 0;}}}}

/* ---------- pipeline chips ---------- */
.chips {{display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin: 6px 0 16px 0;}}
.chip {{display:flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px; font-size:.86rem; color:#E8ECF8;
       background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12);}}
.chip b {{display:inline-flex; width:20px; height:20px; border-radius:50%; align-items:center; justify-content:center;
       font-size:.72rem; color:#0B1020;}}
.chip span {{font-variant-numeric: tabular-nums; color:#9FB0E8; font-size:.8rem;}}
.arrow {{color:#4A5784;}}

/* ---------- KPI cards ---------- */
.kpis {{display:grid; grid-template-columns: repeat(5, 1fr); gap:14px; margin-bottom: 14px;}}
@media (max-width: 1000px) {{.kpis {{grid-template-columns: repeat(2, 1fr);}}}}
.kpi {{position:relative; padding:16px 18px 14px 18px; border-radius:16px; overflow:hidden;
       background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
       border:1px solid rgba(255,255,255,.10); backdrop-filter: blur(6px);}}
.kpi:before {{content:""; position:absolute; left:0; top:0; right:0; height:3px; background: var(--c);}}
.kpi .l {{font-size:.74rem; text-transform:uppercase; letter-spacing:1.4px; color:#8E9ABF; font-weight:600;}}
.kpi .v {{font-variant-numeric: tabular-nums; font-size:1.75rem; font-weight:700; color:#fff; margin-top:6px;
       text-shadow: 0 0 18px var(--c);}}
.kpi .s {{font-size:.78rem; color:#8E9ABF; margin-top:2px;}}

/* ---------- tabs as pills ---------- */
[data-testid="stTabs"] [role="tablist"] {{gap: 6px; border-bottom: 1px solid rgba(255,255,255,.08);}}
[data-testid="stTab"] {{border-radius: 12px 12px 0 0; padding: 8px 16px; font-weight: 600;}}
[data-testid="stTab"] p {{color:#8E9ABF !important; font-size: .98rem;}}
[data-testid="stTab"][aria-selected="true"] {{background: rgba(34,211,238,.10);}}
[data-testid="stTab"][aria-selected="true"] p {{color:#22D3EE !important;}}

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button {{border:0; border-radius:12px; font-weight:600; color:#0B1020 !important;
       background: linear-gradient(90deg,#22D3EE,#8B5CF6); padding:.5rem 1.2rem; transition: transform .12s, box-shadow .12s;}}
.stButton > button:hover, .stDownloadButton > button:hover {{transform: translateY(-1px); box-shadow: 0 6px 20px rgba(34,211,238,.35);}}
.stButton > button p, .stDownloadButton > button p {{color:#0B1020 !important;}}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{border-radius:14px; overflow:hidden;
       border:1px solid rgba(255,255,255,.10);}}
[data-testid="stImage"] img, [data-testid="stPyplot"] img {{border-radius:14px; border:1px solid rgba(255,255,255,.10);}}
div[data-testid="stAlert"] {{border-radius:14px; background: rgba(251,191,36,.08) !important; border:1px solid rgba(251,191,36,.35);}}
div[data-testid="stAlert"] p {{color:#FDE9A8;}}
h2, h3 {{color:#E8ECF8;}}
section[data-testid="stSidebar"] {{border-right: 1px solid rgba(255,255,255,.08);}}
section[data-testid="stSidebar"] h2 {{font-size:1.05rem; letter-spacing:1px; text-transform:uppercase; color:#22D3EE;
       padding-bottom:4px; border-bottom:1px solid rgba(255,255,255,.10);}}
.sec {{display:flex; align-items:center; gap:10px; margin: 6px 0 4px 0; font-size:1.25rem; font-weight:700; color:#E8ECF8;}}
.sec i {{width:6px; height:22px; border-radius:3px; background:{SPECTRUM};}}
.hint {{color:#8E9ABF; font-size:.9rem; margin-bottom:10px;}}
.pill-ok {{color:#34D399;}} .pill-bad {{color:#F43F5E;}}
.rcard {{border-radius:16px; padding:16px 20px; background: linear-gradient(160deg, rgba(34,211,238,.08), rgba(139,92,246,.07));
       border:1px solid rgba(34,211,238,.25);}}
.rcard .rt {{font-weight:700; color:#22D3EE; letter-spacing:1px; text-transform:uppercase; font-size:.78rem; margin-bottom:6px;}}
.rcard ul {{margin:0; padding-left:18px; color:#E8ECF8; line-height:1.55;}}
</style>
"""


def _spark(x, y, w=360, h=84):
    """Tiny SVG of the loaded spectrum for the hero banner."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5:
        return ""
    idx = np.linspace(0, len(x) - 1, 140).astype(int)
    xs, ys = x[idx], y[idx]
    ys = (ys - ys.min()) / (np.ptp(ys) or 1)
    px = (xs.max() - xs) / (np.ptp(xs) or 1) * (w - 8) + 4  # FTIR convention: high wavenumber on the left
    py = h - 8 - ys * (h - 18)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><defs>'
            f'<linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#22D3EE"/>'
            f'<stop offset=".5" stop-color="#A78BFA"/><stop offset="1" stop-color="#F472B6"/></linearGradient></defs>'
            f'<polyline points="{pts}" fill="none" stroke="url(#g)" stroke-width="2.2" stroke-linejoin="round"/></svg>')


PRISM = (
    '<svg class="prism" viewBox="0 0 120 84" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="pg" x1="0" x2="1"><stop offset="0" stop-color="#A78BFA"/><stop offset="1" stop-color="#22D3EE"/></linearGradient></defs>'
    '<line x1="2" y1="46" x2="44" y2="42" stroke="#fff" stroke-width="3" stroke-linecap="round"/>'
    '<polygon points="44,12 78,70 24,70" fill="rgba(255,255,255,.07)" stroke="url(#pg)" stroke-width="2.5" stroke-linejoin="round"/>'
    '<g stroke-width="3" stroke-linecap="round">'
    '<line x1="62" y1="44" x2="116" y2="14" stroke="#F43F5E"/><line x1="62" y1="46" x2="116" y2="26" stroke="#FB923C"/>'
    '<line x1="62" y1="48" x2="116" y2="38" stroke="#FBBF24"/><line x1="62" y1="50" x2="116" y2="50" stroke="#34D399"/>'
    '<line x1="62" y1="52" x2="116" y2="62" stroke="#22D3EE"/><line x1="62" y1="54" x2="116" y2="74" stroke="#7C3AED"/></g></svg>')


def hero(x=None, y=None) -> str:
    svg = _spark(x, y) if x is not None else ""
    return (f'<div class="hero"><div class="hero-row"><div class="hero-left">{PRISM}<div>'
            f'<div class="badge">● R&amp;D CELL · PREMIER</div>'
            f'<div class="title">FTIR Data Analysis</div>'
            f'<div class="tag">Smooth · Baseline · Deconvolve · Select · Quantify</div></div></div>{svg}</div>'
            f'<div class="specbar"></div></div>')


def kpi(label, value, sub="", color="#22D3EE") -> str:
    return (f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def kpi_row(*cards) -> str:
    return '<div class="kpis">' + "".join(cards) + "</div>"


def chips(items) -> str:
    cols = ["#7C3AED", "#3B82F6", "#22D3EE", "#34D399", "#FBBF24"]
    parts = []
    for i, (name, val) in enumerate(items):
        parts.append(f'<div class="chip"><b style="background:{cols[i % len(cols)]}">{i + 1}</b>{name} <span>{val}</span></div>')
    return '<div class="chips">' + '<span class="arrow">→</span>'.join(parts) + "</div>"


def section(title: str, hint: str = "") -> str:
    h = f'<div class="hint">{hint}</div>' if hint else ""
    return f'<div class="sec"><i></i>{title}</div>{h}'


def result_card(title: str, lines: list[str]) -> str:
    body = "".join(f"<li>{l}</li>" for l in lines)
    return (f'<div class="rcard"><div class="rt">{title}</div><ul>{body}</ul></div>')

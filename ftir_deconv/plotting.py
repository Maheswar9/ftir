"""Matplotlib figures shared by the CLI and the Streamlit app (light and dark themes)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .core import FitResult, PipelineOutput  # noqa: E402

THEMES = {
    "light": dict(bg="#FFFFFF", fg="#1B2240", muted="#888888", spine="#444444", data="#888888",
                  fit="#1E2761", total="#444444", raw="#999999", smooth="#1E2761", base="#D4AF37",
                  corr="#1E2761", zero="#bbbbbb", box="#FFFFFF",
                  comps=["#D4AF37", "#A7BEAE", "#B85042", "#6C8EBF", "#8E6C8A"]),
    "dark": dict(bg="#0F1630", fg="#DCE3F8", muted="#7E8BB5", spine="#3A4672", data="#7E8BB5",
                 fit="#22D3EE", total="#CBD5E1", raw="#7E8BB5", smooth="#22D3EE", base="#FBBF24",
                 corr="#34D399", zero="#3A4672", box="#0F1630",
                 comps=["#FBBF24", "#F472B6", "#34D399", "#A78BFA", "#FB923C"]),
}

# kept for backwards compatibility
NAVY, GOLD, SAGE, BRICK = "#1E2761", "#D4AF37", "#A7BEAE", "#B85042"


def _theme(name: str) -> dict:
    return THEMES.get(name, THEMES["light"])


def _style(fig, ax, t):
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["spine"])
    ax.tick_params(colors=t["fg"])
    ax.xaxis.label.set_color(t["fg"])
    ax.yaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()


def _legend(ax, t, **kw):
    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(frameon=False, fontsize=8, **kw)
        for txt in leg.get_texts():
            txt.set_color(t["fg"])


def _shade(ax, regions, t, label="Anchor regions"):
    for i, (a, b) in enumerate(regions):
        ax.axvspan(min(a, b), max(a, b), color=t["base"], alpha=0.18, lw=0, label=label if i == 0 else None)


def _apply_limits(ax, xlim=None, ylim=None):
    """User axis limits; None entries stay automatic. The wavenumber axis keeps its right-to-left orientation."""
    if xlim and any(v is not None for v in xlim):
        a, b = ax.get_xlim()
        lo = min(a, b) if xlim[0] is None else float(xlim[0])
        hi = max(a, b) if xlim[1] is None else float(xlim[1])
        ax.set_xlim(max(lo, hi), min(lo, hi))
    if ylim and any(v is not None for v in ylim):
        ax.set_ylim(bottom=None if ylim[0] is None else float(ylim[0]), top=None if ylim[1] is None else float(ylim[1]))


def _mark_peaks(ax, items, t):
    """Boxed labels above the axes ('P<n> · name' over '<centre> cm-1') joined to the curve by a dotted line.
    items = [(number, name, centre, y_on_curve, colour)]. Rows are chosen from the measured box widths so that
    labels never overlap (call after the layout and axis limits are final). Returns the label height in points."""
    import textwrap
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_h = ax.get_window_extent().height * 72.0 / fig.dpi
    made = []
    for n, name, cx, yy, col in sorted(items, key=lambda m: m[2]):
        label = "\n".join(textwrap.wrap(f"P{int(n)} · {name}", 26) or [""]) + f"\n{cx:.0f} cm$^{{-1}}$"
        ann = ax.annotate(label, xy=(cx, yy), xycoords="data", xytext=(cx, 1.03), textcoords=("data", "axes fraction"),
                          ha="center", va="bottom", fontsize=9, fontweight="bold", color=t["fg"], annotation_clip=False,
                          zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc=t["box"], ec=col, lw=1.2),
                          arrowprops=dict(arrowstyle="-", color=col, lw=1, ls=":"))
        ann.set_clip_on(False)
        ax.plot([cx], [yy], "o", color=col, mec=t["fg"], ms=6, zorder=5)
        made.append((ann, label.count("\n") + 1))
    fig.canvas.draw()
    rows_end, placed = [], []                       # right edge (px) of the last label in each row
    max_h = 0.0
    ext = [(ann.get_bbox_patch().get_window_extent(rend), ann, nl) for ann, nl in made]
    for bb, ann, nlines in sorted(ext, key=lambda e: e[0].x0):
        r = 0
        while r < len(rows_end) and bb.x0 < rows_end[r] + 6:
            r += 1
        if r == len(rows_end):
            rows_end.append(0)
        rows_end[r] = bb.x1
        placed.append((ann, r, nlines))
    line_pt = 9 * 1.25
    row_h = max(n for _, _, n in placed) * line_pt + 12 if placed else 0
    for ann, r, _ in placed:
        ann.set_position((ann.xyann[0], 1.03 + r * row_h / ax_h))
    return (len(rows_end) * row_h + 8) if placed else 0.0


def fit_figure(fit: FitResult, title: str | None = None, show_literature: bool = True,
               show_data: bool = True, show_fit: bool = True, show_components=None,
               show_labels: bool = True, show_selected_sum: bool = False, theme: str = "light",
               xlim=None, ylim=None, texts=None):
    """Fit plot with every layer switchable.

    show_components: None -> all peaks, otherwise an iterable of peak indices (0-based) to draw.
    show_labels / show_literature apply to the drawn peaks only.
    """
    t = _theme(theme)
    comp_cols = t["comps"]
    n = len(fit.components)
    chosen = list(range(n)) if show_components is None else sorted(set(show_components))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ymax = float(max(fit.y.max(), fit.best_fit.max()))
    if show_data:
        ax.plot(fit.x, fit.y, ls="none", marker=".", ms=3, color=t["data"], label="Data (smoothed, baseline-corrected)")
    if show_fit:
        ax.plot(fit.x, fit.best_fit, color=t["fit"], lw=2.6, label="Cumulative fit")
    if show_selected_sum and chosen:
        ax.plot(fit.x, sum(fit.components[i] for i in chosen), color=t["total"], lw=1.8, ls="-.",
                label="Sum of selected peaks")
    rows = fit.peaks.reset_index(drop=True)
    for i in chosen:
        col = comp_cols[i % len(comp_cols)]
        cx, name, lit = rows.at[i, "center_cm-1"], rows.at[i, "peak"], rows.at[i, "literature_cm-1"]
        ax.fill_between(fit.x, fit.components[i], color=col, alpha=0.12 if theme == "dark" else 0.0)
        ax.plot(fit.x, fit.components[i], "--", color=col, lw=2, label=f"P{i + 1} {name}: {cx:.0f} cm$^{{-1}}$")
        if show_literature:
            ax.axvline(lit, color=col, lw=1, ls=":", alpha=0.8)
        j = abs(fit.x - cx).argmin()
        ax.plot([cx], [fit.components[i][j]], "o", color=col, mec=t["fg"], ms=7, zorder=5)
    if show_labels and chosen:
        ax.set_ylim(top=ymax * 1.12)
    texts = texts or {}
    ax.set_ylabel(texts.get("ylabel") or "Absorbance")
    ax.set_title(texts.get("title") or title or f"{fit.k}-component Gaussian fit (R$^2$={fit.r2:.3f})", pad=12)
    _style(fig, ax, t)
    if texts.get("xlabel"):
        ax.set_xlabel(texts["xlabel"])
    _apply_limits(ax, xlim, ylim)
    _legend(ax, t, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    if show_labels and chosen:
        items = [(i + 1, rows.at[i, "peak"], float(rows.at[i, "center_cm-1"]),
                  float(fit.components[i][abs(fit.x - rows.at[i, "center_cm-1"]).argmin()]),
                  comp_cols[i % len(comp_cols)]) for i in chosen]
        need = _mark_peaks(ax, items, t)
        fh = fig.get_size_inches()[1] * 72
        ax.set_title(ax.get_title(), pad=need + 8)
        fig.subplots_adjust(top=min(0.9, 1 - (need + 40) / fh))
    return fig


def _shade_touchups(ax, touchups, t):
    for i, tu in enumerate(touchups or []):
        a, b = sorted((float(tu[0]), float(tu[1])))
        ax.axvspan(a, b, facecolor="none", edgecolor=t["comps"][2], hatch="///", alpha=0.6, lw=0,
                   label="Manual touch-up" if i == 0 else None)


def processing_figure(out: PipelineOutput, show_raw: bool = True, show_smooth: bool = True,
                      show_baseline: bool = True, show_corrected: bool = True, theme: str = "light",
                      marks=None, xlim=None, ylim_top=None, ylim_bottom=None, texts=None):
    """Pre-processing view; every curve is switchable. Returns None if nothing is selected."""
    t = _theme(theme)
    top = show_raw or show_smooth or show_baseline
    if not (top or show_corrected):
        return None
    panels = int(top) + int(show_corrected)
    marks = list(marks or []) if show_corrected else []
    fig, axs = plt.subplots(panels, 1, figsize=(9, 3.6 * panels + 0.6 + (0.9 if marks else 0)), sharex=True,
                            squeeze=False)
    axs = axs[:, 0]
    k = 0
    if top:
        ax = axs[k]; k += 1
        for _, _, cx in marks:
            ax.axvline(cx, color=t["muted"], lw=0.8, ls=":", alpha=0.6)
        if show_raw:
            ax.plot(out.x_view, out.y_raw, color=t["raw"], lw=1, label="Raw")
        if show_smooth:
            ax.plot(out.x_view, out.y_smooth, color=t["smooth"], lw=1.5, label="Savitzky-Golay smoothed")
        if show_baseline:
            ax.plot(out.x_view, out.baseline, color=t["base"], lw=2, label="Baseline")
        ax.set_ylabel("Absorbance")
        if show_baseline and getattr(out, "anchors", None):
            _shade(ax, out.anchors, t, label="Anchor regions")
        if show_baseline:
            _shade_touchups(ax, getattr(out, "touchups", None), t)
            for i, pn in enumerate(getattr(out, "pins", None) or []):
                ax.plot([pn[0]], [pn[1]], "o", ms=7, color=t["comps"][2], mec=t["fg"], zorder=6,
                        label="Baseline pin" if i == 0 else None)
    if show_corrected:
        ax = axs[k]
        ax.plot(out.x_view, out.y_corrected, color=t["corr"], lw=1.5, label="Baseline-corrected")
        ax.axhline(0, color=t["zero"], lw=0.8)
        ax.set_ylabel("Corrected absorbance")
        if getattr(out, "anchors", None):
            _shade(ax, out.anchors, t, label="Anchor regions")
        _shade_touchups(ax, getattr(out, "touchups", None), t)
    for ax in axs:
        _style(fig, ax, t)
        ax.set_xlabel("")
        _legend(ax, t)
    axs[-1].set_xlabel(texts.get("xlabel") or "Wavenumber (cm$^{-1}$)" if texts else "Wavenumber (cm$^{-1}$)")
    if texts:
        if top and texts.get("ylabel_top"):
            axs[0].set_ylabel(texts["ylabel_top"])
        if show_corrected and texts.get("ylabel_bottom"):
            axs[-1].set_ylabel(texts["ylabel_bottom"])
        if texts.get("title"):
            axs[0].set_title(texts["title"], pad=10)
    _apply_limits(axs[0], xlim, ylim_top if top else ylim_bottom)
    if top and show_corrected:
        _apply_limits(axs[-1], xlim, ylim_bottom)
    fig.tight_layout()
    if marks:
        import numpy as np
        ax = axs[-1]
        xs, ys = np.asarray(out.x_view), np.asarray(out.y_corrected)
        o = np.argsort(xs)
        cols = t["comps"]
        items = [(n, nm, cx, float(np.interp(cx, xs[o], ys[o])), cols[(int(n) - 1) % len(cols)]) for n, nm, cx in marks]
        need = _mark_peaks(ax, items, t)
        fh = fig.get_size_inches()[1] * 72
        if panels > 1:
            ah = axs[0].get_window_extent().height * 72 / fig.dpi
            fig.subplots_adjust(hspace=(need + 30) / ah)
        else:
            fig.subplots_adjust(top=min(0.9, 1 - (need + 14) / fh))
    return fig


def survey_figure(res, show_raw=True, show_smooth=True, show_baseline=True, show_corrected=True,
                  show_bands=True, show_peak_labels=True, theme: str = "light"):
    """Whole-spectrum view: raw/smoothed/baseline on top, corrected + band shading + labelled peaks below."""
    t = _theme(theme)
    top = show_raw or show_smooth or show_baseline
    if not (top or show_corrected):
        return None
    panels = int(top) + int(show_corrected)
    fig, axs = plt.subplots(panels, 1, figsize=(11, 3.8 * panels + 0.6), sharex=True, squeeze=False)
    axs = axs[:, 0]
    k = 0
    if top:
        ax = axs[k]; k += 1
        if show_raw:
            ax.plot(res.x, res.y_raw, color=t["raw"], lw=0.8, label="Raw")
        if show_smooth:
            ax.plot(res.x, res.y_smooth, color=t["smooth"], lw=1.4, label="Smoothed")
        if show_baseline:
            ax.plot(res.x, res.baseline, color=t["base"], lw=2, label="Baseline")
        ax.set_ylabel("Absorbance")
    if show_corrected:
        ax = axs[k]
        ax.plot(res.x, res.y_corr, color=t["corr"], lw=1.4, label="Baseline-corrected")
        ax.axhline(0, color=t["zero"], lw=0.8)
        ax.set_ylabel("Corrected absorbance")
        ymax = float(res.y_corr.max()) or 1.0
        if show_bands and len(res.bands):
            for i, b in res.bands.iterrows():
                if not b["covered"]:
                    continue
                a, c = (float(v) for v in str(b["range_cm-1"]).replace("–", "-").split("-"))
                ax.axvspan(a, c, color=t["comps"][i % len(t["comps"])], alpha=0.10, lw=0)
                ax.text((a + c) / 2, ax.get_ylim()[0], b["band"], rotation=90, ha="center", va="bottom",
                        fontsize=7, color=t["muted"])
        if show_peak_labels and len(res.peaks):
            for r, (_, p) in enumerate(res.peaks.iterrows()):
                ax.plot([p["position_cm-1"]], [p["height"]], "o", color=t["fg"], ms=4, zorder=5)
                ax.annotate(f"{p['position_cm-1']:.0f}", xy=(p["position_cm-1"], p["height"]),
                            xytext=(0, 6 + 10 * (r % 2)), textcoords="offset points", ha="center",
                            fontsize=8, color=t["fg"])
        ax.set_ylim(top=ymax * 1.25 if ymax > 0 else None)
    for ax in axs:
        _style(fig, ax, t)
        ax.set_xlabel("")
        _legend(ax, t, loc="upper right")
    axs[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    return fig


def interactive_baseline_figure(out, zoom_lo: float, zoom_hi: float, pins=None, theme: str = "dark", texts=None):
    """Plotly chart for click-to-place baseline pins.

    A grid of invisible markers covers the plot area, so a click anywhere returns the (wavenumber, absorbance) of
    the nearest grid point. Returns (figure, y_lo, y_hi).
    """
    import numpy as np
    import plotly.graph_objects as go

    t = _theme(theme)
    m = (out.x_view >= zoom_lo) & (out.x_view <= zoom_hi)
    if m.sum() < 5:
        m = np.ones_like(out.x_view, bool)
    x, ys, base, raw = out.x_view[m], out.y_smooth[m], out.baseline[m], out.y_raw[m]
    lo, hi = float(min(ys.min(), base.min())), float(max(ys.max(), base.max()))
    pad = 0.12 * (hi - lo or 1e-3)
    y_lo, y_hi = lo - pad, hi + pad
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), 160), np.linspace(y_lo, y_hi, 70))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, mode="lines", name="Raw", line=dict(color=t["raw"], width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=ys, mode="lines", name="Smoothed", line=dict(color=t["smooth"], width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=base, mode="lines", name="Baseline", line=dict(color=t["base"], width=3), hoverinfo="skip"))
    if pins:
        fig.add_trace(go.Scatter(x=[p[0] for p in pins], y=[p[1] for p in pins], mode="markers+text", name="Pins",
                                 text=[str(i + 1) for i in range(len(pins))], textposition="top center",
                                 marker=dict(size=12, color=t["comps"][2], line=dict(color=t["fg"], width=1.5)),
                                 textfont=dict(color=t["fg"]), hoverinfo="skip"))
    fig.add_trace(go.Scattergl(x=gx.ravel(), y=gy.ravel(), mode="markers", name="click-grid", showlegend=False,
                               marker=dict(size=9, opacity=0.01, color=t["fg"]),
                               hovertemplate="%{x:.1f} cm⁻¹ · %{y:.5f}<extra>click to pin</extra>"))
    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font=dict(color=t["fg"]), height=480, clickmode="event+select",
        margin=dict(l=60, r=20, t=20, b=50), hovermode="closest", dragmode="zoom",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title=(texts or {}).get("xlabel") or "Wavenumber (cm⁻¹)", autorange="reversed", range=[x.max(), x.min()], gridcolor=t["spine"],
                   zeroline=False),
        yaxis=dict(title=(texts or {}).get("ylabel") or "Absorbance", range=[y_lo, y_hi], gridcolor=t["spine"], zeroline=False))
    return fig, y_lo, y_hi


def window_preview_figure(x, y, lo, hi, markers=(), theme: str = "dark"):
    """Corrected spectrum with the deconvolution window shaded and literature positions (name, cm-1) marked."""
    t = _theme(theme)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(x, y, color=t["corr"], lw=1.4, label="Baseline-corrected")
    ax.axhline(0, color=t["zero"], lw=0.8)
    ax.axvspan(min(lo, hi), max(lo, hi), color=t["base"], alpha=0.14, lw=0, label="Deconvolution window")
    for k, (name, pos) in enumerate(markers):
        c = t["comps"][k % len(t["comps"])]
        ax.axvline(pos, color=c, lw=1, ls=":")
        ax.annotate(name or f"{pos:g}", xy=(pos, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -4),
                    textcoords="offset points", rotation=90, va="top", ha="left", fontsize=8, color=c)
    ax.set_ylabel("Corrected absorbance")
    _style(fig, ax, t)
    _legend(ax, t, loc="upper left")
    fig.tight_layout()
    return fig

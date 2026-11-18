"""Streamlit front-end for the Si-H FTIR deconvolution pipeline.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
from dataclasses import replace
import random
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks

from ftir_deconv import (ANCHOR_METHODS, LITERATURE_PRESETS, SIGMA_SIH_CM2, TABLE_COLUMNS, PipelineConfig,
                         load_spectrum, model_selection, peaks_from_table, run_pipeline, smooth_savgol,
                         suggest_anchors)
from ftir_deconv import TOUCHUP_MODES, suggest_touchups
from ftir_deconv.plotting import (fit_figure, interactive_baseline_figure, processing_figure,
                                  window_preview_figure)
from help_guide import render_help
from ui_style import CSS, chips, hero, kpi, kpi_row, result_card, section

st.set_page_config(page_title="FTIR data analysis", page_icon=":rainbow:", layout="wide")
def _password_gate() -> None:
    """Optional login: active only if FTIR_APP_PASSWORD (env var) or APP_PASSWORD (Streamlit secret) is set."""
    import hmac
    import os
    pw = os.environ.get("FTIR_APP_PASSWORD", "")
    if not pw:
        try:
            pw = str(st.secrets.get("APP_PASSWORD", ""))
        except Exception:
            pw = ""
    if not pw or st.session_state.get("_authed"):
        return
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state["_authed"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


_password_gate()
st.markdown(CSS, unsafe_allow_html=True)
hero_slot = st.empty()
hero_slot.markdown(hero(), unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Spectrum (CSV/TXT: wavenumber, absorbance)", type=["csv", "txt", "CSV"])

    st.header("2 · Pre-processing")
    c1, c2 = st.columns(2)
    view_lo = c1.number_input("View from (cm⁻¹)", value=1950.0, step=10.0)
    view_hi = c2.number_input("View to (cm⁻¹)", value=2300.0, step=10.0)
    window = st.slider("Savitzky–Golay window (points)", 5, 101, 21, step=2)
    poly = st.slider("Polynomial order", 1, 5, 3)
    BASELINES = {"linear": "Linear (window ends)", "als": "ALS (asymmetric least squares)",
                 "airpls": "airPLS (adaptive, no p)", "poly": "Polynomial (function fit)",
                 "autolevel": "Auto-level (iterative line)", "anchor": "Anchor regions (you choose)",
                 "rubberband": "Rubberband (never negative)", "none": "None"}
    baseline = st.selectbox("Baseline method", list(BASELINES), format_func=BASELINES.get,
                            help="linear: line between the window ends. ALS / airPLS: smooth curve pushed under the data "
                                 "(airPLS adapts its own weights). Polynomial: iterative polynomial that settles under the "
                                 "peaks. Auto-level: straight line fitted while discarding points that do not fit. "
                                 "Anchor: baseline through flat regions you pick. Rubberband: lower envelope, never negative.")
    lam, p_asym, poly_order, autolevel_k = 1e6, 0.01, 3, 2.0
    if baseline in ("als", "airpls"):
        log_lam = st.number_input("log10(λ) – smoothness", min_value=0.0, max_value=12.0, value=6.0,
                                  step=0.01, format="%.2f", help="Larger = stiffer baseline. Step 0.01.")
        lam = 10 ** log_lam
        st.caption(f"λ = {lam:.3g}")
    if baseline == "als":
        p_asym = st.select_slider("ALS p – asymmetry", [0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
    if baseline == "poly":
        poly_order = st.slider("Polynomial order", 1, 6, 3, help="Higher orders follow curvature but can start eating the band.")
    if baseline == "autolevel":
        autolevel_k = st.slider("Rejection threshold k (σ)", 1.0, 4.0, 2.0, 0.1,
                                help="Points further than k·σ above the line are discarded and the line is refitted.")

    invert = st.checkbox("Invert spectrum (×−1)", value=False,
        help="Use when the band points DOWN (negative peak) because the reference was subtracted the wrong way round. "
             "No baseline can turn a downward band into a peak; flipping the sign does.")
    baseline_shift = st.number_input("Lower baseline by (absorbance)", value=0.0, step=0.0001, format="%.5f",
                                     help="Moves the WHOLE baseline down by this amount (negative = up), after the method above. "
                                          "The corrected curve rises by the same amount, so the area grows by shift × window width.")
    floor_zero = st.checkbox("Lift curve to zero", value=False,
        help="After the baseline, lowers it by the smallest amount that leaves no negative point in the fit window. "
             "Guarantees a non-negative curve for ANY baseline, but adds area equal to the lift × the window width.")
    lift_scope = "fit"
    if floor_zero:
        lift_scope = st.radio("Lift looks at", ["fit", "view"], horizontal=True,
                              format_func={"fit": "deconvolution window", "view": "whole view range"}.get,
                              help="Which part of the curve must end up with no negative point.")
    anchors, anchor_method = None, "poly1"

    # the deconvolution window lives in the Analysis tab (own inputs, independent of the baseline range);
    # the sidebar only reads its last value
    fit_lo = float(st.session_state.get("fit_lo", 2000.0))
    fit_hi = float(st.session_state.get("fit_hi", 2250.0))

    if baseline == "anchor":
        st.header("2b · Anchor regions")
        anchor_method = st.selectbox("Anchor baseline type", list(ANCHOR_METHODS), format_func=ANCHOR_METHODS.get)
        st.session_state.setdefault("anc_ver", 0)
        st.session_state.setdefault("anc_rows", [(view_lo, min(view_lo + 30, view_hi)), (max(view_hi - 30, view_lo), view_hi)])
        a1, a2 = st.columns(2)
        if a1.button("Ends", help="Anchors = the first and last 30 cm⁻¹ of the view range"):
            st.session_state.anc_rows = [(view_lo, min(view_lo + 30, view_hi)), (max(view_hi - 30, view_lo), view_hi)]
            st.session_state.anc_ver += 1
        if a2.button("Suggest", help="Flattest 30 cm⁻¹ stretch on each side of the deconvolution window"):
            try:
                _d = load_spectrum(up)
                _m = (_d["wn"] >= view_lo) & (_d["wn"] <= view_hi)
                _x, _y = _d["wn"][_m].to_numpy(), smooth_savgol(_d["abs"][_m].to_numpy(), window, poly)
                _sug = suggest_anchors(_x, _y, fit_lo, fit_hi)
                if _sug:
                    st.session_state.anc_rows = _sug
                    st.session_state.anc_ver += 1
            except Exception:  # noqa: BLE001
                st.warning("Upload a spectrum first.")
        _adf = st.data_editor(pd.DataFrame(st.session_state.anc_rows, columns=["from", "to"]),
                              key=f"anc_{st.session_state.anc_ver}", num_rows="dynamic", hide_index=True,
                              column_config={"from": st.column_config.NumberColumn("From (cm⁻¹)", format="%.1f"),
                                             "to": st.column_config.NumberColumn("To (cm⁻¹)", format="%.1f")})
        anchors = [(float(r.to_dict()["from"]), float(r.to_dict()["to"])) for _, r in _adf.dropna().iterrows()]
        st.caption("Pick regions where the spectrum should be flat (no absorption band).")

    st.header("3 · Manual touch-up")
    use_touch = st.checkbox("Correct the baseline by hand after the method above", value=False,
                            help="Fix the stretches where the corrected curve goes negative: pick a region and a mode. "
                                 "'Lower' pulls the baseline down to the data there (clips at zero locally); "
                                 "'Shift' moves it by a value you type. Changes the area — the app reports by how much.")
    touchups, pins, pin_taper = None, None, 15.0
    if use_touch:
        st.session_state.setdefault("tu_ver", 0)
        st.session_state.setdefault("tu_rows", [])
        if st.button("Detect negative stretches", help="Adds a 'Lower' region for every clearly negative stretch "
                                                       "(deeper than 2× the noise) inside the deconvolution window."):
            try:
                _d = load_spectrum(up)
                _o = run_pipeline(_d, PipelineConfig(view_lo=view_lo, view_hi=view_hi, fit_lo=fit_lo, fit_hi=fit_hi,
                                                     smooth_window=window, smooth_poly=poly, baseline=baseline,
                                                     als_lam=lam, als_p=p_asym, anchors=anchors, anchor_method=anchor_method,
                                                     invert=invert, poly_order=poly_order, autolevel_k=autolevel_k,
                                                     skip_fit=True))
                st.session_state.tu_rows = [(a, b, TOUCHUP_MODES[m], v) for a, b, m, v in
                                            suggest_touchups(_o.x_view, _o.y_corrected, fit_lo, fit_hi)]
                st.session_state.tu_ver += 1
                if not st.session_state.tu_rows:
                    st.info("No clearly negative stretch found.")
            except Exception:  # noqa: BLE001
                st.warning("Upload a spectrum first.")
        _tdf = st.data_editor(pd.DataFrame(st.session_state.tu_rows, columns=["from", "to", "mode", "value"]),
                              key=f"tu_{st.session_state.tu_ver}", num_rows="dynamic", hide_index=True,
                              column_config={"from": st.column_config.NumberColumn("From (cm⁻¹)", format="%.1f"),
                                             "to": st.column_config.NumberColumn("To (cm⁻¹)", format="%.1f"),
                                             "mode": st.column_config.SelectboxColumn("Mode", options=list(TOUCHUP_MODES.values()),
                                                                                      default=TOUCHUP_MODES["lower"]),
                                             "value": st.column_config.NumberColumn("Shift value", format="%.5f",
                                                                                    help="Only used by 'Shift'.")})
        _inv = {v: k for k, v in TOUCHUP_MODES.items()}
        touchups = [(float(r["from"]), float(r["to"]), _inv.get(r["mode"], "lower"),
                     0.0 if pd.isna(r["value"]) else float(r["value"]))
                    for _, r in _tdf.dropna(subset=["from", "to"]).iterrows()] or None

        st.markdown("**Baseline pins** — click on the graph (Pre-processing tab → *Click to adjust*) to drag the "
                    "baseline through a point; or type values here.")
        st.session_state.setdefault("pin_ver", 0)
        st.session_state.setdefault("pin_rows", [])
        pin_taper = st.number_input("Pin fade width (cm⁻¹)", min_value=1.0, value=15.0, step=1.0, key="pin_taper",
                                    help="Outside the outermost pins the change fades out over this distance.")
        _pdf = st.data_editor(pd.DataFrame(st.session_state.pin_rows, columns=["wavenumber", "absorbance"]),
                              key=f"pin_{st.session_state.pin_ver}", num_rows="dynamic", hide_index=True,
                              column_config={"wavenumber": st.column_config.NumberColumn("Wavenumber (cm⁻¹)", format="%.1f"),
                                             "absorbance": st.column_config.NumberColumn("Absorbance", format="%.5f")})
        pins = [(float(r["wavenumber"]), float(r["absorbance"])) for _, r in _pdf.dropna().iterrows()] or None
        st.session_state.pin_rows = pins or []
        if st.button("Clear pins"):
            st.session_state.pin_rows = []
            st.session_state.pin_ver += 1
            st.rerun()

    st.header("4 · Quantification")
    do_density = st.checkbox("Compute bond density (Eq. 1)", value=True,
                             help="C = A/(σ·t) only makes sense for an absorption band with a known cross-section "
                                  "(e.g. Si–H, N–H). Untick for any other band: you then get areas and positions only.")
    thickness = st.number_input("Film thickness (nm)", min_value=0.0, value=80.0, step=1.0,
                                help="Eq. (1): C = A / (σ·t). Concentration scales as 1/t.")
    sigma = st.number_input("Absorption cross-section σ (cm²)", value=SIGMA_SIH_CM2, format="%.2e",
                            help="Si–H: 7.4e-18 (Lanford & Rand, as in Kim et al.). N–H: 5.3e-18. "
                                 "Use the literature value for your bond.")

    st.header("5 · Display")
    plot_theme = st.radio("Plot style", ["dark", "light"], horizontal=True,
                          help="Dark matches the app. Use light for reports and slides (affects the PNG downloads too).")


def _text_inputs(prefix: str, fields) -> dict:
    """Optional title / axis-label overrides (blank = default). fields = [(key, label, placeholder)]."""
    out = {}
    cols = st.columns(len(fields))
    for c, (k, lab, ph) in zip(cols, fields):
        out[k] = c.text_input(lab, value="", placeholder=ph, key=f"{prefix}_txt_{k}").strip() or None
    return out


def _axes_inputs(prefix: str, y_names=("Y from", "Y to")):
    """Optional axis limits and labels (blank = automatic). Returns (xlim, ylim, texts)."""
    with st.expander("Edit axes & labels  (blank = automatic)"):
        c = st.columns(4)
        vals = [c[i].number_input(lab, value=None, placeholder="auto", key=f"{prefix}_{i}", format="%g")
                for i, lab in enumerate(("X from (cm⁻¹)", "X to (cm⁻¹)", y_names[0], y_names[1]))]
        texts = _text_inputs(prefix, [("title", "Title", "default"), ("xlabel", "X-axis label", "Wavenumber (cm⁻¹)"),
                                      ("ylabel", "Y-axis label", "Absorbance")])
    return (vals[0], vals[1]), (vals[2], vals[3]), texts


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    return buf.getvalue()


FACTS = [
    "Si–H stretching bands climb in wavenumber as more electronegative neighbours attach to the silicon: from about "
    "2000 cm⁻¹ for Si₃Si–H up to about 2220 cm⁻¹ for N₃Si–H.",
    "FTIR spectra are traditionally plotted right-to-left, with the highest wavenumber on the left.",
    "The Savitzky–Golay filter you are about to use was published in 1964 in Analytical Chemistry, and it is still a "
    "standard way to smooth spectra without flattening peaks.",
    "A Gaussian's FWHM is 2.3548 × σ, so the σ values in the fit table convert straight to the width you would read "
    "off the plot.",
    "AIC (Akaike, 1974) and BIC (Schwarz, 1978) both reward a good fit but charge a penalty for every extra peak, "
    "which is why R² alone cannot tell you how many peaks to use.",
    "Hydrogen released from Si–H and N–H bonds during firing is what passivates dangling bonds in many solar-cell "
    "dielectric layers, which is why the bond density is worth quantifying.",
]


@st.dialog("Welcome to FTIR Data Analysis", width="large")
def welcome_dialog():
    st.markdown("### 👋 Hey there, welcome!")
    st.write("Turn a raw FTIR spectrum into peak positions, areas and bond densities in a few clicks.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Upload**\n\nDrop a two-column CSV (wavenumber, absorbance) in the sidebar.")
    c2.markdown("**2 · Set peaks**\n\nType the literature peak names and positions you expect.")
    c3.markdown("**3 · Read results**\n\nSee the fit, the densities and an automatic summary.")
    st.info("💡 " + random.choice(FACTS))
    if st.button("Let's go", key="welcome_go"):
        st.rerun()


@st.dialog("Spectrum loaded", width="large")
def loaded_dialog(name, stats, fact):
    st.markdown(f"### 🎉 Nice — **{name}** is in!")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", f"{stats['n']:,}")
    c2.metric("Range (cm⁻¹)", f"{stats['lo']:.0f}–{stats['hi']:.0f}")
    c3.metric("Spacing (cm⁻¹)", f"{stats['dx']:.2f}")
    c4.metric("Signal / noise", f"≈ {stats['snr']:.0f}")
    if stats["bands"]:
        st.write("**Bands spotted in your view window:** " + ", ".join(f"{b:.0f} cm⁻¹" for b in stats["bands"]) +
                 f" — the strongest is at **{stats['top']:.0f} cm⁻¹**.")
    st.info("💡 " + fact)
    st.caption("Next: check the peak table below, then scroll to the Results section.")
    if st.button("Start analysing", key="loaded_go"):
        st.rerun()


def _spectrum_stats(df, lo, hi, window, poly):
    m = (df["wn"] >= lo) & (df["wn"] <= hi)
    x, y = df["wn"][m].to_numpy(), df["abs"][m].to_numpy()
    ys = smooth_savgol(y, window, poly)
    noise = float(np.std(np.diff(y)) / np.sqrt(2)) or 1e-12
    idx, _ = find_peaks(ys, prominence=0.12 * float(np.ptp(ys) or 1))
    idx = sorted(idx, key=lambda i: -ys[i])[:5]
    return dict(n=len(df), lo=float(df["wn"].min()), hi=float(df["wn"].max()),
                dx=float(np.median(np.diff(df["wn"]))),
                snr=float(np.ptp(ys) / noise), bands=sorted(float(x[i]) for i in idx),
                top=float(x[int(np.argmax(ys))]) if len(x) else float("nan"))


def _interpretation(fit, out, tot, neg_pct):
    rows = fit.peaks.reset_index(drop=True)
    d = rows.loc[rows["area"].idxmax()]
    L = [f"<b>{d['peak']}</b> is the largest component: {d['area_pct']:.0f}% of the fitted area, centred at "
         f"{d['center_cm-1']:.0f} cm⁻¹ ({d['shift_cm-1']:+.0f} cm⁻¹ from its literature value of {d['literature_cm-1']:.0f})."]
    if tot is not None:
        L.append(f"Total bond density: <b>{tot:.2e} cm⁻³</b> for t = {thickness:g} nm, σ = {sigma:.2e} cm².")
    L.append(f"The {len(rows)}-peak fit reproduces the band with R² = {fit.r2:.3f}; "
             + ("every peak is resolved." if fit.identifiable else "at least one peak is <b>not resolved</b> — avoid quoting individual areas."))
    shaky = rows[rows["area_unc_pct"] > 50]
    if len(shaky):
        L.append("Large area uncertainty (&gt;50%) for: " + ", ".join(shaky["peak"]) + " — quote their sum.")
    far = rows[rows["shift_cm-1"].abs() > 20]
    if len(far):
        L.append("Moved more than 20 cm⁻¹ from literature: " + ", ".join(far["peak"]) + " — check assignment and bounds.")
    if neg_pct > 10:
        L.append(f"{neg_pct:.0f}% of the corrected points are below zero — check the anchors, or tick “Invert spectrum” if the band points down.")
    return L


def _results_zip(out, fit, fig_fit, fig_pre, summary):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("peaks.csv", fit.peaks.to_csv(index=False))
        if out.concentration is not None:
            z.writestr("concentration.csv", out.concentration.to_csv(index=False))
        z.writestr("summary.json", json.dumps(summary, indent=2, default=float))
        z.writestr("fit.png", _png(fig_fit))
        if fig_pre is not None:
            z.writestr("preprocessing.png", _png(fig_pre))
    return buf.getvalue()


tab_main, tab_help = st.tabs(["Analysis", "Help & Guide"])
with tab_help:
    render_help()

with tab_main:
    if up is None:
        if not st.session_state.get("welcomed"):
            st.session_state["welcomed"] = True
            welcome_dialog()
        st.info("Upload a spectrum in the sidebar to begin (a synthetic example is in `examples/sample_spectrum.csv`). "
                "Open **Help & Guide** for a walkthrough.")
        st.stop()
    try:
        df = load_spectrum(up)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")
        st.stop()
    _m = (df["wn"] >= view_lo) & (df["wn"] <= view_hi)
    hero_slot.markdown(hero(df["wn"][_m], df["abs"][_m]), unsafe_allow_html=True)

    _fkey = f"{getattr(up, 'name', 'spectrum')}-{getattr(up, 'size', len(df))}"
    if st.session_state.get("seen_file") != _fkey:
        st.session_state["seen_file"] = _fkey
        st.session_state["welcomed"] = True
        try:
            loaded_dialog(getattr(up, "name", "your spectrum"), _spectrum_stats(df, view_lo, view_hi, window, poly),
                          random.choice(FACTS))
        except Exception:  # noqa: BLE001
            pass

    # --------------------------------------------------- deconvolution window ---
    st.markdown(section("Deconvolution window", "Independent of the baseline: the baseline is computed on the <b>view range</b> "
                        "(sidebar); only the part inside this window is fitted. Change either freely — results stay on screen."),
                unsafe_allow_html=True)
    st.caption("How to choose it: (1) cover every peak you want, including their full width (about ±2σ beyond the outer "
               "centres); (2) put both edges where the corrected curve is flat and close to zero; (3) leave out neighbouring "
               "bands you are not fitting. The **Window check** below the peak table shows exactly this.")
    dw1, dw2, dw3 = st.columns(3)
    fit_lo = dw1.number_input("Deconvolute from (cm⁻¹)", value=2000.0, step=10.0, key="fit_lo")
    fit_hi = dw2.number_input("Deconvolute to (cm⁻¹)", value=2250.0, step=10.0, key="fit_hi")
    default_tol = dw3.number_input("Default centre tolerance ± (cm⁻¹)", min_value=1.0, value=30.0, step=1.0, key="tol",
                                   help="Used when a row has no lo/hi bounds.")
    if fit_lo >= fit_hi:
        fit_lo, fit_hi = min(fit_lo, fit_hi), max(fit_lo, fit_hi)
        if fit_lo == fit_hi:
            fit_hi = fit_lo + 10.0
        st.info(f"'From' was not below 'to' — using {fit_lo:g}–{fit_hi:g} cm⁻¹.")

    # ------------------------------------------------------------ peak table ---
    st.markdown(section("Peak table", "One row per peak: name + literature wavenumber. Tick <b>Free centre</b> to let a "
                                      "peak move anywhere in the fit window."), unsafe_allow_html=True)
    pc1, pc2 = st.columns([2, 2])
    preset = pc1.selectbox("Start from", list(LITERATURE_PRESETS), help="Loads a starting table; edit freely.")
    tbl_up = pc2.file_uploader("…or load a saved table (CSV)", type=["csv"], key="tbl_up")
    if tbl_up is not None:
        try:
            _t = pd.read_csv(tbl_up)
            if "use" not in _t:
                _t.insert(0, "use", True)
            base_rows = _t[TABLE_COLUMNS]
            tbl_key = f"tbl_file_{tbl_up.name}_{tbl_up.size}"
        except Exception as e:  # noqa: BLE001
            st.error(f"Table CSV must have columns {TABLE_COLUMNS}: {e}")
            st.stop()
    else:
        base_rows = pd.DataFrame(LITERATURE_PRESETS[preset])[TABLE_COLUMNS]
        tbl_key = f"tbl_{preset}"
    edited = st.data_editor(
        base_rows, key=tbl_key, num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "use": st.column_config.CheckboxColumn("Include in fit", default=True,
                                                   help="Untick to remove this peak from the model completely "
                                                        "(fit, areas, totals, bond density)."),
            "name": st.column_config.TextColumn("Peak name"),
            "literature_cm1": st.column_config.NumberColumn("Literature (cm⁻¹)", format="%.1f"),
            "lo": st.column_config.NumberColumn("Lower bound", format="%.1f"),
            "hi": st.column_config.NumberColumn("Upper bound", format="%.1f"),
            "free": st.column_config.CheckboxColumn("Free centre"),
            "sigma0": st.column_config.NumberColumn("σ start", format="%.1f"),
            "sigma_max": st.column_config.NumberColumn("σ max", format="%.1f"),
        })
    st.download_button("Save this table (CSV)", edited.to_csv(index=False), "peak_table.csv", "text/csv")
    _skipped: list = []
    try:
        peaks = peaks_from_table(edited, fit_lo, fit_hi, default_tol, skipped=_skipped)
    except ValueError as e:
        st.error(str(e))
        peaks = None
    if _skipped:
        st.warning("Outside the deconvolution window, so left out of this fit: " + ", ".join(_skipped) +
                   ". Widen the window to include them.")
    k = len(peaks) if peaks else 0

    cfg = PipelineConfig(view_lo=view_lo, view_hi=view_hi, fit_lo=fit_lo, fit_hi=fit_hi,
                         smooth_window=window, smooth_poly=poly, baseline=baseline,
                         als_lam=lam, als_p=p_asym, k=k, peaks=peaks,
                         anchors=anchors, anchor_method=anchor_method, invert=invert, floor_zero=floor_zero, baseline_shift=baseline_shift, lift_scope=lift_scope, touchups=touchups, pins=pins, pin_taper=pin_taper,
                         poly_order=poly_order, autolevel_k=autolevel_k,
                         thickness_nm=(thickness or None) if do_density else None, sigma_cm2=sigma)

    # ------------------------------------------------------------ window check ---
    st.markdown(section("Window check", "Your corrected spectrum with the deconvolution window shaded and the table's "
                                        "literature positions marked."), unsafe_allow_html=True)
    try:
        _pre = run_pipeline(df, replace(cfg, skip_fit=True, peaks=None))
    except Exception as e:  # noqa: BLE001
        _pre = None
        st.warning(f"Window check unavailable: {e}")
    if _pre is not None:
        _rows = []
        for _, r in edited.iterrows():
            if pd.isna(r.get("literature_cm1")) or (r.get("use") is not None and not pd.isna(r.get("use")) and not bool(r.get("use"))):
                continue
            _lit = float(r["literature_cm1"])
            _a = _lit - default_tol if pd.isna(r.get("lo")) else float(r["lo"])
            _b = _lit + default_tol if pd.isna(r.get("hi")) else float(r["hi"])
            _rows.append((str(r.get("name") or ""), _lit, _a, _b))
        _wf = window_preview_figure(_pre.x_view, _pre.y_corrected, fit_lo, fit_hi,
                                    [(n, lit) for n, lit, _, _ in _rows], plot_theme)
        st.pyplot(_wf)
        _in = (_pre.x_view >= fit_lo) & (_pre.x_view <= fit_hi)
        if _in.sum() > 10:
            _yw = _pre.y_corrected[_in]
            _amp = float(_yw.max() - min(_yw.min(), 0.0)) or 1.0
            _el = float(_yw[np.argsort(_pre.x_view[_in])][:max(3, _in.sum() // 20)].mean())
            _er = float(_yw[np.argsort(_pre.x_view[_in])][-max(3, _in.sum() // 20):].mean())
            _eh = max(_el, _er) / _amp
            if _eh > 0.25:
                st.warning(f"The curve is still at {100 * _eh:.0f}% of its height at a window edge, so the fit has to cut a "
                           "band or background short. Move that edge outward to where the curve is flat and near zero.")
            else:
                st.success("Both window edges sit on low, flat curve — a sensible window.")
        if _rows:
            _sl = int(5 * np.floor((min(r[2] for r in _rows) - 15) / 5))
            _sh = int(5 * np.ceil((max(r[3] for r in _rows) + 15) / 5))
            def _set_window(lo, hi):
                st.session_state["fit_lo"], st.session_state["fit_hi"] = float(lo), float(hi)
            st.button(f"Set window from my table  ({_sl}–{_sh} cm⁻¹)", on_click=_set_window, args=(_sl, _sh),
                      help="Spans all included peaks' bounds plus 15 cm⁻¹ on each side.")

    out = None
    if peaks is not None:
        try:
            out = run_pipeline(df, cfg)
        except Exception as e:  # noqa: BLE001
            st.error(f"The deconvolution could not run: {e}")
    if out is None:
        # keep the pre-processing visible even when the fit cannot run
        try:
            out = run_pipeline(df, replace(cfg, skip_fit=True, peaks=None))
        except Exception as e:  # noqa: BLE001
            st.error(f"Pre-processing failed too: {e}")
            st.stop()
        st.info("Showing the baseline and pre-processing only — fix the deconvolution settings above to get the fit.")
        _pf = processing_figure(out, True, True, True, True, theme=plot_theme)
        if _pf is not None:
            st.pyplot(_pf)
        st.stop()
    if out.view_extended:
        st.info("The deconvolution window reaches outside the view range, so the baseline range was widened to cover it.")
    fit = out.fit
    tot = None if out.concentration is None else out.concentration["C_cm-3"].iloc[-1]
    _fm = (out.x_view >= fit_lo) & (out.x_view <= fit_hi)
    neg_pct = 100 * float((out.y_corrected[_fm] < 0).mean())

    # --------------------------------------------------------------- results ---
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(section("Results", "Live summary of the current settings — updates as you edit anything above or in the sidebar."),
                unsafe_allow_html=True)
    st.markdown(chips([("Smooth", f"SG {window}·{poly}"), ("Baseline", baseline),
                       ("Fit", f"{k} Gaussians"), ("Quality", f"R² {fit.r2:.3f}"),
                       ("Quantify", "—" if tot is None else f"{tot:.2e} cm⁻³")]), unsafe_allow_html=True)
    st.markdown(kpi_row(
        kpi("R²", f"{fit.r2:.4f}", "goodness of fit", "#22D3EE"),
        kpi("AIC", f"{fit.aic:.1f}", "lower is better", "#8B5CF6"),
        kpi("BIC", f"{fit.bic:.1f}", "penalises extra peaks", "#3B82F6"),
        kpi("Identifiable", "YES" if fit.identifiable else "NO",
            "all peaks resolved" if fit.identifiable else "see warnings below",
            "#34D399" if fit.identifiable else "#F43F5E"),
        kpi("Total density", "—" if tot is None else f"{tot:.2e}", (f"cm⁻³ · t = {thickness:g} nm" if do_density else "bond density switched off"), "#FBBF24"),
    ), unsafe_allow_html=True)
    for w in fit.warnings:
        st.warning(w)

    rl, rr = st.columns([3, 2])
    with rl:
        st.markdown("**Fitted components**")
        st.dataframe(fit.peaks.style.format({c: "{:.4g}" for c in fit.peaks.select_dtypes("number").columns}),
                     width="stretch", hide_index=True)
        if out.concentration is not None:
            st.markdown(f"**Bond density** · t = {thickness:g} nm · σ = {sigma:.2e} cm²")
            _c = out.concentration.copy()
            _c["center_cm-1"] = _c["center_cm-1"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
            _c["area"] = _c["area"].map(lambda v: f"{v:.4f}")
            _c["C_cm-3"] = _c["C_cm-3"].map(lambda v: f"{v:.3e}")
            st.dataframe(_c, width="stretch", hide_index=True)
    with rr:
        st.markdown(result_card("What the numbers say", _interpretation(fit, out, tot, neg_pct)), unsafe_allow_html=True)
        st.caption("Individual-peak splits are only as trustworthy as their uncertainties; "
                   "when neighbouring areas are strongly correlated, quote the total.")

    # ---------------------------------------------------------- detail tabs ---
    t_fit, t_proc, t_sel = st.tabs(["Fit plot", "Pre-processing", "Model selection"])

    with t_proc:
        st.markdown(section("Pre-processing", "Choose which curves to show; the PNG contains exactly what you tick."),
                    unsafe_allow_html=True)
        pc = st.columns(5)
        p_raw = pc[0].checkbox("Raw", value=True, key="pp_raw")
        p_smooth = pc[1].checkbox("Smoothed", value=True, key="pp_smooth")
        p_base = pc[2].checkbox("Baseline", value=True, key="pp_base")
        p_corr = pc[3].checkbox("Baseline-corrected", value=True, key="pp_corr")
        p_marks = pc[4].checkbox("Peak marks", value=True, key="pp_marks",
                                 help="Boxed labels (P number · name · wavenumber) for the fitted peaks, above the corrected curve.")
        st.caption(f"Points below zero inside the fit window after correction: **{neg_pct:.1f}%**. "
                   "Some negatives are normal (noise, fringes); large dips mean the baseline or anchors need adjusting.")
        if (touchups or pins or baseline_shift) and abs(out.touchup_area) > 0:
            st.info(f"Manual touch-up changed the baseline inside the deconvolution window by "
                    f"{out.touchup_area:+.3g} absorbance·cm⁻¹ of area (fitted total {fit.peaks['area'].sum():.3g}). "
                    "Report that you used it, and use the same regions for samples you compare.")
        if floor_zero and out.lift > 0:
            _extra = out.lift * (fit_hi - fit_lo)
            st.info(f"Curve lifted by {out.lift:.2e} absorbance units so nothing is negative. This adds about "
                    f"{_extra:.3f} to the total area (the fitted total is {fit.peaks['area'].sum():.3f}) — the lifted part "
                    "is a pedestal, not real absorption.")
        if baseline == "rubberband":
            st.info("Rubberband never goes negative, but it is pulled down by every dip (ripple, subtraction artefacts), "
                    "so peak areas are larger than with other baselines. Use the same baseline for every sample you compare.")
        if baseline == "anchor" and anchors and any(not (max(r) < fit_lo or min(r) > fit_hi) for r in anchors):
            st.warning("An anchor region overlaps the fit window — part of the band will be subtracted. "
                       "Keep anchors outside the band.")
        _w = (out.x_view >= fit_lo) & (out.x_view <= fit_hi)
        _yc = out.y_corrected[_w]
        if len(_yc) and abs(_yc.min()) > 1.5 * max(_yc.max(), 1e-12):
            st.warning("The band in the fit window points **downward** (deepest dip "
                       f"{_yc.min():.3g} vs highest point {_yc.max():.3g}). A baseline cannot fix this — it is usually "
                       "a subtraction in the wrong order. Tick **Invert spectrum (×−1)** in the sidebar.")
        with st.expander("Edit axes & labels  (blank = automatic)"):
            _ac = st.columns(6)
            _axv = [_ac[i].number_input(lab, value=None, placeholder="auto", key=f"pp_ax_{i}", format="%g")
                    for i, lab in enumerate(("X from (cm⁻¹)", "X to (cm⁻¹)", "Top Y from", "Top Y to",
                                             "Bottom Y from", "Bottom Y to"))]
            _ptx = _text_inputs("pp", [("title", "Title", "none"), ("xlabel", "X-axis label", "Wavenumber (cm⁻¹)"),
                                       ("ylabel_top", "Top Y label", "Absorbance"),
                                       ("ylabel_bottom", "Bottom Y label", "Corrected absorbance")])
        _marks = [(i + 1, str(r["peak"]), float(r["center_cm-1"])) for i, r in fit.peaks.reset_index(drop=True).iterrows()] if p_marks else None
        pfig = processing_figure(out, p_raw, p_smooth, p_base, p_corr, theme=plot_theme, marks=_marks,
                                 xlim=(_axv[0], _axv[1]), ylim_top=(_axv[2], _axv[3]), ylim_bottom=(_axv[4], _axv[5]), texts=_ptx)
        if pfig is None:
            st.info("Tick at least one curve.")
        else:
            st.pyplot(pfig)
            st.download_button("Download preprocessing.png", _png(pfig), "preprocessing.png", "image/png")

        st.markdown(section("Click to adjust", "Click on the graph where the baseline should pass — it is dragged through that "
                            "point. Two clicks just under the smoothed curve at both ends of a dip give a straight segment."),
                    unsafe_allow_html=True)
        if not use_touch:
            st.info("Tick **Correct the baseline by hand** in the sidebar (section 3) to enable click-to-adjust.")
        else:
            z1, z2 = st.columns(2)
            _zl = z1.number_input("Zoom from (cm⁻¹)", value=float(fit_lo), step=10.0, key="zoom_lo")
            _zh = z2.number_input("Zoom to (cm⁻¹)", value=float(fit_hi), step=10.0, key="zoom_hi")
            _itx = _text_inputs("ia", [("xlabel", "X-axis label", "Wavenumber (cm⁻¹)"), ("ylabel", "Y-axis label", "Absorbance")])
            try:
                _ifig, _, _ = interactive_baseline_figure(out, min(_zl, _zh), max(_zl, _zh), pins, plot_theme, texts=_itx)
            except ImportError:
                st.warning("The interactive chart needs the **plotly** package, which is not installed yet. Close the app, "
                           "run  `py -m pip install plotly`  in the same terminal, then start the app again. "
                           "Everything else keeps working meanwhile.")
                _ifig = None
        if use_touch and _ifig is not None:
            _ckey = f"pinchart_{st.session_state.pin_ver}"
            _ev = st.plotly_chart(_ifig, key=_ckey, on_select="rerun", selection_mode="points", width="stretch")
            _pts = [p for p in (_ev.selection.points if _ev and _ev.selection else [])
                    if p.get("curve_number") == len(_ifig.data) - 1]
            if _pts:
                st.session_state.pin_rows = list(st.session_state.pin_rows) + [(round(float(q["x"]), 2), float(q["y"])) for q in _pts[:1]]
                st.session_state.pin_ver += 1
                st.rerun()
            st.caption("Hover shows the coordinates; each click adds a pin (see the pin table in the sidebar — edit or delete "
                       "rows there). Scroll or drag-zoom the chart to place pins precisely.")

    with t_fit:
        st.markdown(section("Deconvolution", "Tick the layers and peaks you want on the plot. This only changes the DISPLAY — to remove a peak from the fit and the totals, untick <b>Include in fit</b> in the peak table."), unsafe_allow_html=True)
        names = list(fit.peaks["peak"])
        r1 = st.columns(4)
        show_data = r1[0].checkbox("Data", value=True, key="sh_data")
        show_fit = r1[1].checkbox("Cumulative fit", value=True, key="sh_fit")
        show_lab = r1[2].checkbox("Peak names", value=True, key="sh_lab")
        show_lit = r1[3].checkbox("Literature lines", value=False, key="sh_lit")
        cols = st.columns(min(len(names), 4) or 1)
        chosen = [i for i, nm in enumerate(names)
                  if cols[i % len(cols)].checkbox(f"{nm} ({fit.peaks['center_cm-1'].iloc[i]:.0f})", value=True,
                                                  key=f"sh_pk_{i}_{nm}")]
        show_sum = st.checkbox("Sum of ticked peaks only", value=False, key="sh_sum")
        _fx, _fy, _ftx = _axes_inputs("fit_ax", ("Y from", "Y to"))
        fig = fit_figure(fit, show_literature=show_lit, show_data=show_data, show_fit=show_fit,
                         show_components=chosen, show_labels=show_lab, show_selected_sum=show_sum, theme=plot_theme,
                         xlim=_fx, ylim=_fy, texts=_ftx)
        st.pyplot(fig)
        st.download_button("Download fit.png", _png(fig), "fit.png", "image/png")
        if chosen:
            _sel = fit.peaks.iloc[chosen]
            _msg = (f"Ticked peaks: area **{_sel['area'].sum():.4g}** = **{100 * _sel['area'].sum() / fit.peaks['area'].sum():.0f}%** "
                    "of the fitted total (the fit itself always contains every included peak).")
            if out.concentration is not None:
                _msg += f" Bond density of the ticked peaks: **{out.concentration['C_cm-3'].iloc[chosen].sum():.3e} cm⁻³**."
            st.caption(_msg)

    with t_sel:
        st.markdown(section("Model selection", "Your table vs. every leave-one-peak-out variant, ranked by BIC among "
                                               "identifiable fits."), unsafe_allow_html=True)
        st.write("AIC/BIC keep improving as components are added, so each fit is also checked for *identifiability* "
                 "(no centre pinned at a bound, every area has an uncertainty). Recommended = lowest-BIC identifiable model.")
        if st.button("Run model selection"):
            cands = {f"Your table ({k} peaks)": peaks}
            if k > 1:
                for i, pk in enumerate(peaks):
                    cands[f"without {pk.name or i + 1}"] = [q for j, q in enumerate(peaks) if j != i]
            sel = model_selection(fit.x, fit.y, candidates=cands)
            st.dataframe(sel.style.format({"R2": "{:.4f}", "AIC": "{:.1f}", "BIC": "{:.1f}"}),
                         width="stretch", hide_index=True)
            st.download_button("Download model_selection.csv", sel.to_csv(index=False), "model_selection.csv", "text/csv")

    # ------------------------------------------------------ download everything ---
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    _summary = {"file": getattr(up, "name", ""), "settings": {"smooth_window": window, "smooth_poly": poly,
                "baseline": baseline, "anchors": anchors, "fit_window": [fit_lo, fit_hi], "thickness_nm": thickness,
                "sigma_cm2": sigma}, "R2": fit.r2, "AIC": fit.aic, "BIC": fit.bic,
                "identifiable": fit.identifiable, "total_density_cm3": tot, "warnings": fit.warnings}
    _pf = processing_figure(out, True, True, True, True, theme="light", texts=_ptx,
                            marks=[(i + 1, str(r["peak"]), float(r["center_cm-1"])) for i, r in fit.peaks.reset_index(drop=True).iterrows()])
    st.download_button("⬇ Download everything (ZIP: tables, plots, settings)",
                       _results_zip(out, fit, fit_figure(fit, theme="light", texts=_ftx), _pf, _summary),
                       "ftir_results.zip", "application/zip")

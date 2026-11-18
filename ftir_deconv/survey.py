"""Whole-spectrum survey: baseline -> peak detection -> assignment against an editable band table -> summary.

Independent of the Si-H Gaussian fit. The default band table holds *typical* literature ranges for common
Si/Al/O/N/H bands; broad bands overlap and positions shift with film composition, so edit the table for your material.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from .core import (airpls_baseline, als_baseline, linear_edge_baseline, polyfit_baseline, rubberband_baseline,
                   smooth_savgol)

BAND_COLUMNS = ["name", "lo", "hi", "assignment"]

DEFAULT_BANDS = [
    ("Si–O–Si rocking", 430, 500, "Si–O–Si rocking (SiOx / interface oxide)"),
    ("Al–O", 500, 780, "Al–O stretching / bending in AlOx (broad; AlO6 low, AlO4 high end)"),
    ("Si–O–Si bend", 780, 830, "Si–O–Si bending"),
    ("Si–N", 830, 900, "Si–N stretching (SiNx:H)"),
    ("Si–OH", 900, 1000, "Si–OH (silanol) / Si–O non-bridging"),
    ("Si–O–Si stretch", 1000, 1150, "Si–O–Si asymmetric stretch (TO)"),
    ("LO / N–H bend", 1150, 1300, "Si–O–Si LO shoulder / N–H bending"),
    ("H2O bend", 1600, 1680, "H–O–H bending (adsorbed water)"),
    ("C=O", 1680, 1780, "C=O stretching (organic contamination)"),
    ("Si–H", 2000, 2250, "Si–H stretching (Si–H, Si–H2, back-bonded variants)"),
    ("CO2", 2300, 2380, "Atmospheric CO2 (usually an artefact)"),
    ("C–H", 2800, 3000, "C–H stretching (organic contamination)"),
    ("O–H / N–H", 3200, 3700, "O–H stretching (water, Al–OH, Si–OH) / N–H stretching"),
]

SURVEY_BASELINES = {"rubberband": "Rubberband (never negative)", "poly": "Polynomial (function fit)",
                    "als": "ALS", "airpls": "airPLS", "linear": "Linear (range ends)", "none": "None"}


def default_band_table() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_BANDS, columns=BAND_COLUMNS)


@dataclass
class SurveyResult:
    x: np.ndarray
    y_raw: np.ndarray
    y_smooth: np.ndarray
    baseline: np.ndarray
    y_corr: np.ndarray
    peaks: pd.DataFrame
    bands: pd.DataFrame
    summary: list[str]


def _baseline(x, y, method, lam, order):
    if method == "none":
        return np.zeros_like(y)
    if method == "rubberband":
        return rubberband_baseline(x, y)
    if method == "poly":
        return polyfit_baseline(x, y, order, 100)
    if method == "als":
        return als_baseline(y, lam, 0.01)
    if method == "airpls":
        return airpls_baseline(y, lam)
    if method == "linear":
        return linear_edge_baseline(x, y)
    raise ValueError(f"Unknown baseline: {method!r}")


def survey_spectrum(df: pd.DataFrame, lo: float, hi: float, window: int = 21, poly: int = 3,
                    baseline: str = "rubberband", lam: float = 1e6, poly_order: int = 3,
                    bands: pd.DataFrame | None = None, prominence_pct: float = 2.0,
                    min_sep_cm1: float = 15.0, invert: bool = False,
                    min_fwhm_cm1: float = 8.0) -> SurveyResult:
    d = df.sort_values("wn")
    m = (d["wn"] >= lo) & (d["wn"] <= hi)
    x = d["wn"].to_numpy(float)[m.to_numpy()]
    y = d["abs"].to_numpy(float)[m.to_numpy()]
    if len(x) < 30:
        raise ValueError("Fewer than 30 points in the chosen range.")
    if invert:
        y = -y
    ys = smooth_savgol(y, window, poly)
    base = _baseline(x, ys, baseline, lam, poly_order)
    yc = ys - base
    step = float(np.median(np.diff(x)))

    span = float(np.ptp(yc)) or 1.0
    idx, props = find_peaks(yc, prominence=prominence_pct / 100 * span, distance=max(1, int(min_sep_cm1 / step)))
    bands = default_band_table() if bands is None else bands.dropna(subset=["lo", "hi"]).reset_index(drop=True)
    w = peak_widths(yc, idx, rel_height=0.5) if len(idx) else (np.array([]),) * 4
    rows = []
    for n, i in enumerate(idx):
        lb, rb = int(props["left_bases"][n]), int(props["right_bases"][n])
        seg = yc[lb:rb + 1] - np.interp(x[lb:rb + 1], [x[lb], x[rb]], [yc[lb], yc[rb]])
        hit = bands[(bands["lo"] <= x[i]) & (bands["hi"] >= x[i])]
        if len(hit):
            centres = (hit["lo"] + hit["hi"]) / 2
            name = hit.loc[(centres - x[i]).abs().idxmin(), "name"]
        else:
            name = "unassigned"
        if w[0][n] * step < min_fwhm_cm1:   # narrower than any real solid-state band -> noise spike
            continue
        rows.append({"position_cm-1": round(float(x[i]), 1), "height": float(yc[i]),
                     "prominence": float(props["prominences"][n]), "fwhm_cm-1": float(w[0][n] * step),
                     "area": float(np.trapezoid(seg, x[lb:rb + 1])), "band": name})
    peaks = pd.DataFrame(rows, columns=["position_cm-1", "height", "prominence", "fwhm_cm-1", "area", "band"])

    brow = []
    for _, b in bands.iterrows():
        a, c = min(b["lo"], b["hi"]), max(b["lo"], b["hi"])
        mm = (x >= a) & (x <= c)
        if mm.sum() < 3:
            brow.append({"band": b["name"], "range_cm-1": f"{a:.0f}–{c:.0f}", "covered": False, "area": np.nan,
                         "max_height": np.nan, "max_at_cm-1": np.nan, "peaks_found": 0, "assignment": b.get("assignment", "")})
            continue
        j = np.argmax(yc[mm])
        npk = int(((peaks["position_cm-1"] >= a) & (peaks["position_cm-1"] <= c)).sum()) if len(peaks) else 0
        brow.append({"band": b["name"], "range_cm-1": f"{a:.0f}–{c:.0f}", "covered": True,
                     "area": float(np.trapezoid(yc[mm], x[mm])), "max_height": float(yc[mm][j]),
                     "max_at_cm-1": round(float(x[mm][j]), 1), "peaks_found": npk, "assignment": b.get("assignment", "")})
    btab = pd.DataFrame(brow)
    pos_total = btab.loc[btab["area"] > 0, "area"].sum()
    btab["share_%"] = np.where(btab["area"] > 0, 100 * btab["area"] / (pos_total or np.nan), np.nan)

    summary = _summary(x, yc, peaks, btab, baseline, prominence_pct)
    return SurveyResult(x, y, ys, base, yc, peaks, btab, summary)


def _summary(x, yc, peaks, btab, baseline, prom) -> list[str]:
    L = [f"Surveyed {x.min():.0f}–{x.max():.0f} cm⁻¹ ({len(x)} points); baseline: {baseline}; "
         f"{len(peaks)} peak(s) above {prom:g}% of the corrected range (noise spikes narrower than the minimum width are ignored)."]
    if len(peaks):
        t = peaks.loc[peaks["prominence"].idxmax()]
        L.append(f"Strongest feature: {t['position_cm-1']:.0f} cm⁻¹ ({t['band']}), height {t['height']:.3g}, "
                 f"FWHM ≈ {t['fwhm_cm-1']:.0f} cm⁻¹.")
    cov = btab[btab["covered"]]
    found = cov[cov["peaks_found"] > 0]["band"].tolist()
    if found:
        L.append("Bands with a detected peak: " + ", ".join(found) + ".")
    missing = cov[cov["peaks_found"] == 0]["band"].tolist()
    if missing:
        L.append("No distinct peak in: " + ", ".join(missing) + ".")
    unc = btab[~btab["covered"]]["band"].tolist()
    if unc:
        L.append("Outside the data range (not assessed): " + ", ".join(unc) + ".")
    if cov["area"].notna().any() and (cov["area"] > 0).any():
        t = cov.loc[cov["area"].idxmax()]
        L.append(f"Largest integrated band: {t['band']} ({t['range_cm-1']} cm⁻¹), {t['share_%']:.0f}% of the positive band area.")
    neg = cov[cov["area"] < 0]["band"].tolist()
    if neg:
        L.append("Net-negative area in: " + ", ".join(neg) + " — a baseline that is too high, or a subtraction artefact; check before quoting.")
    if len(peaks) and (peaks["band"] == "unassigned").any():
        u = peaks[peaks["band"] == "unassigned"]["position_cm-1"].tolist()
        L.append("Peaks outside every table band (add rows if real): " + ", ".join(f"{v:.0f}" for v in u) + " cm⁻¹.")
    return L

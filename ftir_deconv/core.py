"""
Core FTIR Si-H peak-deconvolution pipeline.

Pipeline (same order as the analysis deck):
    load -> crop -> smooth (Savitzky-Golay) -> baseline (none | linear | ALS)
         -> multi-Gaussian fit (Levenberg-Marquardt via lmfit)
         -> fit diagnostics (R^2, AIC, BIC, identifiability)
         -> bond concentration, Eq. (1) of Kim et al., Thin Solid Films 675 (2019) 109-114

Everything here is plain NumPy/pandas so it can be reused from the CLI,
the Streamlit app, or a notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lmfit.models import GaussianModel
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------------------
# Constants (Lanford & Rand cross-sections, as used in Kim et al. 2019, Eq. 1)
# ---------------------------------------------------------------------------
SIGMA_SIH_CM2 = 7.4e-18  # Si-H stretching absorption cross-section, cm^2
SIGMA_NH_CM2 = 5.3e-18   # N-H stretching absorption cross-section, cm^2

# Table 2, Kim et al. (2019): Si-H stretching bonding structures (cm^-1)
TABLE2_SIH = {
    "Si3Si-H": 2005,
    "Si2Si-H2": 2065,
    "NSi2Si-H": 2082,
    "NSiSi-H2 / N2SiSi-H": 2140,
    "N2Si-H2": 2175,
    "N3Si-H": 2220,
}


@dataclass
class PeakSpec:
    """Initial guess + bounds for one Gaussian component (all in cm^-1)."""

    center: float
    lo: float
    hi: float
    sigma0: float = 20.0
    sigma_min: float = 5.0
    sigma_max: float = 50.0
    name: str = ""          # label shown on the curve / in tables
    free: bool = False      # True -> centre may move anywhere in the fit window


# Presets used in the original analysis (centers anchored to Table 2).
PRESETS: dict[int, list[PeakSpec]] = {
    2: [PeakSpec(2070, 2020, 2110), PeakSpec(2170, 2110, 2230)],
    3: [PeakSpec(2065, 2020, 2100), PeakSpec(2140, 2100, 2165), PeakSpec(2180, 2165, 2230)],
    4: [PeakSpec(2010, 1995, 2040), PeakSpec(2065, 2040, 2100),
        PeakSpec(2140, 2100, 2165), PeakSpec(2185, 2165, 2230)],
    5: [PeakSpec(2005, 1995, 2030), PeakSpec(2065, 2030, 2095), PeakSpec(2082, 2075, 2110),
        PeakSpec(2140, 2110, 2165), PeakSpec(2190, 2165, 2230)],
}



# ---------------------------------------------------------------------------
# User-editable literature table -> PeakSpec list.  Works for any material
# (SiNx:H, AlOx, ...): the user types name + literature wavenumber per row.
# ---------------------------------------------------------------------------
TABLE_COLUMNS = ["use", "name", "literature_cm1", "lo", "hi", "free", "sigma0", "sigma_max"]


def _row(name, lit, lo=None, hi=None, free=False, sigma0=20.0, sigma_max=50.0):
    return {"use": True, "name": name, "literature_cm1": lit, "lo": lo, "hi": hi, "free": free,
            "sigma0": sigma0, "sigma_max": sigma_max}


LITERATURE_PRESETS: dict[str, list[dict]] = {
    "Si-H in SiNx:H - 3 peaks (Kim et al. 2019, Table 2)": [
        _row("Si2Si-H2", 2065, 2020, 2100), _row("NSiSi-H2 / N2SiSi-H", 2140, 2100, 2165),
        _row("N2Si-H2", 2175, 2165, 2230)],
    "Si-H in SiNx:H - 2 peaks": [
        _row("Si-H (Si-rich)", 2070, 2020, 2110), _row("Si-H (N-rich)", 2170, 2110, 2230)],
    "Si-H in SiNx:H - 4 peaks": [
        _row("Si3Si-H", 2005, 1995, 2040), _row("Si2Si-H2", 2065, 2040, 2100),
        _row("NSiSi-H2 / N2SiSi-H", 2140, 2100, 2165), _row("N2Si-H2", 2175, 2165, 2230)],
    "Blank (enter your own literature peaks)": [_row("Peak 1", 1000.0), _row("Peak 2", 1100.0)],
}


def peaks_from_table(df: pd.DataFrame, fit_lo: float, fit_hi: float, default_tol: float = 30.0,
                     skipped: list | None = None) -> list[PeakSpec]:
    """Convert the user's literature table into PeakSpec objects.

    * start value of each centre = the literature wavenumber
    * bounds = [lo, hi] if given, else literature +/- default_tol
    * free=True -> centre may move anywhere inside the fit window
    * bounds are always clipped to the fit window
    * a row whose literature value lies outside the window raises ValueError, unless `skipped` (a list) is given:
      then the row is left out and its name appended to `skipped`
    """
    specs: list[PeakSpec] = []
    for i, r in enumerate(df.to_dict("records")):
        lit = r.get("literature_cm1")
        if lit is None or pd.isna(lit):
            continue  # blank row
        use = r.get("use", True)
        if use is not None and not pd.isna(use) and not bool(use):
            continue  # row switched off -> not part of the model at all
        lit = float(lit)
        name = str(r.get("name") or "").strip() or f"Peak {i + 1}"
        free = bool(r.get("free")) if not pd.isna(r.get("free", False)) else False
        lo, hi = r.get("lo"), r.get("hi")
        lo = lit - default_tol if lo is None or pd.isna(lo) else float(lo)
        hi = lit + default_tol if hi is None or pd.isna(hi) else float(hi)
        if free:
            lo, hi = fit_lo, fit_hi
        lo, hi = max(lo, fit_lo), min(hi, fit_hi)
        if not (fit_lo <= lit <= fit_hi):
            if skipped is not None:
                skipped.append(f"{name} ({lit:g})")
                continue
            raise ValueError(f"'{name}': literature value {lit:g} cm-1 lies outside the fit window "
                             f"({fit_lo:g}-{fit_hi:g}).")
        if lo >= hi:
            raise ValueError(f"'{name}': lower bound must be below upper bound (got {lo:g}, {hi:g}).")
        s0 = r.get("sigma0"); sm = r.get("sigma_max")
        s0 = 20.0 if s0 is None or pd.isna(s0) else float(s0)
        sm = 50.0 if sm is None or pd.isna(sm) else float(sm)
        sm = max(sm, 6.0)
        s0 = min(max(s0, 5.0), sm)  # keep the start inside [sigma_min=5, sigma_max]
        specs.append(PeakSpec(center=min(max(lit, lo), hi), lo=lo, hi=hi, sigma0=s0, sigma_max=sm,
                              name=name, free=free))
    if not specs:
        raise ValueError("No peak of the table lies inside the deconvolution window (or none is ticked 'Include'). "
                         "Widen the window or edit the table.")
    return specs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_spectrum(source) -> pd.DataFrame:
    """Read a 2-column (wavenumber, absorbance) CSV/TXT. Header optional.

    `source` may be a path or a file-like object (e.g. a Streamlit upload).
    Returns a DataFrame with columns ['wn', 'abs'] sorted by ascending wavenumber.
    """
    if hasattr(source, "seek"):
        source.seek(0)  # a stream may already have been read (e.g. by the app's anchor suggestion)
    df = pd.read_csv(source, header=None, sep=None, engine="python", comment="#")
    if hasattr(source, "seek"):
        source.seek(0)
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = ["wn", "abs"]
    if len(df) < 20:
        raise ValueError("Spectrum has fewer than 20 numeric rows - check the file format.")
    return df.sort_values("wn").reset_index(drop=True)


def crop(wn: np.ndarray, ab: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    m = (wn >= lo) & (wn <= hi)
    if m.sum() < 10:
        raise ValueError(f"Fewer than 10 points between {lo} and {hi} cm^-1.")
    return wn[m], ab[m]


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def smooth_savgol(y: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing. Window is forced odd and clipped to the data length."""
    n = len(y)
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    poly = min(int(poly), window - 1)
    return savgol_filter(y, window_length=window, polyorder=poly)


def als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005).

    lam : smoothness (larger -> stiffer baseline)
    p   : asymmetry (small -> baseline hugs the lower envelope of the data)
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def airpls_baseline(y: np.ndarray, lam: float = 1e6, itermax: int = 15, tol: float = 1e-3) -> np.ndarray:
    """airPLS baseline (Zhang, Chen & Liang, Analyst 135 (2010) 1138).

    Adaptive iteratively reweighted penalised least squares: points above the current baseline get weight 0,
    points below get weight exp(i*|d|/sum|d-|) so the baseline is pulled under the data without a hand-set
    asymmetry parameter p. Stops when the negative residual is below `tol` x the signal magnitude.
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y
    for i in range(1, itermax + 1):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + H).tocsc(), w * y)
        d = y - z
        neg = d < 0
        dssn = abs(d[neg].sum())
        if dssn < tol * abs(y).sum() or not neg.any():
            break
        w = np.zeros(L)
        w[neg] = np.exp(i * np.abs(d[neg]) / dssn)
        w[0] = np.exp(i * np.abs(d[neg]).max() / dssn)
        w[-1] = w[0]
    return z


def polyfit_baseline(x: np.ndarray, y: np.ndarray, order: int = 3, iterations: int = 100,
                     tol: float = 1e-4) -> np.ndarray:
    """'Function fit': polynomial baseline, fitted iteratively (modified polyfit,
    Lieber & Mahadevan-Jansen, Appl. Spectrosc. 57 (2003) 1363).

    A plain least-squares polynomial would run through the middle of the band and swallow it. Here, after every
    fit the data are clipped down to the fitted curve wherever they lie above it, so the polynomial settles
    underneath the peaks.
    """
    x = np.asarray(x, float)
    xm = (x - x.mean()) / (np.ptp(x) / 2 or 1.0)
    yy = np.asarray(y, float).copy()
    fit = yy
    for _ in range(iterations):
        fit = np.polyval(np.polyfit(xm, yy, order), xm)
        new = np.minimum(yy, fit)
        if np.linalg.norm(new - yy) <= tol * (np.linalg.norm(yy) or 1.0):
            break
        yy = new
    return np.polyval(np.polyfit(xm, yy, order), xm)


def autolevel_baseline(x: np.ndarray, y: np.ndarray, k: float = 2.0, iterations: int = 50,
                       one_sided: bool = True) -> np.ndarray:
    """Auto-leveling: remove slope and offset with a straight line fitted iteratively.

    Points that do not fit the line (residual > k x standard deviation) are discarded and the line is refitted
    until the set of kept points stops changing. one_sided=True discards only points ABOVE the line (absorption
    bands), which is what a baseline needs; False also discards points far below it.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.ones(len(x), bool)
    for _ in range(iterations):
        coef = np.polyfit(x[keep], y[keep], 1)
        res = y - np.polyval(coef, x)
        sd = float(np.std(res[keep])) or 1e-12
        new = res <= k * sd if one_sided else np.abs(res) <= k * sd
        if new.sum() < max(10, 0.1 * len(x)) or np.array_equal(new, keep):
            break
        keep = new
    return np.polyval(np.polyfit(x[keep], y[keep], 1), x)


def linear_edge_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Straight line through the first and last point of the window."""
    return y[0] + (x - x[0]) * (y[-1] - y[0]) / (x[-1] - x[0])


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def rubberband_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lower convex hull of the spectrum (the 'rubber band' stretched underneath it).

    The baseline never rises above the data, so (data - baseline) is >= 0 everywhere by construction.
    Caveat: it touches the deepest dips, so ripple makes the baseline too low and areas too large.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a]) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(x, x[hull], y[hull])


ANCHOR_METHODS = {"poly1": "Line through anchors", "poly2": "Quadratic through anchors",
                  "poly3": "Cubic through anchors", "interp": "Piecewise-linear through anchor medians"}


def _anchor_mask(x: np.ndarray, regions) -> np.ndarray:
    m = np.zeros(len(x), bool)
    for a, b in regions:
        lo, hi = (a, b) if a <= b else (b, a)
        m |= (x >= lo) & (x <= hi)
    return m


def anchor_baseline(x: np.ndarray, y: np.ndarray, regions, method: str = "poly1") -> np.ndarray:
    """Baseline drawn only through user-chosen 'flat' regions (list of (from, to) in cm^-1).

    poly1/2/3 : least-squares polynomial of that order fitted to the anchor points
    interp    : straight segments between the median of each anchor region
    """
    regions = [(float(a), float(b)) for a, b in regions if a is not None and b is not None]
    if not regions:
        raise ValueError("Anchor baseline needs at least one anchor region.")
    m = _anchor_mask(x, regions)
    if method == "interp":
        pts = []
        for a, b in regions:
            lo, hi = min(a, b), max(a, b)
            mm = (x >= lo) & (x <= hi)
            if mm.sum() >= 1:
                pts.append((float(np.median(x[mm])), float(np.median(y[mm]))))
        if len(pts) < 2:
            raise ValueError("'interp' baseline needs at least two anchor regions containing data.")
        pts.sort()
        px, py = zip(*pts)
        return np.interp(x, px, py)  # constant extrapolation beyond the outer anchors
    order = {"poly1": 1, "poly2": 2, "poly3": 3}.get(method)
    if order is None:
        raise ValueError(f"Unknown anchor method: {method!r}")
    if m.sum() < order + 2:
        raise ValueError(f"Anchor regions contain only {int(m.sum())} data points - widen them or lower the order.")
    n_regions = sum(1 for a, b in regions if ((x >= min(a, b)) & (x <= max(a, b))).sum() >= 3)
    if order > 1 and n_regions < order + 1:
        raise ValueError(f"A {'quadratic' if order == 2 else 'cubic'} anchor baseline needs at least {order + 1} "
                         f"separate anchor regions (you have {n_regions}); the curve is unconstrained between "
                         "distant regions. Add a region or use the line / piecewise-linear type.")
    xm = (x - x.mean()) / (np.ptp(x) / 2 or 1.0)  # conditioning
    coef = np.polyfit(xm[m], y[m], order)
    return np.polyval(coef, xm)


def suggest_anchors(x: np.ndarray, y_smooth: np.ndarray, fit_lo: float, fit_hi: float, width: float = 30.0):
    """Pick the flattest `width` cm^-1 window on each side of the fit window.

    Flatness = standard deviation after removing a straight line. Returns [(lo_side), (hi_side)] regions.
    """
    out = []
    for lo, hi in ((x.min(), fit_lo), (fit_hi, x.max())):
        mask = (x >= lo) & (x <= hi)
        xs, ys = x[mask], y_smooth[mask]
        if len(xs) < 8:
            continue
        w = min(width, np.ptp(xs))
        best, best_s = None, np.inf
        for start in np.arange(xs.min(), xs.max() - w + 1e-9, max(np.ptp(xs) / 60, 0.5)):
            mm = (xs >= start) & (xs <= start + w)
            if mm.sum() < 6:
                continue
            res = ys[mm] - np.polyval(np.polyfit(xs[mm], ys[mm], 1), xs[mm])
            if res.std() < best_s:
                best, best_s = (float(start), float(start + w)), float(res.std())
        if best:
            out.append(best)
    return out


@dataclass
class FitResult:
    k: int
    x: np.ndarray
    y: np.ndarray
    best_fit: np.ndarray
    components: list[np.ndarray]
    peaks: pd.DataFrame
    r2: float
    aic: float
    bic: float
    identifiable: bool
    warnings: list[str] = field(default_factory=list)
    lmfit_result: object = None


# ---------------------------------------------------------------------------
# Manual touch-up of the baseline (after any automatic method)
# ---------------------------------------------------------------------------
TOUCHUP_MODES = {"lower": "Lower baseline to the data", "shift": "Shift baseline by a value"}


def apply_touchups(x: np.ndarray, y: np.ndarray, base: np.ndarray, touchups, taper: float = 4.0) -> np.ndarray:
    """Local manual corrections to a baseline. touchups = [(from, to, mode, value), ...] (cm^-1).

    lower : inside the region the baseline is pulled down to the smoothed data wherever it lies above it,
            so the corrected curve cannot go negative there (it is clipped at zero in that region only).
    shift : the baseline is moved by `value` absorbance units (negative lowers it, positive raises it).
    Each correction fades in/out with a cosine ramp of `taper` cm^-1 outside the region, so no kinks are created.
    """
    new = np.array(base, float)
    for t in touchups or []:
        a, b = sorted((float(t[0]), float(t[1])))
        mode = t[2] if len(t) > 2 else "lower"
        val = float(t[3]) if len(t) > 3 and t[3] is not None and not np.isnan(t[3]) else 0.0
        d = np.where(x < a, a - x, np.where(x > b, x - b, 0.0))
        w = np.where(d <= 0, 1.0, np.where(d >= taper, 0.0, 0.5 * (1 + np.cos(np.pi * d / max(taper, 1e-9)))))
        if mode == "lower":
            new = new + w * (np.minimum(new, y) - new)
        elif mode == "shift":
            new = new + w * val
        else:
            raise ValueError(f"Unknown touch-up mode: {mode!r}")
    return new


def apply_pins(x: np.ndarray, base: np.ndarray, pins, taper: float = 15.0) -> np.ndarray:
    """Drag the baseline through user-clicked points. pins = [(wavenumber, absorbance), ...].

    At each pin the baseline is moved so that it passes exactly through (x_i, y_i). Between pins the shift is
    interpolated linearly; outside the outermost pins it fades to zero with a cosine ramp of `taper` cm^-1.
    One pin = a local bump/dent, two pins = a straight-ish segment, more pins = a hand-drawn baseline.
    """
    pts = sorted((float(a), float(b)) for a, b in (pins or []) if a is not None and b is not None
                 and not (np.isnan(a) or np.isnan(b)))
    if not pts:
        return np.array(base, float)
    px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
    delta_at = py - np.interp(px, x, base)
    inside = np.interp(x, px, delta_at)  # constant beyond the ends
    left = np.clip((px[0] - x) / max(taper, 1e-9), 0, 1)
    right = np.clip((x - px[-1]) / max(taper, 1e-9), 0, 1)
    fade = np.where(x < px[0], 0.5 * (1 + np.cos(np.pi * left)), np.where(x > px[-1], 0.5 * (1 + np.cos(np.pi * right)), 1.0))
    return np.array(base, float) + inside * fade


def suggest_touchups(x: np.ndarray, y_corr: np.ndarray, lo: float, hi: float, depth_sigma: float = 2.0,
                     pad: float = 2.0, merge_gap: float = 6.0):
    """Find stretches inside [lo, hi] where the corrected curve is clearly negative (deeper than depth_sigma noise)."""
    o = np.argsort(x)
    xs, ys = np.asarray(x)[o], np.asarray(y_corr)[o]
    sig = 1.4826 * np.median(np.abs(np.diff(ys) - np.median(np.diff(ys)))) / np.sqrt(2)
    m = (xs >= lo) & (xs <= hi) & (ys < 0)
    runs, i, n = [], 0, len(xs)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if ys[i:j + 1].min() < -depth_sigma * sig:
                runs.append([float(xs[i] - pad), float(xs[j] + pad)])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and a <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(round(float(a), 1), round(float(b), 1), "lower", 0.0) for a, b in merged]


def fit_gaussians(x: np.ndarray, y: np.ndarray, peaks: Sequence[PeakSpec]) -> FitResult:
    """Fit a sum of Gaussians. `amplitude` is the integrated area (absorbance*cm^-1)."""
    if not peaks:
        raise ValueError("Need at least one PeakSpec.")
    model = None
    for i in range(len(peaks)):
        comp = GaussianModel(prefix=f"p{i + 1}_")
        model = comp if model is None else model + comp
    params = model.make_params()
    ymax = float(np.max(np.abs(y))) or 1e-6
    for i, pk in enumerate(peaks):
        pre = f"p{i + 1}_"
        params[pre + "center"].set(value=pk.center, min=pk.lo, max=pk.hi)
        params[pre + "sigma"].set(value=pk.sigma0, min=pk.sigma_min, max=pk.sigma_max)
        # data-scaled start: height ~ ymax/2, area = height * sigma * sqrt(2*pi)
        params[pre + "amplitude"].set(value=0.5 * ymax * pk.sigma0 * 2.5066, min=0)

    res = model.fit(y, params, x=x)
    comps = res.eval_components(x=x)

    rows, warns = [], []
    total_area = sum(res.params[f"p{i + 1}_amplitude"].value for i in range(len(peaks)))
    for i, pk in enumerate(peaks):
        pre = f"p{i + 1}_"
        c, a, s = res.params[pre + "center"], res.params[pre + "amplitude"], res.params[pre + "sigma"]
        at_bound = abs(c.value - pk.lo) < 0.5 or abs(c.value - pk.hi) < 0.5
        unc = (a.stderr / a.value * 100) if (a.stderr is not None and a.value) else np.nan
        rows.append({
            "peak": pk.name or f"p{i + 1}",
            "literature_cm-1": pk.center,
            "shift_cm-1": c.value - pk.center,
            "center_cm-1": c.value,
            "sigma_cm-1": s.value,
            "fwhm_cm-1": 2.3548 * s.value,
            "area": a.value,
            "area_pct": 100 * a.value / total_area if total_area else np.nan,
            "area_unc_pct": unc,
            "at_bound": bool(at_bound),
        })
        if at_bound:
            warns.append(f"{pk.name or f'p{i + 1}'} centre pinned at a bound ({c.value:.1f} cm-1) - component is not resolved.")
        if total_area > 0 and a.value < 0.005 * total_area:
            rows[-1]["collapsed"] = True
            warns.append(f"{pk.name or f'p{i + 1}'} collapsed to ~zero area - the data do not need this peak; "
                         "untick it in the peak table.")
        if np.isnan(unc):
            warns.append(f"{pk.name or f'p{i + 1}'} has no uncertainty estimate (degenerate fit).")
    peaks_df = pd.DataFrame(rows).drop(columns="collapsed", errors="ignore")

    resid = res.residual
    r2 = 1.0 - float(np.var(resid) / np.var(y))
    identifiable = (not any(r["at_bound"] for r in rows) and not any(np.isnan(r["area_unc_pct"]) for r in rows)
                    and not any(r.get("collapsed") for r in rows))

    # strong amplitude correlations between neighbouring peaks -> sum is more trustworthy than split
    try:
        for i in range(len(peaks) - 1):
            ci = res.params[f"p{i + 1}_amplitude"].correl or {}
            cj = ci.get(f"p{i + 2}_amplitude")
            if cj is not None and abs(cj) > 0.95:
                warns.append(f"{peaks[i].name or f'p{i + 1}'} / {peaks[i + 1].name or f'p{i + 2}'} areas are strongly correlated (r={cj:+.2f}); trust their sum, not the split.")
    except Exception:
        pass

    return FitResult(
        k=len(peaks), x=x, y=y, best_fit=res.best_fit,
        components=[comps[f"p{i + 1}_"] for i in range(len(peaks))],
        peaks=peaks_df, r2=r2, aic=float(res.aic), bic=float(res.bic),
        identifiable=identifiable, warnings=warns, lmfit_result=res,
    )



def model_selection(x: np.ndarray, y: np.ndarray, ks: Iterable[int] = (2, 3, 4, 5),
                    candidates: dict[str, Sequence[PeakSpec]] | None = None) -> pd.DataFrame:
    """Fit candidate peak sets and tabulate R^2, AIC, BIC and identifiability.

    Default candidates are the Si-H presets with k components. Pass `candidates`
    ({label: [PeakSpec, ...]}) to compare your own peak sets (any material).
    AIC/BIC alone keep improving as peaks are added; the `identifiable` column flags
    fits whose peaks collapse onto bounds or lose uncertainty estimates (overfitting).
    """
    if candidates is None:
        candidates = {f"{k} peaks (preset)": PRESETS[k] for k in ks if k in PRESETS}
    rows = []
    for label, specs in candidates.items():
        fr = fit_gaussians(x, y, specs)
        rows.append({"model": label, "k": len(specs), "n_params": 3 * len(specs), "R2": fr.r2,
                     "AIC": fr.aic, "BIC": fr.bic, "identifiable": fr.identifiable})
    df = pd.DataFrame(rows)
    ok = df[df["identifiable"]]
    df["recommended"] = False
    if not ok.empty:
        df.loc[ok["BIC"].idxmin(), "recommended"] = True
    return df


# ---------------------------------------------------------------------------
# Quantification - Eq. (1), Kim et al. 2019:  C = A / (sigma * t)
# ---------------------------------------------------------------------------
def bond_concentration(area: float | np.ndarray, thickness_nm: float,
                       sigma_cm2: float = SIGMA_SIH_CM2) -> float | np.ndarray:
    """Bond density in cm^-3 from integrated absorbance area (cm^-1) and thickness (nm)."""
    if thickness_nm <= 0:
        raise ValueError("Thickness must be positive.")
    t_cm = thickness_nm * 1e-7
    return np.asarray(area) / (sigma_cm2 * t_cm)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    view_lo: float = 1950.0       # region loaded around the band (gives the baseline context)
    view_hi: float = 2300.0
    fit_lo: float = 2000.0        # region actually fitted
    fit_hi: float = 2250.0
    smooth_window: int = 21
    smooth_poly: int = 3
    baseline: str = "linear"      # none|linear|als|airpls|poly|autolevel|anchor|rubberband
    als_lam: float = 1e6
    als_p: float = 0.01
    k: int = 3
    peaks: list[PeakSpec] | None = None   # overrides PRESETS[k] when given
    anchors: list | None = None           # [(from, to), ...] flat regions for baseline='anchor'
    anchor_method: str = "poly1"          # poly1 | poly2 | poly3 | interp
    poly_order: int = 3                   # baseline="poly"
    poly_iter: int = 100
    autolevel_k: float = 2.0              # baseline="autolevel"
    invert: bool = False                  # multiply the spectrum by -1 (band points down because of subtraction order)
    floor_zero: bool = False              # lower the baseline so the corrected fit window never goes below 0
    thickness_nm: float | None = None
    sigma_cm2: float = SIGMA_SIH_CM2
    baseline_shift: float = 0.0           # absorbance units the whole baseline is LOWERED by (negative raises it)
    lift_scope: str = "fit"               # floor_zero looks at the deconvolution window ('fit') or the whole view ('view')
    pins: list | None = None              # [(wavenumber, absorbance), ...] clicked points the baseline is dragged through
    pin_taper: float = 15.0
    touchups: list | None = None          # [(from, to, 'lower'|'shift', value), ...] applied after the baseline
    skip_fit: bool = False                # baseline/pre-processing only (used when the fit cannot run)


@dataclass
class PipelineOutput:
    x_view: np.ndarray
    y_raw: np.ndarray
    y_smooth: np.ndarray
    baseline: np.ndarray
    y_corrected: np.ndarray
    fit: FitResult | None
    concentration: pd.DataFrame | None
    anchors: list | None = None
    lift: float = 0.0
    view_extended: bool = False
    touchups: list | None = None
    pins: list | None = None
    touchup_area: float = 0.0


def run_pipeline(df: pd.DataFrame, cfg: PipelineConfig) -> PipelineOutput:
    wn, ab = df["wn"].to_numpy(float), df["abs"].to_numpy(float)
    # the baseline is computed on the view range; the deconvolution window is independent of it, so the view is
    # widened (never narrowed) whenever the window reaches outside it
    v_lo, v_hi = min(cfg.view_lo, cfg.fit_lo), max(cfg.view_hi, cfg.fit_hi)
    extended = (v_lo, v_hi) != (cfg.view_lo, cfg.view_hi)
    x, y = crop(wn, ab, v_lo, v_hi)
    if cfg.invert:
        y = -y

    y_s = smooth_savgol(y, cfg.smooth_window, cfg.smooth_poly)
    if cfg.baseline == "als":
        base = als_baseline(y_s, cfg.als_lam, cfg.als_p)
    elif cfg.baseline == "linear":
        base = linear_edge_baseline(x, y_s)
    elif cfg.baseline == "airpls":
        base = airpls_baseline(y_s, cfg.als_lam)
    elif cfg.baseline == "poly":
        base = polyfit_baseline(x, y_s, cfg.poly_order, cfg.poly_iter)
    elif cfg.baseline == "autolevel":
        base = autolevel_baseline(x, y_s, cfg.autolevel_k)
    elif cfg.baseline == "rubberband":
        base = rubberband_baseline(x, y_s)
    elif cfg.baseline == "anchor":
        base = anchor_baseline(x, y_s, cfg.anchors or [], cfg.anchor_method)
    elif cfg.baseline == "none":
        base = np.zeros_like(y_s)
    else:
        raise ValueError(f"Unknown baseline mode: {cfg.baseline!r}")
    base_auto = base
    if cfg.pins:
        base = apply_pins(x, base, cfg.pins, cfg.pin_taper)
    if cfg.touchups:
        base = apply_touchups(x, y_s, base, cfg.touchups)
    if cfg.baseline_shift:
        base = base - cfg.baseline_shift
    base_touch = base
    y_c = y_s - base
    lift = 0.0
    if cfg.floor_zero:
        win = (x >= cfg.fit_lo) & (x <= cfg.fit_hi) if cfg.lift_scope != "view" else np.ones_like(x, bool)
        lift = max(0.0, -float(y_c[win].min())) if win.any() else 0.0
        base = base - lift
        y_c = y_c + lift

    if cfg.skip_fit:
        out = PipelineOutput(x, y, y_s, base, y_c, None, None, view_extended=extended)
        out.anchors = list(cfg.anchors) if cfg.baseline == "anchor" and cfg.anchors else None
        out.lift = lift
        out.touchups = list(cfg.touchups) if cfg.touchups else None
        out.pins = list(cfg.pins) if cfg.pins else None
        _w0 = (x >= cfg.fit_lo) & (x <= cfg.fit_hi)
        out.touchup_area = (float(np.trapezoid((base_auto - base_touch)[_w0], x[_w0]))
                            if (cfg.touchups or cfg.pins or cfg.baseline_shift) and _w0.sum() > 1 else 0.0)
        return out
    xf, yf = crop(x, y_c, cfg.fit_lo, cfg.fit_hi)
    peaks = cfg.peaks if cfg.peaks else PRESETS.get(cfg.k)
    if peaks is None:
        raise ValueError(f"No preset for k={cfg.k}; supply cfg.peaks.")
    fit = fit_gaussians(xf, yf, peaks)

    conc = None
    if cfg.thickness_nm:
        conc = fit.peaks[["peak", "center_cm-1", "area"]].copy()
        conc["C_cm-3"] = bond_concentration(conc["area"].to_numpy(), cfg.thickness_nm, cfg.sigma_cm2)
        total = pd.DataFrame([{"peak": "Total", "center_cm-1": np.nan,
                               "area": conc["area"].sum(), "C_cm-3": conc["C_cm-3"].sum()}])
        conc = pd.concat([conc, total], ignore_index=True)

    out = PipelineOutput(x, y, y_s, base, y_c, fit, conc, view_extended=extended)
    out.lift = lift
    out.touchups = list(cfg.touchups) if cfg.touchups else None
    out.pins = list(cfg.pins) if cfg.pins else None
    _w = (x >= cfg.fit_lo) & (x <= cfg.fit_hi)
    out.touchup_area = float(np.trapezoid((base_auto - base_touch)[_w], x[_w])) if (cfg.touchups or cfg.pins or cfg.baseline_shift) and _w.sum() > 1 else 0.0
    out.anchors = list(cfg.anchors) if cfg.baseline == "anchor" and cfg.anchors else None
    return out

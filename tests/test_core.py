import sys, pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
from make_sample import synthetic  # noqa: E402

from ftir_deconv import (PipelineConfig, als_baseline, bond_concentration, load_spectrum,  # noqa: E402
                         model_selection, run_pipeline, smooth_savgol)


def test_concentration_formula():
    # C = A / (sigma * t): A=0.1 cm^-1, sigma=7.4e-18, t=80 nm=8e-6 cm
    c = bond_concentration(0.1, 80, 7.4e-18)
    assert c == pytest.approx(0.1 / (7.4e-18 * 8e-6))


def test_concentration_scales_inverse_thickness():
    assert bond_concentration(0.1, 40, 7.4e-18) == pytest.approx(2 * bond_concentration(0.1, 80, 7.4e-18))


def test_savgol_forces_odd_window_and_reduces_noise():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 500)
    assert smooth_savgol(y, 20, 3).std() < y.std()


def test_als_removes_curved_baseline():
    x = np.linspace(0, 1, 600)
    peak = np.exp(-0.5 * ((x - 0.5) / 0.03) ** 2)
    base = 2 * x ** 2 + 0.5
    est = als_baseline(peak + base, lam=1e5, p=0.01)
    assert np.abs((peak + base - est) - peak).max() < 0.15


def test_pipeline_recovers_centers_and_total_area():
    df = synthetic(seed=3)
    out = run_pipeline(df, PipelineConfig(baseline="linear", k=3, thickness_nm=80))
    centers = np.sort(out.fit.peaks["center_cm-1"].to_numpy())
    truth = np.array([2080, 2126, 2184])
    assert np.all(np.abs(centers - truth) < 25)
    total = out.fit.peaks["area"].sum()
    assert total == pytest.approx(0.214, rel=0.25)
    assert out.concentration.iloc[-1]["peak"] == "Total"


def test_model_selection_returns_expected_columns():
    out = run_pipeline(synthetic(seed=3), PipelineConfig(k=3))
    sel = model_selection(out.fit.x, out.fit.y, ks=(2, 3))
    assert {"k", "R2", "AIC", "BIC", "identifiable", "recommended"} <= set(sel.columns)
    assert len(sel) == 2


def test_load_spectrum_sorted(tmp_path):
    p = tmp_path / "s.csv"
    wn = list(range(2100, 1900, -10))  # descending, 20 rows
    p.write_text("\n".join(f"{w},0.1" for w in wn))
    df = load_spectrum(str(p))
    assert list(df["wn"]) == sorted(wn)


def test_bad_baseline_mode():
    with pytest.raises(ValueError):
        run_pipeline(synthetic(), PipelineConfig(baseline="bogus"))


def test_peaks_from_table_bounds_and_free():
    import pandas as pd
    from ftir_deconv import peaks_from_table
    df = pd.DataFrame([
        {"name": "A", "literature_cm1": 2080, "lo": None, "hi": None, "free": False},
        {"name": "B", "literature_cm1": 2180, "lo": 2150, "hi": 2220, "free": False},
        {"name": "C", "literature_cm1": 2130, "lo": None, "hi": None, "free": True},
        {"name": "blank", "literature_cm1": None, "lo": None, "hi": None, "free": False},
    ])
    a, b, c = peaks_from_table(df, 2000, 2250, default_tol=25)
    assert (a.lo, a.hi) == (2055, 2105)
    assert (b.lo, b.hi) == (2150, 2220)
    assert (c.lo, c.hi) == (2000, 2250) and c.free


def test_peaks_from_table_rejects_out_of_window():
    import pandas as pd
    from ftir_deconv import peaks_from_table
    with pytest.raises(ValueError):
        peaks_from_table(pd.DataFrame([{"name": "x", "literature_cm1": 900}]), 2000, 2250)


def test_names_and_literature_in_fit_output():
    import pandas as pd
    from ftir_deconv import LITERATURE_PRESETS, peaks_from_table
    specs = peaks_from_table(pd.DataFrame(LITERATURE_PRESETS["Si-H in SiNx:H - 3 peaks (Kim et al. 2019, Table 2)"]), 2000, 2250)
    out = run_pipeline(synthetic(seed=3), PipelineConfig(peaks=specs, thickness_nm=80))
    assert list(out.fit.peaks["peak"]) == ["Si2Si-H2", "NSiSi-H2 / N2SiSi-H", "N2Si-H2"]
    assert {"literature_cm-1", "shift_cm-1"} <= set(out.fit.peaks.columns)
    assert list(out.concentration["peak"])[:1] == ["Si2Si-H2"]


def test_free_centre_can_leave_literature_window():
    import pandas as pd
    from ftir_deconv import peaks_from_table
    # literature value deliberately 40 cm-1 off the true band; free centre should still find ~2080
    df = pd.DataFrame([{"name": "A", "literature_cm1": 2120, "free": True}])
    out = run_pipeline(synthetic(seed=3), PipelineConfig(peaks=peaks_from_table(df, 2000, 2250)))
    assert abs(out.fit.peaks["center_cm-1"].iloc[0] - 2120) > 5


def test_anchor_baseline_recovers_polynomial_background():
    from ftir_deconv import anchor_baseline
    x = np.linspace(1950, 2300, 800)
    bg = 1e-3 + 2e-6 * (x - 1950) - 4e-9 * (x - 1950) ** 2
    band = 5e-4 * np.exp(-0.5 * ((x - 2120) / 12) ** 2)
    est = anchor_baseline(x, bg + band, [(1950, 2010), (2030, 2050), (2200, 2220), (2260, 2300)], "poly2")
    with pytest.raises(ValueError):
        anchor_baseline(x, bg + band, [(1950, 2010), (2240, 2300)], "poly2")
    assert np.abs(est - bg)[(x > 2000) & (x < 2250)].max() < 6e-5


def test_anchor_baseline_interp_and_errors():
    from ftir_deconv import anchor_baseline
    x = np.linspace(1950, 2300, 400)
    y = 1e-4 * (x - 1950) / 350
    est = anchor_baseline(x, y, [(1950, 1970), (2280, 2300)], "interp")
    assert np.abs(est - y)[(x > 1970) & (x < 2280)].max() < 5e-6
    with pytest.raises(ValueError):
        anchor_baseline(x, y, [], "poly1")
    with pytest.raises(ValueError):
        anchor_baseline(x, y, [(1950, 1950.5)], "poly1")


def test_suggest_anchors_picks_flat_side_regions():
    from ftir_deconv import suggest_anchors
    x = np.linspace(1950, 2300, 700)
    y = np.where(x < 1990, 0.0005 * np.sin((x - 1950) / 3), 0.0) + 1e-6 * np.random.default_rng(0).normal(size=x.size)
    (lo_a, lo_b), (hi_a, hi_b) = suggest_anchors(x, y, 2000, 2250)
    assert lo_a >= 1950 and lo_b <= 2000 and hi_a >= 2250 and hi_b <= 2300


def test_pipeline_with_anchor_baseline_and_cli_parse():
    df = synthetic(seed=3)
    out = run_pipeline(df, PipelineConfig(baseline="anchor", anchors=[(1950, 2000), (2260, 2300)], k=3))
    assert out.anchors == [(1950, 2000), (2260, 2300)]
    from ftir_deconv.cli import _parse_anchors
    assert _parse_anchors("1950-2000,2260-2300") == [(1950.0, 2000.0), (2260.0, 2300.0)]


def test_load_spectrum_can_be_called_twice_on_same_stream():
    import io
    from ftir_deconv import load_spectrum
    buf = io.BytesIO("\n".join(f"{2000 + i},{0.001 * i}" for i in range(50)).encode())
    a = load_spectrum(buf)
    b = load_spectrum(buf)
    assert len(a) == len(b) == 50


def test_rubberband_never_above_data_and_corrected_nonnegative():
    from ftir_deconv import rubberband_baseline
    x = np.linspace(1950, 2300, 700)
    y = 1e-3 * np.exp(-0.5 * ((x - 2120) / 30) ** 2) + 2e-6 * (x - 1950) + 2e-4 * np.sin(x / 7)
    b = rubberband_baseline(x, y)
    assert np.all(y - b >= -1e-12)
    out = run_pipeline(synthetic(seed=3), PipelineConfig(baseline="rubberband", k=3))
    assert out.y_corrected.min() >= -1e-12


def test_floor_zero_removes_negatives_in_fit_window():
    df = synthetic(seed=3)
    base = run_pipeline(df, PipelineConfig(baseline="linear", k=3))
    out = run_pipeline(df, PipelineConfig(baseline="linear", k=3, floor_zero=True))
    win = (out.x_view >= 2000) & (out.x_view <= 2250)
    assert out.y_corrected[win].min() >= -1e-12
    assert out.lift >= 0 and np.allclose(out.y_corrected - base.y_corrected, out.lift)


def test_new_baselines_remove_known_background():
    from ftir_deconv import airpls_baseline, autolevel_baseline, polyfit_baseline
    x = np.linspace(1950, 2300, 700)
    bg = 5e-4 + 1.5e-6 * (x - 1950) - 2e-9 * (x - 1950) ** 2
    band = 1e-3 * np.exp(-0.5 * ((x - 2120) / 20) ** 2)
    y = bg + band
    inside = (x > 2060) & (x < 2180)
    pb = polyfit_baseline(x, y, 2)
    plain = np.polyval(np.polyfit(x, y, 2), x)  # unconstrained fit runs through the band
    assert np.abs(pb - bg).max() < 0.8 * np.abs(plain - bg).max()
    ab = airpls_baseline(y, 1e6)
    assert np.abs(ab - bg)[~inside].max() < 1e-4
    lin = 5e-4 + 1.5e-6 * (x - 1950)
    lv = autolevel_baseline(x, lin + band, k=2.0)
    assert np.abs(lv - lin)[~inside].max() < 2e-5


def test_pipeline_runs_every_baseline_mode():
    df = synthetic(seed=3)
    for mode, extra in [("linear", {}), ("als", {}), ("airpls", {}), ("poly", {"poly_order": 2}), ("autolevel", {}),
                        ("anchor", {"anchors": [(1950, 2000), (2260, 2300)]}), ("rubberband", {}), ("none", {})]:
        out = run_pipeline(df, PipelineConfig(baseline=mode, k=3, **extra))
        assert np.isfinite(out.fit.r2), mode


def test_invert_flips_a_downward_band():
    df = synthetic(seed=3)
    df = df.assign(abs=-df["abs"])
    a = run_pipeline(df, PipelineConfig(baseline="anchor", anchors=[(1950, 2000), (2260, 2300)], k=3))
    b = run_pipeline(df, PipelineConfig(baseline="anchor", anchors=[(1950, 2000), (2260, 2300)], k=3, invert=True))
    assert a.y_corrected.min() < -abs(a.y_corrected.max())
    assert b.y_corrected.max() > 5 * abs(b.y_corrected.min())
    assert b.fit.r2 > 0.9


def test_survey_finds_and_assigns_known_bands():
    import pandas as pd
    from ftir_deconv import survey_spectrum
    x = np.linspace(450, 4000, 3000)
    g = lambda c, h, w: h * np.exp(-((x - c) / w) ** 2)
    y = 0.05 + 3e-5 * x + g(1070, 0.9, 40) + g(2150, 0.25, 25) + g(3400, 0.4, 150) + np.random.default_rng(0).normal(0, 2e-3, x.size)
    df = pd.DataFrame({"wn": x[::-1], "abs": y[::-1]})  # descending order, as most instruments export
    r = survey_spectrum(df, 450, 4000, window=21, baseline="poly", poly_order=3, prominence_pct=5)
    pos = r.peaks["position_cm-1"].to_numpy()
    for c in (1070, 2150, 3400):
        assert np.min(np.abs(pos - c)) < 25, (c, pos)
    assert "Si–O–Si stretch" in set(r.peaks["band"]) and "Si–H" in set(r.peaks["band"])
    assert r.bands.set_index("band").loc["Si–H", "peaks_found"] >= 1
    assert any("Strongest" in s for s in r.summary)


def test_unticked_table_row_is_excluded_from_the_model():
    import pandas as pd
    from ftir_deconv import LITERATURE_PRESETS, TABLE_COLUMNS, peaks_from_table
    tbl = pd.DataFrame(LITERATURE_PRESETS[list(LITERATURE_PRESETS)[0]])[TABLE_COLUMNS]
    assert len(peaks_from_table(tbl, 2000, 2250)) == 3
    tbl.loc[1, "use"] = False
    specs = peaks_from_table(tbl, 2000, 2250)
    assert [p.name for p in specs] == ["Si2Si-H2", "N2Si-H2"]
    tbl2 = tbl.drop(columns="use")              # old CSVs without the column keep working (everything included)
    assert len(peaks_from_table(tbl2, 2000, 2250)) == 3
    out = run_pipeline(synthetic(seed=3), PipelineConfig(baseline="linear", peaks=specs, thickness_nm=80))
    assert list(out.fit.peaks["peak"]) == ["Si2Si-H2", "N2Si-H2"]
    assert len(out.concentration) == 3           # 2 peaks + Total


def test_peak_with_no_signal_collapses_and_is_flagged():
    from ftir_deconv import PeakSpec, fit_gaussians
    x = np.linspace(2000, 2250, 400)
    y = 0.01 * np.exp(-((x - 2100) / 15) ** 2) + np.random.default_rng(0).normal(0, 1e-4, x.size)
    r = fit_gaussians(x, y, [PeakSpec(2100, 2050, 2150), PeakSpec(2200, 2170, 2240)])
    assert not r.identifiable
    assert any("collapsed" in w for w in r.warnings)


def test_deconvolution_window_is_independent_of_the_view_range():
    import pandas as pd
    from ftir_deconv import LITERATURE_PRESETS, TABLE_COLUMNS, peaks_from_table
    df = synthetic(seed=3)
    tbl = pd.DataFrame(LITERATURE_PRESETS[list(LITERATURE_PRESETS)[0]])[TABLE_COLUMNS]
    peaks = peaks_from_table(tbl, 2000, 2250)
    # fit window reaches outside the view range -> view is widened, nothing fails
    out = run_pipeline(df, PipelineConfig(view_lo=2050, view_hi=2200, fit_lo=2000, fit_hi=2250, baseline="linear",
                                          peaks=peaks))
    assert out.view_extended and out.x_view.min() <= 2002 and out.x_view.max() >= 2248
    # a narrower fit window than the view is fine too
    out2 = run_pipeline(df, PipelineConfig(view_lo=1950, view_hi=2300, fit_lo=2030, fit_hi=2200, baseline="als",
                                           peaks=peaks_from_table(tbl, 2030, 2200, skipped=[])))
    assert not out2.view_extended and out2.fit is not None
    # rows outside the window are skipped (reported), not fatal, when a list is supplied
    sk: list = []
    kept = peaks_from_table(tbl, 2100, 2250, skipped=sk)
    assert len(sk) >= 1 and all(p.center >= 2100 for p in kept)
    # preprocessing-only mode never needs a fit
    out3 = run_pipeline(df, PipelineConfig(baseline="als", skip_fit=True))
    assert out3.fit is None and out3.y_corrected.size


def test_manual_touchup_removes_negatives_locally_and_reports_area():
    from ftir_deconv import apply_touchups, suggest_touchups
    df = synthetic(seed=3)
    cfg = dict(baseline="als", als_lam=1e5, als_p=0.1, skip_fit=True, fit_lo=2000, fit_hi=2250)
    o0 = run_pipeline(df, PipelineConfig(**cfg))
    tu = suggest_touchups(o0.x_view, o0.y_corrected, 2000, 2250)
    o1 = run_pipeline(df, PipelineConfig(touchups=tu or [(2100, 2120, "lower", 0.0)], **cfg))
    w = (o1.x_view >= 2000) & (o1.x_view <= 2250)
    for a, b, *_ in o1.touchups:
        m = (o1.x_view >= a) & (o1.x_view <= b)
        assert o1.y_corrected[m].min() >= -1e-12          # 'lower' leaves nothing negative inside the region
    outside = ~np.any([(o1.x_view >= a - 4) & (o1.x_view <= b + 4) for a, b, *_ in o1.touchups] or [np.zeros_like(w)], axis=0)
    assert np.allclose(o1.baseline[outside], o0.baseline[outside])   # untouched elsewhere
    assert o1.touchup_area >= 0
    # shift mode moves the baseline by the stated value inside the region
    x = np.linspace(2000, 2250, 300); b0 = np.zeros_like(x)
    nb = apply_touchups(x, b0, b0, [(2100, 2150, "shift", -0.01)])
    assert np.isclose(nb[np.argmin(abs(x - 2125))], -0.01) and nb[0] == 0


def test_pins_drag_the_baseline_through_clicked_points():
    from ftir_deconv import apply_pins
    x = np.linspace(2000, 2250, 501); base = 0.01 + 1e-5 * (x - 2000)
    nb = apply_pins(x, base, [(2100.0, 0.0), (2150.0, 0.0)], taper=10)
    for xp in (2100.0, 2150.0):
        assert abs(np.interp(xp, x, nb) - 0.0) < 1e-12           # passes exactly through the pins
    assert np.allclose(nb[x < 2080], base[x < 2080]) and np.allclose(nb[x > 2170], base[x > 2170])  # fades out
    assert apply_pins(x, base, [], 10).tolist() == base.tolist()
    df = synthetic(seed=3)
    o = run_pipeline(df, PipelineConfig(baseline="als", skip_fit=True, pins=[(2100, 0.0), (2120, 0.0)]))
    assert o.pins and np.isfinite(o.touchup_area)


def test_baseline_shift_and_lift_scope():
    df = synthetic(seed=3)
    base = dict(baseline="als", als_lam=1e5, als_p=0.1, skip_fit=True, fit_lo=2000, fit_hi=2250)
    o0 = run_pipeline(df, PipelineConfig(**base))
    o1 = run_pipeline(df, PipelineConfig(baseline_shift=0.002, **base))
    assert np.allclose(o0.baseline - o1.baseline, 0.002) and np.allclose(o1.y_corrected - o0.y_corrected, 0.002)
    assert o1.touchup_area > 0
    ofit = run_pipeline(df, PipelineConfig(floor_zero=True, **base))
    oview = run_pipeline(df, PipelineConfig(floor_zero=True, lift_scope="view", **base))
    w = (ofit.x_view >= 2000) & (ofit.x_view <= 2250)
    assert ofit.y_corrected[w].min() >= -1e-12 and oview.y_corrected.min() >= -1e-12
    assert oview.lift >= ofit.lift


def test_axis_limits_and_peak_marks_in_figures():
    import matplotlib
    from ftir_deconv.plotting import fit_figure, processing_figure
    df = synthetic(seed=3)
    out = run_pipeline(df, PipelineConfig(baseline="linear", peaks=PRESETS[list(PRESETS)[0]]) if False else PipelineConfig(baseline="linear", k=3))
    marks = [(1, "A", 2065.0), (2, "B", 2140.0)]
    fig = processing_figure(out, marks=marks, xlim=(2050, 2200), ylim_top=(None, None), ylim_bottom=(-0.1, 0.5))
    ax_top, ax_bot = fig.axes[0], fig.axes[1]
    assert sorted(ax_bot.get_xlim()) == [2050.0, 2200.0] and ax_bot.get_xlim()[0] > ax_bot.get_xlim()[1]   # still right-to-left
    assert ax_bot.get_ylim() == (-0.1, 0.5)
    assert sum("P" in t.get_text() for t in ax_bot.texts) == 2
    f2 = fit_figure(out.fit, xlim=(2100, 2200), ylim=(0, 0.3))
    assert sorted(f2.axes[0].get_xlim()) == [2100.0, 2200.0] and f2.axes[0].get_ylim() == (0.0, 0.3)


def test_custom_titles_and_labels():
    import matplotlib; matplotlib.use("Agg")
    from ftir_deconv.plotting import fit_figure, processing_figure
    out = run_pipeline(synthetic(seed=3), PipelineConfig(baseline="linear", k=3))
    f = fit_figure(out.fit, texts={"title": "T", "xlabel": "X!", "ylabel": "Y!"})
    a = f.axes[0]
    assert (a.get_title(), a.get_xlabel(), a.get_ylabel()) == ("T", "X!", "Y!")
    g = processing_figure(out, texts={"title": "PT", "xlabel": "XX", "ylabel_top": "A", "ylabel_bottom": "B"})
    assert g.axes[0].get_ylabel() == "A" and g.axes[1].get_ylabel() == "B" and g.axes[1].get_xlabel() == "XX"
    assert processing_figure(out).axes[1].get_xlabel().startswith("Wavenumber")

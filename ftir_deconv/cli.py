"""Command-line interface:  python -m ftir_deconv.cli spectrum.csv --thickness-nm 80 --select"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import PipelineConfig, load_spectrum, model_selection, run_pipeline
from .plotting import fit_figure, processing_figure


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ftir-deconv", description="Si-H FTIR Gaussian deconvolution")
    p.add_argument("spectrum", nargs="+", help="CSV/TXT file(s): wavenumber, absorbance")
    p.add_argument("--out", default="results", help="output directory (default: results)")
    p.add_argument("--view", nargs=2, type=float, default=[1950, 2300], metavar=("LO", "HI"))
    p.add_argument("--fit", nargs=2, type=float, default=[2000, 2250], metavar=("LO", "HI"))
    p.add_argument("--baseline", choices=["none", "linear", "als", "airpls", "poly", "autolevel", "anchor", "rubberband"], default="linear")
    p.add_argument("--anchors", default=None, help='flat regions for --baseline anchor, e.g. "1950-2000,2260-2300"')
    p.add_argument("--anchor-method", choices=["poly1", "poly2", "poly3", "interp"], default="poly1")
    p.add_argument("--window", type=int, default=21, help="Savitzky-Golay window (odd)")
    p.add_argument("--poly", type=int, default=3, help="Savitzky-Golay polynomial order")
    p.add_argument("--als-lam", type=float, default=1e6)
    p.add_argument("--als-p", type=float, default=0.01)
    p.add_argument("--poly-order", type=int, default=3, help="order for --baseline poly")
    p.add_argument("--autolevel-k", type=float, default=2.0, help="rejection threshold (sigma) for --baseline autolevel")
    p.add_argument("-k", type=int, default=3, choices=[2, 3, 4, 5], help="number of Gaussians")
    p.add_argument("--thickness-nm", type=float, default=None, help="film thickness for Eq. (1)")
    p.add_argument("--select", action="store_true", help="also run AIC/BIC model selection (k=2..5)")
    return p


def _parse_anchors(text):
    if not text:
        return None
    out = []
    for part in text.split(","):
        lo, hi = part.replace(":", "-").split("-")
        out.append((float(lo), float(hi)))
    return out


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    out_root = Path(a.out)
    for path in a.spectrum:
        stem = Path(path).stem
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = PipelineConfig(view_lo=a.view[0], view_hi=a.view[1], fit_lo=a.fit[0], fit_hi=a.fit[1],
                             smooth_window=a.window, smooth_poly=a.poly, baseline=a.baseline,
                             als_lam=a.als_lam, als_p=a.als_p, k=a.k, thickness_nm=a.thickness_nm,
                             anchors=_parse_anchors(a.anchors), anchor_method=a.anchor_method,
                             poly_order=a.poly_order, autolevel_k=a.autolevel_k)
        out = run_pipeline(load_spectrum(path), cfg)

        out.fit.peaks.to_csv(out_dir / "peaks.csv", index=False)
        if out.concentration is not None:
            out.concentration.to_csv(out_dir / "concentration.csv", index=False)
        fit_figure(out.fit).savefig(out_dir / "fit.png", dpi=200)
        processing_figure(out).savefig(out_dir / "processing.png", dpi=200)

        summary = {"file": str(path), "k": out.fit.k, "R2": out.fit.r2, "AIC": out.fit.aic,
                   "BIC": out.fit.bic, "identifiable": out.fit.identifiable, "warnings": out.fit.warnings}
        if a.select:
            sel = model_selection(out.fit.x, out.fit.y)
            sel.to_csv(out_dir / "model_selection.csv", index=False)
            summary["model_selection"] = sel.to_dict("records")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

        print(f"\n== {stem} ==  k={out.fit.k}  R2={out.fit.r2:.4f}  identifiable={out.fit.identifiable}")
        print(out.fit.peaks[["peak", "literature_cm-1", "center_cm-1", "shift_cm-1", "area", "area_pct", "area_unc_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        if out.concentration is not None:
            print(out.concentration.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        for w in out.fit.warnings:
            print("  ! " + w)
        print(f"  -> {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

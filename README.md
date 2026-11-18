# ftir-sih-deconvolution

Automated Gaussian deconvolution of the **Si–H stretching band (~2000–2250 cm⁻¹)** in FTIR spectra of
a-Si / SiNx / AlOx-type PECVD films, with bond-density quantification. Available as a **Streamlit app**,
a **command-line tool** and a **Python library**.

Methodology follows Kim et al., *Thin Solid Films* 675 (2019) 109–114 (peak assignments, Table 2; Eq. 1).

## Pipeline

1. **Load** headerless 2-column CSV/TXT (wavenumber, absorbance); auto-detects separator, sorts ascending.
2. **Smooth** – Savitzky–Golay (default window 21, order 3) to suppress interference-fringe ripple.
3. **Baseline** – `linear` (edge-to-edge, for already corrected data), `als` (Eilers–Boelens asymmetric least
   squares; log10 λ adjustable in 0.01 steps, default 6.00, p=0.01, for raw spectra with drift) `airpls` (adaptive reweighting, Zhang et al. 2010), `poly` (iterative function fit, order 1-6), `autolevel` (iterative straight line that discards points not fitting it), `anchor` (baseline drawn through flat regions you choose; line/quadratic/cubic/piecewise-linear; auto-suggest available) or `none`.
4. **Fit** – sum of Gaussians (lmfit, Levenberg–Marquardt), one per row of an **editable literature peak table**
   (name, literature cm⁻¹, optional lo/hi bounds, *free centre* tick). The fit starts at the literature value, the
   name is drawn on the curve, and a free centre may move anywhere in the fit window. Use it for SiNx:H, AlOx or any
   other film. Built-in Si–H starting table (Kim et al. Table 2):

   | Structure | cm⁻¹ |
   |---|---|
   | Si₃Si–H | 2005 |
   | Si₂Si–H₂ | 2065 |
   | NSi₂Si–H | 2082 |
   | NSiSi–H₂ / N₂SiSi–H | 2140 |
   | N₂Si–H₂ | 2175 |
   | N₃Si–H | 2220 |

   `amplitude` is the integrated area (absorbance·cm⁻¹).
5. **Model selection** – R², AIC, BIC for your table vs. each leave-one-peak-out variant **plus identifiability checks** (center pinned at a bound,
   missing uncertainty, |r|≈1 amplitude correlation). Recommended k = lowest-BIC *identifiable* fit.
6. **Quantify** – Eq. (1): `C = A / (σ · t)`, σ(Si–H) = 7.4×10⁻¹⁸ cm², σ(N–H) = 5.3×10⁻¹⁸ cm²
   (Lanford & Rand), `t` in cm. Concentration scales as 1/t, so enter the real thickness.

## App layout

Peak table (editable literature table) → **Results** (summary cards, fitted components, bond density, plain-language
interpretation, ZIP download of everything) → detail tabs (Fit plot with per-curve checkboxes, Pre-processing, Model selection)
plus a **Help & Guide** tab. A welcome pop-up appears on first visit and a "Spectrum loaded" pop-up after each upload.

## Install & run

```bash
pip install -r requirements.txt

# Web app
streamlit run app.py

# CLI (writes peaks.csv, concentration.csv, fit.png, processing.png, summary.json, model_selection.csv)
python -m ftir_deconv.cli examples/sample_spectrum.csv --thickness-nm 80 --select

# Raw spectrum with curved baseline
python -m ftir_deconv.cli raw.csv --baseline als --als-lam 1e6 --als-p 0.01 -k 3
```

```python
from ftir_deconv import load_spectrum, PipelineConfig, run_pipeline, model_selection

out = run_pipeline(load_spectrum("spectrum.csv"), PipelineConfig(baseline="linear", k=3, thickness_nm=80))
print(out.fit.peaks, out.concentration, out.fit.warnings)
print(model_selection(out.fit.x, out.fit.y))
```

## Caveats

- R² alone cannot select the number of components; AIC/BIC improve with every added peak. Check the
  identifiability flag and warnings.
- Overlapping components (typically the 2080 / 2126 cm⁻¹ pair) have strongly correlated areas – quote their
  **sum**, not the split.
- Residual interference fringes are not Gaussian noise; reported uncertainties are lower bounds.
- The smoothing window, baseline mode and fit window change the areas – report them with your results.
- Absolute densities carry the uncertainty of σ and of the thickness.

## Tests

```bash
pip install -r requirements-dev.txt && pytest
```

`examples/make_sample.py` regenerates the synthetic demo spectrum (no measured data is bundled).

## License

MIT

## Library only: whole-spectrum survey

`from ftir_deconv import survey_spectrum` runs baseline -> peak detection -> assignment against an editable band table
for a whole spectrum (no fitting). It is not part of the web app; use the app's Pre-processing tab to view full spectra.

## Sharing with another person / PC

- **Hosted (recommended):** deploy on Streamlit Community Cloud; the other person just opens the link. Set the app to private and add their email, or set the secret `APP_PASSWORD` to require a password.
- **Local:** `pip install -r requirements.txt`, then `streamlit run app.py`. To serve other PCs on your network use `--server.address 0.0.0.0` and set `FTIR_APP_PASSWORD`.

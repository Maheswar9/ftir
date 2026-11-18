"""Help & Guide tab content for the Streamlit app."""
import streamlit as st


def render_help() -> None:
    st.header("Help & Guide")

    st.subheader("1 · Quick start")
    st.markdown("""
1. **Upload** a two-column CSV/TXT (wavenumber, absorbance) in the sidebar. No header row is needed.
2. **Pre-processing** (sidebar): choose the smoothing and the baseline. Check the *Pre-processing* tab: the
   corrected curve should be flat outside the band and positive across it.
3. **Deconvolution window** (top of the Analysis tab): cover every peak you want with its full width, put both edges on flat, near-zero curve, and leave out bands you are not fitting. The **Window check** under the peak table shows the window on your spectrum and warns when an edge sits on a band; **Set window from my table** fills it from your peak bounds.
4. **Literature peak table**: one row per expected band. Type its **name** and **literature wavenumber**.
   Pick a built-in starting table or start blank. Save and reload tables as CSV for other materials.
5. **Quantification**: enter the film thickness (nm) and the cross-section σ for your bond.
6. Read the **Fit & quantification** tab. Tick or untick curves in *Show on plot*, then download the PNG/CSV.
7. Press **Run model selection** to test whether every peak is really needed.
""")

    st.subheader("2 · What each algorithm does")
    st.markdown("**A. Savitzky–Golay smoothing** — suppresses noise and interference-fringe ripple.")
    st.markdown("""
A polynomial of order *p* is fitted by least squares to every window of *w* consecutive points. The fitted value
at the window centre replaces the original point. Unlike a moving average, it keeps peak height and width.
Window in cm⁻¹ ≈ *w* × point spacing (for example 21 points × 0.48 cm⁻¹ ≈ 10 cm⁻¹).
""")
    st.markdown("**B. Baseline removal** — removes the slow background so only the absorption band remains.")
    st.markdown("""
- **Linear**: a straight line through the two ends of the view window is subtracted. Use it for spectra that are
  already baseline-corrected and only tilted.
- **ALS (asymmetric least squares, Eilers & Boelens)**: finds a smooth curve *z* that stays *under* the peaks by
  minimising
""")
    st.latex(r"S=\sum_i w_i\,(y_i-z_i)^2+\lambda\sum_i(\Delta^2 z_i)^2,\qquad w_i=\begin{cases}p & y_i>z_i\\ 1-p & y_i\le z_i\end{cases}")
    st.markdown("""
  The weights are updated and the fit repeated 10 times. Use it for raw spectra with a curved background.
- **airPLS (adaptive iteratively reweighted penalised least squares, Zhang et al. 2010)**: like ALS, but the weights
  are not set by a fixed asymmetry *p*. After every pass, points above the baseline get weight 0 and points below
  get a weight that grows with how far below they are, so the baseline is pulled under the data automatically.
  Only λ needs to be chosen. Good for raw spectra with curved, drifting baselines.
- **Polynomial (function fit)**: a polynomial of the order you choose, fitted iteratively (modified polyfit,
  Lieber & Mahadevan-Jansen 2003): after each fit the data are clipped down to the curve wherever they lie above
  it, so the polynomial ends up under the peaks instead of through them. Orders 2–3 are usually enough; high
  orders start to follow the band itself.
- **Auto-level (iterative line)**: removes slope and offset with a straight line, refitted repeatedly while points
  that sit more than *k*σ above it (the absorption bands) are discarded. Works only when the true background is
  close to a straight line.
- **Anchor regions (recommended when the ends of the spectrum are not flat)**: you give one or more wavenumber
  ranges where *no band should exist*. The baseline is drawn only through the data inside those ranges:
  a straight line, quadratic or cubic polynomial (least squares), or straight segments between the median of each
  region. The band itself is never used to place the baseline. Different samples need different ranges, so
  choose them per spectrum; **Auto-suggest** picks the flattest 30 cm⁻¹ stretch on each side of the fit window and
  **Use window ends** resets them.
- **Rubberband**: the lower convex hull of the smoothed spectrum, like a rubber band stretched under it. The
  corrected curve is never negative, but the band touches every deep dip, so ripple makes the baseline too low and
  the areas too large. Use it when you need a non-negative curve and apply it identically to all samples.
- **None**: no baseline is subtracted.
""")
    st.markdown("**C. Multi-Gaussian fit** — splits the band into overlapping components.")
    st.latex(r"y(x)=\sum_{j=1}^{k}\frac{A_j}{\sigma_j\sqrt{2\pi}}\exp\!\left[-\frac{(x-\mu_j)^2}{2\sigma_j^2}\right],\qquad \mathrm{FWHM}_j=2.3548\,\sigma_j")
    st.markdown("""
Parameters (centre μ, width σ, area A) are found with the Levenberg–Marquardt algorithm, which minimises the
sum of squared residuals. **A is the integrated area** (absorbance·cm⁻¹), the quantity used for concentration.
Centres start at your literature values and are limited to the bounds you set, or are free inside the fit window.
""")
    st.markdown("**D. Fit quality and model selection**")
    st.latex(r"R^2=1-\frac{\mathrm{Var}(\text{residual})}{\mathrm{Var}(y)},\quad \mathrm{AIC}=n\ln\frac{RSS}{n}+2K,\quad \mathrm{BIC}=n\ln\frac{RSS}{n}+K\ln n")
    st.markdown("""
*n* = number of points, *K* = number of fitted parameters (3 per peak). **R² only ever rises as peaks are added**, so it
cannot choose the number of peaks. AIC and BIC subtract a penalty for extra parameters (lower is better; BIC
penalises more strongly). The app also checks **identifiability**: a model is rejected if a centre is stuck on a
bound or a peak has no uncertainty estimate. *Recommended* = lowest-BIC identifiable model.
""")
    st.markdown("**E. Bond concentration** — Eq. (1), Kim et al. (2019).")
    st.latex(r"C=\frac{A}{\sigma\,t}\qquad (A\ \text{in cm}^{-1},\ \sigma\ \text{in cm}^2,\ t\ \text{in cm}=\text{nm}\times10^{-7})")

    st.subheader("3 · Parameter guide")
    st.markdown("""
| Parameter | What it controls | Too low | Too high | Typical |
|---|---|---|---|---|
| **View window** | Region used for smoothing and baseline | Baseline has no anchor points | Other bands distort the baseline | Band ± 50–100 cm⁻¹ |
| **Fit window** | Region where Gaussians are fitted | Part of the band is missed | Includes noise/other bands, biasing areas | Just around the band |
| **SG window** | Smoothing strength | Fringes and noise remain | Real shoulders are flattened | 15–35 points |
| **SG polynomial order** | How closely the smoother follows curvature | Peaks are over-flattened | Noise passes through | 2–4 |
| **ALS log10 λ** | Stiffness of the baseline (steps of 0.01) | Baseline follows the peaks and eats area | Baseline cannot follow real curvature | 5–7 |
| **Anchor regions** | Where the baseline touches the data | Few points → noisy, unstable baseline | Region overlaps the band → band area is subtracted | 30–60 cm⁻¹ wide, outside the band |
| **Lift curve to zero** | Lowers the baseline until nothing in the fit window is negative | — | Adds a pedestal: area grows by lift × window width | Off unless a non-negative curve is required |
| **airPLS λ** | Stiffness of the adaptive baseline | Follows the band and removes area | Cannot follow real curvature | 1e5–1e7 (area can change a lot between them, so test) |
| **Polynomial order** | Flexibility of the function-fit baseline | Cannot follow the background | Starts fitting the band | 2–3 |
| **Auto-level k** | How far above the line a point must be to be discarded | Keeps band points, baseline rides high | Few points rejected, noise drives the line | 1.5–3 |
| **ALS p** | How strongly the baseline is pushed under the peaks | Baseline sits too low | Baseline rides into the peaks | 0.001–0.05 |
| **Default tolerance ±** | Search range around a literature value if bounds are blank | Peak cannot reach its true position | Neighbouring peaks swap or merge | 15–40 cm⁻¹ |
| **Thickness** | Divides the area in Eq. (1) | Concentration overestimated | Concentration underestimated | Measured value |
| **σ** | Absorption cross-section of the bond | — | — | Si–H 7.4e-18, N–H 5.3e-18 cm² |

**Peak table columns**

- **Peak name**: the label drawn on the curve and used in all tables.
- **Literature (cm⁻¹)**: the known position; the fit starts here.
- **Lower / upper bound**: the allowed range for the fitted centre. Leave blank to use ± default tolerance.
- **Free centre**: tick to let the centre move anywhere in the fit window. Use it when you do not trust the
  literature value; the *shift* column then shows how far the peak moved.
- **σ start / σ max**: starting width and the largest allowed width (cm⁻¹). A very large σ max lets one peak absorb the
  whole band; a very small σ start can stall the fit. The start is kept inside the allowed range automatically.
""")

    st.subheader("4 · Reading the results")
    st.markdown("""
- **R²**: how much of the band the fit reproduces. Compare only within the same data and fit window.
- **AIC / BIC**: only differences matter. A drop of more than about 10 in BIC is strong evidence for the lower model.
- **Parameters identifiable = No**: at least one peak is not resolved. Do not quote individual peak areas.
- **Area uncertainty (%)**: if it is large (above about 50%), that peak's share is poorly determined.
- **Shift (cm⁻¹)**: fitted centre minus literature. Shifts of a few cm⁻¹ are normal; large shifts suggest a wrong
  assignment or a wrong bound.
- **Strong correlation warning**: two neighbouring peaks trade area with each other. Trust their **sum**, not the split.
- **Total concentration** is more reliable than the individual components.
""")

    st.subheader("5 · Troubleshooting")
    st.markdown("""
| Symptom | Likely cause | Fix |
|---|---|---|
| A peak is flat or has zero area | Its literature value has no band in your data | Delete that row or tighten bounds |
| Centre stuck at a bound | Bound too tight, or peak not present | Widen the bound or remove the peak |
| Peaks swap positions | Overlapping bounds or free centres | Use non-overlapping bounds |
| Huge uncertainty or "degenerate fit" | Too many peaks for the data | Remove a peak; run model selection |
| Fit ignores the shoulder | σ max too small, or smoothing too strong | Raise σ max or reduce the SG window |
| Baseline cuts into the peak | λ too low or p too high | Raise log10 λ, lower p |
| Corrected curve dips below zero | Tilted window ends (linear), ripple or subtraction artefacts | *Anchor* baseline on flat regions, or *rubberband* for a never-negative curve (areas then increase); small negatives from noise are normal |
| R² is low but the shape looks right | Fringe ripple on the data | Increase smoothing; quote AIC/BIC with R² |
""")

    st.subheader("6 · Good practice")
    st.markdown("""
- Report the smoothing window, baseline mode, fit window and peak table with every result: all of them change the areas.
- Use the **same settings** for samples you want to compare.
- Absolute concentrations carry the uncertainty of σ and of the thickness.
- Interference fringes are not Gaussian noise, so the reported uncertainties are lower bounds.

**References** — Savitzky & Golay, *Anal. Chem.* 36 (1964) 1627 · Eilers & Boelens, *Baseline correction with
asymmetric least squares smoothing* (2005) · Moré, *The Levenberg–Marquardt algorithm* (1978) · Akaike, *IEEE
Trans. Autom. Control* 19 (1974) 716 · Schwarz, *Ann. Stat.* 6 (1978) 461 · Lanford & Rand, *J. Appl. Phys.* 49 (1978) 2473 ·
Kim et al., *Thin Solid Films* 675 (2019) 109–114.
""")

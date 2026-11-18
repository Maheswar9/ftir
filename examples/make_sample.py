"""Generate a synthetic Si-H spectrum (3 Gaussians + fringe ripple + noise) for demos/tests."""
import numpy as np, pandas as pd, pathlib

def synthetic(seed=0, noise=2e-5):
    rng = np.random.default_rng(seed)
    wn = np.arange(1800, 2400, 0.4822)
    g = lambda a, mu, s: a / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wn - mu) / s) ** 2)
    y = g(0.14, 2080, 28) + g(0.05, 2126, 17) + g(0.024, 2184, 14)           # areas: 0.14 / 0.05 / 0.024
    y += 1.2e-4 * np.sin(2 * np.pi * wn / 22.0)                              # etalon-like fringes
    y += 2e-6 * (wn - 1800)                                                  # small residual tilt
    y += rng.normal(0, noise, wn.size)
    return pd.DataFrame({"wn": wn, "abs": y})

if __name__ == "__main__":
    out = pathlib.Path(__file__).with_name("sample_spectrum.csv")
    synthetic().to_csv(out, header=False, index=False, float_format="%.6e")
    print("wrote", out)

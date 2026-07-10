# Generalized Tikhonov Regularization & Error Analysis

Image deblurring in the frequency domain for *Mathematics for Imaging and Signal Processing*
(A.A. 2025/2026). We recover a sharp image `f` from a blurred, noisy observation `g = A f + w`
using **generalized Tikhonov regularization**, compare the `L²`, `H¹` and `H²` penalties against a
hard spectral cutoff, and analyze the **bias–variance trade-off**.

## Layout

```
tikhonov-deblurring/
├── tikhonov_deblurring.py     # core library (all the math, one sectioned module)
├── deblurring_project.ipynb   # MAIN deliverable: Parts 1–4 with inline visualizations
├── requirements.txt
├── data/                      # drop your own grayscale image here (optional)
└── results/figures/           # PNG copies of the notebook's figures (all are embedded inline)
```

The notebook only imports the module (`import tikhonov_deblurring as td`) and calls it — all
computation lives in `tikhonov_deblurring.py`.

## Setup

Requires Python 3.10+. Create a virtual environment with the dependencies:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m ipykernel install --user --name tikhonov-deblurring \
    --display-name "Python (tikhonov-deblurring)"
```

## Run

**Notebook (main deliverable):** open `deblurring_project.ipynb`, select the
**Python (tikhonov-deblurring)** kernel, and *Restart & Run All*. Every figure renders inline and a
verification cell at the end asserts the key invariants.

Headless (re-execute end to end):
```bash
./.venv/bin/python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 deblurring_project.ipynb
```

**Module self-test** (no notebook needed):
```bash
./.venv/bin/python tikhonov_deblurring.py
```

## Use your own image

By default the built-in `scipy.datasets.ascent` photo is used (with a synthetic fallback if it
cannot be fetched offline). To use your own picture, drop a grayscale image into `data/` and change
the setup cell:

```python
f0 = td.load_image("data/your_image.png")   # resized to 256×256, normalized to [0, 1]
```

## What each part shows

1. **Forward problem** — Gaussian and linear-motion blur at 40 dB and 20 dB; the naive inverse
   `ĝ/K̂` blows up (ill-posedness).
2. **Generalized Tikhonov** — `L²`/`H¹`/`H²` reconstructions (motion blur). The edge zoom makes the
   trade-off explicit: `L²` sharpest but grainiest, `H²` smoothest but softest edges, `H¹` in
   between (best PSNR here).
3. **Spectral windowing** — a hard frequency cutoff (TSVD) produces ringing / Gibbs oscillations,
   whereas the smooth Tikhonov filter does not.
4. **Bias–variance trade-off** — the total error splits into an approximation (bias) term that grows
   with `µ` and a noise (variance) term that shrinks with `µ`; the optimum sits at their crossover
   (verified to coincide with the total-error minimum).

## Key implementation notes

- Physical kernels/penalties are built on a **centered** frequency grid, then converted once to FFT
  layout via `to_fft_layout` (`ifftshift`).
- **Motion blur** uses the DFT angular frequency `ω₁ = 2π·index/N`, so the motion length `L` is a
  true pixel count (Eq. 21); the linear phase is dropped by default (centered PSF, no spurious
  translation; it cancels in the Tikhonov filter anyway).
- The bias/variance terms are computed exactly in Fourier via Parseval's identity and cross-checked
  against a spatial-domain reference.
- The useful `µ` scale differs by orders of magnitude across penalties (`|P|²` is `1` vs `|ω|²` vs
  `|ω|⁴`), so each method is shown at its own PSNR-optimal `µ`, and the Part 4 `µ` grid is chosen to
  bracket the `H¹` optimum.

"""
tikhonov_deblurring.py
======================

Core library for the assignment *Generalized Tikhonov Regularization & Error
Analysis* (Mathematics for Imaging and Signal Processing, A.A. 2025/2026).

The inverse problem is image deblurring:

    g = A f + w            (spatial domain)
    g_hat = K_hat * f_hat + w_hat     (Fourier domain, pointwise product)

where ``A`` is convolution with a Point Spread Function (PSF), ``K_hat`` is its
transfer function and ``w`` is additive Gaussian noise.  Because ``A`` is a
convolution, every operator involved is *diagonal in the Fourier basis*, so all
the heavy lifting reduces to pointwise multiplications by frequency masks.

This module exposes the building blocks; the notebook ``deblurring_project.ipynb``
imports it and runs the four parts of the assignment with inline visualizations.

Conventions (enforced throughout):
    * FFTs via ``numpy.fft.fft2`` / ``ifft2``; the DC term sits at index (0, 0).
    * Physical kernels / penalties are built on a *centered* grid (DC in the
      middle) and converted to FFT layout with a single ``ifftshift``.
    * Images are float64, grayscale, normalized to [0, 1].
    * Reconstructions are returned as ``real(ifft2(...))``.

"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# ============================================================================
# Section 1: Configuration
# ----------------------------------------------------------------------------
# Every tunable parameter lives here so experiments are reproducible and easy
# to sweep from the notebook (e.g. ``td.GAUSS_SIGMA = 3``).
# ============================================================================

IMAGE_SIZE: int = 256          # work on IMAGE_SIZE x IMAGE_SIZE images (N, M even)

# Noise scenarios required by the assignment (Part 1).
SNR_HIGH_DB: float = 40.0      # high SNR  (~ visually clean)
SNR_LOW_DB: float = 20.0       # low  SNR  (~ visibly noisy)

# Blur parameters (Part 1). Units are *pixel/index* units on the frequency grid.
GAUSS_SIGMA: float = 4.0       # sigma_blur of the Gaussian transfer function
MOTION_L: int = 21             # motion length L in pixels (odd looks cleaner)

# Regularization-parameter grid for the bias-variance study (Part 4, H1 penalty).
# The optimal mu scales with the penalty order. With the literal raw-index penalty
# |P|^2 = |omega|^2 (omega in DFT index units, up to ~(N/2)^2 ~ 3e4), the H1
# optimum lies near 1e-8..1e-6 -- below the assignment's *example* range (1e-6..1,
# which presumes normalized frequencies). We bracket the actual optimum so the
# trade-off curve is centered with visible variance- and bias-dominated tails.
# (L2/H2 would need different ranges; Part 4 fixes the penalty to H1.)
MU_GRID: np.ndarray = np.logspace(-9, -1, 60)

# Master random seed for reproducibility (noise realizations, etc.).
SEED: int = 0


# ============================================================================
# Section 2: Image loading
# ----------------------------------------------------------------------------
# ``load_image`` returns a float64 grayscale image in [0, 1]. By default it
# uses the built-in ``scipy.datasets.ascent`` photo; if that is unavailable
# (no ``pooch`` / no network) it falls back to a synthetic image with sharp
# edges so the notebook always runs offline. Pass ``path=`` to use your own
# picture later (the "swap point").
# ============================================================================

def _to_float_gray(img: np.ndarray) -> np.ndarray:
    """Convert an arbitrary image array to a 2-D float64 grayscale array.

    RGB(A) inputs are collapsed with the standard luminosity weights
    (0.2989, 0.5870, 0.1140); an alpha channel, if present, is dropped.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.2989, 0.5870, 0.1140])
    elif img.ndim != 2:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {img.shape}.")
    return img


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resample a 2-D image to ``(size, size)`` via bilinear interpolation.

    Bilinear (``order=1``) does not overshoot, so it keeps edges clean and does
    not introduce ringing into the source image.
    """
    height, width = img.shape
    if (height, width) == (size, size):
        return img
    return ndimage.zoom(img, (size / height, size / width), order=1)


def _normalize01(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] so that the peak value is exactly 1.0.

    A peak of 1.0 makes the PSNR definition (Eq. 14) use ``peak=1.0`` directly.
    Constant images map to all-zeros.
    """
    img = img.astype(np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _synthetic_image(size: int) -> np.ndarray:
    """Build a synthetic grayscale phantom with several sharp-edged shapes.

    The shapes (squares, a thin bar, a disk) give strong straight and curved
    edges, which makes the edge zoom-ins of Part 2/3 and the ringing of Part 3
    easy to see. Geometry is specified relative to a 256-px reference and
    scaled to ``size``.
    """
    img = np.full((size, size), 0.12, dtype=np.float64)
    s = size / 256.0  # scale factor relative to the 256-px reference layout

    def box(y0: int, y1: int, x0: int, x1: int, value: float) -> None:
        img[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = value

    box(36, 116, 36, 116, 0.85)    # large bright square
    box(150, 205, 55, 110, 0.55)   # medium gray square
    box(165, 210, 150, 232, 0.95)  # bright rectangle
    box(120, 128, 28, 228, 0.40)   # thin horizontal bar (fine edges)

    yy, xx = np.ogrid[:size, :size]
    cy, cx, radius = int(95 * s), int(180 * s), int(42 * s)
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 0.70  # disk (curved edge)

    return img


def load_image(path: str | None = None, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load a float64 grayscale image in [0, 1] of shape ``(size, size)``.

    Parameters
    ----------
    path:
        If ``None`` (default), use the built-in ``scipy.datasets.ascent`` photo,
        falling back to a synthetic phantom if it cannot be fetched. If a path
        is given, load that file (this is the swap point for using your own
        image, e.g. ``load_image("data/my_photo.png")``).
    size:
        Output side length in pixels (assumed even elsewhere in the module).
    """
    if path is not None:
        import imageio.v3 as iio  # optional dependency, imported lazily
        img = iio.imread(path)
        return _normalize01(_resize(_to_float_gray(img), size))

    # Built-in photo. ``ascent()`` needs ``pooch`` and downloads on first call;
    # any failure (missing dependency / offline) falls back to the synthetic image.
    try:
        from scipy.datasets import ascent
        img = np.asarray(ascent(), dtype=np.float64)  # 512 x 512 grayscale
        return _normalize01(_resize(img, size))
    except Exception as exc:
        print(
            f"[load_image] built-in ascent() unavailable "
            f"({type(exc).__name__}: {exc}); using synthetic test image instead."
        )
        return _normalize01(_synthetic_image(size))


# ============================================================================
# Section 3: Frequency grid
# ----------------------------------------------------------------------------
# Implements Appendix Algorithm 1. Physical kernels (Gaussian, motion, ...) and
# penalty polynomials are defined on a *centered* grid where the zero frequency
# (DC) sits in the middle of the array. numpy's ``fft2``/``ifft2``, however,
# place the DC term at index (0, 0). The rule used everywhere in this module:
#
#     build on the centered grid  ->  ``to_fft_layout`` (ifftshift) once
#                                 ->  multiply with ``fft2`` outputs
#
# Getting this shift wrong (or applying it twice) is the single most common bug
# in frequency-domain deblurring, so the conversion goes through one helper.
# ============================================================================

def make_centered_grid(n_rows: int, n_cols: int):
    """Return the centered 2-D frequency grid ``(wX, wY, w_radius)``.

    The grid matches an image of shape ``(n_rows, n_cols)`` (rows = axis 0,
    columns = axis 1), with the DC term at the center:

        * ``wX`` -- horizontal frequency omega_1, varies along the columns (axis 1)
        * ``wY`` -- vertical   frequency omega_2, varies along the rows    (axis 0)
        * ``w_radius`` -- the magnitude ``|omega| = sqrt(wX**2 + wY**2)``

    Frequencies are integer index units (``-n/2 .. n/2 - 1``); ``n_rows`` and
    ``n_cols`` are assumed even. Convert the result to FFT layout with
    :func:`to_fft_layout` before combining it with ``fft2`` outputs.

    Reference: Lecture Notes, Appendix Algorithm 1.
    """
    x_range = np.arange(-n_cols // 2, n_cols // 2)  # columns -> horizontal (wX)
    y_range = np.arange(-n_rows // 2, n_rows // 2)  # rows    -> vertical   (wY)
    # 'xy' indexing makes meshgrid return shape (len(y), len(x)) = (n_rows, n_cols).
    wX, wY = np.meshgrid(x_range, y_range, indexing="xy")
    w_radius = np.sqrt(wX ** 2 + wY ** 2)
    return wX, wY, w_radius


def to_fft_layout(centered: np.ndarray) -> np.ndarray:
    """Shift a DC-centered array to numpy FFT layout (DC moved to index (0, 0)).

    Wrapper around ``numpy.fft.ifftshift`` so the centered-grid to FFT-layout
    conversion is applied consistently, once, wherever a kernel or penalty
    enters the frequency domain.
    """
    return np.fft.ifftshift(centered)


# ============================================================================
# Section 4: Transfer functions / PSF
# ----------------------------------------------------------------------------
# Each function takes the *centered* grid (wX, wY) and returns the transfer
# function K_hat already in FFT layout (DC at (0,0)), ready to multiply with
# ``fft2`` outputs. Two physical models are implemented (assignment scope:
# Gaussian + Motion); defocus (Bessel J1) is intentionally omitted.
#
# Frequency convention
# --------------------
# The shared grid holds integer DFT indices (-n/2 .. n/2-1). The Gaussian uses
# them directly, exactly as in Appendix Algorithm 1 (so ``sigma`` is a width in
# index units). The motion model instead converts an index to the true DFT
# angular frequency  omega_1 = 2*pi*index/N  (rad/sample), so that the motion
# length ``L`` is a genuine *pixel* count (Eq. 21). With L pixels, the first
# sinc zero falls at index N/L -- a visible yet invertible blur.
# ============================================================================

def gaussian_tf(wX: np.ndarray, wY: np.ndarray, sigma: float = GAUSS_SIGMA) -> np.ndarray:
    """Gaussian blur transfer function ``K_hat = exp(-|omega|^2 / (2 sigma^2))``.

    A smooth, strictly positive low-pass filter with **no zeros** (max = 1 at
    DC). ``sigma`` is the spectral width in index units; larger sigma => the TF
    decays faster => stronger blur.

    Returns ``K_hat`` in FFT layout. Reference: Appendix Algorithm 1.
    """
    K_centered = np.exp(-(wX ** 2 + wY ** 2) / (2.0 * sigma ** 2))
    return to_fft_layout(K_centered)


def motion_tf(
    wX: np.ndarray,
    wY: np.ndarray,
    length: int = MOTION_L,
    with_phase: bool = False,
) -> np.ndarray:
    """Linear motion blur transfer function (Eq. 21), motion along the x1 axis.

    With angular frequency ``omega_1 = 2*pi*wX/N`` (N = number of columns):

        K_hat(omega) = exp(-i * L * omega_1 / 2) * sinc(L * omega_1 / (2*pi))

    where ``sinc`` is numpy's normalized sinc, so the magnitude reduces to
    ``np.sinc(L * wX / N)``. The real, sign-alternating sinc has zeros at
    ``|wX| = N/L, 2N/L, ...`` -- the frequencies where motion blur destroys
    information (division-by-zero for naive inversion, ringing for hard cutoffs).
    Only the horizontal (``wX``) axis is affected; ``wY`` is passed for a uniform
    signature but unused.

    The linear phase ``exp(-i*L*omega_1/2)`` encodes the L/2-pixel shift of the
    causal boxcar PSF (sweep 0 -> L). It is **omitted by default** (``with_phase
    =False``), i.e. we use a PSF centered at the origin: this avoids a spurious
    L/2 translation of the displayed blurred image and makes ``K_hat`` exactly
    real (perfect Hermitian symmetry). The magnitude -- hence every zero and all
    the deblurring physics -- is identical, and the phase would cancel anyway in
    the Tikhonov filter (which applies ``conj(K_hat)`` and ``|K_hat|^2``). Set
    ``with_phase=True`` to reproduce Eq. 21 verbatim.

    Returns ``K_hat`` in FFT layout. Reference: Lecture Notes, Eq. (20)-(21).
    """
    n_cols = wX.shape[1]
    omega1 = 2.0 * np.pi * wX / n_cols                  # angular freq along x1 (rad/sample)
    K_centered = np.sinc(length * omega1 / (2.0 * np.pi))  # == sinc(L*wX/N), real
    if with_phase:
        K_centered = K_centered.astype(np.complex128) * np.exp(-1j * length * omega1 / 2.0)
    return to_fft_layout(K_centered)


def get_transfer_function(
    model: str,
    wX: np.ndarray,
    wY: np.ndarray,
    sigma: float = GAUSS_SIGMA,
    length: int = MOTION_L,
) -> np.ndarray:
    """Dispatch to the requested blur transfer function.

    ``model`` is ``"gaussian"`` or ``"motion"`` (case-insensitive). Returns
    ``K_hat`` in FFT layout.
    """
    key = model.lower()
    if key == "gaussian":
        return gaussian_tf(wX, wY, sigma)
    if key == "motion":
        return motion_tf(wX, wY, length)
    raise ValueError(
        f"Unknown blur model {model!r}; expected 'gaussian' or 'motion'."
    )


# ============================================================================
# Section 5: Metrics
# ----------------------------------------------------------------------------
# Quality / error metrics used to rank reconstructions and to confirm that the
# forward problem injects noise at the requested SNR. All operate on float
# images of identical shape and use every pixel.
# ============================================================================

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error ``mean((a - b)**2)`` (Eq. 10)."""
    diff = a - b
    return float(np.mean(diff * diff))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root Mean Squared Error ``sqrt(MSE)`` (Eq. 11)."""
    return float(np.sqrt(mse(a, b)))


def psnr(reference: np.ndarray, estimate: np.ndarray, peak: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in **decibels** (Eq. 14 on the dB scale, Eq. 13).

        PSNR_dB = 20 * log10(peak / RMSE) = 10 * log10(peak**2 / MSE)

    ``peak`` is the signal's dynamic range (1.0 for images normalized to [0, 1]).
    Higher is better; identical images give ``+inf``. Rule of thumb from the
    notes: 30-40 dB is good-to-excellent restoration quality.
    """
    err = rmse(reference, estimate)
    if err == 0.0:
        return float("inf")
    return float(20.0 * np.log10(peak / err))


def relative_l2_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 (Frobenius) error ``||estimate - truth|| / ||truth||``.

    A scale-free reconstruction-quality score (0 = perfect); convenient for
    ranking methods independently of image energy.
    """
    denom = np.linalg.norm(truth)
    if denom == 0.0:
        return float("inf")
    return float(np.linalg.norm(estimate - truth) / denom)


def measure_snr_db(g_clean: np.ndarray, g_noisy: np.ndarray) -> float:
    """Empirical SNR in dB, consistent with the noise-injection convention.

        SNR_dB = 20 * log10( std(g_clean) / std(noise) ),  noise = g_noisy - g_clean

    This matches Appendix Algorithm 2, which sets
    ``sigma_noise = 10**(-SNR_dB/20) * std(g_clean)``; hence the measured SNR of
    a freshly degraded image returns (up to sampling fluctuations) the target SNR.

    Note: this *std-based* SNR differs from the *mean-based* definition of Eq. (7)
    (``mean(image)/sigma``). We follow the appendix's injection formula so that
    the target and the measured SNR coincide throughout.
    """
    noise_std = float(np.std(g_noisy - g_clean))
    if noise_std == 0.0:
        return float("inf")
    return float(20.0 * np.log10(np.std(g_clean) / noise_std))


# ============================================================================
# Section 6: Forward problem
# ----------------------------------------------------------------------------
# Implements Appendix Algorithm 2: degrade a sharp image with a known blur and
# additive Gaussian noise to produce the observed data ``g``. Because the kernel
# is already in FFT layout, convolution is a single pointwise product.
# ============================================================================

def blur_image(f0: np.ndarray, K_hat: np.ndarray) -> np.ndarray:
    """Noise-free blurred image ``g_clean = real(ifft2(fft2(f0) * K_hat))``.

    Pointwise multiplication in frequency = convolution in space (Eq. 15).
    ``K_hat`` must be in FFT layout. Reference: Appendix Algorithm 2.
    """
    return np.real(np.fft.ifft2(np.fft.fft2(f0) * K_hat))


def sigma_for_snr(g_clean: np.ndarray, snr_db: float) -> float:
    """Noise standard deviation for a target SNR (Appendix Algorithm 2):

        sigma_noise = 10**(-SNR_dB / 20) * std(g_clean)

    Centralizes the injection formula so :func:`add_noise` and
    :func:`measure_snr_db` stay mutually consistent.
    """
    return float(10.0 ** (-snr_db / 20.0) * np.std(g_clean))


def add_noise(g_clean: np.ndarray, snr_db: float, rng: np.random.Generator):
    """Add zero-mean Gaussian noise at the target SNR. Returns ``(g_noisy, noise)``.

    The raw ``noise`` array is returned alongside the noisy image because Part 4
    needs the *actual* noise realization to compute the variance error term
    exactly (not just its standard deviation). Reference: Appendix Algorithm 2.
    """
    sigma = sigma_for_snr(g_clean, snr_db)
    noise = rng.normal(0.0, sigma, size=g_clean.shape)
    return g_clean + noise, noise


def simulate_forward(
    f0: np.ndarray,
    K_hat: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> dict:
    """Run the full forward problem (blur, then add noise).

    Returns a dict with:
        * ``g_clean`` -- noise-free blurred image ``A f0``
        * ``g_noisy`` -- observed data ``g = A f0 + w``
        * ``noise``   -- the noise realization ``w`` (for Part 4's variance term)
        * ``sigma``   -- the (theoretical) injected noise std

    Reference: Appendix Algorithm 2.
    """
    g_clean = blur_image(f0, K_hat)
    sigma = sigma_for_snr(g_clean, snr_db)
    g_noisy, noise = add_noise(g_clean, snr_db, rng)
    return {"g_clean": g_clean, "g_noisy": g_noisy, "noise": noise, "sigma": sigma}


# ============================================================================
# Section 7: Reconstruction
# ----------------------------------------------------------------------------
# Implements Appendix Algorithm 3 (generalized Tikhonov) plus the Part 3 hard
# spectral window and the naive inverse used to motivate regularization. All
# reconstructions are pointwise operations in the Fourier domain followed by a
# single inverse FFT.
# ============================================================================

def penalty_squared(method: str, wX: np.ndarray, wY: np.ndarray) -> np.ndarray:
    """Squared penalty polynomial ``|P(omega)|^2`` in FFT layout (Appendix Alg. 3).

        * ``"L2"`` -> 1                      (order 0: penalizes energy ||f||^2)
        * ``"H1"`` -> wX^2 + wY^2            (order 1: penalizes ||grad f||^2, Eq. 108)
        * ``"H2"`` -> (wX^2 + wY^2)^2        (order 2: penalizes the Hessian, Eq. 110)

    Built on the centered grid, then shifted to FFT layout. Note that H1/H2
    vanish at DC, so the image mean is never penalized (the mean has no
    derivative) -- it is recovered by the plain inverse there.
    """
    r2 = wX.astype(np.float64) ** 2 + wY.astype(np.float64) ** 2
    key = method.upper()
    if key == "L2":
        P_sq = np.ones_like(r2)
    elif key == "H1":
        P_sq = r2
    elif key == "H2":
        P_sq = r2 ** 2
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'L2', 'H1' or 'H2'.")
    return to_fft_layout(P_sq)


def tikhonov_filter(
    g: np.ndarray,
    K_hat: np.ndarray,
    P_sq_fft: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Generalized Tikhonov reconstruction (Eq. 107 / Appendix Algorithm 3):

        f_hat_mu = conj(K_hat) * g_hat / (|K_hat|^2 + mu * |P|^2)

    ``K_hat`` and ``P_sq_fft`` are in FFT layout. The denominator is strictly
    positive for ``mu > 0`` (|P|^2 vanishes only at DC, where |K_hat|^2 = 1), so
    no guarding is needed. Returns the real-space reconstruction.
    """
    g_hat = np.fft.fft2(g)
    denom = np.abs(K_hat) ** 2 + mu * P_sq_fft
    f_hat = np.conj(K_hat) * g_hat / denom
    return np.real(np.fft.ifft2(f_hat))


def window_function(K_hat: np.ndarray, P_sq_fft: np.ndarray, mu: float) -> np.ndarray:
    """Tikhonov window ``W_mu = |K_hat|^2 / (|K_hat|^2 + mu*|P|^2)`` (Eq. 112).

    This is the multiplier of the operator ``R_mu A`` (a soft, SNR-dependent
    low-pass): ``~1`` where the signal dominates, ``~0`` where noise does. Real,
    in [0, 1]. Used for analysis/plots and for the exact bias term in Part 4.
    """
    Kmag2 = np.abs(K_hat) ** 2
    return Kmag2 / (Kmag2 + mu * P_sq_fft)


def spectral_window_reconstruction(
    g: np.ndarray,
    K_hat: np.ndarray,
    w_radius_fft: np.ndarray,
    Omega: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Hard spectral cutoff reconstruction (Part 3, Eq. 3 / TSVD equivalent):

        f_hat = W_Omega * g_hat / K_hat,   W_Omega(omega) = 1 if |omega| < Omega else 0

    ``w_radius_fft`` is the frequency magnitude in FFT layout (i.e.
    ``to_fft_layout(w_radius)``). Division by ``K_hat`` is guarded only against
    literal zeros (``|K_hat| < eps``); within the passband the genuine noise
    amplification is preserved: that, together with the abrupt edge
    of the brick-wall window, is what produces the ringing / Gibbs artifacts we
    want to contrast against the smooth Tikhonov roll-off.
    """
    g_hat = np.fft.fft2(g)
    inv = np.zeros_like(g_hat)
    safe = np.abs(K_hat) > eps
    inv[safe] = g_hat[safe] / K_hat[safe]
    W = (w_radius_fft < Omega).astype(np.float64)  # brick-wall mask, FFT layout
    return np.real(np.fft.ifft2(W * inv))


def naive_inverse(g: np.ndarray, K_hat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Naive inverse filter ``f_hat = g_hat / K_hat`` (Eq. 32).

    The unregularized solution. ``f_hat = f_hat_true + w_hat / K_hat``: dividing
    the (white) noise by a transfer function that decays to ~0 amplifies high
    frequencies without bound, so this is expected to blow up. ``eps`` only
    prevents literal division by zero; the catastrophic amplification is kept to
    demonstrate why regularization is necessary.
    """
    g_hat = np.fft.fft2(g)
    K_safe = np.where(np.abs(K_hat) < eps, eps, K_hat)
    return np.real(np.fft.ifft2(g_hat / K_safe))


# ============================================================================
# Section 8: Error analysis
# ----------------------------------------------------------------------------
# Implements Part 4: split the reconstruction error into the competing
# Approximation (bias) and Noise-propagation (variance) terms as a function of
# the regularization parameter mu (Eq. 100-101).
#
# Because every operator is diagonal in Fourier, with
#     W_mu  = |K|^2 / (|K|^2 + mu|P|^2)        (= R_mu A, the window function)
#     Phi_mu = conj(K) / (|K|^2 + mu|P|^2)     (= R_mu, the reconstruction filter)
# Parseval's identity gives the two terms exactly and cheaply (1/N^2 is the
# DFT-Parseval normalization for an N*M image):
#     Bias^2(mu)     = (1/N^2) * sum |W_mu - 1|^2 |f0_hat|^2
#     Variance^2(mu) = (1/N^2) * sum |Phi_mu|^2  |w_hat|^2
# where w is the *actual* noise realization from the forward problem.
# ============================================================================

def bias_variance_curves(
    f0: np.ndarray,
    K_hat: np.ndarray,
    noise: np.ndarray,
    P_sq_fft: np.ndarray,
    mu_grid: np.ndarray,
) -> dict:
    """Bias, variance and total error vs mu (Fourier/Parseval form, Eq. 101).

    Returns a dict of arrays (same length as ``mu_grid``):
        * ``bias``     -- ||(R_mu A - I) f0||^2  (approximation error)
        * ``variance`` -- ||R_mu w||^2           (noise-propagation error)
        * ``total``    -- bias + variance
    """
    n_pixels = f0.size  # N*M, the Parseval normalization
    f0_mag2 = np.abs(np.fft.fft2(f0)) ** 2
    w_mag2 = np.abs(np.fft.fft2(noise)) ** 2
    Kmag2 = np.abs(K_hat) ** 2

    bias = np.empty(mu_grid.shape, dtype=np.float64)
    variance = np.empty(mu_grid.shape, dtype=np.float64)
    for i, mu in enumerate(mu_grid):
        denom = Kmag2 + mu * P_sq_fft
        W = Kmag2 / denom                 # window function W_mu  (real)
        Phi_mag2 = Kmag2 / denom ** 2     # |Phi_mu|^2 = |K|^2 / denom^2
        bias[i] = np.sum((W - 1.0) ** 2 * f0_mag2) / n_pixels
        variance[i] = np.sum(Phi_mag2 * w_mag2) / n_pixels
    total = bias + variance
    return {"bias": bias, "variance": variance, "total": total}


def optimal_mu(mu_grid: np.ndarray, total: np.ndarray) -> float:
    """Return the ``mu`` in ``mu_grid`` that minimizes the total error."""
    return float(mu_grid[int(np.argmin(total))])


def best_mu_by_psnr(
    f0: np.ndarray,
    g: np.ndarray,
    K_hat: np.ndarray,
    P_sq_fft: np.ndarray,
    mu_grid: np.ndarray,
    peak: float = 1.0,
):
    """Pick the ``mu`` in ``mu_grid`` whose Tikhonov reconstruction maximizes PSNR.

    A simple, robust way to choose a near-optimal regularization parameter per
    method/scenario for the Part 2 / Part 3 comparisons (so each penalty is shown
    at its own best setting, since the useful ``mu`` scale differs by orders of
    magnitude between L2, H1 and H2). Returns ``(best_mu, best_psnr)``.
    """
    best_mu, best_psnr = float(mu_grid[0]), -np.inf
    for mu in mu_grid:
        score = psnr(f0, tikhonov_filter(g, K_hat, P_sq_fft, mu), peak)
        if score > best_psnr:
            best_psnr, best_mu = score, float(mu)
    return best_mu, best_psnr


def bias_variance_spatial(
    f0: np.ndarray,
    K_hat: np.ndarray,
    noise: np.ndarray,
    P_sq_fft: np.ndarray,
    mu: float,
):
    """Reference bias/variance for a single ``mu``, computed in the spatial domain.

    Used to validate :func:`bias_variance_curves`: it forms the actual error
    images and sums their squared pixels, so agreement with the Fourier formulas
    confirms the Parseval bookkeeping (normalization, layout). Returns
    ``(bias2, variance2)``.

        bias image     = real(ifft2(W_mu   * f0_hat)) - f0   = (R_mu A - I) f0
        variance image = real(ifft2(Phi_mu * w_hat))         = R_mu w
    """
    denom = np.abs(K_hat) ** 2 + mu * P_sq_fft
    W = np.abs(K_hat) ** 2 / denom
    Phi = np.conj(K_hat) / denom
    bias_img = np.real(np.fft.ifft2(W * np.fft.fft2(f0))) - f0
    var_img = np.real(np.fft.ifft2(Phi * np.fft.fft2(noise)))
    return float(np.sum(bias_img ** 2)), float(np.sum(var_img ** 2))


# ============================================================================
# Section 9: Plot helpers
# ----------------------------------------------------------------------------
# Thin, reusable plotting utilities. They create (and return) their figure/axes
# so the notebook stays in control of layout and inline display. matplotlib is
# imported lazily so the numerical core has no plotting dependency at import time.
# ============================================================================

def show_images(
    images,
    titles=None,
    ncols=None,
    cmap: str = "gray",
    vmin=0.0,
    vmax=1.0,
    figsize=None,
    suptitle=None,
):
    """Display grayscale images in a grid with shared intensity scaling.

    ``vmin``/``vmax`` default to (0, 1) so panels are directly comparable; pass
    ``None`` for either to autoscale per image (useful for the naive inverse,
    whose values blow up). Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.2 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()
    for k, ax in enumerate(flat):
        if k < n:
            ax.imshow(images[k], cmap=cmap, vmin=vmin, vmax=vmax)
            if titles is not None:
                ax.set_title(titles[k], fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig, axes


def zoom(image: np.ndarray, box) -> np.ndarray:
    """Crop ``image`` to ``box = (row0, row1, col0, col1)`` for edge zoom-ins."""
    r0, r1, c0, c1 = box
    return image[r0:r1, c0:c1]


def plot_bias_variance(
    mu_grid,
    bias,
    variance,
    total,
    mu_opt=None,
    ax=None,
    title=None,
):
    """Plot the Part 4 bias / variance / total error curves on a log-log scale.

    Draws the three curves, a vertical line at ``mu_opt`` (if given) and a marker
    at the Bias=Variance crossover (whose mu should match the total-error
    minimum). Accepts an existing ``ax`` for composition; returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    bias = np.asarray(bias)
    variance = np.asarray(variance)
    ax.loglog(mu_grid, bias, "--", color="tab:blue", label=r"Bias$^2$ (approximation)")
    ax.loglog(mu_grid, variance, "--", color="tab:orange", label=r"Variance$^2$ (noise)")
    ax.loglog(mu_grid, total, "-", color="black", lw=2, label=r"Total error$^2$")

    # Crossover marker: the mu where the two terms are closest.
    i_cross = int(np.argmin(np.abs(bias - variance)))
    ax.plot(mu_grid[i_cross], total[i_cross], "o", color="tab:red", ms=7,
            label=r"Bias$^2$ = Variance$^2$", zorder=5)
    if mu_opt is not None:
        ax.axvline(mu_opt, color="tab:green", ls=":", label=rf"$\mu_{{opt}}$ = {mu_opt:.1e}")

    ax.set_xlabel(r"regularization parameter $\mu$")
    ax.set_ylabel("squared error")
    ax.set_title(title or "Bias-Variance trade-off")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return fig, ax


# ============================================================================
# Smoke test
# ----------------------------------------------------------------------------
# Runnable without the notebook: ``python tikhonov_deblurring.py``.
# Exercises every phase with asserts: image loading, frequency grid, transfer
# functions, metrics, forward problem, reconstruction, bias-variance, plotting.
# ============================================================================

if __name__ == "__main__":
    # --- image loading -----------------------------------------------
    f0 = load_image()
    print("Loaded image:")
    print(f"  shape  = {f0.shape}")
    print(f"  dtype  = {f0.dtype}")
    print(f"  min/max = {f0.min():.3f} / {f0.max():.3f}")
    print(f"  mean   = {f0.mean():.3f}")
    assert f0.shape == (IMAGE_SIZE, IMAGE_SIZE), "unexpected image shape"
    assert f0.dtype == np.float64, "image must be float64"
    assert 0.0 <= f0.min() and f0.max() <= 1.0 + 1e-9, "image must lie in [0, 1]"
    print("  ok")

    # --- frequency grid ----------------------------------------------
    wX, wY, w_radius = make_centered_grid(IMAGE_SIZE, IMAGE_SIZE)
    cy, cx = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    print("\nFrequency grid:")
    print(f"  shape   = {w_radius.shape}")
    print(f"  DC index (center) = ({cy}, {cx}) -> "
          f"wX={wX[cy, cx]}, wY={wY[cy, cx]}, |w|={w_radius[cy, cx]}")
    assert wX.shape == wY.shape == w_radius.shape == (IMAGE_SIZE, IMAGE_SIZE)
    # DC must sit at the center on the centered grid ...
    assert wX[cy, cx] == 0 and wY[cy, cx] == 0, "DC not centered"
    assert w_radius[cy, cx] == 0.0, "|omega| should be 0 at DC"
    # ... and move to index (0, 0) once shifted to FFT layout.
    assert to_fft_layout(w_radius)[0, 0] == 0.0, "to_fft_layout must put DC at (0,0)"
    # Sanity on the axis semantics: wX varies along columns, wY along rows.
    assert np.all(wX[0, :] == np.arange(-IMAGE_SIZE // 2, IMAGE_SIZE // 2))
    assert np.all(wY[:, 0] == np.arange(-IMAGE_SIZE // 2, IMAGE_SIZE // 2))
    print("  ok")

    # --- transfer functions ------------------------------------------
    K_gauss = gaussian_tf(wX, wY, GAUSS_SIGMA)
    K_motion = motion_tf(wX, wY, MOTION_L)
    print("\nTransfer functions:")
    print(f"  Gaussian: DC={K_gauss[0, 0].real:.3f}, real={np.allclose(K_gauss.imag, 0)}, "
          f"min={K_gauss.real.min():.2e}, peak-at-DC={np.abs(K_gauss).argmax() == 0}")
    print(f"  Motion:   DC={K_motion[0, 0].real:.3f}, min|K|={np.abs(K_motion).min():.2e}, "
          f"first zero index ~ N/L = {IMAGE_SIZE / MOTION_L:.1f}")
    # Gaussian: real, non-negative, low-pass with global peak exactly at DC.
    assert np.isclose(K_gauss[0, 0], 1.0), "Gaussian DC must be 1"
    assert np.allclose(K_gauss.imag, 0.0), "Gaussian TF must be real"
    assert K_gauss.real.min() >= 0.0, "Gaussian TF must be non-negative"
    assert np.abs(K_gauss).argmax() == 0, "Gaussian peak must be at DC (low-pass)"
    # Motion: unit gain at DC, real (zero-phase default), and has near-zeros.
    assert np.isclose(np.abs(K_motion[0, 0]), 1.0), "Motion DC gain must be 1"
    assert np.allclose(K_motion.imag, 0.0), "Default motion TF must be real (zero-phase)"
    assert np.abs(K_motion).min() < 0.05, "Motion TF must have near-zeros (lost frequencies)"
    # Dispatcher returns the same arrays.
    assert np.allclose(get_transfer_function("gaussian", wX, wY), K_gauss)
    assert np.allclose(get_transfer_function("motion", wX, wY), K_motion)
    print("  ok")

    # --- metrics -----------------------------------------------------
    rng = np.random.default_rng(SEED)
    # Inject noise at a known target SNR (same formula add_noise will use) and
    # check that measure_snr_db recovers it.
    target = 30.0
    sigma = 10.0 ** (-target / 20.0) * np.std(f0)
    noisy = f0 + rng.normal(0.0, sigma, f0.shape)
    measured = measure_snr_db(f0, noisy)
    print("\nMetrics:")
    print(f"  identical: MSE={mse(f0, f0):.1e}, PSNR={psnr(f0, f0)}, "
          f"relL2={relative_l2_error(f0, f0):.1e}")
    print(f"  noisy@{target:.0f}dB: measured SNR={measured:.2f} dB, "
          f"PSNR={psnr(f0, noisy):.2f} dB, relL2={relative_l2_error(noisy, f0):.3f}")
    assert mse(f0, f0) == 0.0 and psnr(f0, f0) == float("inf")
    assert relative_l2_error(f0, f0) == 0.0
    assert rmse(f0, f0) == 0.0
    assert abs(measured - target) < 1.0, "measured SNR must match the injected target"
    assert psnr(f0, noisy) < float("inf"), "PSNR of a noisy image must be finite"
    print("  ok")

    # --- forward problem ---------------------------------------------
    sim = simulate_forward(f0, K_gauss, SNR_HIGH_DB, np.random.default_rng(SEED))
    g_clean, g_noisy, noise, sigma = (
        sim["g_clean"], sim["g_noisy"], sim["noise"], sim["sigma"])
    print("\nForward problem (Gaussian, 40 dB):")
    print(f"  g_clean range = [{g_clean.min():.3f}, {g_clean.max():.3f}]")
    print(f"  sigma (theory) = {sigma:.4f}, std(noise) = {noise.std():.4f}")
    print(f"  measured SNR   = {measure_snr_db(g_clean, g_noisy):.2f} dB")
    assert g_clean.shape == f0.shape and np.isrealobj(g_clean)
    assert np.allclose(g_noisy, g_clean + noise), "g_noisy must equal g_clean + noise"
    assert abs(measure_snr_db(g_clean, g_noisy) - SNR_HIGH_DB) < 1.0, "forward SNR off target"
    assert abs(noise.std() - sigma) / sigma < 0.05, "realized noise std must match sigma"
    # Determinism: same seed -> identical observation.
    sim2 = simulate_forward(f0, K_gauss, SNR_HIGH_DB, np.random.default_rng(SEED))
    assert np.allclose(sim2["g_noisy"], g_noisy), "forward must be deterministic for fixed seed"
    print("  ok")

    # --- reconstruction ----------------------------------------------
    P_L2 = penalty_squared("L2", wX, wY)
    P_H1 = penalty_squared("H1", wX, wY)
    P_H2 = penalty_squared("H2", wX, wY)
    assert np.allclose(P_L2, 1.0)
    # H1/H2 must not penalize DC (index (0,0) in FFT layout).
    assert P_H1[0, 0] == 0.0 and P_H2[0, 0] == 0.0, "H1/H2 must vanish at DC"

    # Reconstruct a hard case: motion blur at 20 dB.
    simM = simulate_forward(f0, K_motion, SNR_LOW_DB, np.random.default_rng(SEED))
    g = simM["g_noisy"]
    rec_tik = tikhonov_filter(g, K_motion, P_H1, mu=1e-2)
    rec_naive = naive_inverse(g, K_motion)
    print("\nReconstruction (Motion, 20 dB):")
    print(f"  observed (blurred+noisy) PSNR = {psnr(f0, g):.2f} dB")
    print(f"  Tikhonov H1 (mu=1e-2)    PSNR = {psnr(f0, rec_tik):.2f} dB")
    print(f"  naive inverse            PSNR = {psnr(f0, rec_naive):.2f} dB")
    assert np.isrealobj(rec_tik) and rec_tik.shape == f0.shape
    # Checklist: Tikhonov must beat the (unstable) naive inverse.
    assert psnr(f0, rec_tik) > psnr(f0, rec_naive), "Tikhonov must beat naive inverse"
    # Window function is a real low-pass in [0, 1].
    W = window_function(K_motion, P_H1, mu=1e-2)
    assert np.allclose(W.imag if np.iscomplexobj(W) else 0.0, 0.0)
    assert W.min() >= 0.0 and W.max() <= 1.0 + 1e-9, "window must lie in [0, 1]"
    # Checklist: mu -> infinity drives the reconstruction to ~0.
    rec_big = tikhonov_filter(g, K_motion, P_L2, mu=1e12)
    assert np.linalg.norm(rec_big) < 1e-3 * np.linalg.norm(f0), "mu->inf must give ~0 image"
    # Hard spectral window runs and returns a real image.
    rec_win = spectral_window_reconstruction(g, K_motion, to_fft_layout(w_radius), Omega=10.0)
    assert np.isrealobj(rec_win) and rec_win.shape == f0.shape
    print("  ok")

    # --- error analysis (bias-variance) ---------------------------------------
    curves = bias_variance_curves(f0, K_motion, simM["noise"], P_H1, MU_GRID)
    bias, var, total = curves["bias"], curves["variance"], curves["total"]
    mu_opt = optimal_mu(MU_GRID, total)
    i_opt = int(np.argmin(total))
    print("\nBias-Variance (Motion, 20 dB, H1):")
    print(f"  mu_opt = {mu_opt:.2e}  (grid index {i_opt}/{len(MU_GRID) - 1})")
    print(f"  min total error = {total.min():.4f}")
    # Monotonicity: bias up, variance down (Part 4 interpretation).
    assert np.all(np.diff(bias) >= -1e-9), "bias must be non-decreasing in mu"
    assert np.all(np.diff(var) <= 1e-9), "variance must be non-increasing in mu"
    # The total error must have an interior minimum (the trade-off).
    assert 0 < i_opt < len(MU_GRID) - 1, "total-error minimum should be interior to the grid"
    # Fourier vs spatial cross-check at a few mu values (validates Parseval).
    for i in (5, i_opt, len(MU_GRID) - 5):
        b2, v2 = bias_variance_spatial(f0, K_motion, simM["noise"], P_H1, MU_GRID[i])
        assert np.isclose(b2, bias[i], rtol=1e-8), (b2, bias[i])
        assert np.isclose(v2, var[i], rtol=1e-8), (v2, var[i])
    print("  ok")

    # --- plot helpers (headless) -------------------------------------
    import matplotlib
    matplotlib.use("Agg")  # no display needed for the smoke test
    import matplotlib.pyplot as plt

    fig_a, axes_a = show_images([f0, g, rec_tik], ["f0", "observed", "Tikhonov H1"])
    assert axes_a.size >= 3
    crop = zoom(f0, (100, 150, 100, 150))
    assert crop.shape == (50, 50)
    fig_b, ax_b = plot_bias_variance(MU_GRID, bias, var, total, mu_opt)
    assert ax_b is not None
    plt.close("all")
    print("  ok")
    print("\nAll smoke tests passed.")

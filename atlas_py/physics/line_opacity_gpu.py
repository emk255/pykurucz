"""MLX Metal backend for LINOP1 wing accumulation (Phase 2 pilot).

Bug-fixed re-implementation:
  - C1: Voigt profile selects out_mid for high-adamp not-far cases
        (Fortran _voigt_nb has DISJOINT regimes adamp>=0.2 vs adamp<0.2).
  - C2: Negative blue-wing indices clamped before gather to avoid garbage reads.
  - P1: Scatter-add on GPU via mx.array.at[idx].add(vals) (no np.add.at).
  - P2: No CPU readback per batch (xlines stays on GPU until end of LINOP1 call).
  - P3: Default batch size = 524288 (much larger; better GPU utilisation).
  - P5: Eager mx.eval(xlines_mx) per batch keeps memory bounded.
"""

from __future__ import annotations

import os

import numpy as np

_MLX_AVAILABLE = False
try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore[assignment,misc]


def mlx_available() -> bool:
    return _MLX_AVAILABLE


def _require_mlx() -> None:
    if not _MLX_AVAILABLE:
        raise RuntimeError(
            "ATLAS_LINOP1_BACKEND=mlx requires mlx (pip install mlx); not installed"
        )


def _voigt_profile_mlx(v, a, h0, h1, h2):
    """Vectorized REAL*4 Voigt profile matching ``_voigt_nb`` in line_opacity.py.

    Fortran has two disjoint regimes:
      - a >= 0.2: far formula (a>1.4 OR a+v>3.2) else mid expansion
      - a <  0.2: lorentz (v>10) else table
    """
    iv = (v * np.float32(200.0) + np.float32(1.5)).astype(mx.int32)
    iv = mx.clip(iv, 1, 2001)
    i = iv - 1
    h0v = h0[i]
    h1v = h1[i]
    h2v = h2[i]

    aa = a * a
    vv = v * v

    u = (aa + vv) * np.float32(1.4142)
    out_far = a * np.float32(0.79788) / u
    aau = aa / u
    vvu = vv / u
    uu = u * u
    out_far = (
        ((((aau - np.float32(10.0) * vvu) * aau * np.float32(3.0)
            + np.float32(15.0) * vvu * vvu)
           + np.float32(3.0) * vv - aa) / uu + np.float32(1.0)) * out_far
    )

    hh1 = h1v + h0v * np.float32(1.12838)
    hh2 = h2v + hh1 * np.float32(1.12838) - h0v
    hh3 = (np.float32(1.0) - h2v) * np.float32(0.37613) - hh1 * np.float32(0.66667) * vv + hh2 * np.float32(1.12838)
    hh4 = (np.float32(3.0) * hh3 - hh1) * np.float32(0.37613) + h0v * np.float32(0.66667) * vv * vv
    poly = ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0v)
    damp = (((np.float32(-0.122727278) * a + np.float32(0.532770573)) * a - np.float32(0.96284325)) * a + np.float32(0.979895032))
    out_mid = poly * damp

    out_lorentz = np.float32(0.5642) * a / vv
    out_table = (h2v * a + h1v) * a + h0v

    high = a >= np.float32(0.2)
    use_far = high & ((a > np.float32(1.4)) | (a + v > np.float32(3.2)))
    use_mid = high & ~use_far
    use_lorentz = (~high) & (v > np.float32(10.0))

    out = mx.where(
        use_far,
        out_far,
        mx.where(
            use_mid,
            out_mid,
            mx.where(use_lorentz, out_lorentz, out_table),
        ),
    )
    return out.astype(mx.float32)


def _wing_cv_mlx(vvoigt, center, adamp, h0, h1, h2):
    """CV for one side of a wing; mirrors ``_accwings_nb`` REAL*4 semantics.

    Fortran fast path for adamp <= 0.2:
      vvoigt > 10: cv = center * 0.5642 * adamp / vvoigt^2
      else:        cv = center * ((h2*a + h1)*a + h0)
    Slow path (adamp > 0.2):
      cv = center * _voigt_nb(...)
    """
    f32_5642 = np.float32(0.5642)
    f10 = np.float32(10.0)
    f02 = np.float32(0.2)
    low_adamp = adamp <= f02

    iv = (vvoigt * np.float32(200.0) + np.float32(1.5)).astype(mx.int32)
    iv = mx.clip(iv, 1, 2001)
    ii = iv - 1
    cv_table = center * ((h2[ii] * adamp + h1[ii]) * adamp + h0[ii])
    cv_lorentz = center * f32_5642 * adamp / (vvoigt * vvoigt)
    cv_low = mx.where(vvoigt > f10, cv_lorentz, cv_table)

    voigt = _voigt_profile_mlx(vvoigt, adamp, h0, h1, h2)
    cv_high = center * voigt.astype(mx.float32)

    return mx.where(low_adamp, cv_low.astype(mx.float32), cv_high)


def _wing_break_mask(cv, tabcont, valid):
    """Add-then-break mask matching Fortran ``xlines += cv; if cv<tabcont: break``.

    Includes the FIRST column where cv<tabcont (already-added), excludes the rest.
    Cumprod of (cv >= tabcont) over previous columns; True where prefix product is
    still 1 AND the current column is valid.
    """
    ge = (cv >= tabcont[:, None]).astype(mx.float32)
    ones_col = mx.ones((cv.shape[0], 1), dtype=mx.float32)
    shifted = mx.concatenate([ones_col, ge[:, :-1]], axis=1)
    prefix = mx.cumprod(shifted, axis=1) > np.float32(0.0)
    return valid & prefix


def apply_wings_mlx(
    *,
    wing_j0: np.ndarray,
    wing_nu0: np.ndarray,
    wing_wlvac: np.ndarray,
    wing_center: np.ndarray,
    wing_adamp: np.ndarray,
    wing_dopwave: np.ndarray,
    wing_tabcont: np.ndarray,
    xlines: np.ndarray,
    waveset: np.ndarray,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
    batch_size: int | None = None,
    xlines_mx_in=None,
):
    """Apply recorded wing ops on MLX Metal; returns (xlines_np, xlines_mx).

    If xlines_mx_in is provided, it is the persistent GPU-resident accumulator;
    we keep accumulating into it and return both the final numpy view (None for
    perf if caller doesn't need it yet) and the GPU array for chained calls.
    """
    _require_mlx()

    n_wings = int(wing_j0.shape[0])
    numnu = int(waveset.shape[0])
    nrhox = int(xlines.shape[0])

    if batch_size is None:
        batch_size = int(os.environ.get("ATLAS_LINOP1_MLX_BATCH", "524288"))

    if xlines_mx_in is None:
        xlines_mx = mx.array(np.asarray(xlines, dtype=np.float32, order="C").reshape(-1))
    else:
        xlines_mx = xlines_mx_in

    if n_wings == 0:
        return xlines, xlines_mx

    waveset_mx = mx.array(np.asarray(waveset, dtype=np.float32))
    h0 = mx.array(np.asarray(h0tab, dtype=np.float32))
    h1 = mx.array(np.asarray(h1tab, dtype=np.float32))
    h2 = mx.array(np.asarray(h2tab, dtype=np.float32))

    red_off = mx.arange(101, dtype=mx.int32)
    blue_off = mx.arange(1, 101, dtype=mx.int32)
    zero_f32 = mx.array(np.float32(0.0))

    for start in range(0, n_wings, batch_size):
        end = min(start + batch_size, n_wings)
        sl = slice(start, end)

        j0 = mx.array(np.ascontiguousarray(wing_j0[sl], dtype=np.int32))
        nu0 = mx.array(np.ascontiguousarray(wing_nu0[sl], dtype=np.int32))
        wlvac = mx.array(np.ascontiguousarray(wing_wlvac[sl], dtype=np.float32))
        center = mx.array(np.ascontiguousarray(wing_center[sl], dtype=np.float32))
        adamp = mx.array(np.ascontiguousarray(wing_adamp[sl], dtype=np.float32))
        dopwave = mx.array(np.ascontiguousarray(wing_dopwave[sl], dtype=np.float32))
        tabcont = mx.array(np.ascontiguousarray(wing_tabcont[sl], dtype=np.float32))

        active = dopwave > np.float32(0.0)

        # Red wing: iw = nu0 + offset (0..100), capped to numnu-1 for safe gather.
        iw_red_raw = nu0[:, None] + red_off[None, :]
        valid_red = (iw_red_raw < numnu) & active[:, None]
        iw_red = mx.minimum(iw_red_raw, numnu - 1)
        w_red = waveset_mx[iw_red]
        v_red = (w_red - wlvac[:, None]) / dopwave[:, None]
        cv_red = _wing_cv_mlx(v_red, center[:, None], adamp[:, None], h0, h1, h2)
        cv_red = mx.where(valid_red, cv_red, zero_f32)
        mask_red = _wing_break_mask(cv_red, tabcont, valid_red)
        cv_red = mx.where(mask_red, cv_red, zero_f32)

        # Blue wing: iw = nu0 - offset (1..100), clamped to 0 for safe gather.
        iw_blue_raw = nu0[:, None] - blue_off[None, :]
        valid_blue = (iw_blue_raw >= 0) & active[:, None]
        iw_blue = mx.maximum(iw_blue_raw, 0)
        w_blue = waveset_mx[iw_blue]
        v_blue = (wlvac[:, None] - w_blue) / dopwave[:, None]
        cv_blue = _wing_cv_mlx(v_blue, center[:, None], adamp[:, None], h0, h1, h2)
        cv_blue = mx.where(valid_blue, cv_blue, zero_f32)
        mask_blue = _wing_break_mask(cv_blue, tabcont, valid_blue)
        cv_blue = mx.where(mask_blue, cv_blue, zero_f32)

        flat_red = (j0[:, None] * numnu + iw_red).reshape(-1)
        flat_blue = (j0[:, None] * numnu + iw_blue).reshape(-1)
        flat_idx = mx.concatenate([flat_red, flat_blue])
        vals = mx.concatenate([cv_red.reshape(-1), cv_blue.reshape(-1)])

        # GPU scatter-add — duplicate indices summed by MLX.
        xlines_mx = xlines_mx.at[flat_idx].add(vals)
        mx.eval(xlines_mx)

    return xlines, xlines_mx


def finalize_xlines(xlines_mx, nrhox: int, numnu: int) -> np.ndarray:
    """Materialise the GPU-resident xlines accumulator to numpy (host)."""
    return np.array(xlines_mx).reshape(nrhox, numnu)

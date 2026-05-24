#!/usr/bin/env python3
"""End-to-end parity + benchmark: Numba prange vs Python fallback.

Runs the full SYNTHE pipeline (opacity compilation + radiative transfer)
on a real atmosphere model and compares Numba vs Python-loop results.
"""
import os
import sys
import time
import tempfile
import numpy as np

# Use the local pykurucz
sys.path.insert(0, os.path.dirname(__file__))

# --- Configuration ---
ATM_FILE = "/Users/ElliotKim/Desktop/Research/all_kurucz/samples/at12_feh-1.50_afe+0.2_t05500g5.00.atm"
ATOMIC_LINES = "/Users/ElliotKim/Desktop/Research/all_kurucz/pykurucz/lines/gfallvac.latest"

# Small range for parity test
WL_START_PARITY = 500.0   # nm
WL_END_PARITY = 502.0     # nm
RESOLUTION = 300_000.0


def run_synthe(wl_start, wl_end, use_numba=True, label=""):
    """Run the SYNTHE pipeline and return (wavelengths, intensity, continuum, elapsed)."""
    if use_numba:
        os.environ.pop("PY_RT_NO_NUMBA", None)
    else:
        os.environ["PY_RT_NO_NUMBA"] = "1"

    from synthe_py.config import SynthesisConfig
    from synthe_py.engine.opacity import run_synthesis
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".spec", delete=False) as f:
        spec_path = Path(f.name)

    try:
        cfg = SynthesisConfig.from_cli(
            spec_path=spec_path,
            diagnostics_path=None,
            atmosphere_path=Path(ATM_FILE),
            atomic_catalog=Path(ATOMIC_LINES),
            wl_start=wl_start,
            wl_end=wl_end,
            resolution=RESOLUTION,
            vacuum=True,
            cutoff=1e-3,
            linout=30,
            nlte=False,
            n_workers=1,
            npz_path=Path("/tmp/test_atm.npz"),
        )

        t0 = time.perf_counter()
        result = run_synthesis(cfg)
        elapsed = time.perf_counter() - t0

        print(f"\n{'='*60}")
        print(f"[{label}] wl={wl_start}-{wl_end} nm, numba={use_numba}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Wavelengths: {result.wavelength.shape[0]:,}")
        print(f"  Intensity range: [{result.intensity.min():.6e}, {result.intensity.max():.6e}]")
        print(f"  Continuum range: [{result.continuum.min():.6e}, {result.continuum.max():.6e}]")
        print(f"{'='*60}")

        return result.wavelength, result.intensity, result.continuum, elapsed
    finally:
        try:
            os.unlink(spec_path)
        except OSError:
            pass


if __name__ == "__main__":
    print("=" * 70)
    print("END-TO-END NUMBA PARITY + BENCHMARK TEST")
    print("=" * 70)
    print(f"Atmosphere: {ATM_FILE}")
    print(f"Line list:  {ATOMIC_LINES}")

    # --- Step 1: Parity test (small range) ---
    print(f"\n\n>>> STEP 1: PARITY TEST ({WL_START_PARITY}–{WL_END_PARITY} nm)")

    wl_nb, int_nb, cont_nb, t_nb = run_synthe(
        WL_START_PARITY, WL_END_PARITY, use_numba=True, label="Numba"
    )
    wl_py, int_py, cont_py, t_py = run_synthe(
        WL_START_PARITY, WL_END_PARITY, use_numba=False, label="Python"
    )

    # Compare
    assert np.array_equal(wl_nb, wl_py), "Wavelength grids differ!"

    mask_int = int_py != 0
    if mask_int.any():
        rel_err_int = np.abs(int_nb[mask_int] - int_py[mask_int]) / np.abs(int_py[mask_int])
        max_rel_int = rel_err_int.max()
        mean_rel_int = rel_err_int.mean()
    else:
        max_rel_int = mean_rel_int = 0.0

    mask_cont = cont_py != 0
    if mask_cont.any():
        rel_err_cont = np.abs(cont_nb[mask_cont] - cont_py[mask_cont]) / np.abs(cont_py[mask_cont])
        max_rel_cont = rel_err_cont.max()
        mean_rel_cont = rel_err_cont.mean()
    else:
        max_rel_cont = mean_rel_cont = 0.0

    print(f"\n{'='*60}")
    print("PARITY RESULTS:")
    print(f"  intensity — max rel error: {max_rel_int:.3e}, mean: {mean_rel_int:.3e}")
    print(f"  continuum — max rel error: {max_rel_cont:.3e}, mean: {mean_rel_cont:.3e}")
    PARITY_THRESH = 1e-10
    if max_rel_int < PARITY_THRESH and max_rel_cont < PARITY_THRESH:
        print(f"  ✓ PASS (threshold: {PARITY_THRESH})")
    else:
        print(f"  ✗ FAIL (threshold: {PARITY_THRESH})")
    print(f"  Numba: {t_nb:.2f}s  (includes Numba JIT compilation on first call)")
    print(f"  Python: {t_py:.2f}s")
    speedup = t_py / max(t_nb, 0.01)
    print(f"  RT speedup: {speedup:.1f}x")
    print(f"{'='*60}")

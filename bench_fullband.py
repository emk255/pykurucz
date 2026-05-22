#!/usr/bin/env python3
"""Full-band benchmark: 300-1800nm cold-start comparison.

Usage:
    python bench_fullband.py --run         # Run synthesis and save reference
    python bench_fullband.py --compare     # Compare against saved reference
"""
import logging
logging.basicConfig(level=logging.WARNING)

import sys
import os
import time
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

ATM_FILE = "/Users/ElliotKim/Desktop/Research/all_kurucz/samples/at12_feh-1.50_afe+0.2_t05500g5.00.atm"
ATOMIC_LINES = "/Users/ElliotKim/Desktop/Research/all_kurucz/pykurucz/lines/gfallvac.latest"
NPZ_PATH = "/tmp/test_atm.npz"
REF_FILE = "/tmp/fullband_optimized_ref.npz"

WL_START = 300.0
WL_END = 1800.0
RESOLUTION = 300_000.0


def run_pipeline():
    from synthe_py.config import SynthesisConfig
    from synthe_py.engine.opacity import run_synthesis
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".spec", delete=False) as f:
        spec_path = Path(f.name)

    cfg = SynthesisConfig.from_cli(
        spec_path=spec_path,
        diagnostics_path=None,
        atmosphere_path=Path(ATM_FILE),
        atomic_catalog=Path(ATOMIC_LINES),
        wl_start=WL_START,
        wl_end=WL_END,
        resolution=RESOLUTION,
        vacuum=True,
        cutoff=1e-3,
        linout=30,
        nlte=False,
        n_workers=1,
        npz_path=Path(NPZ_PATH),
    )

    t0 = time.perf_counter()
    result = run_synthesis(cfg)
    elapsed = time.perf_counter() - t0
    os.unlink(spec_path)
    return result, elapsed


def main():
    if "--run" in sys.argv:
        print(f"Running 300-1800nm synthesis...")
        result, elapsed = run_pipeline()
        print(f"Time: {elapsed:.2f}s")
        print(f"Wavelengths: {result.wavelength.shape[0]:,}")
        print(f"Intensity: [{result.intensity.min():.6e}, {result.intensity.max():.6e}]")
        print(f"Continuum: [{result.continuum.min():.6e}, {result.continuum.max():.6e}]")
        np.savez(REF_FILE,
            wavelength=result.wavelength,
            intensity=result.intensity,
            continuum=result.continuum)
        print(f"Saved to {REF_FILE}")

    elif "--compare" in sys.argv:
        if not os.path.exists(REF_FILE):
            print(f"ERROR: {REF_FILE} not found. Run with --run first.")
            sys.exit(1)

        ref = np.load(REF_FILE)
        print(f"Running 300-1800nm synthesis for comparison...")
        result, elapsed = run_pipeline()

        assert np.array_equal(result.wavelength, ref["wavelength"]), "Wavelength grids differ!"

        mask_int = ref["intensity"] != 0
        if mask_int.any():
            rel_err_int = np.abs(result.intensity[mask_int] - ref["intensity"][mask_int]) / np.abs(ref["intensity"][mask_int])
            max_rel_int = rel_err_int.max()
            mean_rel_int = rel_err_int.mean()
        else:
            max_rel_int = mean_rel_int = 0.0

        mask_cont = ref["continuum"] != 0
        if mask_cont.any():
            rel_err_cont = np.abs(result.continuum[mask_cont] - ref["continuum"][mask_cont]) / np.abs(ref["continuum"][mask_cont])
            max_rel_cont = rel_err_cont.max()
            mean_rel_cont = rel_err_cont.mean()
        else:
            max_rel_cont = mean_rel_cont = 0.0

        print(f"\nTime: {elapsed:.2f}s")
        print(f"Max rel error intensity: {max_rel_int:.3e}")
        print(f"Max rel error continuum: {max_rel_cont:.3e}")
        THRESH = 1e-13
        if max_rel_int < THRESH and max_rel_cont < THRESH:
            print(f"✓ PASS (threshold: {THRESH})")
        else:
            print(f"✗ FAIL (threshold: {THRESH})")
    else:
        print("Usage: bench_fullband.py [--run | --compare]")
        sys.exit(1)


if __name__ == "__main__":
    main()

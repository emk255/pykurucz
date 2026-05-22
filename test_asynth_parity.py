#!/usr/bin/env python3
"""Save ASYNTH reference output from current wing kernel, then verify parity after changes."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

ATM_FILE = "/Users/ElliotKim/Desktop/Research/all_kurucz/samples/at12_feh-1.50_afe+0.2_t05500g5.00.atm"
ATOMIC_LINES = "/Users/ElliotKim/Desktop/Research/all_kurucz/pykurucz/lines/gfallvac.latest"
NPZ_PATH = "/tmp/test_atm.npz"
REF_FILE = "/tmp/asynth_reference.npz"

WL_START = 500.0
WL_END = 510.0
RESOLUTION = 300_000.0


def run_pipeline():
    """Run SYNTHE pipeline and return (asynth, wavelength, timings)."""
    from synthe_py.config import SynthesisConfig
    from synthe_py.engine.opacity import run_synthesis
    from pathlib import Path
    import tempfile

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

    result = run_synthesis(cfg)
    os.unlink(spec_path)
    return result


def save_reference():
    """Run pipeline and save reference spectrum."""
    print("Running pipeline to save reference...")
    result = run_pipeline()
    np.savez(
        REF_FILE,
        wavelength=result.wavelength,
        intensity=result.intensity,
        continuum=result.continuum,
    )
    print(f"Reference saved to {REF_FILE}")
    print(f"  Wavelengths: {result.wavelength.shape[0]:,}")
    print(f"  Intensity range: [{result.intensity.min():.6e}, {result.intensity.max():.6e}]")
    return result


def verify_parity():
    """Run pipeline and compare against saved reference."""
    if not os.path.exists(REF_FILE):
        print(f"ERROR: Reference file {REF_FILE} not found. Run with --save first.")
        return False

    ref = np.load(REF_FILE)
    print("Running pipeline to verify parity...")
    result = run_pipeline()

    assert np.array_equal(result.wavelength, ref["wavelength"]), "Wavelength grids differ!"

    # Compare intensity (flux_total)
    mask_int = ref["intensity"] != 0
    if mask_int.any():
        rel_err_int = np.abs(result.intensity[mask_int] - ref["intensity"][mask_int]) / np.abs(
            ref["intensity"][mask_int]
        )
        max_rel_int = rel_err_int.max()
        mean_rel_int = rel_err_int.mean()
    else:
        max_rel_int = mean_rel_int = 0.0

    # Compare continuum (flux_cont)
    mask_cont = ref["continuum"] != 0
    if mask_cont.any():
        rel_err_cont = np.abs(result.continuum[mask_cont] - ref["continuum"][mask_cont]) / np.abs(
            ref["continuum"][mask_cont]
        )
        max_rel_cont = rel_err_cont.max()
        mean_rel_cont = rel_err_cont.mean()
    else:
        max_rel_cont = mean_rel_cont = 0.0

    # Max absolute difference
    max_abs_int = np.max(np.abs(result.intensity - ref["intensity"]))
    max_abs_cont = np.max(np.abs(result.continuum - ref["continuum"]))

    print(f"\n{'='*60}")
    print("PARITY RESULTS:")
    print(f"  intensity — max rel: {max_rel_int:.3e}, mean rel: {mean_rel_int:.3e}, max abs: {max_abs_int:.3e}")
    print(f"  continuum — max rel: {max_rel_cont:.3e}, mean rel: {mean_rel_cont:.3e}, max abs: {max_abs_cont:.3e}")

    THRESH = 1e-13
    passed = max_rel_int < THRESH and max_rel_cont < THRESH
    if passed:
        print(f"  ✓ PASS (threshold: {THRESH})")
    else:
        print(f"  ✗ FAIL (threshold: {THRESH})")
    print(f"{'='*60}")
    return passed


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    if "--save" in sys.argv:
        save_reference()
    elif "--verify" in sys.argv:
        ok = verify_parity()
        sys.exit(0 if ok else 1)
    else:
        print("Usage: test_asynth_parity.py [--save | --verify]")
        print("  --save    Run pipeline and save reference output")
        print("  --verify  Run pipeline and compare against saved reference")
        sys.exit(1)

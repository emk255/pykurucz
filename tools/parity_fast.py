#!/usr/bin/env python3
"""SYNTHE-only parity check against the 6 cached Fortran spectra in fortran_specs/.

Runs synthesize_from_atm.py for each of the 6 sample atmospheres (using the
pre-iterated .atm files in samples/), then compares against the cached Fortran
spectrum in fortran_specs/ using normalized-flux error.

Parity threshold: max|F/C_py(λ) − F/C_ft(λ)| < 0.10 for all λ in 300–1800 nm.

Usage:
    python tools/parity_fast.py [--out-dir <dir>] [--threshold 0.10]

Returns exit code 0 if all 6 cases pass, 1 if any fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"
FORTRAN_SPECS_DIR = REPO_ROOT / "fortran_specs"

FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")

# When running against a different repo (e.g. ting_pykurucz), pass
# --synth-root to point at that repo's synthesize_from_atm.py.
# --samples-dir and --fortran-specs-dir override the reference data paths.
SAMPLE_FILES = [
    "at12_aaaaa_t04500g2.00.atm",
    "at12_aaaaa_t05000g3.00.atm",
    "at12_aaaaa_t05500g4.00.atm",
    "at12_aaaaa_t05770g4.44.atm",
    "at12_aaaaa_t06000g4.50.atm",
    "at12_aaaaa_t08250g4.00.atm",
]


def _load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelengths, fluxes, continua = [], [], []
    with open(path) as f:
        for line in f:
            parts = FLOAT_RE.findall(line)
            if len(parts) >= 3:
                try:
                    wavelengths.append(float(parts[0]))
                    fluxes.append(float(parts[1]))
                    continua.append(float(parts[2]))
                except ValueError:
                    continue
    return np.array(wavelengths), np.array(fluxes), np.array(continua)


def _max_norm_abs_error(
    py_wl: np.ndarray,
    py_flux: np.ndarray,
    py_cont: np.ndarray,
    ft_wl: np.ndarray,
    ft_flux: np.ndarray,
    ft_cont: np.ndarray,
) -> float:
    """Return max |F/C_py - F/C_ft| over the overlapping wavelength range."""
    wl_min = max(py_wl.min(), ft_wl.min())
    wl_max = min(py_wl.max(), ft_wl.max())

    py_mask = (py_wl >= wl_min) & (py_wl <= wl_max)
    ft_mask = (ft_wl >= wl_min) & (ft_wl <= wl_max)

    py_wl_c = py_wl[py_mask]
    py_flux_c = py_flux[py_mask]
    py_cont_c = py_cont[py_mask]
    ft_wl_c = ft_wl[ft_mask]
    ft_flux_c = ft_flux[ft_mask]
    ft_cont_c = ft_cont[ft_mask]

    ft_flux_i = np.interp(py_wl_c, ft_wl_c, ft_flux_c)
    ft_cont_i = np.interp(py_wl_c, ft_wl_c, ft_cont_c)

    py_norm = py_flux_c / np.maximum(py_cont_c, 1e-30)
    ft_norm = ft_flux_i / np.maximum(ft_cont_i, 1e-30)
    return float(np.max(np.abs(py_norm - ft_norm)))


def run_one_case(
    atm_file: Path,
    *,
    out_dir: Path,
    threshold: float,
    wl_start: float = 300.0,
    wl_end: float = 1800.0,
    fortran_specs_dir: Path = FORTRAN_SPECS_DIR,
    synth_root: Path = REPO_ROOT,
) -> dict:
    stem = atm_file.stem
    wl_tag = f"{int(wl_start)}_{int(wl_end)}"
    spec_out = out_dir / f"{stem}_{wl_tag}.spec"
    out_dir.mkdir(parents=True, exist_ok=True)

    fortran_spec = fortran_specs_dir / f"{stem}.spec"
    if not fortran_spec.exists():
        return {
            "case": stem,
            "status": "ERROR",
            "error": f"{fortran_specs_dir}/{stem}.spec not found",
            "max_norm_abs": None,
            "elapsed_s": 0.0,
        }

    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(synth_root / "synthesize_from_atm.py"),
        str(atm_file),
        "--wl-start", str(wl_start),
        "--wl-end", str(wl_end),
        "--output-dir", str(out_dir.parent),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(synth_root))
    elapsed = round(time.perf_counter() - t0, 1)

    if result.returncode != 0:
        return {
            "case": stem,
            "status": "ERROR",
            "error": (result.stderr or result.stdout)[:2000],
            "max_norm_abs": None,
            "elapsed_s": elapsed,
        }

    if not spec_out.exists():
        return {
            "case": stem,
            "status": "ERROR",
            "error": f"Expected output not found: {spec_out}",
            "max_norm_abs": None,
            "elapsed_s": elapsed,
        }

    py_wl, py_flux, py_cont = _load_spectrum(spec_out)
    ft_wl, ft_flux, ft_cont = _load_spectrum(fortran_spec)

    max_err = _max_norm_abs_error(py_wl, py_flux, py_cont, ft_wl, ft_flux, ft_cont)
    status = "PASS" if max_err < threshold else "FAIL"

    return {
        "case": stem,
        "status": status,
        "max_norm_abs": max_err,
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast SYNTHE-only parity check (6 samples)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Dir for synthesized spectra (default: <synth-root>/results/parity_fast/)")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Max allowed normalized-flux error (default: 0.10)")
    parser.add_argument("--wl-start", type=float, default=300.0)
    parser.add_argument("--wl-end", type=float, default=1800.0)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write results to this JSON file")
    parser.add_argument("--samples-dir", type=Path, default=None,
                        help="Override path to directory containing *.atm sample files")
    parser.add_argument("--fortran-specs-dir", type=Path, default=None,
                        help="Override path to directory containing cached Fortran *.spec files")
    parser.add_argument("--synth-root", type=Path, default=None,
                        help="Override repo root for synthesize_from_atm.py (default: parent of this script)")
    parser.add_argument("--label", type=str, default=None,
                        help="Label to include in JSON output (e.g. 'C0', 'T1')")
    parser.add_argument("--sample-files", nargs="+", default=None,
                        help="Subset of sample filenames to run (default: all 6). "
                             "E.g. --sample-files at12_aaaaa_t04500g2.00.atm at12_aaaaa_t05770g4.44.atm at12_aaaaa_t08250g4.00.atm")
    args = parser.parse_args()

    synth_root = (args.synth_root or REPO_ROOT).resolve()
    samples_dir = (args.samples_dir or SAMPLES_DIR).resolve()
    fortran_specs_dir = (args.fortran_specs_dir or FORTRAN_SPECS_DIR).resolve()
    out_dir = args.out_dir or (synth_root / "results" / "parity_fast")
    sample_files = args.sample_files or SAMPLE_FILES

    print("=" * 72)
    print(f"PARITY_FAST — {len(sample_files)} atmospheres, WL {args.wl_start}–{args.wl_end} nm, threshold={args.threshold}")
    print(f"  synth root: {synth_root}")
    print(f"  samples/: {samples_dir}")
    print(f"  fortran_specs/: {fortran_specs_dir}")
    print(f"  output: {out_dir}")
    print("=" * 72)

    results = []
    n_pass = n_fail = n_err = 0
    total_t0 = time.perf_counter()

    for fname in sample_files:
        atm_file = samples_dir / fname
        if not atm_file.exists():
            r = {"case": fname, "status": "ERROR",
                 "error": f"{samples_dir}/{fname} not found", "max_norm_abs": None, "elapsed_s": 0.0}
            results.append(r)
            n_err += 1
            print(f"  ERROR  {fname}: not found")
            continue

        print(f"\n  Running {fname} ...", flush=True)
        r = run_one_case(
            atm_file,
            out_dir=out_dir / "spec",
            threshold=args.threshold,
            wl_start=args.wl_start,
            wl_end=args.wl_end,
            fortran_specs_dir=fortran_specs_dir,
            synth_root=synth_root,
        )
        results.append(r)

        icon = "PASS" if r["status"] == "PASS" else r["status"]
        err_str = f"{r['max_norm_abs']:.4e}" if r["max_norm_abs"] is not None else "—"
        print(f"  {icon:5s}  max_err={err_str}  {r['elapsed_s']}s")

        if r["status"] == "PASS":
            n_pass += 1
        elif r["status"] == "FAIL":
            n_fail += 1
        else:
            n_err += 1

    total_elapsed = round(time.perf_counter() - total_t0, 1)
    overall = "PASS" if n_fail == 0 and n_err == 0 else "FAIL"

    summary = {
        "label": args.label,
        "synth_root": str(synth_root),
        "sample_files": list(sample_files),
        "threshold": args.threshold,
        "wl_start": args.wl_start,
        "wl_end": args.wl_end,
        "num_pass": n_pass,
        "num_fail": n_fail,
        "num_error": n_err,
        "overall_status": overall,
        "total_elapsed_s": total_elapsed,
        "cases": results,
    }

    print(f"\n{'='*72}")
    print(f"RESULT: {overall}  ({n_pass} pass, {n_fail} fail, {n_err} error)  total={total_elapsed}s")
    print("=" * 72)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {args.json_out}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

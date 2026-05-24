#!/usr/bin/env python3
"""5-case end-to-end parity check (ATLAS + SYNTHE, Python vs cached Fortran).

For each of the 5 curated cases in results/<case>/, this script:
  1. Re-runs Python ATLAS (atlas_py.cli) from the stored emulator warm-start .atm,
     using the cached Fortran fort12.bin for line selection.
  2. Runs Python SYNTHE (synthe_py.cli) on the resulting atmosphere.
  3. Compares normalized flux against the cached fortran_synthe_300_1800.spec.

Parity threshold: max|F/C_py(λ) − F/C_ft(λ)| < 0.10 for all λ in 300–1800 nm.

Usage:
    python tools/parity_e2e.py [--threshold 0.10] [--json-out FILE]

Returns exit code 0 if all 5 cases pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")

E2E_CASES = [
    "t04500_g+2.50_mh-1.00",
    "t05500_g+4.45_mh+0.00",
    "t05770_g+4.44_mh-1.00",
    "t06200_g+4.20_mh+0.30",
    "t08250_g+4.00_mh+0.00",
]

WL_START = 300.0
WL_END = 1800.0


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
    wl_min = max(py_wl.min(), ft_wl.min())
    wl_max = min(py_wl.max(), ft_wl.max())

    py_mask = (py_wl >= wl_min) & (py_wl <= wl_max)
    ft_mask = (ft_wl >= wl_min) & (ft_wl <= wl_max)

    py_wl_c, py_flux_c, py_cont_c = py_wl[py_mask], py_flux[py_mask], py_cont[py_mask]
    ft_wl_c, ft_flux_c, ft_cont_c = ft_wl[ft_mask], ft_flux[ft_mask], ft_cont[ft_mask]

    ft_flux_i = np.interp(py_wl_c, ft_wl_c, ft_flux_c)
    ft_cont_i = np.interp(py_wl_c, ft_wl_c, ft_cont_c)

    py_norm = py_flux_c / np.maximum(py_cont_c, 1e-30)
    ft_norm = ft_flux_i / np.maximum(ft_cont_i, 1e-30)
    return float(np.max(np.abs(py_norm - ft_norm)))


def run_one_case(
    case_name: str,
    *,
    threshold: float,
    n_workers: int | None = None,
    atlas_iterations: int = 30,
    convergence_epsilon: float | None = 1.0e-3,
    convergence_min_iterations: int = 5,
    convergence_consecutive: int = 1,
    use_fortran_fort12: bool = False,
) -> dict:
    case_dir = RESULTS_DIR / case_name
    fortran_dir = case_dir / "fortran"
    python_dir = case_dir / "python"
    logs_dir = case_dir / "logs"
    inputs_dir = case_dir / "inputs"

    wl_tag = f"{int(WL_START)}_{int(WL_END)}"
    fortran_spec = fortran_dir / f"fortran_synthe_{wl_tag}.spec"
    input_atm = inputs_dir / "emulator_warmstart.atm"
    fort12_bin = fortran_dir / "fortran_iter1_fort12.bin"

    for req in (fortran_spec, input_atm):
        if not req.exists():
            return {
                "case": case_name,
                "status": "ERROR",
                "error": f"Required file missing: {req}",
                "max_norm_abs": None,
                "elapsed_s": 0.0,
            }

    python_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = RESULTS_DIR / "atlas_fort12_cache"

    t0 = time.perf_counter()

    # Step 1: Python ATLAS
    from pykurucz import run_atlas_py, run_synthe_py

    python_atm = python_dir / "python_iter1.atm"
    atlas_log = logs_dir / "python_atlas_parity_e2e.log"
    try:
        run_atlas_py(
            input_atm=input_atm,
            output_atm=python_atm,
            log_path=atlas_log,
            iterations=atlas_iterations,
            convergence_epsilon=convergence_epsilon,
            convergence_min_iterations=convergence_min_iterations,
            convergence_consecutive=convergence_consecutive,
            fort12_bin=fort12_bin if (use_fortran_fort12 and fort12_bin.exists()) else None,
            n_workers=n_workers,
            cache_dir=cache_dir if not use_fortran_fort12 else None,
        )
    except Exception as exc:
        return {
            "case": case_name,
            "status": "ERROR",
            "error": f"atlas_py failed: {exc}",
            "max_norm_abs": None,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }

    # Step 2: Python SYNTHE
    python_spec = python_dir / f"python_synthe_{wl_tag}.spec"
    python_npz = python_dir / "python_iter1_synthe.npz"
    synthe_log = logs_dir / f"python_synthe_{wl_tag}_parity_e2e.log"
    try:
        run_synthe_py(
            atm=python_atm,
            spec=python_spec,
            npz=python_npz,
            log_path=synthe_log,
            wl_start=WL_START,
            wl_end=WL_END,
            n_workers=n_workers,
        )
    except Exception as exc:
        return {
            "case": case_name,
            "status": "ERROR",
            "error": f"synthe_py failed: {exc}",
            "max_norm_abs": None,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }

    elapsed = round(time.perf_counter() - t0, 1)

    if not python_spec.exists():
        return {
            "case": case_name,
            "status": "ERROR",
            "error": f"Python spec not produced: {python_spec}",
            "max_norm_abs": None,
            "elapsed_s": elapsed,
        }

    py_wl, py_flux, py_cont = _load_spectrum(python_spec)
    ft_wl, ft_flux, ft_cont = _load_spectrum(fortran_spec)

    max_err = _max_norm_abs_error(py_wl, py_flux, py_cont, ft_wl, ft_flux, ft_cont)
    status = "PASS" if max_err < threshold else "FAIL"

    return {
        "case": case_name,
        "status": status,
        "max_norm_abs": max_err,
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="5-case end-to-end parity check (ATLAS + SYNTHE, Python vs Fortran)"
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--atlas-iterations", type=int, default=30,
                        help="Max ATLAS iterations (default: 30, same as pykurucz.py)")
    parser.add_argument(
        "--atlas-convergence-epsilon",
        type=float,
        default=1.0e-3,
        help="Early-stop epsilon on physical columns (default: 1e-3, prod pykurucz.py)",
    )
    parser.add_argument(
        "--no-atlas-convergence",
        action="store_true",
        help="Disable early stopping; run exactly --atlas-iterations (debug only)",
    )
    parser.add_argument(
        "--use-fortran-fort12",
        action="store_true",
        help="Replay Fortran fort12 line selection (strict harness; NOT prod pykurucz.py)",
    )
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Override which cases to run (default: all 5)")
    args = parser.parse_args()

    conv_eps = None if args.no_atlas_convergence else args.atlas_convergence_epsilon

    cases = args.cases or E2E_CASES

    print("=" * 72)
    conv_label = "off" if conv_eps is None else f"{conv_eps:g}"
    fort12_label = "fort12 replay" if args.use_fortran_fort12 else "prod line selection"
    print(
        f"PARITY_E2E — {len(cases)} cases, WL {WL_START}–{WL_END} nm, "
        f"threshold={args.threshold}, max_iter={args.atlas_iterations}, "
        f"conv_eps={conv_label}, {fort12_label}"
    )
    print("=" * 72)

    results = []
    n_pass = n_fail = n_err = 0
    total_t0 = time.perf_counter()

    for case_name in cases:
        print(f"\n  [{case_name}] running Python ATLAS + SYNTHE ...", flush=True)
        r = run_one_case(
            case_name,
            threshold=args.threshold,
            n_workers=args.n_workers,
            atlas_iterations=args.atlas_iterations,
            convergence_epsilon=conv_eps,
            use_fortran_fort12=args.use_fortran_fort12,
        )
        results.append(r)

        icon = r["status"]
        err_str = f"{r['max_norm_abs']:.4e}" if r["max_norm_abs"] is not None else "—"
        print(f"  {icon:5s}  max_err={err_str}  {r['elapsed_s']}s")

        if r["status"] == "PASS":
            n_pass += 1
        elif r["status"] == "FAIL":
            n_fail += 1
        else:
            n_err += 1
            print(f"         {r.get('error', '')[:200]}")

    total_elapsed = round(time.perf_counter() - total_t0, 1)
    overall = "PASS" if n_fail == 0 and n_err == 0 else "FAIL"

    summary = {
        "threshold": args.threshold,
        "wl_start": WL_START,
        "wl_end": WL_END,
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

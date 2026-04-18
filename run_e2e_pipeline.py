#!/usr/bin/env python3
"""Fortran ↔ Python parity harness for the ATLAS + SYNTHE pipeline.

Each case starts from the *same* atmosphere — a warm-start written by the
kurucz-a1 emulator from stellar parameters — and runs both pipelines on it:

    (teff, logg, [M/H])                               results/<case>/
        └─► emulator ──► warm-start .atm ──┬──►  inputs/emulator_warmstart.atm
                                           │
                                           ├──► Fortran ATLAS → Fortran SYNTHE
                                           │        fortran/fortran_iter1.atm
                                           │        fortran/fortran_synthe_<wl>.spec
                                           │
                                           └──► Python  ATLAS → Python  SYNTHE
                                                    python/python_iter1.atm
                                                    python/python_synthe_<wl>.spec

When ``--mode both`` (default) the two spectra are compared on normalized
flux; the threshold is ``--norm-frac-threshold`` (default 0.10).

Single case:
    python run_e2e_pipeline.py --teff 5600 --logg 4.42 --mh 0.50 [--wipe]

Batch (one ``TEFF LOGG MH`` per line in ``e2e_batch_cases.txt``):
    python run_e2e_pipeline.py --batch [--wipe]

Re-run existing results/ folders (parameters are parsed from the case name):
    python run_e2e_pipeline.py --batch --batch-from-results
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from atlas_py.tools.validate_synthe_e2e import (
    _compute_norm_flux_errors,
    _run_fortran_synthe,
)
from synthe_py.tools.compare_spectra import compare_spectra

from pykurucz import (
    _default_kurucz_root,
    _ensure_gfpred_assembled,
    emulator_warmstart_atm,
    run_atlas_py,
    run_synthe_py,
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_BATCH_MANIFEST = REPO_ROOT / "e2e_batch_cases.txt"

_CASE_NAME_RE = re.compile(
    r"^t(\d{5})_g([+-]\d+\.\d{2})_mh([+-]\d+\.\d{2})$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    import subprocess
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output}"
        )
    return output


def _case_name_from_params(teff: int, logg: float, mh: float) -> str:
    return f"t{teff:05d}_g{logg:+.2f}_mh{mh:+.2f}"


def _parse_case_name_to_stellar(case_name: str) -> tuple[int, float, float]:
    """Parse ``t#####_g±#.##_mh±#.##`` folder names (as under ``results/``)."""
    m = _CASE_NAME_RE.match(case_name)
    if not m:
        raise ValueError(
            f"Case name {case_name!r} must match t#####_g±#.##_mh±#.## "
            "(e.g. t05770_g+4.44_mh+0.00)"
        )
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


def _ensure_emulator_warmstart(
    out_case_dir: Path,
    *,
    stellar: dict,
) -> Path:
    """Write the emulator warm-start ``.atm`` for *stellar* if not present."""
    warm = out_case_dir / "inputs" / "emulator_warmstart.atm"
    if warm.exists():
        return warm
    print(f"  [{out_case_dir.name}] writing emulator warm-start → {warm.name}")
    emulator_warmstart_atm(
        warm,
        teff=float(stellar["teff"]),
        logg=float(stellar["logg"]),
        mh=float(stellar["mh"]),
        am=float(stellar["am"]),
        vturb=float(stellar["vturb"]),
    )
    return warm


def _wipe_case_outputs(out_case_dir: Path) -> None:
    for subdir in ("fortran", "python", "logs", "inputs"):
        p = out_case_dir / subdir
        if p.exists():
            shutil.rmtree(p)


# ---------------------------------------------------------------------------
# ATLAS steps
# ---------------------------------------------------------------------------

def _run_fortran_atlas(
    *,
    input_atm: Path,
    out_fortran_dir: Path,
    log_path: Path,
    kurucz_root: Path,
) -> Path:
    """Invoke run_single_iteration to run Fortran atlas12.exe once.

    Binary paths mirror validate_phase1.py lines 152-167.
    """
    bin_dir = kurucz_root / "bin_macos"
    if not bin_dir.exists():
        bin_dir = kurucz_root / "bin_linux"
    lines_dir = kurucz_root / "lines"
    mol_dir = kurucz_root / "molecules"

    atlas12_exe = bin_dir / "atlas12.exe"
    molecules_new = lines_dir / "molecules.new"
    gfpred_bin = lines_dir / "gfpred29dec2014.bin"
    lowobs_bin = lines_dir / "lowobsat12.bin"
    hilines_bin = lines_dir / "hilines.bin"
    diatomics_bin = lines_dir / "diatomicspacksrt.bin"
    tio_bin = mol_dir / "tio" / "schwenke.bin"
    h2o_bin = mol_dir / "h2o" / "h2ofastfix.bin"
    nltelinobsat12_bin = lines_dir / "nltelinobsat12.bin"

    _ensure_gfpred_assembled(gfpred_bin)

    out_fortran_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    output_atm = out_fortran_dir / "fortran_iter1.atm"
    output_fort12 = out_fortran_dir / "fortran_iter1_fort12.bin"

    cmd = [
        sys.executable, "-m", "atlas_py.tools.run_single_iteration",
        "--atlas12-exe", str(atlas12_exe),
        "--input-atm", str(input_atm.resolve()),
        "--output-atm", str(output_atm),
        "--molecules-new", str(molecules_new),
        "--gfpred-bin", str(gfpred_bin),
        "--lowobs-bin", str(lowobs_bin),
        "--hilines-bin", str(hilines_bin),
        "--diatomics-bin", str(diatomics_bin),
        "--tio-bin", str(tio_bin),
        "--h2o-bin", str(h2o_bin),
        "--nltelinobsat12-bin", str(nltelinobsat12_bin),
        "--log-path", str(log_path),
        "--output-lines-bin", str(output_fort12),
    ]
    _run(cmd, cwd=REPO_ROOT)
    return output_atm


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def _run_case(
    case_name: str,
    *,
    stellar: dict,
    mode: str,
    wl_start: float,
    wl_end: float,
    resolution: float,
    wipe: bool,
    force_rerun_fortran: bool,
    force_rerun_python: bool,
    kurucz_root: Path,
    line_list_dir: str,
    n_workers: int | None,
    norm_frac_threshold: float,
) -> dict:
    """Run the full e2e pipeline for one case. Returns a result dict.

    ``stellar`` is ``{teff, logg, mh, am, vturb}``; both Fortran and Python
    branches consume the same emulator warm-start ``.atm``.
    """
    t0 = time.perf_counter()

    out_case_dir = RESULTS_DIR / case_name
    out_fortran_dir = out_case_dir / "fortran"
    out_python_dir = out_case_dir / "python"
    out_logs_dir = out_case_dir / "logs"

    if wipe:
        _wipe_case_outputs(out_case_dir)

    out_fortran_dir.mkdir(parents=True, exist_ok=True)
    out_python_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)

    input_atm = _ensure_emulator_warmstart(out_case_dir, stellar=stellar)

    wl_tag = f"{int(wl_start)}_{int(wl_end)}"

    # ------------------------------------------------------------------
    # Fortran pipeline
    # ------------------------------------------------------------------
    if mode in ("fortran", "both"):
        fortran_atm = out_fortran_dir / "fortran_iter1.atm"
        fortran_atlas_log = out_logs_dir / "fortran_atlas.log"

        if force_rerun_fortran or not fortran_atm.exists():
            print(f"  [{case_name}] running Fortran ATLAS ...")
            _run_fortran_atlas(
                input_atm=input_atm,
                out_fortran_dir=out_fortran_dir,
                log_path=fortran_atlas_log,
                kurucz_root=kurucz_root,
            )

        fortran_spec = out_fortran_dir / f"fortran_synthe_{wl_tag}.spec"
        fortran_synthe_log = out_logs_dir / f"fortran_synthe_{wl_tag}.log"

        if force_rerun_fortran or not fortran_spec.exists():
            print(f"  [{case_name}] running Fortran SYNTHE ...")
            _run_fortran_synthe(
                kurucz_root=kurucz_root,
                atm_file=fortran_atm,
                output_spec=fortran_spec,
                line_list_dir=line_list_dir,
                log_path=fortran_synthe_log,
            )

    # ------------------------------------------------------------------
    # Python pipeline
    # ------------------------------------------------------------------
    if mode in ("python", "both"):
        python_atm = out_python_dir / "python_iter1.atm"
        python_atlas_log = out_logs_dir / "python_atlas.log"

        # Line-selection binary from Fortran ATLAS (same case), if Fortran ran.
        fort12_from_fortran = out_fortran_dir / "fortran_iter1_fort12.bin"

        if force_rerun_python or not python_atm.exists():
            print(f"  [{case_name}] running Python ATLAS ...")
            out_python_dir.mkdir(parents=True, exist_ok=True)
            run_atlas_py(
                input_atm=input_atm,
                output_atm=python_atm,
                log_path=python_atlas_log,
                kurucz_root=kurucz_root,
                fort12_bin=fort12_from_fortran if fort12_from_fortran.exists() else None,
            )

        python_spec = out_python_dir / f"python_synthe_{wl_tag}.spec"
        python_npz = out_python_dir / "python_iter1_synthe.npz"
        python_synthe_log = out_logs_dir / f"python_synthe_{wl_tag}.log"

        if force_rerun_python or not python_spec.exists():
            print(f"  [{case_name}] running Python SYNTHE ...")
            run_synthe_py(
                python_atm,
                spec=python_spec,
                npz=python_npz,
                log_path=python_synthe_log,
                wl_start=wl_start,
                wl_end=wl_end,
                resolution=resolution,
                n_workers=n_workers,
                kurucz_root=kurucz_root,
            )

    # ------------------------------------------------------------------
    # Compare (both mode only)
    # ------------------------------------------------------------------
    status = "OK"
    max_norm_abs = None

    if mode == "both":
        fortran_spec = out_fortran_dir / f"fortran_synthe_{wl_tag}.spec"
        python_spec = out_python_dir / f"python_synthe_{wl_tag}.spec"
        compare_txt = out_logs_dir / f"compare_e2e_{wl_tag}.txt"
        compare_json_path = out_logs_dir / f"compare_e2e_{wl_tag}.json"

        summary = compare_spectra(
            python_file=python_spec,
            fortran_file=fortran_spec,
            wl_range=(wl_start, wl_end),
            top_n=10,
            quiet=True,
        )
        max_norm_abs, outliers, n_points, top = _compute_norm_flux_errors(
            python_spec=python_spec,
            fortran_spec=fortran_spec,
            wl_start=wl_start,
            wl_end=wl_end,
        )
        status = "PASS" if max_norm_abs <= norm_frac_threshold else "FAIL"

        report_lines = [
            f"case={case_name}",
            f"python_spec={python_spec}",
            f"fortran_spec={fortran_spec}",
            f"wavelength_window_nm={wl_start:.1f}-{wl_end:.1f}",
            f"norm_frac_threshold={norm_frac_threshold:.6f}",
            f"status={status}",
            f"n_points={n_points}",
            f"max_norm_abs={max_norm_abs:.6e}",
            f"outliers_over_10pct={outliers}",
            f"summary_norm_rms={summary['norm_rms']:.6e}",
            "top_norm_outliers:",
        ]
        for item in top:
            report_lines.append(
                "  wl={wavelength_nm:.6f} abs={abs_err:.6e} "
                "py_norm={py_norm:.6e} ft_norm={ft_norm:.6e}".format(**item)
            )
        compare_txt.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        payload = {
            "case": case_name,
            "status": status,
            "wavelength_window_nm": [wl_start, wl_end],
            "norm_frac_threshold": norm_frac_threshold,
            "max_norm_abs": max_norm_abs,
            "outliers_over_10pct": outliers,
            "n_points": n_points,
            "summary": {k: float(v) for k, v in summary.items()},
            "top_norm_outliers": top,
            "fortran_spec": str(fortran_spec),
            "python_spec": str(python_spec),
        }
        compare_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    return {
        "case": case_name,
        "status": status,
        "max_norm_abs": max_norm_abs,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_batch_manifest_rows(path: Path) -> list[tuple[int, float, float]]:
    """Parse lines ``TEFF LOGG MH`` from a manifest file."""
    if not path.is_file():
        return []
    rows: list[tuple[int, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return rows


def _batch_case_names_from_results() -> list[str]:
    """Folders under ``results/`` named ``t#####_g±#.##_mh±#.##``."""
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and _CASE_NAME_RE.match(d.name)
    )


def _resolve_batch_cases(
    *,
    batch_manifest: Path,
    batch_from_results: bool,
    am: float,
    vturb: float,
) -> list[tuple[str, dict]]:
    """Return list of ``(case_name, stellar_dict)`` for a batch run."""
    def _from_name(name: str) -> tuple[str, dict]:
        teff_i, logg_f, mh_f = _parse_case_name_to_stellar(name)
        return (
            name,
            {
                "teff": float(teff_i),
                "logg": logg_f,
                "mh": mh_f,
                "am": am,
                "vturb": vturb,
            },
        )

    if batch_from_results:
        names = _batch_case_names_from_results()
        if not names:
            raise FileNotFoundError(
                "No case folders matching t#####_g±#.##_mh±#.## under results/"
            )
        return [_from_name(n) for n in names]

    rows = _load_batch_manifest_rows(batch_manifest.resolve())
    if rows:
        return [
            (
                _case_name_from_params(t, g, m),
                {
                    "teff": float(t),
                    "logg": g,
                    "mh": m,
                    "am": am,
                    "vturb": vturb,
                },
            )
            for t, g, m in rows
        ]

    names = _batch_case_names_from_results()
    if not names:
        raise FileNotFoundError(
            f"No batch manifest at {batch_manifest} (lines: TEFF LOGG MH) and no "
            "matching case folders under results/. Create e2e_batch_cases.txt or use "
            "--batch-from-results after a first run."
        )
    return [_from_name(n) for n in names]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fortran ↔ Python parity harness (emulator warm-start → ATLAS → SYNTHE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single case — runs both Fortran and Python from the same emulator warm-start
  python run_e2e_pipeline.py --teff 5600 --logg 4.42 --mh 0.50 --wipe

  # Batch — reads e2e_batch_cases.txt (lines: TEFF LOGG MH)
  python run_e2e_pipeline.py --batch --wipe

  # Re-run every existing results/t*_g*_mh*/ folder (params parsed from name)
  python run_e2e_pipeline.py --batch --batch-from-results --force-rerun

  # Python-only (skip Fortran) — same flow, faster iteration while debugging
  python run_e2e_pipeline.py --teff 5770 --logg 4.44 --mh 0 --mode python
""",
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--teff", type=int,
                     help="Effective temperature (K) for a single-case run")
    src.add_argument("--batch", action="store_true",
                     help="Run many cases (see --batch-manifest / --batch-from-results)")

    p.add_argument("--logg", type=float, help="log g (required with --teff)")
    p.add_argument("--mh", type=float, help="[M/H] (required with --teff)")
    p.add_argument("--am", type=float, default=0.0,
                   help="[alpha/M] passed to emulator (default 0)")
    p.add_argument("--vturb", type=float, default=2.0,
                   help="Microturbulence km/s passed to emulator (default 2)")

    p.add_argument("--batch-manifest", type=Path, default=None,
                   help=f"Batch list (default: {DEFAULT_BATCH_MANIFEST.name} at repo root)")
    p.add_argument("--batch-from-results", action="store_true",
                   help="Batch over existing results/t#####_g*_mh*/ folders")

    p.add_argument("--mode", default="both",
                   choices=["fortran", "python", "both"],
                   help="Pipeline(s) to run (default: both)")
    p.add_argument("--wl-start", type=float, default=300.0, help="Wavelength start (nm)")
    p.add_argument("--wl-end", type=float, default=1800.0, help="Wavelength end (nm)")
    p.add_argument("--resolution", type=float, default=300000.0, help="Spectral resolution R")

    p.add_argument("--wipe", action="store_true",
                   help="Delete results/<case>/{fortran,python,logs,inputs} before running")
    p.add_argument("--force-rerun-fortran", action="store_true",
                   help="Re-run Fortran ATLAS+SYNTHE even if outputs exist")
    p.add_argument("--force-rerun-python", action="store_true",
                   help="Re-run Python ATLAS+SYNTHE even if outputs exist")
    p.add_argument("--force-rerun", action="store_true",
                   help="Shorthand for --force-rerun-fortran --force-rerun-python")

    p.add_argument("--n-workers", type=int, default=None,
                   help="Workers for synthe_py.cli (default: all logical CPUs)")
    p.add_argument("--kurucz-root", type=Path, default=None,
                   help="Data tree (default: data/ inside the pykurucz repo)")
    p.add_argument("--line-list-dir", type=str, default="linelists_full",
                   help="Subdir under data/synthe/ holding Fortran SYNTHE line lists")
    p.add_argument("--norm-frac-threshold", type=float, default=0.10,
                   help="Max allowed normalized-flux error when --mode both (default 0.10)")
    p.add_argument("--no-smoke-gate", action="store_true",
                   help="In batch mode, keep going even if the first (smoke) case fails")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.force_rerun:
        args.force_rerun_fortran = True
        args.force_rerun_python = True

    kurucz_root = (args.kurucz_root or _default_kurucz_root()).resolve()

    common_kwargs: dict = dict(
        mode=args.mode,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
        resolution=args.resolution,
        wipe=args.wipe,
        force_rerun_fortran=args.force_rerun_fortran,
        force_rerun_python=args.force_rerun_python,
        kurucz_root=kurucz_root,
        line_list_dir=args.line_list_dir,
        n_workers=args.n_workers,
        norm_frac_threshold=args.norm_frac_threshold,
    )

    wl_tag = f"{int(args.wl_start)}_{int(args.wl_end)}"
    manifest_path = args.batch_manifest or DEFAULT_BATCH_MANIFEST

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    if args.batch:
        try:
            batch_items = _resolve_batch_cases(
                batch_manifest=manifest_path,
                batch_from_results=args.batch_from_results,
                am=args.am,
                vturb=args.vturb,
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        cases = [item[0] for item in batch_items]

        print(f"{'='*72}")
        print(f"E2E PIPELINE BATCH — mode={args.mode} — {len(cases)} cases")
        print(f"  Outputs: {RESULTS_DIR}")
        print(f"  WL: {args.wl_start}–{args.wl_end} nm  wipe={args.wipe}")
        print(f"{'='*72}")

        # Smoke test: run the first case before committing to the full batch
        smoke_name, smoke_stellar = batch_items[0]
        print(f"\n[smoke 1/{len(cases)}] {smoke_name}")
        try:
            smoke_result = _run_case(
                smoke_name,
                stellar=smoke_stellar,
                **common_kwargs,
            )
        except Exception as exc:
            print(f"  SMOKE TEST FAILED: {exc}", file=sys.stderr)
            return 1

        smoke_status = smoke_result.get("status", "?")
        err = smoke_result.get("max_norm_abs")
        err_str = f"{err:.4e}" if err is not None else "—"
        icon = "✓" if smoke_status == "PASS" else ("✗" if smoke_status == "FAIL" else " ")
        print(f"  {icon} {smoke_status:6s}  max_err={err_str:>10s}  {smoke_result['elapsed_s']}s")

        if args.mode == "both" and smoke_status != "PASS":
            if not args.no_smoke_gate:
                print(
                    f"\n  Smoke test FAILED (max_err={err_str} > threshold "
                    f"{args.norm_frac_threshold}). Aborting batch.\n"
                    f"  Re-run with --no-smoke-gate to process all cases anyway.",
                    file=sys.stderr,
                )
                return 2
            else:
                print(
                    f"  Smoke test FAILED — continuing anyway (--no-smoke-gate)",
                    file=sys.stderr,
                )

        # Remaining cases
        results = [smoke_result]
        for i, (case_name, stellar) in enumerate(batch_items[1:], start=2):
            print(f"\n[{i}/{len(cases)}] {case_name}")
            try:
                result = _run_case(
                    case_name,
                    stellar=stellar,
                    **common_kwargs,
                )
                status = result.get("status", "?")
                err = result.get("max_norm_abs")
                err_str = f"{err:.4e}" if err is not None else "—"
                icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else " ")
                print(f"  {icon} {status:6s}  max_err={err_str:>10s}  {result['elapsed_s']}s")
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                result = {
                    "case": case_name,
                    "status": "ERROR",
                    "error": str(exc),
                    "max_norm_abs": None,
                }
            results.append(result)

        n_pass = sum(1 for r in results if r.get("status") == "PASS")
        n_fail = sum(1 for r in results if r.get("status") == "FAIL")
        n_err = sum(1 for r in results if r.get("status") == "ERROR")
        n_ok = sum(1 for r in results if r.get("status") == "OK")

        overall = "PASS" if (n_fail == 0 and n_err == 0) else "FAIL"

        summary_payload = {
            "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": args.mode,
            "wl_start": args.wl_start,
            "wl_end": args.wl_end,
            "threshold": args.norm_frac_threshold,
            "num_pass": n_pass,
            "num_fail": n_fail,
            "num_error": n_err,
            "overall_status": overall if args.mode == "both" else "N/A",
            "cases": results,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_summary = RESULTS_DIR / f"validation_summary_e2e_{wl_tag}.json"
        out_summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        print(f"\n{'='*72}")
        if args.mode == "both":
            print(
                f"SUMMARY: {n_pass} PASS  {n_fail} FAIL  {n_err} ERROR"
                f"  overall={overall}  ({len(results)} total)"
            )
            if n_fail > 0 or n_err > 0:
                print("\nFailing / error cases:")
                for r in results:
                    if r.get("status") not in ("PASS", "OK"):
                        err = r.get("max_norm_abs")
                        err_str = f"{err:.4e}" if err is not None else "—"
                        print(f"  {r.get('status'):7s}  max_err={err_str}  {r['case']}")
                        if "error" in r:
                            print(f"    {r['error']}")
        else:
            print(
                f"SUMMARY: {n_ok} OK  {n_err} ERROR  mode={args.mode}"
                f"  ({len(results)} total)"
            )
        print(f"Results: {RESULTS_DIR}")
        print(f"Summary JSON: {out_summary}")
        print(f"{'='*72}")

        return 0 if (n_fail == 0 and n_err == 0) else 1

    # ------------------------------------------------------------------
    # Single case
    # ------------------------------------------------------------------
    if args.teff is None:
        print("One of --batch or --teff is required.", file=sys.stderr)
        return 1
    if args.logg is None or args.mh is None:
        print("--logg and --mh are required with --teff", file=sys.stderr)
        return 1

    case_name = _case_name_from_params(args.teff, args.logg, args.mh)
    stellar = {
        "teff": float(args.teff),
        "logg": float(args.logg),
        "mh": float(args.mh),
        "am": float(args.am),
        "vturb": float(args.vturb),
    }

    print(f"Running e2e pipeline: {case_name}  mode={args.mode}")
    try:
        result = _run_case(
            case_name,
            stellar=stellar,
            **common_kwargs,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    status = result.get("status", "?")
    err = result.get("max_norm_abs")
    err_str = f"{err:.6e}" if err is not None else "—"
    print(f"Done: status={status}  max_norm_abs={err_str}  elapsed={result['elapsed_s']}s")

    if args.mode == "both":
        out_case_dir = RESULTS_DIR / case_name
        print(f"Compare report: {out_case_dir / 'logs' / f'compare_e2e_{wl_tag}.txt'}")

    return 0 if status in ("PASS", "OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())

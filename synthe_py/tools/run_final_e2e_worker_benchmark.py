#!/usr/bin/env python3
"""Regenerate ``results/final_e2e_workers/worker_benchmark_300_1800_and_parity.json``.

Re-runs ``synthe_py.cli`` for each converged case × worker count (300–1800 nm,
R=300000), and recomputes parity vs ``results/<case>/fortran/fortran_synthe_300_1800.spec``
using ``results/<case>/python/python_synthe_300_1800.spec`` on the Fortran grid.

Usage (from ``pykurucz`` repo root)::

    python synthe_py/tools/run_final_e2e_worker_benchmark.py
    python synthe_py/tools/run_final_e2e_worker_benchmark.py --update-report

Options ``--cases`` / ``--workers`` support smoke subsets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_norm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, usecols=(0, 1, 2))
    w = arr[:, 0]
    n = arr[:, 1] / np.clip(arr[:, 2], 1e-300, None)
    return w, n


def _parity_for_case(repo: Path, case: str) -> dict[str, Any]:
    case_dir = repo / "results" / case
    f_spec = case_dir / "fortran" / "fortran_synthe_300_1800.spec"
    p_spec = case_dir / "python" / "python_synthe_300_1800.spec"
    if not f_spec.is_file():
        raise FileNotFoundError(f"Missing Fortran spec: {f_spec}")
    if not p_spec.is_file():
        raise FileNotFoundError(f"Missing Python ref spec: {p_spec}")
    fw, fn = _load_norm(f_spec)
    pw, pn = _load_norm(p_spec)
    pni = np.interp(fw, pw, pn)
    adiff = np.abs(pni - fn)
    return {
        "case": case,
        "max_norm_abs": float(adiff.max()),
        "p95_norm_abs": float(np.percentile(adiff, 95)),
        "pass_0p10_abs": bool(float(adiff.max()) < 0.10),
    }


def _write_report(repo: Path, payload: dict[str, Any], report_path: Path) -> None:
    cases: list[str] = payload["cases"]
    workers: list[int] = payload["workers"]
    parity_rows: list[dict[str, Any]] = payload["parity"]
    rows: list[dict[str, Any]] = payload["benchmarks"]

    by_case: dict[str, dict[int, float]] = {}
    for r in rows:
        if r.get("status") != "OK":
            continue
        by_case.setdefault(r["case"], {})[int(r["workers"])] = float(r["elapsed_s"])

    def fmt_cell(case: str, nw: int) -> str:
        m = by_case.get(case, {})
        if nw not in m:
            return "—"
        return f"{m[nw]:.2f}"

    def best_nw(case: str) -> str:
        m = by_case.get(case, {})
        if not m:
            return "—"
        best = min(m, key=lambda k: m[k])
        return f"`nw={best}`"

    wmax = max(workers) if workers else 1
    n1 = [by_case[c][1] for c in cases if 1 in by_case.get(c, {})]
    n_hi = [by_case[c][wmax] for c in cases if wmax in by_case.get(c, {})]
    med1 = statistics.median(n1) if n1 else float("nan")
    med_hi = statistics.median(n_hi) if n_hi else float("nan")
    geo = (
        math.prod(n1[i] / n_hi[i] for i in range(len(n1)))
        ** (1.0 / len(n1))
        if len(n1) == len(n_hi) and n_hi
        else float("nan")
    )

    wcols = " | ".join(str(w) for w in workers)
    sep = " | ".join(["---:"] * len(workers))

    lines = [
        "# Final Parity + Runtime Report",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "Repo: `pykurucz`",
        "Primary data source: `results/final_e2e_workers/worker_benchmark_300_1800_and_parity.json`",
        "",
        "## Scope",
        "",
        "- Converged/validated cases in `results/`:",
    ]
    for c in cases:
        lines.append(f"  - `{c}`")
    lines.extend(
        [
            "",
            "## Parity Results (full 300–1800 nm, normalized flux absolute error)",
            "",
            "Criterion: `max |py_norm - fortran_norm| < 0.10` "
            "(Python reference: `results/<case>/python/python_synthe_300_1800.spec` "
            "interpolated onto the Fortran wavelength grid).",
            "",
        ]
    )
    for p in parity_rows:
        st = "PASS" if p.get("pass_0p10_abs") else "FAIL"
        lines.append(
            f"- `{p['case']}`: max={p['max_norm_abs']:.6f}, "
            f"p95={p['p95_norm_abs']:.6f}, {st}"
        )
    n_pass = sum(1 for p in parity_rows if p.get("pass_0p10_abs"))
    lines.extend(
        [
            "",
            f"Summary: {n_pass}/{len(parity_rows)} PASS under the `<0.10` threshold.",
            "",
            "## Runtime Benchmarks (`n_workers = " + ", ".join(str(w) for w in workers) + "`)",
            "",
            "Benchmark window: **300–1800 nm** (`R=300000`)  ",
            "Runner: `synthe_py.cli` on `python_iter30.atm` + `python_iter30_synthe.npz` per case. "
            "Single cold run per cell.",
            "",
            "### Per-case elapsed time (seconds)",
            "",
            f"| Case | {' | '.join(f'nw={w}' for w in workers)} | Best |",
            f"|---|{sep}|---|",
        ]
    )
    for c in cases:
        cells = " | ".join(fmt_cell(c, w) for w in workers)
        lines.append(f"| `{c}` | {cells} | {best_nw(c)} |")
    lines.extend(
        [
            "",
            "### Aggregate runtime findings (full band)",
            "",
            f"- Median elapsed at `nw=1`: **{med1:.1f} s**" if not math.isnan(med1) else "- Median elapsed at `nw=1`: **n/a**",
            (
                f"- Median elapsed at `nw={wmax}`: **{med_hi:.1f} s**"
                if not math.isnan(med_hi)
                else f"- Median elapsed at `nw={wmax}`: **n/a**"
            ),
        ]
    )
    if not math.isnan(geo) and wmax != 1:
        lines.append(
            f"- Geometric mean of `elapsed(nw=1) / elapsed(nw={wmax})` across cases: **{geo:.2f}**"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- See JSON/logs for per-run status; worker scaling is workload- and machine-dependent.",
            "- Full-band runs are often dominated by catalog/molecular stages; RT pool size is only one term.",
            "",
            "## Artifacts",
            "",
            "- `results/final_e2e_workers/worker_benchmark_300_1800_and_parity.json`",
            "- `results/final_e2e_workers/<case>/bench300_1800_nw*/`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", report_path, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="pykurucz repo root (default: infer from this file)",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path",
    )
    p.add_argument(
        "--update-report",
        action="store_true",
        help="Rewrite results/final_e2e_workers/report_final_parity_runtime.md from JSON",
    )
    p.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Markdown report path (default: results/final_e2e_workers/report_final_parity_runtime.md)",
    )
    p.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated case names under results/ (default: five final_e2e cases)",
    )
    p.add_argument(
        "--workers",
        type=str,
        default=None,
        help="Comma-separated worker counts (default: 1,2,4,8,cpu_count)",
    )
    p.add_argument(
        "--parity-only",
        action="store_true",
        help="Only recompute parity rows; skip synthe_py.cli benchmarks",
    )
    args = p.parse_args()

    repo = (args.repo_root or _repo_root()).resolve()
    out_json = (
        args.out_json
        if args.out_json is not None
        else repo / "results" / "final_e2e_workers" / "worker_benchmark_300_1800_and_parity.json"
    )
    report_md = (
        args.report_md
        if args.report_md is not None
        else repo / "results" / "final_e2e_workers" / "report_final_parity_runtime.md"
    )

    default_cases = [
        "t04000_g+5.00_mh+0.00",
        "t05600_g+4.42_mh+0.50",
        "t05770_g+4.44_mh-1.00",
        "t06200_g+4.20_mh+0.30",
        "t08250_g+4.00_mh+0.00",
    ]
    cases = (
        [c.strip() for c in args.cases.split(",") if c.strip()]
        if args.cases
        else default_cases
    )
    if args.workers:
        workers = [int(x.strip()) for x in args.workers.split(",") if x.strip()]
    else:
        workers = [1, 2, 4, 8, os.cpu_count() or 1]
    workers = list(dict.fromkeys(workers))

    parity_rows = []
    for case in cases:
        pr = _parity_for_case(repo, case)
        parity_rows.append(pr)
        print("PARITY", case, pr["max_norm_abs"], flush=True)

    rows: list[dict[str, Any]] = []
    if not args.parity_only:
        for case in cases:
            case_dir = repo / "results" / case
            atm = case_dir / "python" / "python_iter30.atm"
            npz = case_dir / "python" / "python_iter30_synthe.npz"
            if not atm.is_file():
                print("SKIP missing atm", atm, file=sys.stderr)
                continue
            if not npz.is_file():
                print("SKIP missing npz", npz, file=sys.stderr)
                continue
            for nw in workers:
                out_root = repo / "results" / "final_e2e_workers"
                out_dir = out_root / case / f"bench300_1800_nw{nw}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_spec = out_dir / "synthe_300_1800.spec"
                out_log = out_dir / "synthe_300_1800.log"
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "synthe_py.cli",
                    str(atm),
                    str(repo / "lines" / "gfallvac.latest"),
                    "--npz",
                    str(npz),
                    "--spec",
                    str(out_spec),
                    "--wl-start",
                    "300",
                    "--wl-end",
                    "1800",
                    "--resolution",
                    "300000",
                    "--n-workers",
                    str(nw),
                    "--log-level",
                    "INFO",
                ]
                t0 = time.perf_counter()
                with open(out_log, "w", encoding="utf-8") as lf:
                    r = subprocess.run(cmd, cwd=str(repo), stdout=lf, stderr=subprocess.STDOUT)
                elapsed = time.perf_counter() - t0
                rows.append(
                    {
                        "case": case,
                        "workers": nw,
                        "elapsed_s": elapsed,
                        "status": "OK" if r.returncode == 0 else "ERROR",
                        "returncode": int(r.returncode),
                        "spec": str(out_spec),
                        "log": str(out_log),
                    }
                )
                print(
                    "BENCH",
                    case,
                    "nw",
                    nw,
                    "elapsed",
                    round(elapsed, 2),
                    "rc",
                    r.returncode,
                    flush=True,
                )

    payload: dict[str, Any] = {
        "window_nm": [300, 1800],
        "cases": cases,
        "workers": workers,
        "parity": parity_rows,
        "benchmarks": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", out_json, flush=True)

    if args.update_report:
        if args.parity_only:
            print(
                "WARN: skipped --update-report because --parity-only left benchmarks empty",
                file=sys.stderr,
            )
        elif not rows:
            print("WARN: skipped --update-report (no benchmark rows)", file=sys.stderr)
        else:
            _write_report(repo, payload, report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

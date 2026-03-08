#!/usr/bin/env python
"""End-to-end spectrum synthesis: stellar parameters -> synthetic spectrum.

Combines the ATLAS12 neural network emulator (kurucz-a1) with synthe_py
to go directly from (Teff, logg, [M/H], [alpha/M]) to a synthetic
spectrum — entirely in Python, no Fortran required.

The emulator predicts the closest atmospheric structure for the 4 global
stellar parameters.  Individual element abundances can be overridden in
the resulting .atm file (the atmospheric *structure* is approximate for
the nearest 4-parameter model, but the *line opacities* in SYNTHE will
use the exact abundances you specify).

Note: synthe_py is a standalone SYNTHE reimplementation that works with
any .atm file.  This script is a convenience wrapper that generates the
atmosphere first.  A full Python reimplementation of ATLAS12 (which would
self-consistently solve for atmospheric structure at arbitrary abundances)
is the next step.

Usage:
    python pykurucz.py --teff 5770 --logg 4.44
    python pykurucz.py --teff 4500 --logg 2.0 --mh -1.5 --am 0.3
    python pykurucz.py --teff 5770 --logg 4.44 --abund C:+0.5 --abund Fe:-1.0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# ── Solar abundances and element data (from Kurucz / kurucz.py) ─────────

SOLAR_ABUND = np.array([
    -10.99, -10.66, -9.34, -3.61, -4.21, -3.35,   # 3-8
    -7.48, -4.11, -5.80, -4.44, -5.59, -4.53,     # 9-14
    -6.63, -4.92, -6.54, -5.64, -7.01, -5.70,     # 15-20
    -8.89, -7.09, -8.11, -6.40, -6.61, -4.54,     # 21-26
    -7.05, -5.82, -7.85, -7.48, -9.00, -8.39,     # 27-32
    -9.74, -8.70, -9.50, -8.79, -9.52, -9.17,     # 33-38
    -9.83, -9.46, -10.58, -10.16, -20.00, -10.29,  # 39-44
    -11.13, -10.47, -11.10, -10.33, -11.24, -10.00, # 45-50
    -11.03, -9.86, -10.49, -9.80, -10.96, -9.86,   # 51-56
    -10.94, -10.46, -11.32, -10.62, -20.00, -11.08, # 57-62
    -11.52, -10.97, -11.74, -10.94, -11.56, -11.12, # 63-68
    -11.94, -11.20, -11.94, -11.19, -12.16, -11.19, # 69-74
    -11.78, -10.64, -10.66, -10.42, -11.12, -10.87, # 75-80
    -11.14, -10.29, -11.39, -20.00, -20.00, -20.00, # 81-86
    -20.00, -20.00, -20.00, -12.02, -20.00, -12.58, # 87-92
    -20.00, -20.00, -20.00, -20.00, -20.00, -20.00, -20.00,  # 93-99
], dtype=np.float64)

ALPHA_ELEMENTS = [8, 10, 12, 14, 16, 20, 22]
ALPHA_IDX = np.array([e - 3 for e in ALPHA_ELEMENTS])
HE_ABUNDANCE = 0.078370

ELEM_SYM = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
    65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
    93: 'NP', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',
}

SYM_TO_Z = {sym.lower(): z for z, sym in ELEM_SYM.items()}


# ── Abundance helpers ────────────────────────────────────────────────────

def compute_abundances(mh: float = 0.0, am: float = 0.0,
                       individual: Optional[Dict[int, float]] = None) -> np.ndarray:
    """Compute log abundances for elements 3-99.

    Starts from solar, scales all metals by [M/H], applies additional
    [alpha/M] to alpha elements. Individual offsets (when provided) are
    relative to solar and override the [M/H]/[alpha/M] scaling entirely
    for that element — e.g. {26: -1.0} sets [Fe/H] = -1.0 regardless
    of the --mh value.
    """
    abund = SOLAR_ABUND.copy() + mh
    abund[ALPHA_IDX] += am
    if individual:
        for elem, offset in individual.items():
            if 3 <= elem <= 99:
                abund[elem - 3] = SOLAR_ABUND[elem - 3] + offset
    return abund


def compute_h(mh: float = 0.0, am: float = 0.0,
              individual: Optional[Dict[int, float]] = None) -> float:
    """Compute H abundance: H = 1 - He - sum(10^A_i) for metals."""
    abund = compute_abundances(mh, am, individual)
    z_sum = np.sum(10**abund)
    return 1.0 - HE_ABUNDANCE - z_sum


def derive_emulator_params(
    mh: float,
    am: float,
    individual: Optional[Dict[int, float]],
) -> tuple[float, float]:
    """Derive effective [M/H] and [alpha/M] for the emulator.

    When individual element offsets are provided, the emulator needs
    scalar [M/H] and [alpha/M] to predict the atmospheric structure.
    We derive these as:
      - [M/H] ~ [Fe/H] = offset provided for Fe (if given), else mh
      - [alpha/M] = mean of (alpha_offset_i - eff_mh) for alpha elements
        (O, Ne, Mg, Si, S, Ca, Ti)

    If an element is not individually specified, its abundance follows
    the --mh/--am scaling.
    """
    if not individual:
        return mh, am

    if 26 in individual:
        eff_mh = individual[26]
    else:
        eff_mh = mh

    alpha_offsets = []
    for elem in ALPHA_ELEMENTS:
        if elem in individual:
            alpha_offsets.append(individual[elem] - eff_mh)
        else:
            alpha_offsets.append(mh + am - eff_mh)

    eff_am = float(np.mean(alpha_offsets)) if alpha_offsets else am

    return eff_mh, eff_am


def parse_abund_arg(s: str) -> tuple[int, float]:
    """Parse 'Z:offset' or 'Symbol:offset' into (atomic_number, dex_offset).

    The offset is relative to Asplund solar abundances.
    Examples: 'Fe:+0.3' means [Fe/H]=+0.3, 'C:-0.5' means [C/H]=-0.5
    """
    parts = s.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid abundance format '{s}'. Use Z:offset or Symbol:offset "
            f"(e.g. Fe:+0.3 for [Fe/H]=+0.3)")
    key, val_str = parts[0].strip(), parts[1].strip()
    try:
        value = float(val_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid abundance value '{val_str}' in '{s}'")

    if key.isdigit():
        z = int(key)
    else:
        z = SYM_TO_Z.get(key.lower())
        if z is None:
            raise argparse.ArgumentTypeError(f"Unknown element symbol '{key}' in '{s}'")

    if not (3 <= z <= 99):
        raise argparse.ArgumentTypeError(f"Element Z={z} out of range (3-99)")
    return z, value


# ── .atm file writer ────────────────────────────────────────────────────

def write_atm_file(path: Path, teff: float, logg: float,
                   data: np.ndarray, vturb: float,
                   mh: float = 0.0, am: float = 0.0,
                   individual: Optional[Dict[int, float]] = None) -> None:
    """Write a Kurucz-format .atm file from emulator-predicted atmospheric structure."""
    abund = compute_abundances(mh, am, individual)
    h = compute_h(mh, am, individual)

    lines = []
    lines.append(f'TEFF   {teff:.0f}.  GRAVITY {logg:7.4f} LTE ')
    lines.append('TITLE kurucz-a1 emulator (Li et al. 2025) + pyKurucz                               ')
    lines.append(' OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0')
    lines.append(' CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00')
    lines.append(f'ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 {h:.5f} 2 {HE_ABUNDANCE:.5f}')

    elem = 3
    while elem <= 99:
        end = min(elem + 5, 99)
        parts = [' ABUNDANCE CHANGE']
        for e in range(elem, end + 1):
            val = abund[e - 3]
            if val > -10:
                parts.append(f' {e:2d}  {val:5.2f}')
            else:
                parts.append(f' {e:2d} {val:6.2f}')
        lines.append(''.join(parts))
        elem = end + 1

    lines.append(' ABUNDANCE TABLE')
    lines.append(f'    1H   {h:.6f}       2He  {HE_ABUNDANCE:.6f}')

    for start in range(3, 100, 5):
        end = min(start + 4, 99)
        parts = []
        for e in range(start, end + 1):
            sym = ELEM_SYM[e]
            val = abund[e - 3]
            if val > -10:
                padding = ' ' * (3 - len(sym))
                parts.append(f'{e:5d}{sym}{padding}{val:6.3f} 0.000')
            else:
                padding = ' ' * (3 - len(sym) - 1)
                parts.append(f'{e:5d}{sym}{padding}{val:7.3f} 0.000')
        lines.append(''.join(parts))

    n_layers = data.shape[0]
    lines.append(f'READ DECK6 {n_layers} RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB')

    vturb_cms = vturb * 1.0e5
    for row in data:
        parts = [f' {row[0]:.8E}', f'   {row[1]:.1f}']
        for j in range(2, 6):
            parts.append(f' {row[j]:.3E}')
        parts.append(f' {vturb_cms:.3E}')
        parts.append(f' {row[6]:.3E} {row[7]:.3E}')
        lines.append(''.join(parts))

    lines.append(f'PRADK {0.5:.4E}')
    lines.append('BEGIN                    ITERATION  15 COMPLETED')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ── End-to-end pipeline ─────────────────────────────────────────────────

def synthesize(
    teff: float,
    logg: float,
    mh: float = 0.0,
    am: float = 0.0,
    vturb: float = 2.0,
    wl_start: float = 300.0,
    wl_end: float = 1800.0,
    resolution: float = 300_000.0,
    abundances: Optional[Dict[int, float]] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Generate a synthetic spectrum from stellar parameters.

    This is the main entry point for end-to-end synthesis:
      1. Predict atmospheric structure with the kurucz-a1 neural network
         emulator, using the 4 global parameters (Teff, logg, [M/H],
         [alpha/M]) to find the closest atmospheric model.
      2. Write a Kurucz-format .atm file with the requested abundances
         (including any individual element overrides).
      3. Convert to .npz (atmosphere preprocessing: populations,
         molecular equilibrium, continuous opacity).
      4. Run synthe_py spectrum synthesis.

    The emulator predicts atmospheric *structure* (T, P, etc. vs depth)
    for the nearest 4-parameter model.  When individual abundances are
    provided, [M/H] and [alpha/M] for the emulator are derived from
    the specified abundances (Fe as the metallicity proxy, average of
    alpha elements O/Ne/Mg/Si/S/Ca/Ti for [alpha/M]).  The exact
    individual abundances are written into the .atm abundance table
    and used by SYNTHE for line opacities.  For self-consistent
    treatment of non-standard abundances in the atmospheric structure
    itself, a full ATLAS12 iteration would be needed (future work).

    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin.
    logg : float
        Surface gravity (log10 cgs).
    mh : float
        Overall metallicity [M/H] (default: 0.0, solar).  Scales all
        metals uniformly.  When --abund is used without --mh, this
        stays at 0 (solar baseline); only the specified elements change.
    am : float
        Alpha enhancement [alpha/M] (default: 0.0).  Additional offset
        applied to alpha elements (O, Ne, Mg, Si, S, Ca, Ti) on top
        of [M/H].
    vturb : float
        Microturbulent velocity in km/s (default: 2.0).
    wl_start, wl_end : float
        Wavelength range in nanometers (default: 300-1800 nm).
    resolution : float
        Resolving power lambda/delta_lambda (default: 300,000).
    abundances : dict, optional
        Individual element offsets {Z: dex_offset_from_solar}.
        E.g. {26: -1.0, 6: +0.5} to set [Fe/H]=-1.0 and [C/H]=+0.5.
        These override the [M/H]/[alpha/M] scaling for those elements.
    output_dir : str, optional
        Directory for output files. Defaults to results/.

    Returns
    -------
    Path
        Path to the output .spec file.
    """
    repo_root = Path(__file__).resolve().parent

    if output_dir is None:
        output_dir = str(repo_root / "results")
    output_path = Path(output_dir)

    npz_dir = output_path / "npz"
    spec_dir = output_path / "spec"
    log_dir = output_path / "logs"
    atm_dir = output_path / "atm"
    for d in [npz_dir, spec_dir, log_dir, atm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Derive effective [M/H] and [alpha/M] for the emulator from
    # individual abundances (if any).  Fe -> [M/H] proxy, average of
    # alpha elements -> [alpha/M] proxy.
    eff_mh, eff_am = derive_emulator_params(mh, am, abundances)

    # Warn if parameters are outside the emulator's training range
    # (Li et al. 2025: Teff 2500-50000, logg -1 to 5.5,
    #  [Fe/H] -4 to +1.46, [alpha/Fe] -0.2 to +0.62)
    warnings = []
    if not (2500 <= teff <= 50000):
        warnings.append(f"Teff={teff:.0f} K outside training range [2500, 50000]")
    if not (-1.0 <= logg <= 5.5):
        warnings.append(f"logg={logg:.2f} outside training range [-1.0, 5.5]")
    if not (-4.0 <= eff_mh <= 1.5):
        warnings.append(f"[M/H]={eff_mh:+.2f} outside training range [-4.0, +1.5]")
    if not (-0.2 <= eff_am <= 0.62):
        warnings.append(f"[alpha/M]={eff_am:+.2f} outside training range [-0.2, +0.6]")
    if warnings:
        print("  WARNING: Parameters outside kurucz-a1 emulator training range:")
        for w in warnings:
            print(f"           {w}")
        print("           Results may be unreliable. Consider using your own .atm file")
        print("           with synthesize_from_atm.py instead.")

    stem = f"t{int(teff):05d}g{logg:.2f}_mh{eff_mh:+.2f}_am{eff_am:+.2f}"
    atm_path = atm_dir / f"{stem}.atm"
    npz_path = npz_dir / f"{stem}.npz"
    spec_path = spec_dir / f"{stem}_{int(wl_start)}_{int(wl_end)}.spec"
    log_path = log_dir / f"{stem}_{int(wl_start)}_{int(wl_end)}.log"

    # ── Stage 1: Predict atmosphere with kurucz-a1 emulator ──────────
    print("[1/3] Predicting atmosphere with kurucz-a1 emulator...")
    print(f"      Teff={teff:.0f} K, logg={logg:.2f}, vturb={vturb:.1f} km/s")
    if abundances:
        overrides = ', '.join(
            f'[{ELEM_SYM.get(z, f"Z={z}")}/H]={v:+.2f}' for z, v in sorted(abundances.items()))
        print(f"      Individual abundances: {overrides}")
        if eff_mh != mh or eff_am != am:
            print(f"      Derived emulator params: [M/H]={eff_mh:+.2f}, [alpha/M]={eff_am:+.2f}")
        else:
            print(f"      Emulator params: [M/H]={eff_mh:+.2f}, [alpha/M]={eff_am:+.2f}")
    else:
        print(f"      [M/H]={mh:+.2f}, [alpha/M]={am:+.2f}")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required for the ATLAS12 emulator.")
        print("       Install with: pip install torch")
        sys.exit(1)

    from emulator import load_emulator

    emulator = load_emulator()
    data_9col = emulator.predict_atmosphere_data(teff, logg, eff_mh, eff_am, vturb)

    write_atm_file(atm_path, teff, logg, data_9col, vturb,
                   mh=mh, am=am, individual=abundances)
    print(f"      Atmosphere: {atm_path}")

    # ── Stage 2: Convert .atm to .npz ────────────────────────────────
    print("[2/3] Converting atmosphere to NPZ...")
    atlas_tables = repo_root / "synthe_py" / "data" / "atlas_tables.npz"

    cmd_convert = [
        sys.executable,
        str(repo_root / "synthe_py" / "tools" / "convert_atm_to_npz.py"),
        str(atm_path), str(npz_path),
        "--atlas-tables", str(atlas_tables),
    ]
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR in convert_atm_to_npz:\n{result.stderr}")
        sys.exit(1)
    print(f"      NPZ: {npz_path}")

    # ── Stage 3: Run synthe_py ────────────────────────────────────────
    print(f"[3/3] Running synthe_py ({wl_start:.0f}-{wl_end:.0f} nm, R={resolution:.0f})...")
    line_list = repo_root / "lines" / "gfallvac.latest"
    cache_dir = repo_root / "synthe_py" / "out" / "line_cache"
    n_workers = os.cpu_count() or 1

    cmd_synthe = [
        sys.executable, "-m", "synthe_py.cli",
        str(atm_path), str(line_list),
        "--npz", str(npz_path),
        "--spec", str(spec_path),
        "--wl-start", str(wl_start),
        "--wl-end", str(wl_end),
        "--resolution", str(resolution),
        "--n-workers", str(n_workers),
        "--cache", str(cache_dir),
        "--log-level", "INFO",
    ]

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd_synthe, stdout=logf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"ERROR: synthe_py failed. See log: {log_path}")
        sys.exit(1)

    print(f"      Spectrum: {spec_path}")
    print(f"      Log: {log_path}")
    print(f"\nDone! Output: {spec_path}")
    return spec_path


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end spectrum synthesis: stellar parameters -> synthetic spectrum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solar-type star, full wavelength range
  python pykurucz.py --teff 5770 --logg 4.44

  # Metal-poor K giant with alpha enhancement
  python pykurucz.py --teff 4500 --logg 2.0 --mh -1.5 --am 0.3

  # Override individual element abundances relative to solar
  python pykurucz.py --teff 5770 --logg 4.44 --abund Fe:-1.0 --abund C:+0.5
  python pykurucz.py --teff 5770 --logg 4.44 --abund 26:-1.0 --abund 6:+0.5

  # Hot B star, optical only
  python pykurucz.py --teff 15000 --logg 4.0 --wl-start 350 --wl-end 700

  # Quick low-resolution run
  python pykurucz.py --teff 5770 --logg 4.44 --resolution 50000

Notes:
  --mh scales ALL metals uniformly (like [Fe/H] in a standard grid).
  --am adds an extra offset to alpha elements on top of --mh.
  --abund sets individual element offsets relative to solar (Asplund).
    e.g. --abund Fe:-1.0 means [Fe/H] = -1.0 dex below solar.

  These can be combined:
    --mh -1.0 --abund C:+0.5   # all metals at -1 dex, but [C/H]=+0.5
    --abund Fe:-1.0             # only Fe changes, rest stay solar

  For the emulator's atmospheric structure prediction:
    - [M/H] is proxied by Fe: if --abund Fe is given, [M/H] ~ [Fe/H]_eff
    - [alpha/M] is the mean offset of alpha elements (O,Ne,Mg,Si,S,Ca,Ti)
    - Unspecified elements follow the --mh/--am scaling

  The exact abundances (solar + offsets) are written into the .atm file
  and used by SYNTHE for line opacities.

  A full Python ATLAS12 that self-consistently solves for atmospheric
  structure at arbitrary abundances is planned as future work.
""",
    )
    parser.add_argument("--teff", type=float, required=True,
                        help="Effective temperature (K)")
    parser.add_argument("--logg", type=float, required=True,
                        help="Surface gravity (log10 cgs)")
    parser.add_argument("--mh", type=float, default=0.0,
                        help="Overall metallicity [M/H] in dex (default: 0.0). "
                             "Scales all metals uniformly from solar.")
    parser.add_argument("--am", type=float, default=0.0,
                        help="Alpha enhancement [alpha/M] in dex (default: 0.0). "
                             "Extra offset for O, Ne, Mg, Si, S, Ca, Ti.")
    parser.add_argument("--vturb", type=float, default=2.0,
                        help="Microturbulent velocity in km/s (default: 2.0)")
    parser.add_argument("--abund", type=parse_abund_arg, action="append",
                        metavar="ELEM:OFFSET",
                        help="Override element abundance relative to solar (dex). "
                             "e.g. --abund Fe:-1.0 for [Fe/H]=-1.0, "
                             "--abund C:+0.5 for [C/H]=+0.5. Can be repeated.")
    parser.add_argument("--wl-start", type=float, default=300.0,
                        help="Start wavelength in nm (default: 300)")
    parser.add_argument("--wl-end", type=float, default=1800.0,
                        help="End wavelength in nm (default: 1800)")
    parser.add_argument("--resolution", type=float, default=300_000.0,
                        help="Resolving power (default: 300000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    individual = None
    if args.abund:
        individual = {z: val for z, val in args.abund}

    synthesize(
        teff=args.teff,
        logg=args.logg,
        mh=args.mh,
        am=args.am,
        vturb=args.vturb,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
        resolution=args.resolution,
        abundances=individual,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

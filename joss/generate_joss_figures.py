#!/usr/bin/env python
"""Generate JOSS publication figures: one per atmosphere model.

Each figure has 3 rows:
  Top:    Normalized flux overlay (optical 3500-9000 A)
  Middle: Full flux spectrum (log scale, 3000-18000 A)
  Bottom: Fractional deviation vs wavelength (smoothed + raw)

Requires: pip install SciencePlots
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from scipy.ndimage import median_filter

plt.style.use(['science', 'nature'])

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORTRAN_DIR = os.path.join(REPO_ROOT, "fortran_specs")
PYTHON_DIR = os.path.join(REPO_ROOT, "results", "spec")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("at12_aaaaa_t02500g-1.0",
     r"$T_{\rm eff}=2500\,{\rm K},\;\log g=-1.0$ (cool giant)"),
    ("at12_aaaaa_t04000g5.00",
     r"$T_{\rm eff}=4000\,{\rm K},\;\log g=5.0$ (K dwarf)"),
    ("at12_aaaaa_t08250g4.00",
     r"$T_{\rm eff}=8250\,{\rm K},\;\log g=4.0$ (A star)"),
    ("at12_aaaaa_t10250g5.00",
     r"$T_{\rm eff}=10250\,{\rm K},\;\log g=5.0$ (late B dwarf)"),
    ("at12_aaaaa_t44000g4.50",
     r"$T_{\rm eff}=44000\,{\rm K},\;\log g=4.5$ (O star)"),
]

SMOOTH_PTS = 501

C_FORTRAN = '#0C5DA5'
C_PYTHON = '#FF2C00'


def load(fortran_path, python_path):
    f = np.loadtxt(fortran_path)
    p = np.loadtxt(python_path)
    return (f[:, 0], f[:, 1], f[:, 2]), (p[:, 0], p[:, 1], p[:, 2])


for stem, label in MODELS:
    f_path = f"{FORTRAN_DIR}/{stem}.spec"
    p_path = f"{PYTHON_DIR}/{stem}_300_1800.spec"
    if not (os.path.exists(f_path) and os.path.exists(p_path)):
        print(f"  Skipping {stem} (missing files)")
        continue

    (f_wl, f_flux, f_cont), (p_wl, p_flux, p_cont) = load(f_path, p_path)
    f_wl_a = f_wl * 10
    p_wl_a = p_wl * 10

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(3.5, 4.5),
        gridspec_kw={'height_ratios': [1.2, 0.8, 0.8], 'hspace': 0.08},
    )

    # ── Row 1: Normalized flux overlay (optical) ─────────────────────
    f_norm = np.where(f_cont > 0, f_flux / f_cont, 1.0)
    p_norm = np.where(p_cont > 0, p_flux / p_cont, 1.0)

    optical = (3500, 9000)
    fm = (f_wl_a >= optical[0]) & (f_wl_a <= optical[1])
    pm = (p_wl_a >= optical[0]) & (p_wl_a <= optical[1])

    ax1.plot(f_wl_a[fm], f_norm[fm], color=C_FORTRAN, lw=0.15, alpha=0.8,
             label='Fortran SYNTHE', rasterized=True)
    ax1.plot(p_wl_a[pm], p_norm[pm], color=C_PYTHON, lw=0.1, alpha=0.7,
             label='pyKurucz', rasterized=True)
    ax1.set_xlim(*optical)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel(r'$F_\lambda\,/\,F_{\rm cont}$')
    ax1.legend(fontsize=5, loc='lower right')
    ax1.set_title(label, fontsize=6, pad=4)
    ax1.set_xticklabels([])

    # ── Row 2: Full flux (log scale) ─────────────────────────────────
    ax2.semilogy(f_wl_a, f_flux, color=C_FORTRAN, lw=0.1, alpha=0.7,
                 rasterized=True)
    ax2.semilogy(p_wl_a, p_flux, color=C_PYTHON, lw=0.1, alpha=0.6,
                 rasterized=True)
    ax2.semilogy(f_wl_a, f_cont, color=C_FORTRAN, lw=0.5, alpha=0.5,
                 ls='--')
    ax2.semilogy(p_wl_a, p_cont, color=C_PYTHON, lw=0.5, alpha=0.5,
                 ls=':')
    ax2.set_xlim(3000, 18000)
    ax2.set_ylabel(r'$F_\lambda$ (erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$)')
    ax2.set_xticklabels([])

    # ── Row 3: Fractional deviation ──────────────────────────────────
    p_norm_interp = np.interp(f_wl, p_wl, p_norm)
    denom = np.where(np.abs(f_norm) > 1e-10, f_norm, 1.0)
    frac = (p_norm_interp - f_norm) / denom

    ax3.plot(f_wl_a, frac * 100, color='0.8', lw=0.05, alpha=0.4,
             rasterized=True)
    smoothed = median_filter(frac * 100, size=SMOOTH_PTS)
    ax3.plot(f_wl_a, smoothed, color='k', lw=0.5, label='running median')
    ax3.axhline(0, color='0.5', ls='-', lw=0.2)
    ax3.set_xlim(3000, 18000)
    ax3.set_xlabel(r'Wavelength (\AA)')
    ax3.set_ylabel(r'Deviation (\%)')

    med = np.median(np.abs(frac)) * 100
    p95 = np.percentile(np.abs(frac), 95) * 100
    p99 = np.percentile(np.abs(frac), 99) * 100
    stats = (f'median $|\\Delta|$ = {med:.4f}\\%\n'
             f'95th pctl = {p95:.4f}\\%\n'
             f'99th pctl = {p99:.4f}\\%')
    ax3.text(0.02, 0.95, stats, transform=ax3.transAxes,
             fontsize=4.5, va='top', family='monospace',
             bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                       edgecolor='0.7', linewidth=0.3))
    ax3.legend(fontsize=5, loc='upper right')

    outpath = f"{OUT_DIR}/compare_{stem}.png"
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}  (median={med:.5f}%, 95th={p95:.5f}%)")

print("\nDone.")

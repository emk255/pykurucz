# Synthesizing the Solar Spectrum

This tutorial walks through the generation of a high-fidelity synthetic spectrum for a Sun-like star. We use the canonical solar parameters ($T_{\rm eff}=5770$ K, $\log g=4.44$, ${\rm [M/H]}=0$) and explore both the Python interface and the resulting spectral features.

## What You Will Learn

- How to run the full Stellar Parameters pipeline from Python
- How to read and normalize the `.spec` output
- How to identify classic solar absorption lines in the synthetic spectrum
- How to produce publication-quality plots with matplotlib

## Prerequisites

- pykurucz installed and the data downloaded (see [Downloading Data](../getting-started/downloading-data.md))
- Python 3.10+ with `numpy` and `matplotlib`
- ~8 GB RAM and ~1 GB disk space for outputs

!!! warning "Do not skip data download"
    The atomic line list (`lines/gfallvac.latest`) and molecular binaries are required. Without them, `synthe_py` will raise a `FileNotFoundError`.

## Step 1 — Generate the Spectrum

We will synthesize a 200 nm chunk covering the optical region where many classic absorption lines fall.

=== "Python API (Stellar Parameters)"

    ```python
    from pykurucz import synthesize

    spec_path = synthesize(
        teff=5770,          # Solar effective temperature (K)
        logg=4.44,          # Solar surface gravity (log10 cgs)
        mh=0.0,             # Solar metallicity [M/H]
        am=0.0,             # No alpha enhancement
        vturb=2.0,          # Microturbulence (km/s), typical for the Sun
        wl_start=400.0,     # Start wavelength (nm)
        wl_end=600.0,       # End wavelength (nm)
        resolution=300_000, # High resolving power (echelle-like)
        output_dir="results_solar",
    )
    print(f"Spectrum written to: {spec_path}")
    ```

=== "CLI"

    ```bash
    python pykurucz.py \
        --teff 5770 --logg 4.44 \
        --mh 0.0 --am 0.0 \
        --wl-start 400 --wl-end 600 \
        --resolution 300000 \
        --output-dir results_solar
    ```

!!! note "Runtime estimate"
    On a modern 8-core workstation, the full 200 nm synthesis at $R=300\,000$ takes approximately **5–10 minutes**:

    | Stage | Time |
    |---|---|
    | Emulator warm-start | < 1 s |
    | `atlas_py` iteration (converged) | 2–5 min |
    | `synthe_py` synthesis (400–600 nm) | 3–5 min |

    Narrower ranges run proportionally faster.

## Step 2 — Read and Inspect the Output

The pipeline writes several files under `results_solar/`. The spectrum itself is a whitespace-delimited text file with three columns. See [Output Files](../user-guide/output-files.md) for a full description of every file type.

```python
import numpy as np

# Load the spectrum: wavelength (nm), flux, continuum
wl, flux, cont = np.loadtxt(
    "results_solar/spec/t05770g4.44_mh+0.00_am+0.00_400_600.spec",
    unpack=True
)

# Normalized spectrum
norm = flux / cont

print(f"Wavelength range: {wl.min():.2f} – {wl.max():.2f} nm")
print(f"Number of points: {len(wl):,}")
print(f"Mean spacing: {np.mean(np.diff(wl)) * 1e3:.3f} pm")
```

You will see output similar to:

```
Wavelength range: 400.00 – 600.00 nm
Number of points: 480,026
Mean spacing: 0.417 pm
```

The pixel spacing is set by the resolving power: $\Delta\lambda = \lambda / R \approx 1.3$ pm at 400 nm and 2.0 pm at 600 nm, sampled at roughly two points per resolution element.

## Step 3 — Identify Key Absorption Lines

The solar spectrum is rich with atomic lines. In the 400–600 nm window you should see several famous features:

| Line | Wavelength | Element | Approximate Depth |
|---|---|---|---|
| Ca II H | 396.85 nm | Ca | Very strong, core near zero |
| Ca II K | 393.37 nm | Ca | Very strong, core near zero |
| H$\epsilon$ | 397.01 nm | H | Moderate (Balmer series) |
| H$\delta$ | 410.17 nm | H | Moderate |
| H$\gamma$ | 434.05 nm | H | Moderate |
| CH G-band | 430.5 nm | CH (molecular) | Broad, shallow depression |
| H$\beta$ | 486.13 nm | H | Deep Balmer line |
| Mg b triplet | 516.73, 517.27, 518.36 nm | Mg | Strong, closely spaced |
| Na D$_1$ / D$_2$ | 589.59, 588.99 nm | Na | Deep, easily resolved at $R=300\,000$ |

!!! physics "Why vacuum wavelengths?"
    pykurucz uses vacuum wavelengths internally. The Ca II H&K lines listed above are at their vacuum positions. If you compare against observed solar atlases tabulated in air, apply the standard Edlén conversion: $\lambda_{\rm air} = \lambda_{\rm vac} / n_{\rm air}(\lambda)$.

## Step 4 — Plot the Spectrum

The snippet below generates a two-panel figure: the top panel shows the normalized flux, and the bottom panel zooms into the Na D and Mg b regions.

```python
import numpy as np
import matplotlib.pyplot as plt

wl, flux, cont = np.loadtxt(
    "results_solar/spec/t05770g4.44_mh+0.00_am+0.00_400_600.spec",
    unpack=True
)
norm = flux / cont

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# --- Top panel: full range ---
ax = axes[0]
ax.plot(wl, norm, lw=0.3, c="C0", alpha=0.8)
ax.set_ylabel(r"$F_\lambda / F_{\rm cont}$")
ax.set_ylim(0, 1.1)
ax.set_xlim(400, 600)
ax.set_title("Synthetic Solar Spectrum (400–600 nm, $R=300\,000$)")

# Annotate key lines
for wl_line, label in [
    (410.17, r"H$\delta$"),
    (434.05, r"H$\gamma$"),
    (486.13, r"H$\beta$"),
    (517.27, "Mg b"),
    (589.29, "Na D"),  # centroid of the doublet
]:
    ax.axvline(wl_line, color="red", ls="--", lw=0.5, alpha=0.5)
    ax.text(wl_line + 1.5, 0.15, label, rotation=90, va="bottom", fontsize=8)

# --- Bottom panel: Na D zoom ---
ax = axes[1]
mask = (wl >= 588.0) & (wl <= 590.5)
ax.plot(wl[mask], norm[mask], lw=0.5, c="C0")
ax.axvline(588.99, color="red", ls="--", lw=0.5, alpha=0.5)
ax.axvline(589.59, color="red", ls="--", lw=0.5, alpha=0.5)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"$F_\lambda / F_{\rm cont}$")
ax.set_ylim(0, 1.1)
ax.set_title("Na D Doublet Zoom")

plt.tight_layout()
plt.savefig("solar_spectrum_400_600.png", dpi=250)
```

You will see:

- **Top panel**: A dense forest of metal lines with the deep Balmer lines standing out as the broadest features.
- **Bottom panel**: The Na D$_1$ and D$_2$ lines cleanly separated by ~0.6 nm, each with a sharp core and slightly extended Lorentzian wings.

!!! tip "Experiment with narrower ranges"
    For quick iteration, restrict the wavelength window to 10 nm. A 10 nm chunk at solar parameters runs in ~10–30 seconds and is ideal for checking line depths before committing to a full-range synthesis.

## Expected Output Files

After the run completes, `results_solar/` contains:

```
results_solar/
├── atm/
│   ├── t05770g4.44_mh+0.00_am+0.00_warmstart.atm
│   └── t05770g4.44_mh+0.00_am+0.00.atm
├── npz/
│   └── t05770g4.44_mh+0.00_am+0.00.npz
├── spec/
│   └── t05770g4.44_mh+0.00_am+0.00_400_600.spec
└── logs/
    ├── t05770g4.44_mh+0.00_am+0.00_atlas.log
    └── t05770g4.44_mh+0.00_am+0.00_synthe_400_600.log
```

The iterated atmosphere (`*.atm`) and the `.npz` cache can be reused for other wavelength ranges without rerunning `atlas_py` — see [Existing Atmosphere](../user-guide/from-atmosphere.md) if you want to synthesize from an existing atmosphere.

## Next Steps

- Learn how to [override individual element abundances](custom-abundances.md) to model peculiar stars.
- Compare [different spectral resolutions](resolution-comparison.md) to understand the trade-off between computation time and line detail.
- Read the [Stellar Parameters](../user-guide/from-parameters.md) user-guide page for advanced convergence tuning.

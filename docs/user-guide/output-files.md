# Output Files

pykurucz produces three primary output formats: `.spec` (the synthetic spectrum), `.atm` (the model atmosphere), and `.npz` (the preprocessed atmosphere cache). This page explains the contents of each and how to load them in Python.

## `.spec` — Synthetic Spectrum

The `.spec` file is the final scientific product: a whitespace-delimited text file with three columns.

| Column | Name | Units | Description |
|---|---|---|---|
| 1 | Wavelength | nm (vacuum) | Wavelength of each sample point |
| 2 | \(F_\lambda\) | erg cm⁻² s⁻¹ nm⁻¹ | Total emergent flux (line + continuum) |
| 3 | \(F_{\rm cont}\) | erg cm⁻² s⁻¹ nm⁻¹ | Continuum-only flux |

The normalized spectrum is \(F_\lambda / F_{\rm cont}\). Wavelengths are uniformly spaced in log-space according to the requested resolving power.

### Reading a `.spec` file

```python
import numpy as np

wl, flux, cont = np.loadtxt("results/spec/t05770g4.44_mh+0.00_am+0.00_500_510.spec", unpack=True)

# Normalized spectrum
norm = flux / cont
```

### Plotting

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(wl, norm, lw=0.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel(r"$F_\lambda / F_{\rm cont}$")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("spectrum.png", dpi=200)
```

## `.atm` — Model Atmosphere

The `.atm` file is a Kurucz-format ASCII file that completely describes the model atmosphere. It contains:

- **Header cards**: `TEFF`, `GRAVITY`, `TITLE`, `OPACITY IFOP ...`, `CONVECTION ...`
- **Abundance cards**: `ABUNDANCE CHANGE` and `ABUNDANCE TABLE` for elements 1–99
- **DECK6 block**: 80 layers of atmospheric structure

### DECK6 format

The `READ DECK6` block has 80 rows (layers) and 9 columns:

| Column | Symbol | Units | Description |
|---|---|---|---|
| 1 | `RHOX` | g cm⁻² | Mass column density |
| 2 | `T` | K | Temperature |
| 3 | `P` | dyn cm⁻² | Gas pressure |
| 4 | `XNE` | cm⁻³ | Electron number density |
| 5 | `ABROSS` | cm² g⁻¹ | Rosseland mean opacity |
| 6 | `ACCRAD` | cm s⁻² | Radiative acceleration |
| 7 | `VTURB` | cm s⁻¹ | Microturbulent velocity |
| 8 | `FLXCNV` | — | Convective flux ratio (often 0) |
| 9 | `VCONV` | cm s⁻¹ | Convective velocity (often 0) |

### Reading an `.atm` file

```python
from atlas_py.io.atmosphere import load_atm

atm = load_atm("results/atm/t05770g4.44_mh+0.00_am+0.00.atm")

print(atm.temperature)      # shape (80,)
print(atm.gas_pressure)     # shape (80,)
print(atm.electron_density) # shape (80,)
print(atm.rhox)             # shape (80,)
```

!!! note "Two atmosphere files in Stellar Parameters"
    Stellar Parameters writes both `<stem>_warmstart.atm` (the raw emulator prediction) and `<stem>.atm` (the `atlas_py` iterated atmosphere). For science, always use the iterated atmosphere.

## `.npz` — Preprocessed Atmosphere Cache

The `.npz` file is a NumPy archive produced by `convert_atm_to_npz.py`. It stores precomputed quantities that accelerate synthesis:

- `depth` — Rosseland optical depth grid
- `temperature` — layer temperatures (K)
- `gas_pressure` — gas pressures (dyn cm⁻²)
- `electron_density` — electron densities (cm⁻³)
- `mass_density` — mass densities (g cm⁻³)
- `turbulent_velocity` — microturbulent velocity (km/s)
- `population_per_ion` — Saha–Boltzmann populations for all ions
- `doppler_per_ion` — Doppler widths per ion
- `xnfph` — hydrogen ground-state populations
- `xnf_he1` / `xnf_he2` — helium populations
- `xnfpc`, `xnfpmg`, `xnfpal`, `xnfpsi`, `xnfpfe` — metal ground-state populations
- `xabund` — abundance mass fractions
- `cont_abs_coeff` / `cont_scat_coeff` — continuum opacity interpolation coefficients
- `meta_*` — metadata (Teff, logg, title, abundances, version stamp)

### Loading a `.npz` file

```python
import numpy as np

data = np.load("results/npz/t05770g4.44_mh+0.00_am+0.00.npz")
print(data.files)          # list of arrays
print(data["temperature"]) # shape (80,)
```

!!! tip "Cache invalidation"
    The `.npz` file is tied to a specific `.atm` file by content, not just by filename. If you edit the `.atm` abundances or temperature structure, you must regenerate the `.npz`. `synthe_py` checks a version stamp inside the `.npz` and will auto-refresh stale caches if the converter script is available.

## Log Files

Both `atlas_py` and `synthe_py` write detailed logs:

- `logs/<stem>_atlas.log` — iteration timings, convergence diagnostics, opacity summaries
- `logs/<stem>_synthe_<wl0>_<wl1>.log` — line counts, wavelength progress, worker timings

These are invaluable for debugging convergence issues or understanding runtime bottlenecks.

## Summary Table

| Extension | Producer | Consumer | Purpose |
|---|---|---|---|
| `.spec` | `synthe_py.cli` | User / analysis scripts | Final synthetic spectrum |
| `.atm` | `atlas_py.cli` / emulator | `convert_atm_to_npz.py` | Model atmosphere stratification |
| `.npz` | `convert_atm_to_npz.py` | `synthe_py.cli` | Precomputed populations and opacities |

## Next Steps

- Explore the [CLI Reference](cli-reference.md) to control output paths and filenames.
- Learn about the [Emulator](emulator.md) to understand the warm-start atmosphere.
- Dive into [Architecture](../architecture/data-flows.md) for the full data-flow diagram.

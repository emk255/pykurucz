# Data Flows

This page documents the file formats that carry data between pipeline stages, the caching strategy, and how to inspect intermediate outputs.

## File Format Specifications

Three plain file formats carry data between stages:

- `.atm` (text) — Kurucz model atmosphere; in/out of `atlas_py` and the
  starting input to synthesis preprocessing.
- `.npz` (NumPy archive) — Cached populations + opacities computed once
  from a `.atm`, reused for every wavelength range.
- `.spec` (text) — Final spectrum: wavelength (nm, vacuum), total flux,
  continuum flux.

### `.atm` — Kurucz Atmosphere Format

The `.atm` file is a plain-text file that completely specifies a 1D plane-parallel model atmosphere. It is the lingua franca of the Kurucz ecosystem and is understood by both Fortran ATLAS12/SYNTHE and pykurucz.

**Structure:**

1. **Header cards** — `TEFF`, `GRAVITY`, `TITLE`, `OPACITY IFOP`, `CONVECTION`, `ABUNDANCE CHANGE`
2. **Abundance table** — Log abundances for elements 1–99
3. **`READ DECK6` block** — 80 rows × 9 columns of atmospheric structure
4. **`PRADK` line** — Radiative pressure constant
5. **`BEGIN` / `ITERATION` footer** — Convergence metadata

See [Output Files](../user-guide/output-files.md) for a column-by-column breakdown of the DECK6 block.

### `.npz` — Preprocessed Atmosphere Cache

The `.npz` file is a NumPy archive that stores everything `synthe_py` needs to know about the atmosphere *except* the line list. It is generated once per `.atm` file and can be reused for many syntheses.

**Key arrays:**

| Array | Shape | Description |
|---|---|---|
| `temperature` | (80,) | Temperature (K) |
| `p` | (80,) | Gas pressure (dyn cm⁻²) |
| `xne` | (80,) | Electron density (cm⁻³) |
| `rho` | (80,) | Mass density (g cm⁻³) |
| `xnf` | (80, 1006) | Saha–Boltzmann populations |
| `xnfp` | (80, 1006) | Partition-function-weighted populations |
| `edens` | (80,) | Energy density |
| `dopple` | (80,) | Doppler widths (cm s⁻¹) |
| `xabund` | (80, 99) | Mass fractions |

!!! note "Version stamp"
    The `.npz` contains a `meta_npz_conversion_version` integer. `synthe_py` checks this stamp and auto-regenerates the cache if it is older than `MIN_NPZ_CONVERSION_VERSION` (currently 3). Set `PY_DISABLE_AUTO_NPZ_REFRESH=1` to disable this behavior.

### `.spec` — Synthetic Spectrum

A whitespace-delimited text file with three columns: wavelength (nm, vacuum), total flux, and continuum flux. See [Output Files](../user-guide/output-files.md) for reading and plotting examples.

## Cache Invalidation Strategy

pykurucz uses a hierarchical caching strategy to avoid redundant computation:

1. **`.npz` cache** — Tied to a specific `.atm` file by content (via version stamp). If the `.atm` changes, the `.npz` is regenerated.
2. **Line list cache** — `synthe_py` can cache compiled line records (fort.12-style binaries) to disk. Use `--cache DIR` to enable this.
3. **Molecular binary cache** — Schwenke TiO and Partridge–Schwenke H₂O binaries are read once per synthesis and held in memory.

In words:

- The `.npz` cache is bound to a specific `.atm` by a content/version hash;
  if the `.atm` changes or the cache version is too old, the cache is
  auto-regenerated before synthesis runs.
- The atomic line cache (compiled fort.12-style binaries) is reused across
  runs when `--cache DIR` is set.
- Molecular binary catalogues (Schwenke TiO, Partridge–Schwenke H₂O) are
  loaded once into memory at the start of a synthesis and held until exit.

!!! tip "Manual cache management"
    If you are running many syntheses from the same atmosphere, generate the `.npz` once explicitly with `convert_atm_to_npz.py` and pass it via `--npz`. This avoids the overhead of repeated auto-refresh checks.

## Inspecting Intermediate Outputs

### Debug state from `atlas_py`

Pass `--debug-state debug.npz` to `atlas_py.cli` to dump the full internal state after the final iteration. This includes:

- All opacity arrays (`abross_out`, `tauros_out`, `prad_out`, `flxrad_out`)
- Convection quantities (`flxcnv_out`, `vconv_out`, `grdadb_out`)
- Temperature-correction diagnostics (`dtflux_out`, `dtlamb_out`, `t1_out`)
- NLTE accumulators (if enabled)

```python
import numpy as np
debug = np.load("debug.npz")
print(debug["temperature"])
print(debug["abross_out"])
```

### Diagnostics from `synthe_py`

Use `--diagnostics diag.npz` to save per-wavelength opacity and source function arrays. This is useful for investigating specific lines or spectral regions.

## Data Flow Summary

| Stage | Input | Output | Cached? |
|---|---|---|---|
| Emulator | 4 stellar params | `_warmstart.atm` | No |
| `atlas_py` | `.atm` (warm-start or external) | `.atm` (iterated) | No |
| `convert_atm_to_npz` | `.atm` | `.npz` | Yes (reused) |
| `synthe_py` | `.npz` + line list | `.spec` | No |

## Next Steps

- See [Output Files](../user-guide/output-files.md) for code snippets to read each format in Python.
- Read [Fortran Parity](fortran-parity.md) to understand how intermediate outputs are validated.

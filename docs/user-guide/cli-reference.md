# CLI Reference

pykurucz provides four command-line entry points, each covering a distinct stage of the pipeline. This page documents every flag, option, and environment variable.

---

## `pykurucz.py` — End-to-End Pipeline

Orchestrates the full Stellar Parameters workflow: emulator → `atlas_py` → `synthe_py`.

```bash
python pykurucz.py --teff <Teff> --logg <logg> [options]
```

### Required arguments

| Flag | Type | Description |
|---|---|---|
| `--teff` | float | Effective temperature (K) |
| `--logg` | float | Surface gravity (\(\log_{10}\) cgs) |

### Stellar parameters

| Flag | Default | Description |
|---|---|---|
| `--mh` | 0.0 | Overall metallicity [M/H] (dex) |
| `--am` | 0.0 | Alpha enhancement [α/M] (dex) |
| `--vturb` | 2.0 | Microturbulent velocity (km/s) |
| `--abund` | — | Individual element offset `ELEM:OFFSET` (repeatable). Example: `--abund Fe:-1.0` |

### Wavelength grid

| Flag | Default | Description |
|---|---|---|
| `--wl-start` | 300.0 | Start wavelength (nm) |
| `--wl-end` | 1800.0 | End wavelength (nm) |
| `--resolution` | 300000.0 | Resolving power \(\lambda / \Delta\lambda\) |

### Atmosphere iteration

| Flag | Default | Description |
|---|---|---|
| `--atlas-iterations` | 30 | Maximum `atlas_py` iterations |
| `--atlas-convergence-epsilon` | 1e-3 | Early-stop threshold on column changes |
| `--atlas-convergence-min-iterations` | 5 | Minimum iterations before early stop |
| `--atlas-convergence-consecutive` | 1 | Consecutive converged iterations required |
| `--no-atlas-convergence` | — | Disable early stopping; force all iterations |

### Molecular lines

| Flag | Description |
|---|---|
| `--no-molecular-lines` | Disable all molecular line opacity |
| `--no-tio` | Exclude Schwenke TiO |
| `--no-h2o` | Exclude Partridge–Schwenke H₂O |

!!! warning "H₂O default differs from `synthe_py.cli`"
    `pykurucz.py` ships with **H₂O on by default** (`include_h2o=True`); the
    standalone `synthe_py.cli` ships with **H₂O off by default** to match the
    Fortran reference compilation (see the
    [`synthe_py.cli` molecular-lines section](#python-m-synthe_pycli-spectrum-synthesis)
    below). When chasing parity bugs against the Fortran outputs, double-check
    which CLI you're using.

### Output

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `results/` | Root directory for all outputs |

---

## `python -m atlas_py.cli` — Atmosphere Iteration

Runs the Python ATLAS12 engine on an existing `.atm` file.

```bash
python -m atlas_py.cli <atm_file> --output-atm <out.atm> [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `atm` | Input `.atm` file |

### Core options

| Flag | Default | Description |
|---|---|---|
| `--output-atm` | (required) | Output `.atm` path |
| `--deck` | — | ATLAS12 control deck (stdin input file) |
| `--iterations` | 1 | Number of outer iterations |
| `--enable-molecules` | — | Enable molecular POPS path (`MOLECULES ON`) |
| `--molecules` | — | Path to `molecules.new` / `molecules.dat` |

### Line catalog inputs (for `SELECTLINES`)

| Flag | Description |
|---|---|
| `--fort11` | Kurucz lowlines binary (fort.11) |
| `--fort111` | Kurucz lowlines-observed binary (fort.111) |
| `--fort21` | Kurucz hilines binary (fort.21) |
| `--fort31` | Kurucz diatomics binary (fort.31) |
| `--fort41` | Kurucz TiO binary (fort.41) |
| `--fort51` | Kurucz H₂O binary (fort.51) |
| `--fort61` | Kurucz H₃⁺ binary (fort.61) |
| `--line-selection-bin` | Precomputed fort.12 binary (skips `SELECTLINES`) |
| `--nlteline-bin` | NLTE line binary (fort.19) |

### Convergence

| Flag | Default | Description |
|---|---|---|
| `--convergence-epsilon` | — | Early-stop threshold on physical column changes |
| `--convergence-min-iterations` | 5 | Minimum iterations before early stop |
| `--convergence-consecutive` | 1 | Consecutive converged iterations required |

### Diagnostics

| Flag | Description |
|---|---|
| `--debug-state` | Path to write internal EOS state arrays (`.npz`) |

### Python-only knobs (not exposed on the CLI)

The `AtlasConfig` dataclass in `atlas_py/config.py` exposes a few additional
fields that the command-line wrapper does not surface but that you can set
when driving `atlas_py` from Python:

| Field | Default | Purpose |
|---|---|---|
| `enable_convection` | `True` | Toggle the mixing-length convection step in the temperature correction |
| `enable_scattering` | `True` | Toggle electron / Rayleigh scattering in the radiative transfer |
| `print_level` | `1` | Verbosity of `prnt` output (Fortran-equivalent print cards) |
| `punch_level` | `1` | Verbosity of `punch` output (Fortran-equivalent saved cards) |
| `log_level` | `"INFO"` | Python `logging` level — `DEBUG`/`INFO`/`WARNING`/`ERROR` |

These default to ATLAS12-faithful settings; tweak them only if you know what
the corresponding Fortran knob does.

---

## `python -m synthe_py.cli` — Spectrum Synthesis

The core SYNTHE engine: line opacity + radiative transfer.

```bash
python -m synthe_py.cli <model> <atomic_catalog> [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `model` | Model atmosphere file (`.atm` or `.npz`) |
| `atomic` | Atomic line catalog (e.g., `lines/gfallvac.latest`) |

### Wavelength and sampling

| Flag | Default | Description |
|---|---|---|
| `--wl-start` | 300.0 | Start wavelength (nm) |
| `--wl-end` | 1800.0 | End wavelength (nm) |
| `--resolution` | 300000.0 | Resolving power |
| `--no-vacuum` | — | Treat wavelengths as air instead of vacuum |
| `--cutoff` | 1e-3 | Opacity cutoff factor (fraction of continuum) |
| `--linout` | 30 | Line output control flag |

### Atmosphere and caching

| Flag | Default | Description |
|---|---|---|
| `--npz` | auto | Explicit path to preprocessed `.npz` file |
| `--cache` | — | Optional directory for cached line data |

### Molecular lines

| Flag | Default | Description |
|---|---|---|
| `--no-molecular-lines` | — | Disable all molecular line opacity |
| `--molecules-dir` | auto | Directory containing Kurucz ASCII molecular files (repeatable) |
| `--no-tio` | — | Exclude Schwenke TiO |
| `--h2o` | — | Include Partridge–Schwenke H₂O (default off for Fortran parity) |
| `--no-h2o` | — | Exclude H₂O (default behavior) |
| `--tio-bin` | auto | Explicit path to `schwenke.bin` |
| `--h2o-bin` | auto | Explicit path to `h2ofastfix.bin` |

### Parallelism and performance

| Flag | Default | Description |
|---|---|---|
| `--n-workers` | auto | Number of parallel workers (default: all logical CPUs) |

### Radiative transfer

| Flag | Default | Description |
|---|---|---|
| `--scat-iterations` | 8 | Maximum scattering iterations per frequency |
| `--scat-tol` | 1e-3 | Relative tolerance for scattering convergence |
| `--rhoxj` | 0.0 | Scattering scale height RHOXJ (cm⁻²) |
| `--nlte` | — | Enable NLTE line source handling |

### Other

| Flag | Default | Description |
|---|---|---|
| `--microturb` | 0.0 | Microturbulent velocity (km/s) |
| `--log-level` | INFO | Verbosity: DEBUG, INFO, WARNING, ERROR |
| `--diagnostics` | — | Optional path for diagnostics output |
| `--allow-tfort-runtime` | — | Allow runtime substitution of a Fortran-compiled line cache (`tfort.*` files). Debug-only escape hatch for parity testing against the Fortran SYNTHE; not needed for normal Python-only synthesis. |

### Python-only knobs (not exposed on the CLI)

`SynthesisConfig` (in `synthe_py/config.py`) exposes several extra fields that
the command-line wrapper does not surface. Set them when driving `synthe_py`
from Python:

| Field | Default | Purpose |
|---|---|---|
| `enable_helium_wings` | `True` | Use the tabulated `he1tables.dat` profiles for He I lines (turn off for atomic-only Voigt synthesis) |
| `skip_hydrogen_wings` | `False` | Skip the dedicated hydrogen `HPROF4` profile path (the H lines then fall back to a generic Voigt — used when comparing to codes that don't include the Stark tables) |
| `line_filter` | `True` | Pre-filter the atomic catalog to lines whose centre falls within the requested wavelength range (slight cost saving; set `False` for diagnostics) |
| `wavelength_subsample` | `1` | Stride into the synthesis wavelength grid — `2` means evaluate every other point (useful for fast-iteration debugging; **not** recommended for science) |
| `rhoxj_scale` | `0.0` | Scattering scale-height `RHOXJ` (cm⁻²); also surfaced as `--rhoxj` on the CLI |

---

## `convert_atm_to_npz.py` — Atmosphere Preprocessing

Converts a Kurucz `.atm` file into a cached `.npz` for synthesis.

```bash
python synthe_py/tools/convert_atm_to_npz.py <atm_file> <output.npz> [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `atm_file` | Input Kurucz-format `.atm` file |
| `output.npz` | Output path for the cached archive |

### Options

| Flag | Default | Description |
|---|---|---|
| `--atlas-tables` | `synthe_py/data/atlas_tables.npz` | Path to ATLAS opacity tables |
| `--continua` | `lines/continua.dat` | Path to bound-free edge table |
| `--molecules` | `lines/molecules.dat` | Path to molecular equilibrium data |

!!! tip "Run once per atmosphere"
    The `.npz` cache can be reused for many different wavelength ranges or resolutions. Regenerate only when the `.atm` file changes.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `PYTHONUNBUFFERED=1` | Ensures live log output from child processes (set automatically by `pykurucz.py`) |
| `PY_DISABLE_AUTO_NPZ_REFRESH=1` | Prevents `synthe_py` from auto-regenerating stale `.npz` caches |
| `ATLAS_CONVEC_FD_LOG` | If set, `atlas_py` logs finite-difference convection diagnostics to this path |
| `ATLAS_NMOLEC_EDENS_LOG` | If set, `atlas_py` logs molecular equilibrium electron-density terms |
| `ATLAS_KNU_LOG` | If set, `atlas_py` writes detailed opacity diagnostics at a specific frequency index |

## Next Steps

- See [Existing Atmosphere](from-atmosphere.md) and [Stellar Parameters](from-parameters.md) for practical usage examples.
- Read the [Architecture](../architecture/overview.md) section to understand how the CLI maps to internal modules.

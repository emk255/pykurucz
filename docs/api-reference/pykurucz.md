# pykurucz

Top-level end-to-end API for generating synthetic stellar spectra from stellar parameters.

This module implements the canonical Python pipeline:

```
Teff / logg / [M/H] / [α/M]
    └─► kurucz-a1 emulator  ──► warm-start .atm
                             └─► atlas_py (iterated, MOLECULES ON)
                                 └─► iterated .atm
                                     └─► synthe_py.cli
                                         └─► .spec
```

The public helpers below are the building blocks for this flow.  `synthesize()`
chains them into the full user-facing pipeline; the individual helpers can be
imported directly when you need finer control (e.g. skip the emulator and start
from an existing `.atm` file).

See also:

- [Architecture overview](../architecture/overview.md)
- [User guide — Existing Atmosphere](../user-guide/from-atmosphere.md)

---

## `synthesize()`

```python
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
    use_molecular_lines: bool = True,
    include_tio: bool = True,
    include_h2o: bool = True,
    atlas_iterations: int = 30,
    atlas_convergence_epsilon: Optional[float] = 1.0e-3,
    atlas_convergence_min_iterations: int = 5,
    atlas_convergence_consecutive: int = 1,
    n_workers: Optional[int] = None,
) -> Path
```

Generate a synthetic spectrum from stellar parameters.

Runs the canonical Python pipeline:

1. **Emulator warm-start** — predicts a 9-column atmospheric structure with
   the kurucz-a1 neural network.
2. **atlas_py iteration** — self-consistently iterates the atmospheric
   structure with the same physics as Fortran ATLAS12 (MOLECULES ON).
3. **synthe_py synthesis** — computes line opacity and radiative transfer
   to produce the final `.spec` spectrum.

| Parameter | Description |
|-----------|-------------|
| `teff` | Effective temperature (K). |
| `logg` | Surface gravity log₁₀(g) in cgs. |
| `mh` | Overall metallicity [M/H] (default 0.0, solar). |
| `am` | Alpha enhancement [α/M] (default 0.0). |
| `vturb` | Microturbulent velocity in km/s (default 2.0). |
| `wl_start`, `wl_end` | Wavelength range in nm (default 300–1800 nm). |
| `resolution` | Resolving power λ/Δλ (default 300 000). |
| `abundances` | Individual element offsets `{Z: dex_offset_from_solar}`. |
| `output_dir` | Directory for output files (default `results/`). |
| `use_molecular_lines` | Include molecular catalogs (default `True`). |
| `include_tio`, `include_h2o` | Include Schwenke TiO / Partridge–Schwenke H₂O. |
| `atlas_iterations` | Maximum atlas_py outer iterations (default 30). |
| `atlas_convergence_epsilon` | Early-stop threshold on column changes (default 1e-3). |
| `n_workers` | Parallel workers for synthe_py (default: all CPUs). |

**Returns** `Path` to the output `.spec` file.

---

## `emulator_warmstart_atm()`

```python
def emulator_warmstart_atm(
    dest: Path,
    *,
    teff: float,
    logg: float,
    mh: float = 0.0,
    am: float = 0.0,
    vturb: float = 2.0,
    abundances: Optional[Dict[int, float]] = None,
) -> Path
```

Predict a warm-start atmosphere with kurucz-a1 and write it to *dest*.

The emulator is queried with effective `[M/H]` and `[α/M]` derived from
*abundances* (when provided). The resulting 9-column layer structure is written
as a Kurucz-format `.atm` file that `atlas_py` (and the Fortran pipeline) can
consume as a READ DECK6 starting point.

Requires PyTorch (`pip install torch`).

**Returns** `Path` to the written `.atm` file.

---

## `run_atlas_py()`

```python
def run_atlas_py(
    input_atm: Path,
    output_atm: Path,
    *,
    log_path: Path,
    kurucz_root: Optional[Path] = None,
    iterations: int = 1,
    fort12_bin: Optional[Path] = None,
    convergence_epsilon: Optional[float] = None,
    convergence_min_iterations: int = 5,
    convergence_consecutive: int = 1,
) -> Path
```

Run `atlas_py.cli` on *input_atm* and write the iterated atmosphere to
*output_atm*.

`--enable-molecules` is always passed to match the Fortran deck (MOLECULES ON).
Skipping molecular opacity would silently diverge from Fortran parity — this was
the root cause of the historical 90 % Na D discrepancy for cool stars.

| Parameter | Description |
|-----------|-------------|
| `input_atm` | Starting atmosphere (usually the emulator warm-start). |
| `output_atm` | Destination `.atm` for the iterated model. |
| `log_path` | File receiving combined stdout+stderr. |
| `kurucz_root` | Data tree with `lines/` and `molecules/` (default `data/`). |
| `iterations` | Number of outer iterations (default 1). |
| `fort12_bin` | Optional precomputed Fortran `fort12` line-selection binary. |
| `convergence_epsilon` | Optional early-stop threshold on max normalized changes. |

**Returns** `Path` to *output_atm*.

---

## `run_synthe_py()`

```python
def run_synthe_py(
    atm: Path,
    *,
    spec: Path,
    npz: Path,
    log_path: Path,
    wl_start: float = 300.0,
    wl_end: float = 1800.0,
    resolution: float = 300_000.0,
    n_workers: Optional[int] = None,
    kurucz_root: Optional[Path] = None,
    use_molecular_lines: bool = True,
    include_tio: bool = True,
    include_h2o: bool = True,
) -> Path
```

Convert *atm* to NPZ and run `synthe_py.cli` to produce *spec*.

Two stages are executed and logged:

1. `convert_atm_to_npz.py` — populations, molecular equilibrium, continuous
   opacity → `.npz`.
2. `synthe_py.cli` — line opacity + radiative transfer → `.spec`.

| Parameter | Description |
|-----------|-------------|
| `atm` | Input iterated atmosphere. |
| `spec` | Destination `.spec` spectrum file. |
| `npz` | Intermediate `.npz` cache file. |
| `log_path` | Combined log for both stages. |
| `wl_start`, `wl_end` | Wavelength window in nm. |
| `resolution` | Resolving power. |
| `n_workers` | Workers for parallel radiative transfer (default: all CPUs). |
| `use_molecular_lines` | Pass molecular catalogs to synthe_py. |
| `include_tio`, `include_h2o` | Schwenke TiO / Partridge–Schwenke H₂O toggles. |

**Returns** `Path` to *spec*.

---

## Abundance helpers

```python
def compute_abundances(
    mh: float = 0.0,
    am: float = 0.0,
    individual: Optional[Dict[int, float]] = None,
) -> np.ndarray
```

Compute log abundances for elements 3–99.  Starts from solar, scales all metals
by `[M/H]`, applies additional `[α/M]` to alpha elements, and overrides
individual elements when *individual* is provided.

```python
def compute_h(
    mh: float = 0.0,
    am: float = 0.0,
    individual: Optional[Dict[int, float]] = None,
) -> float
```

Compute H abundance: `H = 1 - He - sum(10^A_i)` for metals.

```python
def derive_emulator_params(
    mh: float,
    am: float,
    individual: Optional[Dict[int, float]],
) -> tuple[float, float]
```

Derive effective `[M/H]` and `[α/M]` for the emulator when individual element
offsets are present.  `[M/H]` is proxied by Fe; `[α/M]` is the mean offset of
alpha elements (O, Ne, Mg, Si, S, Ca, Ti).

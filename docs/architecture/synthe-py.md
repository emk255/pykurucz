# `synthe_py` — The Spectrum Synthesis Engine

`synthe_py` is the heart of pykurucz: a pure Python reimplementation of Kurucz's SYNTHE code that computes emergent stellar spectra wavelength by wavelength. Given a model atmosphere and a line list, it evaluates continuum opacity, accumulates line opacity from ~1.3 million atomic lines and molecular transitions, and solves the radiative transfer equation.

## Purpose

The synthesis engine takes a preprocessed atmosphere (`.npz`) and produces a `.spec` file containing:

- Wavelength (nm, vacuum)
- Total emergent flux \(F_\lambda\) (line + continuum)
- Continuum-only flux \(F_{\rm cont}\)

It is agnostic about the origin of the atmosphere — whether from `atlas_py`, MARCS, PHOENIX, or any other source — as long as the input follows the Kurucz `.atm` / `.npz` conventions.

## Key Modules

### `engine/opacity.py`

This is the main synthesis driver (`run_synthesis`). It implements:

1. **Wavelength grid generation** — geometric spacing based on resolving power
2. **Atmosphere loading** — prefers `.npz` caches, auto-refreshes stale versions
3. **Line list compilation** — reads atomic and molecular catalogs, compiles fort.12-style records
4. **Continuum opacity evaluation** — per-wavelength, per-depth
5. **Line opacity accumulation** — Voigt profile wing addition for atomic and molecular lines
6. **Radiative transfer dispatch** — calls `solve_lte_spectrum` for each wavelength batch

!!! fortran "Fortran parity in opacity.py"
    The line-opacity kernels (`_accumulate_metal_profile_kernel`, `_accumulate_mol_wings_batch`) are JIT-compiled with Numba and contain inline comments mapping every loop index and branching condition to the original Fortran labels (e.g., `synthe.for 320-350`).

### `engine/radiative.py`

Implements the JOSH solver for the radiative transfer equation. Given total opacity (`acont + aline + sigmac + sigmal`) and source functions, it returns the emergent flux at the top of the atmosphere. See [Radiative Transfer](../physics/radiative-transfer.md) for the physics details.

### `io/lines/`

- `io/lines/atomic.py` — Reads `gfallvac.latest` and other atomic catalogs
- `io/lines/molecular_compiler.py` — Compiles molecular lines from Kurucz ASCII and binary formats (Schwenke TiO, Partridge–Schwenke H₂O)
- `io/lines/compiler.py` — Builds the unified fort.12-style record list used by the opacity kernel

### `physics/`

| Module | Purpose |
|---|---|
| `continuum.py` | H⁻, H I, He, metal, and scattering continuous opacity |
| `profiles.py` | Voigt profile evaluation (Faddeeva function) |
| `hydrogen_wings.py` | Stark-broadened hydrogen line wings |
| `helium_profiles.py` | Tabulated helium line broadening |
| `populations.py` | Saha–Boltzmann and partition functions |
| `voigt_jit.py` | Numba-accelerated Voigt kernels |

## Two-Stage Design

`synthe_py` is deliberately split into two stages:

1. **Preprocessing** (`convert_atm_to_npz.py`) — Solves the equation of state (populations, molecular equilibrium, continuous opacity) once per atmosphere.
2. **Wavelength loop** (`synthe_py.cli`) — Repeats the line-opacity and RT steps for every wavelength point.

This separation means you can synthesize many different wavelength ranges or resolutions from the same atmosphere without recomputing populations.

<div class="pk-diagram">
<svg viewBox="0 0 760 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="synthe_py two-stage pipeline">
  <defs>
    <marker id="pk-arr-syn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--pk-muted)"/>
    </marker>
  </defs>
  <g font-family="Inter, sans-serif" font-size="13" fill="var(--pk-ink)">
    <rect x="20"  y="70" width="120" height="40" rx="8" fill="var(--pk-surface)" stroke="var(--pk-rule-strong)"/>
    <text x="34" y="86" font-weight="600">.atm file</text>
    <text x="34" y="100" font-size="10.5" fill="var(--pk-muted)">model atmosphere</text>

    <rect x="180" y="70" width="170" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
    <text x="194" y="86" font-weight="600">convert_atm_to_npz</text>
    <text x="194" y="100" font-size="10.5" fill="var(--pk-muted)">EOS once per atmosphere</text>

    <rect x="390" y="70" width="120" height="40" rx="8" fill="var(--pk-surface)" stroke="var(--pk-rule-strong)"/>
    <text x="404" y="86" font-weight="600">.npz cache</text>
    <text x="404" y="100" font-size="10.5" fill="var(--pk-muted)">populations</text>

    <rect x="550" y="70" width="120" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
    <text x="564" y="86" font-weight="600">synthe_py</text>
    <text x="564" y="100" font-size="10.5" fill="var(--pk-muted)">wavelength loop</text>

    <rect x="700" y="20"  width="50" height="32" rx="6" fill="var(--pk-ink)"/>
    <text x="725" y="40" text-anchor="middle" font-weight="600" fill="var(--pk-bg)" font-size="11">.spec</text>
    <rect x="700" y="74"  width="50" height="32" rx="6" fill="var(--pk-ink)"/>
    <text x="725" y="94" text-anchor="middle" font-weight="600" fill="var(--pk-bg)" font-size="11">.spec</text>
    <rect x="700" y="128" width="50" height="32" rx="6" fill="var(--pk-ink)"/>
    <text x="725" y="148" text-anchor="middle" font-weight="600" fill="var(--pk-bg)" font-size="11">.spec</text>
    <text x="780" y="40" font-size="10.5" fill="var(--pk-muted)">300–400 nm</text>
    <text x="780" y="94" font-size="10.5" fill="var(--pk-muted)">500–600 nm</text>
    <text x="780" y="148" font-size="10.5" fill="var(--pk-muted)">300–1800 nm</text>
  </g>
  <g stroke="var(--pk-muted)" stroke-width="1.25" fill="none">
    <path d="M140 90 L176 90" marker-end="url(#pk-arr-syn)"/>
    <path d="M350 90 L386 90" marker-end="url(#pk-arr-syn)"/>
    <path d="M510 90 L546 90" marker-end="url(#pk-arr-syn)"/>
    <path d="M670 84 L686 36" marker-end="url(#pk-arr-syn)"/>
    <path d="M670 90 L696 90" marker-end="url(#pk-arr-syn)"/>
    <path d="M670 96 L686 144" marker-end="url(#pk-arr-syn)"/>
  </g>
</svg>
</div>

## Parallelism Model

The most expensive operation in synthesis is the line-opacity accumulation, which is \(\mathcal{O}(N_\lambda \times N_{\rm lines})\). `synthe_py` parallelizes this at two levels:

### Wavelength-level parallelism

The wavelength grid is partitioned into chunks, each processed by a worker in a `ThreadPoolExecutor`. Each worker:

1. Receives its subset of wavelengths
2. Evaluates continuum opacity for its subset
3. Searches the line list for lines near its wavelengths
4. Accumulates Voigt profile wings into a local opacity buffer
5. Solves the RT equation (JOSH) for each wavelength

### Numba JIT within workers

The inner loops (Voigt evaluation, wing accumulation, JOSH integration) are JIT-compiled by Numba. This gives single-thread performance comparable to compiled Fortran, while the process pool exploits multi-core parallelism.

!!! tip "Worker count tuning"
    By default, `synthe_py` uses all logical CPUs. For machines with many cores (e.g., 64+), memory bandwidth can become the bottleneck. In that case, reduce `--n-workers` to the number of physical cores.

## Config System

`synthe_py` uses dataclasses defined in `config.py`:

```python
from synthe_py.config import SynthesisConfig, WavelengthGrid, LineDataConfig

cfg = SynthesisConfig(
    wavelength_grid=WavelengthGrid(
        start=300.0, end=1800.0, resolution=300_000.0
    ),
    line_data=LineDataConfig(
        atomic_catalog=Path("lines/gfallvac.latest"),
        molecular_line_dirs=[Path("data/molecules")],
        include_tio=True,
        include_h2o=False,
    ),
    atmosphere=AtmosphereInput(model_path=Path("model.atm")),
    output=OutputConfig(spec_path=Path("spectrum.spec")),
    cutoff=1e-3,
    n_workers=None,  # auto = all CPUs
)
```

## Next Steps

- Read about [`atlas_py`](atlas-py.md), the atmosphere engine that produces the input `.atm`.
- Explore the [Emulator](emulator.md) for the warm-start stage.
- See [Data Flows](data-flows.md) for how `.atm`, `.npz`, and `.spec` relate.

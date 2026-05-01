# Architecture Overview

pykurucz is organized as a modular, two-stage pipeline: `atlas_py` computes self-consistent model atmospheres, and `synthe_py` synthesizes spectra from them. An optional PyTorch emulator accelerates the first stage. Every module is pure Python, with Numba JIT compilation applied to performance-critical loops.

## System Architecture

The system is a 3-stage pipeline with two equivalent entry points; the
canonical visual is the [pipeline diagram on the home page](../index.md#pipeline-at-a-glance).
The next section ("Design Principles") explains the rationale behind each stage,
and the [Module Relationships](#module-relationships) diagram below shows how the
Python packages cross-reference each other internally.

## Design Principles

### ATLAS12, not ATLAS9: arbitrary abundance patterns

The single most important reason `atlas_py` reimplements ATLAS12 (rather than the more widely distributed ATLAS9) is **opacity sampling for arbitrary per-element abundance patterns**.

ATLAS9 atmospheres are computed against pre-tabulated opacity distribution functions (ODFs) keyed on a small set of bulk-metallicity tags (e.g. solar, $\alpha$-enhanced, scaled by [M/H]). Anything outside that grid — a CEMP star, an Ap star, an iron-poor halo giant with a large $[\alpha$/Fe], any specific element jacked up or knocked down by a dex — is unreachable in ATLAS9 without recomputing the ODF, which most users never do.

ATLAS12 was Kurucz's answer to this: instead of pre-tabulating opacity for a finite set of abundance patterns, it **directly samples opacity** at each iteration from the actual line list, using whatever per-element abundances you specify. The atmosphere structure then adjusts self-consistently to that opacity.

`atlas_py` preserves this property: any combination of `--abund` flags (or any `abundances` dict in the Python API) propagates straight into the opacity sampling loop, the temperature correction, and the converged atmosphere. The downstream synthesis then uses the **same** abundance pattern for line opacity. There is no inconsistency between "the atmosphere's abundances" and "the spectrum's abundances".

### Fortran Parity

The single most important design constraint is that `atlas_py` and `synthe_py` must reproduce the original Fortran ATLAS12 and SYNTHE results to within sub-0.1% flux differences. This is enforced by:

- Line-by-line numerical reimplementation of key subroutines (KAPP, COOLOP, NMOLEC, JOSH, etc.)
- Identical interpolation schemes (linear, log-linear, and spline where Fortran uses them)
- Identical wavelength grids, depth discretizations, and opacity cutoffs
- End-to-end validation runs against the Fortran pipeline on identical inputs

!!! fortran "Reference tracking"
    Every major physics module contains inline comments referencing the corresponding Fortran subroutine name (e.g., `# KAPP subroutine equivalent`). This makes it possible to trace any numerical difference back to its source.

### Two-Stage Pipeline

The clean separation between atmosphere (`atlas_py`) and synthesis (`synthe_py`) reflects the original Kurucz design. It allows users to:

- Substitute their own atmospheres ([Existing Atmosphere](../user-guide/from-atmosphere.md)) without touching the synthesis core
- Reuse a preprocessed `.npz` for many different wavelength ranges or resolutions
- Debug atmosphere and spectrum problems independently

### Molecular Lines On by Default

Unlike many synthesis codes that treat molecular opacity as an optional add-on, pykurucz enables molecular lines automatically when `data/molecules/` is populated. This matches the Fortran ATLAS12 deck, which always runs with `MOLECULES ON`. Disabling molecules requires an explicit flag (`--no-molecular-lines`).

### Parallel Radiative Transfer

The JOSH solver and line-opacity accumulation are the two most expensive operations in spectrum synthesis. Both are parallelized:

- **Line opacity**: Wavelength points are distributed across a process pool; each worker accumulates atomic and molecular Voigt profiles for its subset.
- **Radiative transfer**: Within each wavelength worker, the JOSH integration over depth layers is JIT-compiled by Numba for single-thread speed.

## Module Relationships

The top-level orchestrator `pykurucz.py` calls into three sibling
packages in order:

1. **`emulator/`** — `emulator.py` loads `model.py` (the PyTorch network)
   and `normalization.py`, returning a 9-column warm-start atmosphere.
2. **`atlas_py/`** — `cli.py` invokes `engine/driver.py`, which calls the
   physics kernels in `physics/` and the file readers in `io/` to iterate
   the atmosphere to convergence.
3. **`synthe_py/`** — `cli.py` drives `engine/opacity.py`, which calls the
   radiative-transfer solver `engine/radiative.py`, the physics kernels
   in `physics/`, and the line-list loaders in `io/lines/` to produce the
   spectrum.

The directory tree below shows how the pieces map onto files on disk.

## Codebase Layout

```
pykurucz/
├── pykurucz.py                 # End-to-end orchestrator (Stellar Parameters)
├── synthesize_from_atm.py      # Existing Atmosphere entry point
├── atlas_py/                   # Python ATLAS12 atmosphere engine
│   ├── cli.py                  # CLI argument parsing
│   ├── config.py               # Configuration dataclasses
│   ├── engine/                 # Driver loop, iteration logic
│   ├── io/                     # Atmosphere and molecule readers
│   ├── physics/                # Opacity, EOS, convection, T-correction
│   └── data/                   # Pre-extracted physics tables (.npz)
├── synthe_py/                  # Python SYNTHE spectrum synthesis engine
│   ├── cli.py                  # CLI argument parsing
│   ├── config.py               # Configuration dataclasses
│   ├── engine/                 # Core synthesis loop + RT
│   ├── io/                     # Line list, spectrum, and atmosphere I/O
│   ├── physics/                # Opacity, broadening, populations, profiles
│   ├── tools/                  # convert_atm_to_npz, population runtime
│   └── data/                   # Pre-extracted physics tables (.npz)
├── emulator/                   # ATLAS12 neural warm-start
│   ├── model.py                # PyTorch MLP architecture
│   ├── emulator.py             # Prediction interface
│   ├── normalization.py        # Input/output normalization
│   ├── a_one_weights.pt        # Trained weights
│   └── norm_params.pt          # Normalization parameters
├── lines/                      # Small in-repo line data
│   ├── gfallvac.latest         # ~1.3M atomic transitions
│   ├── continua.dat
│   ├── molecules.dat
│   └── he1tables.dat
└── scripts/
    └── download_data.py        # One-step data fetcher
```

## Performance Model

| Bottleneck | Complexity | Mitigation |
|------------|-----------|------------|
| Line-opacity accumulation | \(\mathcal{O}(N_\lambda \times N_{\rm lines})\) | Numba JIT + process parallelism across wavelength |
| Voigt profile evaluation | \(\mathcal{O}(N_{\rm lines} \times N_\tau)\) | Vectorized SciPy `wofz` (Faddeeva function) |
| Radiative transfer (JOSH) | \(\mathcal{O}(N_\lambda \times N_\tau^2)\) | Numba JIT within each wavelength worker |
| Molecular equilibrium | \(\mathcal{O}(N_{\rm species} \times N_\tau)\) | Precomputed in `convert_atm_to_npz` |
| Saha–Boltzmann populations | \(\mathcal{O}(N_{\rm elem} \times N_{\rm ion} \times N_\tau)\) | Precomputed in `convert_atm_to_npz` |

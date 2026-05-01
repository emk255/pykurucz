# User Guide Overview

This guide explains how to use pykurucz to generate synthetic stellar spectra. The package offers two primary workflows — **Existing Atmosphere** and **Stellar Parameters** — which share the same validated spectrum synthesis core but differ in how the model atmosphere is prepared.

!!! physics "Why ATLAS12 — and what makes pykurucz different"
    Two abundance use-cases come up all the time, and pykurucz handles both in the **same pipeline**:

    - **Bulk metallicity / α-enhancement.** The `[M/H]` and `[α/M]` knobs (`--mh`, `--am`) cover the everyday scaled-solar and α-enhanced cases — halo dwarfs, thick-disk giants, anything where the abundance pattern is a uniform multi-element shift. This is the most common workflow.
    - **Individual-element offsets.** `--abund Fe:-1.0 --abund C:+0.4 …` (or `abundances={26: -1.0, 6: +0.4, ...}` from Python) lets you override any element separately for peculiar patterns — CEMP stars, Ap stars, individual α-elements set independently, r-process-enhanced halo stars.

    The reason to drive ATLAS12 (rather than just SYNTHE on a pre-computed atmosphere grid) is that **either kind of abundance change reshapes the atmosphere itself**: heavy elements absorb in the UV, redistribute flux to the optical, shift the temperature stratification by tens to hundreds of kelvin, and change the electron density. ATLAS12 does direct opacity sampling, so whatever abundance pattern you specify, the atmosphere is **rebuilt self-consistently** with the matching opacity. SYNTHE then synthesises a spectrum from *that* atmosphere, not a borrowed one.

    Whether you go bulk-only (`--mh -1.5 --am 0.3`) or per-element (`--abund …`), every offset propagates into the atmosphere iteration and out the other side. See [Stellar Parameters](from-parameters.md#abundances) for the syntax and [Custom Abundances](../examples/custom-abundances.md) for worked examples.

## High-Level Pipeline

The pipeline has **three stages** that produce a self-consistent synthetic
spectrum from either stellar parameters or an existing model atmosphere:

> **inputs → atmosphere → preprocessing → synthesis → `.spec` file**

See the [pipeline diagram on the home page](../index.md#pipeline-at-a-glance)
for the visual schematic. The next three subsections explain each stage in turn.

### Stage 1 — Atmosphere

The atmosphere stage produces a self-consistent model of the stellar photosphere: temperature, pressure, density, and electron density as functions of optical depth.

- **Existing Atmosphere**: You supply a pre-computed `.atm` file (ATLAS12, MARCS, PHOENIX, etc.). The atmosphere is read directly and passed to preprocessing.
- **Stellar Parameters**: The [`kurucz-a1`](emulator.md) emulator predicts a warm-start atmosphere from four stellar parameters. [`atlas_py`](../architecture/atlas-py.md) then iterates this warm-start with full ATLAS12 physics (opacity, convection, hydrostatic equilibrium) until convergence.

!!! physics "Why iteration matters"
    A model atmosphere must be self-consistent: the temperature structure determines the opacity, which determines the radiative flux, which feeds back into the temperature correction. The emulator provides an excellent guess, but `atlas_py` performs the final physical relaxation to ensure the spectrum is computed from a consistent stratification.

### Stage 2 — Preprocessing

[`convert_atm_to_npz.py`](../architecture/synthe-py.md) reads the `.atm` file and precomputes:

- Saha–Boltzmann populations for all elements and ionization stages
- Molecular equilibrium for ~50 molecular species
- Continuous opacity coefficients (H⁻, H I, He, metals, scattering)
- Doppler widths at every atmospheric layer

These quantities are cached in a `.npz` file so that repeated syntheses with the same atmosphere (but different wavelength ranges or resolutions) can skip this step.

### Stage 3 — Synthesis

[`synthe_py`](../architecture/synthe-py.md) performs the line-by-line spectrum synthesis:

1. For each wavelength point, evaluate the local continuum opacity.
2. Search the atomic line list (~1.3 M transitions) and molecular catalogs for lines near the current wavelength.
3. Compute Voigt profiles and accumulate line opacity.
4. Solve the radiative transfer equation with the [JOSH solver](../physics/radiative-transfer.md).
5. Write the emergent flux and continuum to a `.spec` file.

## Two-Mode Operation Summary

| | Existing Atmosphere | Stellar Parameters |
|---|---|---|
| **Input** | Your own `.atm` file | Stellar parameters (Teff, logg, [M/H], [α/M]) |
| **Atmosphere source** | Pre-computed (external) | Emulator warm-start → `atlas_py` self-consistent iteration |
| **Dependencies** | Core only (NumPy, SciPy, Numba) | Core + PyTorch + `data/` binaries |
| **Atmosphere physics** | Exact (whatever generated the `.atm`) | Full Python ATLAS12 (Fortran-parity) |
| **Best for** | Full control, outside emulator range, or reusing an external atmosphere | End-to-end synthesis straight from stellar parameters |

!!! tip "Existing Atmosphere for exotic abundances"
    If you need highly non-solar abundance patterns (e.g., extreme r-process enhancement), Existing Atmosphere with a model computed by a dedicated code is safer than relying on the emulator's 4-parameter warm-start.

## Where to Go Next

- [Existing Atmosphere](from-atmosphere.md) — detailed guide to synthesis from an existing atmosphere
- [Stellar Parameters](from-parameters.md) — detailed guide to end-to-end synthesis from stellar parameters
- [Output Files](output-files.md) — understanding `.atm`, `.npz`, and `.spec`
- [Emulator](emulator.md) — how the neural warm-start works and when to trust it
- [CLI Reference](cli-reference.md) — complete flag documentation for all entry points

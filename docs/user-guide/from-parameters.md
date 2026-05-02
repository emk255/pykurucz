# Stellar Parameters — End-to-End Spectrum Synthesis

The Stellar Parameters workflow is pykurucz's flagship: you provide
**stellar parameters and an abundance specification**, and the
pipeline returns a self-consistent synthetic spectrum. The atmosphere
is rebuilt with the requested opacity and the spectrum is computed
from *that* atmosphere — not a generic scaled-solar template.

There are two abundance flows, both fully supported and both common:

- **Bulk** — set $\rm[M/H]$ and $\rm[\alpha/M]$ via `--mh` / `--am`.
  The standard knobs for halo / thick-disk / α-rich grids.
- **Per-element** — set any individual element offset via `--abund`
  (repeatable). For carbon-enhanced metal-poor stars, peculiar Ap
  stars, individual α-element overrides, etc.

You can mix the two. The bulk knobs set the abundance for **every metal
that you don't otherwise override**: `--mh` shifts all metals, `--am`
adds an extra offset to the α-elements. For each element listed in
`--abund Z:offset`, that bulk-derived value is **replaced** with
`solar + offset` (the per-element offset is absolute against solar,
not additive on top of `--mh`/`--am`). See [Abundances](#abundances)
below for the syntax.

## The Pipeline

This workflow chains four steps; the canonical visual is on the
[home page](../index.md#pipeline-at-a-glance).

1. **Emulator warm-start** — `kurucz-a1` predicts a 9-column atmospheric
   structure (80 layers) from four scalar parameters → `<stem>_warmstart.atm`.
2. **`atlas_py` iteration** — self-consistently refines the atmosphere with
   full ATLAS12 physics (`MOLECULES ON`) → `<stem>.atm`.
3. **Preprocessing** — `convert_atm_to_npz.py` caches Saha–Boltzmann
   populations and continuous opacity → `<stem>.npz`.
4. **Synthesis** — `synthe_py.cli` does the line-by-line radiative transfer
   → `<stem>_<wl0>_<wl1>.spec`.

!!! physics "Why the emulator is not the final atmosphere"
    The emulator is trained on converged ATLAS12 models, but it does not know about the exact line list or abundance pattern you request. `atlas_py` performs the physical relaxation (opacity → flux → temperature correction → hydrostatic equilibrium) to ensure the atmosphere is consistent with the physics and data actually used in the run.

## When to Use This Workflow

- You want a spectrum from stellar parameters without managing external atmosphere codes.
- Your target lies within the emulator training range (see [Emulator](emulator.md)).
- You need a quick, reproducible pipeline for grid calculations or parameter exploration.

!!! warning "Outside the training range"
    If your parameters fall outside the emulator's training box, the warm-start may be unreliable and `atlas_py` may take longer to converge — or fail to converge at all. In that case, use [Existing Atmosphere](from-atmosphere.md) with an externally computed atmosphere.

## Command-line invocation

```bash
python pykurucz.py --teff <Teff> --logg <logg> [options]
```

The most-used flags at a glance:

| Flag | Default | Purpose |
|---|---|---|
| `--teff`, `--logg` | required | stellar parameters in K and cgs `log g` |
| `--mh`, `--am` | 0.0 | metallicity and α-enhancement (dex) |
| `--vturb` | 2.0 | microturbulent velocity (km/s) |
| `--wl-start`, `--wl-end` | 300, 1800 | wavelength range (nm) |
| `--resolution` | 300 000 | resolving power $\lambda/\Delta\lambda$ |
| `--atlas-iterations` | 30 | maximum `atlas_py` iterations |
| `--no-molecular-lines` | off | disable all molecular line opacity |
| `--abund Fe:-1.0` | — | per-element offset, repeatable |
| `--output-dir` | `results/` | output root |

For the **full enumeration** (convergence knobs, TiO/H₂O toggles, parallelism,
diagnostics, environment variables), see the **[CLI reference](cli-reference.md)**.

## Abundances

Two flows, both first-class and frequently used together.

### Bulk: scaled-solar with `--mh` and `--am`

For the everyday case — halo dwarfs, α-rich giants, anywhere the
target is well-described by a uniform metallicity offset plus a uniform
α-enhancement:

=== "Command line"

    ```bash
    # Metal-poor α-enhanced K giant
    python pykurucz.py --teff 4500 --logg 2.0 \
        --mh -1.5 --am 0.3 \
        --wl-start 400 --wl-end 700
    ```

=== "Python"

    ```python
    from pykurucz import synthesize

    spec_path = synthesize(
        teff=4500, logg=2.0,
        mh=-1.5, am=0.3,
        wl_start=400, wl_end=700,
    )
    ```

`--mh` scales every metal uniformly from solar; `--am` adds an extra
offset to the standard α-elements (O, Ne, Mg, Si, S, Ca, Ti). Both go
straight into the atmosphere iteration *and* the line opacities.

### Per-element: `--abund` for non-scaled-solar patterns

When the target is abundance-peculiar (CEMP, Ap, individual α-elements
set separately, r-process enhancement, etc.), specify per-element
offsets in dex relative to solar. `--abund` is repeatable and stacks
on top of any bulk `--mh` / `--am`:

=== "Command line"

    ```bash
    # CEMP-s star: Fe-poor, C and Ba enhanced (no other α offset needed)
    python pykurucz.py --teff 4800 --logg 1.5 \
        --abund Fe:-2.5 \
        --abund C:+1.2 \
        --abund Ba:+1.0 \
        --wl-start 400 --wl-end 700
    ```

=== "Python"

    ```python
    from pykurucz import synthesize

    spec_path = synthesize(
        teff=4800, logg=1.5,
        abundances={26: -2.5, 6: +1.2, 56: +1.0},
        wl_start=400, wl_end=700,
    )
    ```

The `abundances` dict is keyed on atomic number (Z); the CLI accepts
either symbol (`Fe`) or atomic number (`26`).

### What happens internally

Whether you used the bulk knobs, the per-element overrides, or both:

1. The offsets are combined into an **effective scalar** $\rm[M/H]$ and
   $\rm[\alpha/M]$ so the [emulator](emulator.md) can produce a
   sensible warm-start atmosphere.
2. The **exact** per-element abundances — not the effective scalars —
   are written into the `.atm` abundance card.
3. `atlas_py` iterates with the true abundance pattern, recomputing
   opacity and line blanketing self-consistently.
4. `synthe_py` synthesises lines from the **same** abundance pattern.

So the elements you tweak shape both the atmosphere (via opacity
feedback) and the spectrum (via line strengths). See
[Custom Abundances](../examples/custom-abundances.md) for worked
CEMP / Ap / α-enhanced examples.

## Using it from Python

For non-peculiar stars, the API is straightforward:

```python
from pykurucz import synthesize

spec_path = synthesize(
    teff=5770,
    logg=4.44,
    mh=0.0,
    am=0.0,
    vturb=2.0,
    wl_start=500.0,
    wl_end=510.0,
    resolution=300_000,
    atlas_iterations=30,
    atlas_convergence_epsilon=1e-3,
    n_workers=None,  # defaults to all CPUs
)
```

For abundance-peculiar stars, supply `abundances` (see above).

## Convergence Parameters and Tuning

`atlas_py` iterates the atmospheric structure until either the maximum iteration count is reached or the physical columns converge.

### Default behavior

- **Maximum iterations**: 30
- **Early-stop threshold**: `1e-3` (maximum normalized change across `RHOX`, `T`, `P`, `XNE`, `ABROSS`, `VTURB`)
- **Minimum iterations before stop**: 5
- **Consecutive converged iterations required**: 1

### When to change the defaults

| Goal | Recommended flags |
|---|---|
| Fast diagnostic (not for science) | `--atlas-iterations 1` |
| Maximum fidelity, ignore convergence | `--no-atlas-convergence` |
| Tighter convergence | `--atlas-convergence-epsilon 1e-4` |
| Highly non-solar abundances | Increase `--atlas-iterations` to 50+; consider `--no-atlas-convergence` |

!!! physics "What convergence means"
    The early-stop criterion compares the normalized change in the physical atmosphere columns between successive iterations. A threshold of `1e-3` means no layer changes by more than 0.1% in any of the tracked quantities. This is typically sufficient for spectroscopic applications where the flux differences are well below the noise floor.

## Output Layout

All files are written under `results/` (or `--output-dir`):

```
results/
├── atm/<stem>_warmstart.atm       # emulator prediction
├── atm/<stem>.atm                 # atlas_py iterated atmosphere
├── npz/<stem>.npz                 # preprocessed populations / opacity tables
├── spec/<stem>_<wl0>_<wl1>.spec   # final spectrum
└── logs/<stem>_atlas.log          # atlas_py log
    <stem>_synthe_<wl0>_<wl1>.log  # synthe_py log
```

`<stem>` is formatted as `t{Teff:05d}g{logg:.2f}_mh{eff_mh:+.2f}_am{eff_am:+.2f}`.

## Example: Metal-Poor Giant

```bash
python pykurucz.py \
    --teff 4500 --logg 2.0 --mh -1.5 --am 0.3 \
    --wl-start 400 --wl-end 700 \
    --resolution 50000
```

This produces a low-resolution optical spectrum of a metal-poor K giant with alpha enhancement. The emulator warm-start is derived from the scalar parameters, then `atlas_py` iterates with `MOLECULES ON`, and `synthe_py` includes TiO and H₂O automatically if the data files are present.

## Next Steps

- Read the [Emulator](emulator.md) guide for details on the training range and limitations.
- See [Output Files](output-files.md) to learn how to read `.spec`, `.atm`, and `.npz` in Python.
- Compare with [Existing Atmosphere](from-atmosphere.md) if you need to use an externally computed atmosphere.

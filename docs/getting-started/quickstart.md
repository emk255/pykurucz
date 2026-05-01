# Quickstart

This page walks you through the fastest path to a synthetic spectrum. We will show **Stellar Parameters** — the end-to-end pipeline that starts from stellar parameters — and **Existing Atmosphere** — synthesis from an existing `.atm` file.

!!! tip "Not sure which mode to use?"
    If you have stellar parameters (\(T_{\rm eff}\), \(\log g\), [M/H]) and want a complete spectrum, use **Stellar Parameters**. If you already have a model atmosphere from ATLAS12, MARCS, or PHOENIX, use **Existing Atmosphere**.

## Stellar Parameters — End-to-End from Stellar Parameters

The fastest way to generate a spectrum is to let `pykurucz.py` do everything: emulator warm-start → `atlas_py` iteration → `synthe_py` synthesis.

### 1. Full pipeline (recommended)

```bash
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 500 --wl-end 510
```

This produces:

- `results/atm/t05770g4.44_mh+0.00_am+0.00_warmstart.atm` — emulator prediction
- `results/atm/t05770g4.44_mh+0.00_am+0.00.atm` — iterated atmosphere
- `results/npz/t05770g4.44_mh+0.00_am+0.00.npz` — preprocessed populations
- `results/spec/t05770g4.44_mh+0.00_am+0.00_500_510.spec` — final spectrum

### 2. Python API

```python
from pykurucz import synthesize

spec_path = synthesize(
    teff=5770,
    logg=4.44,
    mh=0.0,
    am=0.0,
    wl_start=500.0,
    wl_end=510.0,
    resolution=300_000,
)
print(f"Spectrum written to: {spec_path}")
```

!!! note "Convergence and early stopping"
    By default `atlas_py` runs up to 30 iterations but stops early when the physical columns (`RHOX`, `T`, `P`, `XNE`, `ABROSS`, `VTURB`) change by less than `1e-3` after at least 5 iterations. See [Stellar Parameters](../user-guide/from-parameters.md) for details on tuning these parameters.

## Existing Atmosphere — Synthesis from a Model Atmosphere

If you already have a Kurucz-format `.atm` file, you can skip the emulator and atmosphere iteration entirely.

### 1. Preprocess the atmosphere

```bash
python synthe_py/tools/convert_atm_to_npz.py your_model.atm results/your_model.npz
```

### 2. Run synthesis

```bash
python -m synthe_py.cli your_model.atm lines/gfallvac.latest \
    --npz results/your_model.npz \
    --spec results/your_model.spec \
    --wl-start 500 --wl-end 510
```

Or use the convenience wrapper:

```bash
python synthesize_from_atm.py your_model.atm --wl-start 500 --wl-end 510
```

!!! tip "Molecular lines are on by default"
    If `data/molecules/` is populated, Schwenke TiO and Partridge–Schwenke H₂O are included automatically. Use `--no-molecular-lines` for atomic-only synthesis.

## What to expect for runtimes

End-to-end runtimes vary substantially with stellar type, wavelength range,
line-list density, and the number of CPU cores you have. As rough orientation:

- The emulator warm-start is essentially instantaneous.
- `atlas_py` iteration is the slow part of the Stellar-Parameters mode and
  scales mostly with the number of iterations to convergence.
- `synthe_py` synthesis cost is dominated by the wavelength range and the
  number of lines that fall in it; it parallelises across CPU cores in
  the wavelength dimension by default.

For reproducible benchmarking, time the same parameters on your own
machine — published numbers depend heavily on hardware that's hard to
match.

## Next Steps

- Read the [First Spectrum](first-spectrum.md) walkthrough for a detailed explanation of each output file.
- Explore the [User Guide](../user-guide/overview.md) to learn about convergence tuning, custom abundances, and resolution choices.
- Check the [CLI Reference](../user-guide/cli-reference.md) for every available flag.

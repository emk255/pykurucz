# Quickstart

This page walks you through the fastest path to a synthetic spectrum. The
**Stellar Parameters** workflow takes $T_{\rm eff}$, $\log g$, and an
abundance specification (bulk and/or per-element) and runs the full
emulator → `atlas_py` → `synthe_py` pipeline. The **Existing Atmosphere**
workflow takes a pre-computed `.atm` file and skips straight to synthesis.

!!! tip "Which mode?"
    Use **Stellar Parameters** when you want abundance changes — bulk or
    per-element — to reshape the atmosphere itself. Use **Existing
    Atmosphere** when you already have a model from ATLAS12, MARCS,
    PHOENIX, or another code that you trust as-is.

## Stellar Parameters — End-to-End from Stellar Parameters

`pykurucz.py` does the whole thing: emulator warm-start → `atlas_py`
iteration → `synthe_py` synthesis. Below are the three usage patterns
you'll hit most often.

### 1a. Bulk metallicity (the everyday case)

```bash
# Metal-poor α-enhanced K giant
python pykurucz.py --teff 4500 --logg 2.0 \
    --mh -1.5 --am 0.3 \
    --wl-start 500 --wl-end 510
```

`--mh` scales every metal uniformly from solar; `--am` adds an extra
offset to the standard α-elements. This produces:

- `results/atm/t04500g2.00_mh-1.50_am+0.30_warmstart.atm` — emulator prediction
- `results/atm/t04500g2.00_mh-1.50_am+0.30.atm` — iterated atmosphere
- `results/npz/t04500g2.00_mh-1.50_am+0.30.npz` — preprocessed populations
- `results/spec/t04500g2.00_mh-1.50_am+0.30_500_510.spec` — final spectrum

### 1b. Per-element abundances (peculiar patterns)

For CEMP, Ap, individual α-element overrides, r-process enhancements,
etc. — pass any number of `--abund SYMBOL:OFFSET_DEX` flags. Per-element
overrides stack on top of any `--mh`/`--am` you also supply:

```bash
# CEMP-s star: Fe-poor, C and Ba enhanced
python pykurucz.py --teff 4800 --logg 1.5 \
    --abund Fe:-2.5 --abund C:+1.2 --abund Ba:+1.0 \
    --wl-start 400 --wl-end 700
```

Internally the offsets are also rolled into an effective scalar
$\rm[M/H]$/$\rm[\alpha/M]$ to seed the warm-start emulator, then the
**exact** per-element pattern is written into the `.atm` file's
abundance card so `atlas_py` and `synthe_py` both see the same numbers.
See [Stellar Parameters](../user-guide/from-parameters.md#abundances)
for the full story.

### 2. Same thing from Python

```python
from pykurucz import synthesize

# Same metal-poor α-enhanced K giant as 1a
spec_path = synthesize(
    teff=4500, logg=2.0,
    mh=-1.5, am=0.3,
    wl_start=500.0, wl_end=510.0,
    resolution=300_000,
)

# Or the per-element CEMP-s case (Z → dex offset)
spec_path = synthesize(
    teff=4800, logg=1.5,
    abundances={26: -2.5, 6: +1.2, 56: +1.0},
    wl_start=400.0, wl_end=700.0,
)
```

!!! note "Convergence and early stopping"
    By default `atlas_py` runs up to 30 iterations but stops early when
    the physical columns (`RHOX`, `T`, `P`, `XNE`, `ABROSS`, `VTURB`)
    change by less than `1e-3` after at least 5 iterations. See
    [Stellar Parameters](../user-guide/from-parameters.md) for details
    on tuning these parameters.

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

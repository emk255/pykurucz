# From an Existing `.atm` File

This is the **synthesis-only** workflow: you supply a model atmosphere
(from any code that can write a Kurucz-format `.atm` file — ATLAS12,
ATLAS9, MARCS, PHOENIX, …), and `synthe_py` produces the emergent
spectrum. The atmosphere is taken as-given; no emulator, no
`atlas_py` iteration.

!!! physics "When this is the right workflow — and what it gives up"
    With a pre-supplied `.atm` file the abundance pattern is **frozen**:
    whatever the input atmosphere was computed for is what you get.
    Tweaking abundances *after the fact* changes only the line opacities;
    the temperature structure, electron density, and continuum opacity
    do **not** respond. That is fine when you trust the input atmosphere
    for your science case (e.g. you generated it with a dedicated 3-D
    or NLTE code, or you are reproducing a published grid), but it is
    *not* the right tool for asking "what does adding 1 dex of carbon
    do to this star?". For that, drive the full pipeline through
    [Stellar Parameters](from-parameters.md), where any per-element
    abundance offset propagates into both the atmosphere and the
    spectrum.

## When to use this workflow

- You have a **library of pre-computed `.atm` files** and want spectra
  from them (grid extensions, observational mock catalogues, etc.).
- Your target sits **outside the [emulator's training range](emulator.md#training-range)**
  ($T_{\rm eff} < 2500$ K, $\log g > 5.5$, etc.) and you want to
  ingest an atmosphere from a code that does cover those regimes.
- You need **3-D, NLTE, or otherwise non-LTE-1-D atmospheres** that
  ATLAS12 cannot compute — bring them from the dedicated code, ingest
  here.
- You're **benchmarking** pykurucz against an external atmosphere code
  on shared inputs.

!!! tip "Dependency-light"
    Existing-atmosphere mode needs only the core scientific Python stack
    (NumPy, SciPy, Numba). PyTorch and the 3.9 GB `gfpred29dec2014.bin`
    binary are **not** required. If you only ever use this mode, run
    `python scripts/download_data.py --synthe-only` and skip the rest.

## Step 1 — preprocess the atmosphere

The `.atm` text file is converted to a `.npz` cache that holds the
populations and continuous opacities at every depth. This costs a few
seconds and can be reused for many different wavelength windows.

```bash
python synthe_py/tools/convert_atm_to_npz.py your_model.atm results/your_model.npz
```

What the converter does:

1. parses the Kurucz-format `.atm` (TEFF, GRAVITY, abundance card,
   `READ DECK6` block);
2. solves Saha–Boltzmann populations for every element and ionisation
   stage at every depth;
3. solves molecular equilibrium for ~50 species using the dissociation
   energies in `data/lines/molecules.dat`
   (see [Molecular Equilibrium](../physics/molecular-equilibrium.md));
4. computes continuous opacity coefficients (H⁻, H I, He, metal b–f,
   Rayleigh, Thomson) on the synthesis grid;
5. writes the lot to a single `.npz`.

!!! note "Run once per atmosphere"
    The `.npz` is keyed on the atmosphere content. If you edit the
    `.atm` (abundances, layer structure, …) you **must** regenerate
    the `.npz`; `synthe_py` checks an internal version stamp and will
    refuse to use a stale cache.

!!! warning "Molecular equilibrium needs `molecules.dat`"
    The preprocessor reads `data/lines/molecules.dat` (or
    `lines/molecules.dat` in the repo root). It ships in the synthesis
    bundle of `download_data.py`, so you usually have it; if you don't,
    re-run the data downloader.

## Step 2 — synthesise

With the `.npz` ready:

```bash
python -m synthe_py.cli your_model.atm lines/gfallvac.latest \
    --npz results/your_model.npz \
    --spec results/your_model.spec \
    --wl-start 300 --wl-end 1800 \
    --resolution 300000
```

### Most-used flags

| Flag | Default | Purpose |
|---|---|---|
| `model` | required | path to the `.atm` (provides abundances, structure, header) |
| `atomic` | required | atomic line catalogue, normally `lines/gfallvac.latest` |
| `--npz` | auto | the cache from Step 1 (auto-located by sibling path if omitted) |
| `--spec` | `spectrum.spec` | output `.spec` path |
| `--wl-start`, `--wl-end` | 300, 1800 | wavelength range in nm |
| `--resolution` | 300 000 | resolving power $\lambda/\Delta\lambda$ |
| `--no-molecular-lines` | off | drop molecular line opacity |
| `--no-tio`, `--no-h2o` | off | drop those specific binary catalogues |
| `--n-workers` | auto | parallel workers (defaults to all logical CPUs) |
| `--cutoff` | 1e-3 | opacity cutoff (fraction of continuum); matches Fortran SYNTHE default |

For the **full enumeration** (NLTE, scattering tolerance, microturbulence,
log level, environment variables), see the
[CLI reference](cli-reference.md#python--m-synthe_pycli--spectrum-synthesis).

!!! tip "Molecules are on by default"
    If `data/molecules/` is populated, molecular catalogues are
    discovered automatically. Use `--no-molecular-lines` to disable the
    lot, or `--no-tio` / `--no-h2o` to drop just one binary while
    keeping the ASCII molecular catalogues.

### Convenience wrapper

If you want the two steps in one command:

```bash
python synthesize_from_atm.py your_model.atm --wl-start 300 --wl-end 1800
```

This script calls `convert_atm_to_npz.py` and `synthe_py.cli` for you
with sensible defaults and writes outputs under `results/`.

## Worked example: ATLAS12 → pykurucz

Suppose you have a solar atmosphere `sun.atm` from a legacy ATLAS12 run:

```bash
# 1. preprocess once
python synthe_py/tools/convert_atm_to_npz.py sun.atm results/sun.npz

# 2. full optical + NIR at default R
python -m synthe_py.cli sun.atm lines/gfallvac.latest \
    --npz results/sun.npz \
    --spec results/sun_300_1800.spec \
    --wl-start 300 --wl-end 1800

# 3. narrow optical chunk (re-uses the same .npz, no recompute)
python -m synthe_py.cli sun.atm lines/gfallvac.latest \
    --npz results/sun.npz \
    --spec results/sun_500_510.spec \
    --wl-start 500 --wl-end 510
```

The third call starts synthesising immediately — the `.npz` cache
shortcuts step 1 entirely.

## Understanding the log output

`synthe_py.cli` prints timing and progress to stdout. Typical lines:

```
INFO: Loading atmosphere from NPZ file: results/your_model.npz
INFO: Wavelength grid: 300.00–1800.00 nm, 45012 points, R=300000
INFO: Found 1234567 atomic lines in range
INFO: Synthesis complete: 45012 points in 45.23 s
```

If you see warnings about missing molecules or zero populations,
verify that `data/molecules/` is populated and that the `.atm`
abundance card is physically reasonable.

## Next steps

- Compare with **[Stellar Parameters](from-parameters.md)** if your
  question is about *how a non-solar abundance pattern shapes the
  atmosphere itself* — that workflow rebuilds the atmosphere with the
  abundances you specify, this one does not.
- Read **[Output Files](output-files.md)** to load `.spec`, `.atm`,
  and `.npz` from Python.
- Skim the **[CLI reference](cli-reference.md)** for advanced flags
  (`--nlte`, `--scat-iterations`, `--microturb`, …).

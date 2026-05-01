# Existing Atmosphere — Synthesis from a Model Atmosphere

The Existing Atmosphere workflow is the most direct path to a synthetic spectrum: you supply a model atmosphere, and `synthe_py` computes the emergent flux. This mode is ideal when you already have atmospheres from ATLAS12, MARCS, PHOENIX, or any other source that can write a Kurucz-format `.atm` file.

## When to Use This Workflow

- You have a library of pre-computed `.atm` files and want to synthesize spectra from them.
- Your target star lies outside the [emulator training range](emulator.md) (e.g., \(T_{\rm eff} < 2500\) K or \(\log g > 5.5\)).
- You need fully self-consistent non-solar abundance patterns that the emulator does not cover.
- You want to benchmark pykurucz against an external atmosphere code.

!!! tip "Existing Atmosphere is dependency-light"
    Existing Atmosphere requires only the core scientific Python stack (NumPy, SciPy, Numba). You do **not** need PyTorch or the `gfpred29dec2014.bin` binary.

## Step 1 — Preprocess the Atmosphere

Before synthesis, the `.atm` file must be converted to a `.npz` cache that holds precomputed populations, molecular equilibrium, and continuous opacity coefficients.

```bash
python synthe_py/tools/convert_atm_to_npz.py your_model.atm results/your_model.npz
```

This script:

1. Parses the Kurucz-format `.atm` file (TEFF, GRAVITY, abundances, DECK6 layers).
2. Solves Saha–Boltzmann populations for all elements and ionization stages.
3. Solves molecular equilibrium for ~50 molecular species using dissociation energies from `molecules.dat`.
4. Computes continuous opacity coefficients (H⁻, H I, He, metals, Rayleigh, Thomson).
5. Writes everything to a `.npz` archive.

!!! note "Run this once per atmosphere"
    The `.npz` file can be reused for many different wavelength ranges or resolutions. If you modify the `.atm` file (e.g., change abundances), you must regenerate the `.npz`.

!!! warning "Molecular equilibrium requires `molecules.dat`"
    The preprocessor reads `data/lines/molecules.dat` (or `lines/molecules.dat` in the repo root) for dissociation energies and equilibrium constants. Ensure this file is present.

## Step 2 — Run Synthesis

With the `.npz` cache ready, launch `synthe_py.cli`:

```bash
python -m synthe_py.cli your_model.atm lines/gfallvac.latest \
    --npz results/your_model.npz \
    --spec results/your_model.spec \
    --wl-start 300 --wl-end 1800 \
    --resolution 300000
```

### Key flags explained

| Flag | Default | Description |
|---|---|---|
| `model` | (required) | Path to the `.atm` file (used for abundance metadata and DECK6 structure) |
| `atomic` | (required) | Atomic line catalog, typically `lines/gfallvac.latest` |
| `--npz` | auto | Path to the preprocessed `.npz` file. If omitted, `synthe_py` looks for a sibling `.npz` or a cached atmosphere in `synthe_py/data/`. |
| `--spec` | `spectrum.spec` | Output path for the synthetic spectrum |
| `--wl-start` | 300 | Start wavelength (nm) |
| `--wl-end` | 1800 | End wavelength (nm) |
| `--resolution` | 300000 | Resolving power \(\lambda / \Delta\lambda\) |
| `--no-molecular-lines` | off | Disable all molecular line opacity |
| `--no-tio` | off | Exclude Schwenke TiO |
| `--no-h2o` | off | Exclude Partridge–Schwenke H₂O |
| `--n-workers` | auto | Number of parallel workers (defaults to all logical CPUs) |
| `--cutoff` | 1e-3 | Opacity cutoff factor. Lines whose opacity falls below this fraction of the local continuum are truncated. Matches Fortran SYNTHE default. |

!!! tip "Molecular lines are on by default"
    If `data/molecules/` is populated, molecular catalogs are discovered automatically. Use `--no-molecular-lines` for atomic-only synthesis, or `--no-tio` / `--no-h2o` to drop specific species.

### Convenience wrapper

For a simplified two-step workflow, use the provided wrapper:

```bash
python synthesize_from_atm.py your_model.atm --wl-start 300 --wl-end 1800
```

This script internally calls `convert_atm_to_npz.py` and `synthe_py.cli` with sensible defaults, writing outputs to `results/`.

## Full Example: ATLAS12 → pykurucz

Suppose you have a solar atmosphere `sun.atm` from a legacy ATLAS12 run:

```bash
# 1. Preprocess
python synthe_py/tools/convert_atm_to_npz.py sun.atm results/sun.npz

# 2. Synthesize full optical+NIR
python -m synthe_py.cli sun.atm lines/gfallvac.latest \
    --npz results/sun.npz \
    --spec results/sun_300_1800.spec \
    --wl-start 300 --wl-end 1800

# 3. Synthesize a narrow chunk for quick inspection
python -m synthe_py.cli sun.atm lines/gfallvac.latest \
    --npz results/sun.npz \
    --spec results/sun_500_510.spec \
    --wl-start 500 --wl-end 510
```

Because the `.npz` is reused, the second synthesis starts immediately without repeating the population calculations.

## Understanding the Log Output

`synthe_py.cli` writes timing and progress information to stdout (or the log file if redirected). Typical lines include:

```
INFO: Loading atmosphere from NPZ file: results/your_model.npz
INFO: Wavelength grid: 300.00–1800.00 nm, 45012 points, R=300000
INFO: Found 1234567 atomic lines in range
INFO: Synthesis complete: 45012 points in 45.23 s
```

If you see warnings about missing molecules or zero populations, verify that `data/molecules/` is populated and that the `.atm` abundances are physically reasonable.

## Next Steps

- Compare with [Stellar Parameters](from-parameters.md) for end-to-end synthesis from stellar parameters.
- Learn about [output file formats](output-files.md) to load `.spec` data into your analysis pipeline.
- Explore the [CLI Reference](cli-reference.md) for advanced flags such as `--nlte`, `--scat-iterations`, and `--microturb`.

<div align="center">

```
                                           ....                                           
                              ......:::::::=*#%*-:.                                       
                        ..:=+##**+=+#%@@@%@%%%%%@%*.                                      
                       .=+#@@@@@%%@@@@%%@@@@@%%%%%%%#*-                                   
                     .-*#%%%%%%%%%%%%%@@%%%%@@@@%%%%%%%*--:                               
                   .=#@@@%%%%%@@@@@@@@@@@@@@@@@@@%%%%%%@@@#.                              
                  -%@@%%%@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%*=:                            
                 -%@%%%%%%%@@@%%%%%%%%%%%@@@@@%@@%%%%%%%%%#%@*.                           
               :+%%%%%%%%%%%%%@@@@@@@@@@@@@%%@@%%%##%%%%%%%%%#%+                          
              =#%%@%%%@%%%@@@@%##****##%%%%@@%%%#####%%%%@@@%#%@*                         
             =%#%%%%@%*=+**+=-:..::-----=++**++++++++*#%%%%%%%%%*:                        
            :%%@@@@@%*-...      ..::--:::::----===++++++#%@@%%%%**-                       
            #%%%%%%%#=-:..  ..    ...::--::---==++**###*+#%%%@@%%#%-                      
            +##%#%%%*=:..  .--::::::::-:--:---+**********++**#%@%##+                      
          .+*##%%%%%*-::.:::==----=====::-===+*+++*****###=%@%%%@%%#=.                    
         ::+#%%%%%@%#=-:-:::---=+*++=-::.:=*+*####%@###*##-*%@@@%@@##+-                   
           :#%%%%%%%%#+=:::.:*++%%*++-:..:-+##%%#******+*+=**%%%%%%%##%=                  
           .=#%##%%%%*:::.. ....:---=:....:=+**###******+=+==*%%#%%%%:.                   
             #**#%%%%*:.. .     .:::......::--**+******++**+=+%#+##*%                     
             -*+#**+#*-..   .....::....  ..-=-=##***+**####*==+++*%#+                     
              :++---++-:..    .   . .:.:-::=#%#+%@%#*++=*##*=-=*##@%*                     
                -..--:=:....... .  .=:.+#--+*%@%%%@%***==##*+-=+#*#%+                     
                  .-: .::........ .=: .:--=+=+*#*#%%%##*=*#+=-=#*#+-                      
                  ..::..:--.  ....-=:-==+*********#%%%%*+*#=--==--.                       
                  ..  .::-=:......-=+**###*##*#%%%%%%%#****+-:.                           
                     ..:::-:::::.:=+##*++===+*##%%%%%%#***+*=:                            
                          ----:-:-=*%+::::-=+++*##%%%%###***=.                            
                     .:-++::::-:-=+*%=--=++*++**#%%%@%%###*=*%#*=-:.                      
               .-=+*#%%@@@#+=---==**#**+==++++**#%%%%##**++:.#@@@@%%#+=-.                 
         .:=+*###%%%%%@%%%@#.:-==-=+++**+=++***##%##*#*+==#- =%%%%@@%%%%%#*+-.            
   .:-=+*########%%%%%%%%%*.  .:----=+***##**#######***++%*::*%%%%%@@%%%%%%%%%#*+-.       
=+*#########%%%%%%%%%%%%##*.   ..:::-==+***#**#%#**++#%%%#-:-%%%%@@@@%@@@@%%%%%%@@%#*+=:. 
#######%%%%%%%%%%%%%%%%##*%=    .:::::---==***+*#*++#%%%#+--#@%%@@@@@@@@%%@@@@%%%%%%%%%%#*
##%%%%%%%%%%%%%%%%%%%%####%.     .::::::---=+**###*%%%%##*-*@%@@@@@@@@@@@@@@%@@@@@@%%%%%%%
%%%%%%%%%%%%%%%%%%#%%####%#        .:-------+####%%%%###*=*@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@%
%%%%%%%%%%%%%%%%%##########.         .:--=--#%%%%%##****+*@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%%%%%%%%%%%%%%%%#########*=              :=+##*=+++===++*%%%%%@@@%@@@@@@@@@@@@@@@@@@@@@@@@
%%%%%%%%%%%%%%%###########: .           :*%%#%#=::::--+*#%%#*%@%%%@@@@@@@@@@@@@@@@@@@@@@@@
%%%%%%%%%%%%%%############. -         :*@@%@@%@@%*-:-+*####+*@%%%%@@@@@@@@@@@@@@@@@@@@@@%%
%%%%%%%%%%%%%############%- ::        #@@%%%%%%%@@%*+*###*+-#%%%%%@@@@%%@@@@@@@@@@@@@@%%%%
%%%%%%%%%@@%%#############*  :        .%#%%%%%@@%%@%+###*=-+%%%%%%@@@@%%%@@@@@@@@@@@@%%%%%
%%%%%#%%%@@%##############%+ :.      .+--%%%@@@@@%%#+*+=---%%%%%%%@@@@@%%%@@@@@@@@@@%%%%%%
```

</div>

<p align="center"><em>
This project is dedicated to the memory of <strong>Robert L. Kurucz</strong> (1944–2025), whose nearly six decades of work at the Center for Astrophysics | Harvard & Smithsonian laid the foundations for modern stellar spectroscopy. Bob's ATLAS and SYNTHE codes, his atomic and molecular line lists, and his freely shared data have been used by thousands of astronomers worldwide, accumulating tens of thousands of citations. He received the AAS Van Biesbroeck Prize in 1992 for "long-term extraordinary or unselfish service to astronomy." Bob spent seven days a week in his office for decades, driven by a simple goal he never stopped pursuing: "I wanted to determine stellar effective temperatures, gravities, and abundances to study solar and stellar evolution. I still want to." This Python reimplementation of his SYNTHE code is our tribute to his extraordinary legacy of open science.
</em></p>

<p align="center"><sub>Portrait rendered from <a href="https://www.cfa.harvard.edu/people/robert-kurucz">CfA biography photo</a> — credit: Center for Astrophysics | Harvard & Smithsonian</sub></p>

---

# pyKurucz — Pure Python Stellar Spectrum Synthesis

**A ground-truth reimplementation of Kurucz's SYNTHE in pure Python** — validated to sub-0.01% agreement with the original Fortran. No Fortran compiler needed. Just Python, NumPy, SciPy, and Numba.

**Authors:** Elliot M. Kim (Cornell) and Yuan-Sen Ting (The Ohio State), building on the original Fortran codes by Robert L. Kurucz (CfA/Harvard & Smithsonian).

## Quick start

```bash
git clone https://github.com/tingyuansen/pykurucz.git
cd pykurucz
pip install -r requirements.txt
mkdir -p results
```

**Synthesize from an existing atmosphere file** (no PyTorch needed):

```bash
# Try it now with a bundled sample atmosphere
python synthe_py/tools/convert_atm_to_npz.py samples/at12_aaaaa_t08250g4.00.atm results/model.npz
python -m synthe_py.cli samples/at12_aaaaa_t08250g4.00.atm lines/gfallvac.latest \
    --npz results/model.npz --spec results/output.spec --wl-start 500 --wl-end 510
```

**Or go end-to-end from stellar parameters** (requires `pip install torch`):

```bash
pip install torch
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 500 --wl-end 510
```

Both produce a `.spec` file with wavelength, flux, and continuum columns. Use a narrow wavelength range (like 500–510 nm above) for a quick first run; the full 300–1800 nm range is slower but covers the complete optical+NIR.


## Why this exists

Bob Kurucz's codes — ATLAS, SYNTHE, DFSYNTHE, WIDTH, BALMER — together with his atomic and molecular line lists, form one of the most consequential software ecosystems in astrophysics. They have been used to analyze spectra from nearly every major telescope and spectroscopic survey for decades, accumulating tens of thousands of citations. But the original Fortran codebase, developed continuously since the 1960s, is increasingly difficult to compile, install, modify, and integrate with modern workflows.

pyKurucz is a line-by-line numerical reimplementation — not a wrapper around Fortran — ensuring that this extraordinary body of work remains accessible, extensible, and usable for the next generation of astronomers. SYNTHE is the first piece. A full Python ATLAS12 is the next milestone: arbitrary stellar parameters → self-consistent atmosphere → spectrum synthesis, all in Python.


## Two modes of operation

Spectrum synthesis requires a **model atmosphere** as input — a description of how temperature, pressure, and density vary with depth in the star. `synthe_py` is agnostic about where that atmosphere comes from. This repository offers two ways to supply one:

| | Mode A | Mode B |
|---|---|---|
| **Input** | Your own `.atm` file | Stellar parameters ($T_{\text{eff}}$, $\log g$, [M/H], [α/M]) |
| **Atmosphere source** | Pre-computed (ATLAS12, MARCS, PHOENIX, etc.) | Neural network emulator (kurucz-a1) |
| **Dependencies** | Core only | Core + PyTorch |
| **Best for** | Full control, non-standard abundances, outside emulator range | Quick exploration, no pre-computed atmosphere available |

### Mode A examples

```bash
# Step 1: Preprocess the atmosphere (populations, molecular equilibrium, opacities)
python synthe_py/tools/convert_atm_to_npz.py your_model.atm results/your_model.npz

# Step 2: Run synthesis
python -m synthe_py.cli your_model.atm lines/gfallvac.latest \
    --npz results/your_model.npz --spec results/your_model.spec \
    --wl-start 300 --wl-end 1800
```

### Mode B examples

```bash
# Solar-type star, full wavelength range
python pykurucz.py --teff 5770 --logg 4.44

# Metal-poor K giant with alpha enhancement
python pykurucz.py --teff 4500 --logg 2.0 --mh -1.5 --am 0.3

# Override individual element abundances (relative to solar)
python pykurucz.py --teff 5770 --logg 4.44 --abund C:+0.5 --abund Fe:-1.0

# Optical only, lower resolution (faster)
python pykurucz.py --teff 5770 --logg 4.44 --wl-start 400 --wl-end 700 --resolution 50000
```

The emulator is trained on 104,269 ATLAS12 models ($T_{\text{eff}}$ 2,500–50,000 K, $\log g$ −1.0–5.5, [M/H] −4.0–+1.5, [α/M] −0.2–+0.6). Outside this range, the code warns you — use Mode A instead.

> **Note**: The emulator approximates the nearest 4-parameter atmospheric model. It cannot self-consistently iterate the atmosphere for arbitrary abundance patterns. If your science depends on this, use **Mode A** with an atmosphere from the full ATLAS12 or another atmosphere code.


## What `synthe_py` does under the hood

The heart of this repository is **`synthe_py/`** — a pure Python reimplementation of Kurucz's SYNTHE spectral synthesis code. Given a model atmosphere and an atomic line list, it computes the emergent stellar spectrum wavelength by wavelength:

1. **Continuum opacity** — H⁻ bound-free/free-free (the dominant source in Sun-like stars), H I bound-free (Karsas & Latter cross-sections), He I/II, metal photoionization, Rayleigh scattering (H, He, H₂), Thomson scattering — interpolated from pre-tabulated arrays following the original KAPP subroutine logic.
2. **Line opacity** — every atomic transition in the Kurucz GFALL catalog (~1.3 million lines) near the current wavelength contributes a **Voigt profile** (thermal Doppler + van der Waals + Stark + radiative broadening). Hydrogen Balmer/Lyman lines get dedicated Stark-broadened profiles (HPROF4); helium lines use tabulated BCS/Griem/Dimitrijević profiles.
3. **Radiative transfer** — the JOSH solver (ported from Kurucz's ATLAS) integrates the transfer equation on a fixed log-τ grid with parabolic optical depth quadrature and Lambda iteration for scattering, yielding both line+continuum $F_\lambda$ and continuum-only $F_{\rm cont}$.

Plus supporting physics: Saha–Boltzmann populations with detailed partition functions, molecular equilibrium for ~300 diatomic species, and Doppler widths at every atmospheric layer.

### Validated against the Fortran original

Both codes were run with identical inputs: same `.atm` files, same line list, same wavelength range (300–1800 nm), same resolution (R=300,000).

| Model | $T_{\text{eff}}$ | $\log g$ | Type | Median Δ | 95th pctl | 99th pctl |
|-------|-------|-------|------|----------|-----------|-----------|
| `t02500g-1.0` | 2500 K | −1.0 | Cool giant | 0.0006% | 0.023% | 0.065% |
| `t04000g5.00` | 4000 K | 5.0 | K dwarf | 0.00004% | 0.002% | 0.006% |
| `t08250g4.00` | 8250 K | 4.0 | A star | 0.0008% | 0.023% | 0.056% |
| `t10250g5.00` | 10250 K | 5.0 | Late B | 0.0008% | 0.020% | 0.047% |
| `t44000g4.50` | 44000 K | 4.5 | O star | 0.002% | 0.109% | 0.224% |


## How the pipeline works

```
  MODE A (own .atm)              MODE B (emulator)
 ──────────────────         ───────────────────────────
                            Stellar parameters
  Your .atm file            (Teff, logg, [M/H], [alpha/M])
  (from ATLAS12,                     │
   MARCS, PHOENIX,                   v
   or any source)           [kurucz-a1 emulator]
        │                   (predicts atmospheric
        │                    structure)
        │                            │
        │                    generated .atm file
        │                   (+ optional abundance
        │                     overrides via --abund)
        │                            │
        └────────────┬───────────────┘
                     │
                     v
          [convert_atm_to_npz.py]
          (populations, molecular equil.,
           continuous opacity coefficients)
                     │
                     v
              [synthe_py.cli]
          (line-by-line radiative transfer)
                     │
                     v
                 .spec file
          (wavelength, flux, continuum)
```

**Stage 1 — Atmosphere**: Mode A reads your `.atm` file directly. Mode B runs the kurucz-a1 emulator to generate one.

**Stage 2 — Preprocessing** (`convert_atm_to_npz.py`): Reads the `.atm` file and computes Saha–Boltzmann populations, molecular equilibrium (~300 species), continuous opacity coefficients (H⁻, H I, He, metals, scattering), and Doppler widths at each atmospheric layer.

**Stage 3 — Synthesis** (`synthe_py.cli`): The core loop. For each wavelength point: evaluate continuum opacity, search the line list, compute Voigt profiles, solve the transfer equation. Outputs wavelength, $F_\lambda$, and $F_{\text{cont}}$.


## Individual element abundances (Mode B only)

When using the emulator (Mode B) and specifying individual element abundances via `--abund`, values are **dex offsets relative to Asplund solar** (e.g., `Fe:-1.0` means [Fe/H] = −1.0). The script automatically derives the best-matching 4-parameter model for the emulator:

- **[M/H]**: proxied by the Fe offset (if given): $[\text{M/H}] \approx [\text{Fe/H}]$
- **[α/M]**: computed as the average offset of alpha elements (O, Ne, Mg, Si, S, Ca, Ti) relative to the derived [M/H]

The emulator predicts the closest atmospheric structure for those derived parameters. Your exact abundances (solar + offsets) are written into the `.atm` file's abundance table, so SYNTHE uses the precise values you requested for all line opacity calculations.

```bash
# Example: metal-poor star with enhanced Mg and Ca
python pykurucz.py --teff 5000 --logg 3.0 --abund Fe:-1.0 --abund Mg:+0.4 --abund Ca:+0.3
# -> derives [M/H]=-1.0, [alpha/M]=+1.13 for the emulator
# -> writes exact Fe, Mg, Ca abundances into .atm for SYNTHE
```

For fully self-consistent treatment where the atmospheric structure itself responds to non-standard abundances, a full Python ATLAS12 is planned as future work.


## Output

Each `.spec` file has three whitespace-delimited columns: `wavelength(nm)  F_lambda  F_continuum`. The **normalized spectrum** is $F_\lambda / F_{\text{cont}}$. Wavelengths are in vacuum nm; fluxes in erg cm⁻² s⁻¹ nm⁻¹.


## Input data

Everything is included in this repository. Nothing needs to be downloaded separately.

### Line list (`lines/`)

| File | Description |
|------|-------------|
| `gfallvac.latest` | Kurucz GFALL atomic line list — ~1.3M transitions with wavelength, $\log gf$, excitation energies, damping constants |
| `continua.dat` | Bound-free absorption edge wavelengths and cross-sections for computing continuous opacity |
| `molecules.dat` | Dissociation energies and equilibrium constants for ~300 molecular species |
| `he1tables.dat` | Tabulated helium line broadening profiles |

### Physics tables (`synthe_py/data/`)

Pre-extracted from the original Fortran binary data arrays:

| File | Contents |
|------|----------|
| `atlas_tables.npz` | Rosseland mean opacity interpolation tables |
| `fortran_data.npz` | Physical constants, element properties, ionization stages |
| `pfsaha_ion_pots.npz` | Ionization potentials for all elements and ionization stages |
| `pfsaha_levels.npz` | Energy levels and statistical weights for partition function calculations |
| `pfiron_data.npz` | Detailed iron-group partition functions |
| `kapp_tables.npz` | Continuous opacity coefficient tables (KAPP subroutine data) |
| `he1_tables.npz` | Helium profile interpolation tables |
| `helium_aux.npz` | Auxiliary helium broadening data |

### ATLAS12 emulator (`emulator/`)

| File | Description |
|------|-------------|
| `a_one_weights.pt` | Trained MLP weights (512-dim stellar encoder + 512-dim tau encoder + predictor) |
| `norm_params.pt` | Input/output normalization parameters (min-max scaling with log transforms) |

### Sample atmospheres and Fortran references

- `samples/` — 5 ATLAS12 `.atm` files spanning 2500–44000 K for testing
- `fortran_specs/` — pre-computed Fortran SYNTHE spectra for the same models (LFS), for validation


## Reproducing the validation

```bash
# Run synthesis for all 5 sample atmospheres
for atm in samples/*.atm; do
    python synthesize_from_atm.py "$atm"
done

# Compare against Fortran reference spectra
for atm in samples/*.atm; do
    stem=$(basename "$atm" .atm)
    python synthe_py/tools/compare_spectra.py \
        results/spec/${stem}_300_1800.spec \
        fortran_specs/${stem}.spec \
        --range 300 1800 --top 5
done

# Generate comparison figures
python joss/generate_joss_figures.py
```


## Package structure

```
pykurucz/
├── pykurucz.py                     # End-to-end: stellar params -> spectrum
├── synthesize_from_atm.py          # Mode A helper: .atm file -> spectrum
├── requirements.txt                # Core dependencies (Python 3.10+)
├── LICENSE                         # MIT License
│
├── synthe_py/                      # SYNTHE engine (no PyTorch needed)
│   ├── cli.py                      # Command-line interface
│   ├── config.py                   # Configuration dataclasses
│   ├── data/                       # Pre-extracted physics tables (.npz)
│   ├── engine/                     # Core synthesis loop + radiative transfer
│   ├── io/                         # Atmosphere, line list, and spectrum I/O
│   ├── physics/                    # Opacity, broadening, populations, profiles
│   └── tools/                      # Utilities (converter, comparator, plotter)
│
├── emulator/                       # ATLAS12 neural network (optional, needs PyTorch)
│   ├── model.py                    # PyTorch MLP architecture
│   ├── emulator.py                 # Prediction interface
│   └── normalization.py            # Input/output normalization
│
├── lines/                          # Atomic/molecular input data
├── samples/                        # Sample ATLAS12 atmospheres (.atm)
├── fortran_specs/                  # Fortran reference spectra for validation
│
└── joss/                           # JOSS paper and figures
    ├── paper.md
    ├── paper.bib
    ├── paper.pdf                   # Auto-built draft PDF
    ├── generate_joss_figures.py
    └── compare_*.png               # Validation figures
```


## CLI reference

### `pykurucz.py` — end-to-end from stellar parameters (requires `pip install torch`)

```bash
python pykurucz.py --teff <Teff> --logg <logg> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--teff` | (required) | Effective temperature (K) |
| `--logg` | (required) | Surface gravity (log10 cgs) |
| `--mh` | 0.0 | Overall metallicity [M/H] — scales all metals uniformly |
| `--am` | 0.0 | Alpha enhancement [α/M] — extra offset for O, Ne, Mg, Si, S, Ca, Ti |
| `--vturb` | 2.0 | Microturbulent velocity (km/s) |
| `--abund` | — | Element offset relative to solar: `--abund Fe:-1.0` for [Fe/H]=−1.0 (repeatable) |
| `--wl-start` | 300 | Start wavelength (nm) |
| `--wl-end` | 1800 | End wavelength (nm) |
| `--resolution` | 300000 | Resolving power $\lambda / \Delta\lambda$ |
| `--output-dir` | results/ | Output directory |

### `synthe_py.cli` — synthesis from `.atm` file (no PyTorch needed)

```bash
python synthe_py/tools/convert_atm_to_npz.py <atm_file> <output.npz>
python -m synthe_py.cli <atm_file> lines/gfallvac.latest --npz <output.npz> --spec <output.spec> [options]
```


## Dependencies

**Core** (both modes): NumPy, SciPy, Numba, Matplotlib — Python 3.10+

```bash
pip install -r requirements.txt
```

**Mode B only** (emulator): PyTorch

```bash
pip install torch
```


## Current limitations and roadmap

pyKurucz faithfully reimplements the *atomic-line* synthesis path through Kurucz's SYNTHE. It is already a fully functional tool for most stellar spectroscopy applications. The remaining gaps define a clear roadmap toward a complete Python replacement for the entire Kurucz suite.

### What works today

- Full atomic line synthesis from any `.atm` file (Mode A)
- Neural network atmosphere emulator for end-to-end synthesis (Mode B)
- All continuous opacity sources (H⁻, H I, He I/II, metals, Rayleigh, Thomson)
- Voigt line profiles with van der Waals, Stark, and radiative broadening
- Dedicated hydrogen Stark-broadened (Balmer/Lyman) and helium tabulated profiles
- Saha–Boltzmann populations, molecular equilibrium (~300 species)
- Individual element abundance overrides with automatic emulator parameter derivation

### What comes next

| Limitation | Impact | Next step |
|---|---|---|
| **No molecular line opacity** | TiO, H₂O, VO, CH, CN bands missing — affects cool stars ($T_{\text{eff}} \lesssim 4000$ K) and the infrared. Negligible for FGK+ in the optical. | Parse Kurucz molecular line catalogs into the existing opacity loop. |
| **No self-consistent atmosphere iteration** | Emulator approximates nearest 4-parameter model; cannot iterate T–P–κ for arbitrary abundance patterns. | Full Python ATLAS12 reimplementation (radiative/convective equilibrium + opacity sampling). |
| **LTE only** | NLTE effects matter for specific lines (Li I, Na D, O I triplet) in metal-poor/hot stars. | Ingest departure coefficients from external NLTE codes as correction factors. |
| **1D plane-parallel geometry** | Breaks down for evolved giants with extended atmospheres. | Ingest 3D model atmospheres (Stagger, CO⁵BOLD) as stratifications for post-processing. |

## Relation to tingyuansen/kurucz

[tingyuansen/kurucz](https://github.com/tingyuansen/kurucz) provides the original Fortran ATLAS12 + SYNTHE pipeline with pre-compiled binaries, plus the kurucz-a1 emulator. **This repository** (`pykurucz`) replaces the Fortran SYNTHE with pure Python and bundles the kurucz-a1 emulator weights for convenience. If you need Fortran ATLAS12 or DFSYNTHE, use the `kurucz` repo; if you want a pure Python synthesis engine, use this one.


## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use pyKurucz in your research, please cite the JOSS paper:

```bibtex
@article{kim2026pykurucz,
  title   = {pyKurucz: A Pure Python Reimplementation of Kurucz SYNTHE
             for Stellar Spectrum Synthesis},
  author  = {Kim, Elliot M. and Ting, Yuan-Sen},
  journal = {Journal of Open Source Software},
  year    = {2026},
  note    = {in review}
}
```

If you use the kurucz-a1 atmosphere emulator (Mode B), please also cite:

```bibtex
@article{li2025kurucza1,
  title   = {Differentiable Stellar Atmospheres with Physics-Informed
             Neural Networks},
  author  = {Li, Jiadong and Jian, Mingjie and Ting, Yuan-Sen
             and Green, Gregory M.},
  journal = {arXiv e-prints},
  year    = {2025},
  eprint  = {2507.06357},
  url     = {https://arxiv.org/abs/2507.06357}
}
```

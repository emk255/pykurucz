# `atlas_py` — The Atmosphere Engine

`atlas_py` is a pure Python reimplementation of the ATLAS12 stellar atmosphere code. It iterates a model atmosphere to self-consistency, solving hydrostatic equilibrium, the equation of state, opacity, convection, and the temperature-correction equation.

## Why ATLAS12 (and not ATLAS9)

The point of reimplementing ATLAS**12** specifically — rather than the more widely distributed ATLAS9 — is **how opacity is computed**.

- **ATLAS9** reads pre-tabulated **opacity distribution functions** (ODFs) keyed on a small set of bulk-metallicity tags (solar, α-enhanced, scaled by [M/H]). For targets that sit on one of those tags, this is fast and accurate. For *off-grid* abundance patterns (CEMP, Ap, individual α-elements set independently, r-process enhancements), the ODF would have to be re-tabulated — which most users never do.
- **ATLAS12** does **direct opacity sampling**: at each iteration, the opacity arrays are evaluated from the actual atomic and molecular line lists *at the abundance pattern you supplied*. The atmosphere then relaxes around *that* opacity.

For pykurucz, the practical consequence is that bulk (`--mh`, `--am`) and per-element (`--abund Z:offset`) abundances are **handled identically by `atlas_py`**: both are converted into the per-element abundance vector that `compute_abundances()` writes into the `.atm` abundance card, and the continuum opacity (`kapcont_baseline`) and line opacity (`linop1` / `xlinop`) modules in `engine/driver.py` read those abundances off the layer-by-layer populations on every iteration. Nothing in the iteration code branches on "is this a scaled-solar or peculiar pattern"; to ATLAS, they are the same.

## Purpose

Given an input atmosphere (either from the emulator warm-start or an external `.atm` file), `atlas_py` computes:

- **Saha–Boltzmann populations** for all elements and ionization stages
- **Molecular equilibrium** for ~50 molecular species (when `MOLECULES ON`)
- **Continuous opacity** (H⁻, H I, He, metals, scattering)
- **Line opacity** via `LINOP1` and `XLINOP`
- **Radiative flux** and **Rosseland mean opacity**
- **Convection** and **temperature corrections**
- **Hydrogen line wings** (HLINOP)

The output is a converged `.atm` file ready for spectrum synthesis.

## Key Modules

### `engine/driver.py`

The top-level driver (`run_atlas`) implements the main iteration loop. Each iteration follows the same eight-step sequence as Fortran ATLAS12:

<div class="pk-diagram">
<svg viewBox="0 0 880 90" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="atlas_py iteration loop">
  <defs>
    <marker id="pk-arr-atlas" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--pk-muted)"/>
    </marker>
  </defs>
  <g font-family="Inter, sans-serif" font-size="12.5" fill="var(--pk-ink)" text-anchor="middle">
    <g>
      <rect x="10"  y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="56" y="38" font-weight="600">POPSALL</text>
      <text x="56" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">populations</text>
    </g>
    <g>
      <rect x="115" y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="161" y="38" font-weight="600">KAPCONT</text>
      <text x="161" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">cont. opacity</text>
    </g>
    <g>
      <rect x="220" y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="266" y="38" font-weight="600">LINOP</text>
      <text x="266" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">line opacity</text>
    </g>
    <g>
      <rect x="325" y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="371" y="38" font-weight="600">ROSS</text>
      <text x="371" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">Rosseland</text>
    </g>
    <g>
      <rect x="430" y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="476" y="38" font-weight="600">RADIAP</text>
      <text x="476" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">radiative flux</text>
    </g>
    <g>
      <rect x="535" y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="581" y="38" font-weight="600">TCORR</text>
      <text x="581" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">T correction</text>
    </g>
    <g>
      <rect x="640" y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="686" y="38" font-weight="600">CONVEC</text>
      <text x="686" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">convection</text>
    </g>
    <g>
      <rect x="745" y="20" width="92" height="40" rx="8" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)"/>
      <text x="791" y="38" font-weight="600">JOSH</text>
      <text x="791" y="54" font-size="10" fill="var(--pk-muted)" font-family="JetBrains Mono, monospace">RT solve</text>
    </g>
  </g>
  <g stroke="var(--pk-muted)" stroke-width="1.25" fill="none">
    <path d="M102 40 L113 40" marker-end="url(#pk-arr-atlas)"/>
    <path d="M207 40 L218 40" marker-end="url(#pk-arr-atlas)"/>
    <path d="M312 40 L323 40" marker-end="url(#pk-arr-atlas)"/>
    <path d="M417 40 L428 40" marker-end="url(#pk-arr-atlas)"/>
    <path d="M522 40 L533 40" marker-end="url(#pk-arr-atlas)"/>
    <path d="M627 40 L638 40" marker-end="url(#pk-arr-atlas)"/>
    <path d="M732 40 L743 40" marker-end="url(#pk-arr-atlas)"/>
  </g>
</svg>
</div>

1. **POPS / POPSALL** — Solve Saha–Boltzmann populations. If molecules are enabled, route through NMOLEC.
2. **KAPCONT** — Evaluate continuous opacity on a frequency grid.
3. **LINOP / XLINOP** — Accumulate line opacity from selected lines (fort.12 / fort.19).
4. **ROSS** — Compute Rosseland mean opacity and optical depth scale.
5. **RADIAP** — Integrate radiative flux and acceleration.
6. **TCORR** — Apply temperature corrections based on flux errors.
7. **CONVEC** — Compute convective flux and velocity (if convection is unstable).
8. **JOSH** — Solve the transfer equation depth-by-depth for each frequency.

!!! fortran "Fortran line references"
    The driver contains inline comments mapping each step to the corresponding Fortran label or subroutine name (e.g., `# atlas12.for line 224`). This makes parity debugging straightforward.

### `physics/*.py`

The `physics/` directory contains the numerical kernels:

| Module | Fortran Equivalent | Purpose |
|---|---|---|
| `popsall.py` | `POPSALL` | Saha–Boltzmann + partition functions |
| `nmolec.py` | `NMOLEC` | Molecular equilibrium solver |
| `kapcont.py` | `KAPCONT` | Continuous opacity evaluation |
| `line_opacity.py` | `LINOP1`, `XLINOP` | Line-opacity accumulation |
| `ross.py` | `ROSS` | Rosseland mean opacity |
| `radiap.py` | `RADIAP` | Radiative flux and acceleration |
| `tcorr.py` | `TCORR` | Temperature correction |
| `convec.py` | `CONVEC` | Convective flux and velocity |
| `josh.py` | `JOSH` | Feautrier-style transfer solver |
| `hydrogen_wings.py` | `HLINOP` | Hydrogen line-wing opacity |
| `selectlines.py` | `SELECTLINES` | Line selection from catalogs |

### `io/*.py`

- `io/atmosphere.py` — Reads and writes Kurucz-format `.atm` files (DECK6, abundance cards, metadata)
- `io/molecules.py` — Parses `molecules.new` / `molecules.dat` for NMOLEC
- `io/readin.py` — Parses ATLAS12 control decks (stdin input files)

## The Iteration Loop in Detail

### 1. Setup and hydrostatic equilibrium

On the first iteration, the driver loads the input `.atm`, builds abundance arrays (`XABUND`), and initializes the runtime state (`AtlasRuntimeState`). If `IFPRES=1` (the default), it integrates hydrostatic equilibrium to update gas pressure:

\[
P(J) = g \cdot \rho_X(J) + P_{\rm rad} + P_{\rm turb} + P_{\rm con}
\]

### 2. Populations

`popsall()` solves the Saha equation for every element and ionization stage at every depth layer. If `enable_molecules=True`, `nmolec()` computes the partial pressures of ~50 molecular species via equilibrium constants and dissociation energies.

### 3. Opacity

`kapcont_baseline()` evaluates continuous opacity sources:

- H⁻ bound-free and free-free
- H I bound-free (Karsas & Latter cross-sections)
- He I / He II photoionization
- Metal photoionization
- Rayleigh scattering (H, He, H₂)
- Thomson scattering

For cool atmospheres, the `COOLOP` path adds CH, OH, and H₂ collisional opacity when molecular populations are present.

### 4. Line opacity

`LINOP1` accumulates atomic line opacity from the selected fort.12 records. `XLINOP` adds NLTE/fort.19 lines if `IFOP(17)=1`. Both use Voigt profiles with thermal Doppler, van der Waals, Stark, and radiative broadening.

### 5. Radiative transfer and temperature correction

The JOSH solver integrates the transfer equation for each frequency, yielding the monochromatic flux `HNU`. `RADIAP` accumulates the total radiative flux and acceleration. `TCORR` compares the integrated flux against the expected Stefan-Boltzmann flux and applies a temperature correction at each layer.

### 6. Convergence check

After `TCORR`, the driver computes the maximum normalized change in the physical columns (`RHOX`, `T`, `P`, `XNE`, `ABROSS`, `VTURB`). If this change is below `--convergence-epsilon` for `--convergence-consecutive` iterations (and at least `--convergence-min-iterations` have passed), the loop exits early.

## Config System

`atlas_py` uses three dataclasses defined in `config.py`:

```python
from atlas_py.config import AtlasConfig, AtlasInput, AtlasOutput

cfg = AtlasConfig(
    inputs=AtlasInput(
        atmosphere_path=Path("warmstart.atm"),
        molecules_path=Path("data/lines/molecules.new"),
        fort11_path=Path("data/lines/gfpred29dec2014.bin"),
        # ... other catalog paths
    ),
    outputs=AtlasOutput(
        output_atm_path=Path("iterated.atm"),
        debug_state_path=Path("debug.npz"),  # optional
    ),
    iterations=30,
    enable_molecules=True,
    convergence_epsilon=1e-3,
)
```

!!! tip "Debug state output"
    Pass `--debug-state debug.npz` to dump the full internal state (populations, opacities, flux arrays) after the final iteration. This is invaluable for depth-by-depth parity comparisons with Fortran.

## Next Steps

- Read about [`synthe_py`](synthe-py.md), the synthesis engine that consumes `atlas_py` outputs.
- Explore the [Emulator](emulator.md) to see how the warm-start is generated.
- See [Fortran Parity](fortran-parity.md) for validation methodology and results.

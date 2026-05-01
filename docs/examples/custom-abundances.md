# Custom Chemical Abundances

Stellar spectroscopy often requires non-solar abundance patterns: $\alpha$-enhanced thick-disk stars, carbon-enhanced metal-poor (CEMP) giants, or chemically peculiar A-type stars. This tutorial shows how to override individual element abundances in pykurucz.

## What You Will Learn

- The abundance dictionary format (`{atomic_number: dex_offset}`)
- How individual offsets interact with `[M/H]` and `[α/M]`
- How abundance changes affect line strengths in the synthetic spectrum
- Practical examples: $\alpha$-enhancement and a CEMP-like star

## Prerequisites

- Comfortable with the [Stellar Parameters](../user-guide/from-parameters.md) pipeline
- Understanding of [M/H] and [α/M] concepts (see [First Spectrum](../getting-started/first-spectrum.md) for a refresher)

## The Abundance Format

pykurucz accepts custom abundances as a Python dictionary mapping **atomic number** to a **dex offset relative to solar**:

```python
abundances = {
    26: -1.0,   # [Fe/H] = -1.0
    12: +0.4,   # [Mg/H] = +0.4
    20: +0.3,   # [Ca/H] = +0.3
}
```

Alternatively, the CLI accepts `--abund` flags using element symbols:

```bash
--abund Fe:-1.0 --abund Mg:+0.4 --abund Ca:+0.3
```

!!! note "Solar reference"
    All offsets are relative to Asplund solar abundances. A value of `0.0` means exactly solar. Positive values enhance the element; negative values deplete it.

### How the offsets combine with `[M/H]` and `[α/M]`

The `abundances` dict **overrides** the automatic `[M/H]` / `[α/M]` scaling for the specified elements. Unspecified elements still follow the global scalings.

For example, with `mh=-1.0`, `am=0.3`, and `abundances={26: -1.0}`:

- **Fe** gets exactly `[Fe/H] = -1.0` (the individual offset wins)
- **O, Mg, Si, S, Ca, Ti** get `[-1.0 + 0.3] = -0.7` from the global scaling
- **All other metals** get `[M/H] = -1.0`

The emulator warm-start derives an *effective* `[M/H]` and `[α/M]` from these offsets so that the predicted atmospheric structure remains reasonable. See the [`derive_emulator_params`](../api-reference/pykurucz.md) API reference for the exact proxy formulas.

## Example 1 — Alpha-Enhanced Metal-Poor Star

Thick-disk and halo stars frequently show enhanced $\alpha$-capture elements (O, Mg, Si, Ca, Ti) at sub-solar metallicity. Rather than using the blanket `--am` flag, we can set precise offsets for each $\alpha$ element.

=== "Python"

    ```python
    from pykurucz import synthesize

    # Alpha-enhanced metal-poor K giant
    spec_path = synthesize(
        teff=4500,
        logg=1.5,
        mh=-1.5,            # Global metal scaling
        am=0.0,             # We will override alphas individually
        abundances={
            8:  -0.9,       # [O/H]  = -0.9  (slightly less enhanced)
            12: -0.7,       # [Mg/H] = -0.7
            14: -0.8,       # [Si/H] = -0.8
            20: -0.7,       # [Ca/H] = -0.7
            22: -0.6,       # [Ti/H] = -0.6
        },
        wl_start=500.0,
        wl_end=550.0,       # Mg b triplet + nearby Fe lines
        resolution=300_000,
        output_dir="results_alpha",
    )
    ```

=== "CLI"

    ```bash
    python pykurucz.py \
        --teff 4500 --logg 1.5 --mh -1.5 \
        --abund O:-0.9 --abund Mg:-0.7 --abund Si:-0.8 \
        --abund Ca:-0.7 --abund Ti:-0.6 \
        --wl-start 500 --wl-end 550 \
        --output-dir results_alpha
    ```

You will see in the terminal output that the effective emulator parameters are derived automatically:

```
      Individual abundances: [O/H]=-0.90, [Mg/H]=-0.70, [Si/H]=-0.80, [Ca/H]=-0.70, [Ti/H]=-0.60
      Derived emulator params: [M/H]=-1.50, [alpha/M]=+0.80
```

The `[alpha/M] = +0.80` is the mean enhancement of the specified alpha elements relative to the effective `[M/H]`. The emulator uses these derived scalars to predict the atmospheric structure, while `atlas_py` and `synthe_py` use the exact per-element abundances for opacity.

### What to look for in the spectrum

The Mg b triplet at ~517 nm and nearby Fe I lines (e.g., 516.7, 519.1 nm) will show a clear contrast:

- **Mg b lines**: Deeper than in a non-alpha-enhanced model because Mg is enhanced relative to Fe.
- **Fe I lines**: Shallower because `[Fe/H] = -1.5` while the overall `[M/H]` is partially offset by the alpha enhancement.

```python
import numpy as np
import matplotlib.pyplot as plt

# Load alpha-enhanced spectrum
wl, fl, ct = np.loadtxt(
    "results_alpha/spec/t04500g1.50_mh-1.50_am+0.80_500_550.spec",
    unpack=True
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wl, fl / ct, lw=0.4, c="C0", label="Alpha-enhanced [M/H]=-1.5")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"$F_\lambda / F_{\rm cont}$")
ax.set_ylim(0, 1.1)
ax.set_xlim(500, 550)
ax.legend()
ax.set_title(r"Mg b Region: $\alpha$-Enhanced Metal-Poor Giant")
plt.tight_layout()
plt.savefig("alpha_enhanced.png", dpi=250)
```

## Example 2 — Carbon-Enhanced Metal-Poor (CEMP) Star

CEMP stars are characterized by `[C/Fe] \gtrsim +1.0` at low metallicity. The enhanced carbon dramatically changes molecular equilibrium, strengthening CH, CN, and C$_2$ bands.

```python
from pykurucz import synthesize

spec_path = synthesize(
    teff=5000,          # Warm giant
    logg=2.0,
    mh=-2.0,            # Very metal-poor
    abundances={
        6:  +1.5,       # [C/H] = -0.5  -> [C/Fe] = +1.5
        7:  +0.5,       # [N/H] = -1.5  -> [N/Fe] = +0.5
        8:  +0.5,       # [O/H] = -1.5  -> [O/Fe] = +0.5
    },
    wl_start=400.0,     # CH G-band + CN violet bands
    wl_end=450.0,
    resolution=300_000,
    output_dir="results_cemp",
)
```

!!! physics "Molecular equilibrium feedback"
    Raising carbon abundance increases the partial pressure of CO, CH, CN, and C$_2$ at every atmospheric layer. Because CO is extremely stable, much of the additional carbon locks into CO, but the remaining free carbon still enhances CH and C$_2$ significantly. The CN bands at 388 and 421 nm should appear much stronger than in a normal metal-poor star of the same `[M/H]`.

You will see:

- The **CH G-band** at 430.5 nm is broad and deep, resembling a solar-like G-band despite the low metallicity.
- The **CN violet bands** near 388 and 421 nm are prominent.
- Nearby Fe I lines are weak because `[Fe/H] = -2.0`.

## Example 3 — Barium Star (s-process Enhancement)

Barium stars show overabundances of s-process elements such as Ba, La, and Eu. These produce strong lines in the optical that are easily visible even at moderate resolution.

```python
from pykurucz import synthesize

spec_path = synthesize(
    teff=4750,
    logg=2.5,
    mh=0.0,
    abundances={
        56: +1.0,   # [Ba/H] = +1.0  -> [Ba/Fe] = +1.0
        57: +1.0,   # [La/H] = +1.0
        63: +0.8,   # [Eu/H] = +0.8
    },
    wl_start=450.0,
    wl_end=500.0,   # Ba II 455.4 nm, La II 492.1 nm region
    resolution=300_000,
    output_dir="results_ba_star",
)
```

The Ba II resonance line at 455.4 nm should be exceptionally strong — far deeper than in a solar abundance model — while neighboring Fe I lines remain at normal strength.

## Understanding Line-Strength Changes

The equivalent width ($W_\lambda$) of an absorption line scales approximately with abundance in the linear part of the curve of growth:

$$
W_\lambda \propto N_{\rm line} \propto \epsilon_{\rm elem} \times 10^{{\rm [X/H]}}
$$

where $N_{\rm line}$ is the column density of the absorbing species and $\epsilon_{\rm elem}$ is the elemental abundance. In the saturated (flat) part of the curve of growth, increasing abundance broadens the line wings rather than deepening the core.

!!! tip "Visualizing abundance sensitivity"
    Run two syntheses with identical parameters except for one element offset, then ratio the normalized spectra. Regions where the ratio deviates from 1.0 are lines of that element.

## Next Steps

- Return to the [Solar Spectrum](solar-spectrum.md) example to practice line identification before applying custom abundances.
- Read the [pykurucz API reference](../api-reference/pykurucz.md) for details on `compute_abundances` and `derive_emulator_params`.
- Explore [Existing Atmosphere](../user-guide/from-atmosphere.md) if you need to import an externally computed atmosphere (e.g., from MARCS) with custom abundances already baked in.

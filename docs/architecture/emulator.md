# Emulator Architecture

The `kurucz-a1` emulator is a PyTorch neural network that replaces the traditional grey-atmosphere initial guess in ATLAS12 with a data-driven prediction. This page describes the network architecture, training methodology, and how it integrates with the rest of pykurucz.

## Neural Network Architecture

The model is an MLP with two separate encoders that handle qualitatively different inputs:

1. **Global stellar parameters** — \(T_{\rm eff}\), \(\log g\), [M/H], [α/M] — shared across all depth points.
2. **Local optical depth** — \(\tau\) at a specific layer — varies per depth point.

### Stellar parameter encoder

```python
class StellarParamEncoder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128):  # runtime override to 512
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
```

### Tau position encoder

```python
class TauPositionEncoder(nn.Module):
    def __init__(self, embed_dim=64, depth_points=80):  # runtime override to 512
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.GELU()
        )
```

### Combined predictor

The stellar embedding is broadcast to all 80 depth points and concatenated with the tau embedding. A shallow MLP maps the combined 1024-dim vector to the 6 output quantities:

```python
class AtmosphereNetMLPtau(nn.Module):
    def __init__(self, output_size=6, depth_points=80,
                 stellar_embed_dim=128, tau_embed_dim=64):  # runtime override to 512/512
        super().__init__()
        combined_dim = stellar_embed_dim + tau_embed_dim  # 192 default, 1024 at runtime
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(2048, output_size)
        )
```

| Hyperparameter | Value |
|---|---|
| Stellar embed dim | 512 |
| Tau embed dim | 512 |
| Predictor hidden dim | 2048 |
| Dropout | 1% |
| Output | 80 × 6 |

## Normalization Approach

Stellar atmospheres span many orders of magnitude (temperature ~10³ K, pressure ~10¹⁵ dyn cm⁻²). Direct regression on raw values is unstable. The emulator therefore uses:

- **Log transform** for quantities with exponential dynamic range (pressure, density, opacity)
- **Min-max scaling** to \([-1, 1]\) for all inputs and outputs

The normalization parameters are stored in `emulator/norm_params.pt` and loaded by `NormalizationHelper`:

```python
from emulator.normalization import NormalizationHelper, load_norm_params

norm_params = load_norm_params("emulator/norm_params.pt")
normalizer = NormalizationHelper(norm_params)

# Normalize Teff to [-1, 1]
teff_norm = normalizer.normalize("teff", torch.tensor([[5770.0]]))

# Denormalize predicted pressure back to dyn cm^-2
P_pred = normalizer.denormalize("P", model_output[:, :, 2])
```

!!! physics "Why log scaling matters"
    Without log scaling, the optimizer would be dominated by the largest-magnitude outputs (e.g., pressure) and would ignore small but physically important quantities (e.g., radiative acceleration). Log scaling equalizes the loss contribution across all six predicted variables.

## Integration with `atlas_py`

The emulator is integrated into the Stellar Parameters pipeline via `emulator_warmstart_atm()` in `pykurucz.py`:

1. Derive effective [M/H] and [α/M] from any individual abundance overrides.
2. Load the emulator (`load_emulator()`).
3. Call `predict_atmosphere_data()` to get the 80×9 array.
4. Write a Kurucz-format `.atm` file with proper header cards, abundance tables, and DECK6 block.
5. Pass this file to `atlas_py.cli` as the starting atmosphere.

```python
from emulator import load_emulator

emulator = load_emulator()
data = emulator.predict_atmosphere_data(
    teff=5770, logg=4.44, feh=0.0, afe=0.0, vturb=2.0
)
# data.shape == (80, 9)
```

The warm-start `.atm` is saved alongside the iterated output so users can inspect the emulator's raw prediction if desired.

## Training Data

The emulator was trained on 104,269 converged ATLAS12 models spanning:

| Parameter | Range |
|---|---|
| \(T_{\rm eff}\) | 2500 – 50,000 K |
| \(\log g\) | −1.0 – 5.5 |
| [M/H] | −4.0 – +1.5 |
| [α/M] | −0.2 – +0.62 |

Each training sample consists of the 4 stellar parameters plus the 80 \(\tau\) values from the converged model, mapped to the 6 physical quantities at each depth.

!!! warning "Do not extrapolate"
    Predictions outside the training hyper-rectangle are unsupported. The network has no data to generalize from, and the resulting warm-start may be physically implausible.

## Next Steps

- Read the [User Guide](../user-guide/emulator.md) for the usage and code-level docs.
- See [Stellar Parameters](../user-guide/from-parameters.md) for the full pipeline that uses the emulator.
- Explore [`atlas_py`](atlas-py.md) to understand how the warm-start is iterated to convergence.

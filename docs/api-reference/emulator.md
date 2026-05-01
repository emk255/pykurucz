# emulator

Neural-network warm-start package for Kurucz stellar atmospheres.

The emulator replaces the traditional Fortran READ DECK6 starting guess with a
fast, pre-trained MLP that predicts atmospheric structure (RHOX, T, P, XNE,
ABROSS, ACCRAD) from 4D stellar parameters (Teff, logg, [Fe/H], [α/Fe]).  This
allows `atlas_py` to converge in a single outer iteration rather than starting
from a grey approximation.

Most users invoke the emulator indirectly through
[`pykurucz.emulator_warmstart_atm()`](pykurucz.md).

See also:

- [Architecture — Emulator](../architecture/emulator.md)
- [User guide — Emulator](../user-guide/emulator.md)

---

## Atmosphere Emulator
::: emulator.emulator
    options:
      show_root_heading: false
      members_order: source

## Model

::: emulator.model
    options:
      show_root_heading: false
      members_order: source

## Normalization

::: emulator.normalization
    options:
      show_root_heading: false
      members_order: source

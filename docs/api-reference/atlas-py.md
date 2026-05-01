# atlas_py

Pure-Python ATLAS12 atmosphere engine.  `atlas_py` iterates a 1-D stellar
atmosphere in hydrostatic, radiative, and convective equilibrium using the same
physics as the original Fortran ATLAS12 code.

The top-level entry point is `run_atlas()` in the driver.  Most users invoke it
indirectly through [`pykurucz.run_atlas_py()`](pykurucz.md).

See also:

- [Architecture — atlas_py](../architecture/atlas-py.md)
- [Physics — Opacity](../physics/opacity.md)
- [Physics — Radiative Transfer](../physics/radiative-transfer.md)

---

## Configuration

::: atlas_py.config
    options:
      show_root_heading: false
      members_order: source

## Engine Driver

::: atlas_py.engine.driver
    options:
      show_root_heading: false
      members_order: source

## I/O — Atmosphere

::: atlas_py.io.atmosphere
    options:
      show_root_heading: false
      members_order: source

## I/O — READIN Deck

::: atlas_py.io.readin
    options:
      show_root_heading: false
      members_order: source

## Physics — Hydrostatic Equilibrium

::: atlas_py.engine.hydrostatic
    options:
      show_root_heading: false
      members_order: source

## Physics — Population Orchestration (POPSALL)

::: atlas_py.physics.popsall
    options:
      show_root_heading: false
      members_order: source

## Physics — Molecular Equilibrium (NMOLEC)

::: atlas_py.physics.nmolec
    options:
      show_root_heading: false
      members_order: source

## Physics — Continuum Opacity (KAPP)

::: atlas_py.physics.kapp
    options:
      show_root_heading: false
      members_order: source

## Physics — Temperature Correction (TCORR)

::: atlas_py.physics.tcorr
    options:
      show_root_heading: false
      members_order: source

## Physics — Convection (CONVEC)

::: atlas_py.physics.convec
    options:
      show_root_heading: false
      members_order: source

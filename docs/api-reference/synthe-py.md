# synthe_py

Pure-Python SYNTHE spectrum synthesis engine.  `synthe_py` computes atomic and
molecular line opacity and solves the LTE radiative-transfer equation to produce
a synthetic spectrum from a converged model atmosphere.

The top-level entry point is the opacity engine, which orchestrates wavelength
grid construction, line selection, profile accumulation, and the JOSH radiative
transfer solver.  Most users invoke it indirectly through
[`pykurucz.run_synthe_py()`](pykurucz.md).

See also:

- [Architecture — synthe_py](../architecture/synthe-py.md)
- [Physics — Line Broadening](../physics/line-broadening.md)
- [Physics — Molecular Equilibrium](../physics/molecular-equilibrium.md)

---

## Configuration

::: synthe_py.config
    options:
      show_root_heading: false
      members_order: source

## Engine — Opacity

::: synthe_py.engine.opacity
    options:
      show_root_heading: false
      members_order: source

## Engine — Radiative Transfer

::: synthe_py.engine.radiative
    options:
      show_root_heading: false
      members_order: source

## I/O — Atmosphere

::: synthe_py.io.atmosphere
    options:
      show_root_heading: false
      members_order: source

## I/O — Export

::: synthe_py.io.export
    options:
      show_root_heading: false
      members_order: source

## I/O — spectrv

::: synthe_py.io.spectrv
    options:
      show_root_heading: false
      members_order: source

## Physics — Continuum Opacity

::: synthe_py.physics.continuum
    options:
      show_root_heading: false
      members_order: source

## Physics — Populations

::: synthe_py.physics.populations
    options:
      show_root_heading: false
      members_order: source

## Physics — Line Broadening

::: synthe_py.physics.broadening
    options:
      show_root_heading: false
      members_order: source

## Physics — JOSH Solver

::: synthe_py.physics.josh_solver
    options:
      show_root_heading: false
      members_order: source

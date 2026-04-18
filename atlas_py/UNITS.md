# atlas_py Unit Audit (Phase 1)

This document tracks units used by the current atlas_py implementation and ties
them to the corresponding `atlas12.for` definitions.

## Global constants and COMMON semantics

- `K = 1.38054E-16` erg/K (`atlas12.for` comment line ~100)
- `H = 6.6256E-27` erg*s (`atlas12.for` comment line ~101)
- `C = 2.99792458E10` cm/s (`atlas12.for` comment line ~102)
- `ATMASS = 1.660E-24` g (`atlas12.for` comment line ~104)

## Atmosphere columns (`READ DECK6`)

From `atlas12.for` header and `READ DECK6` output:

- `RHOX`: g cm^-2
- `T`: K
- `P`: dyn cm^-2
- `XNE`: cm^-3
- `ABROSS`: cm^2 g^-1
- `ACCRAD`: cm s^-2
- `VTURB`: cm s^-1

## Frequency/wavelength

From `atlas12.for` lines ~327-329:

- `WAVE`: nm
- `FREQ = 2.99792458D17 / WAVE`: Hz
- `WAVENO = 1.D7 / WAVE`: cm^-1

## Pressure integration

From `atlas12.for` lines ~226-229:

- `P = GRAV * RHOX - PRAD - PTURB - PCON`
- `GRAV`: cm s^-2
- `RHOX`: g cm^-2
- resulting pressure terms in dyn cm^-2

## Population arrays

From `atlas12.for` comments and NELECT/POPS logic:

- `XNF`: number density per volume after POPS scaling path in NELECT context
- `XNFP`: number density divided by partition function in MODE=11 path
- `XNATOM`: total atom number density, cm^-3
- `XABUND`: elemental number fraction (dimensionless)

## Doppler quantities

From `atlas12.for` lines ~275-278:

- `DOPPLE = sqrt(2*TK/mass + VTURB^2) / c` dimensionless (v/c)
- `XNFDOP = XNFP / DOPPLE / RHO`

## Indexing discipline

- Fortran uses 1-based indexing:
  - `XABUND(J,IZ)` -> Python `xabund[j, iz - 1]`
  - `XNF(J,NELION)` -> Python `xnf[j, nelion - 1]`
  - heavy-element packed slots:
    - Z <= 30: triangular packing
    - Z >= 31: starts at slot 496 with stride 5

## Known Phase-1 gaps

- Full `PFSAHA` table logic (`NNN(6,365)` with exact constants) is not yet fully ported.
- `NMOLEC/MOLEC` Newton system is not yet ported.
- `WTMOLE` is currently approximated from abundances and needs exact Fortran parity.


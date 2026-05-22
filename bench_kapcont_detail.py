#!/usr/bin/env python3
"""Add line_profiler-style timing to compute_kapp_continuum frequency loops."""

import time
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from atlas_py.io.atmosphere import load_atm
from atlas_py.physics.kapcont import kapcont_baseline, build_waveset
from atlas_py.physics.kapp import KappAtmosphereAdapter, compute_kapp
from atlas_py.physics.atlas_tables import load_atlas_tables
from atlas_py.physics.populations import pops
from atlas_py.physics.popsall import popsall
from atlas_py.physics.tcorr import init_tcorr

atm_path = Path("samples/at12_aaaaa_t05500g4.00.atm")


def build_adapter(atm):
    from atlas_py.engine.driver import _runtime_from_atm, _build_kapp_adapter
    state = _runtime_from_atm(atm)
    dummy = np.zeros((atm.layers, 1), dtype=np.float64)
    itemp_cache = {"pops_itemp": -1}
    pops(code=0.0, mode=1, out=dummy, ifmol=False, ifpres=True,
         temperature_k=atm.temperature, tk_erg=atm.tk, state=state,
         itemp=1, itemp_cache=itemp_cache)
    popsall(temperature_k=atm.temperature, tk_erg=atm.tk, state=state,
            ifmol=False, ifpres=True, itemp=1, itemp_cache=itemp_cache)
    return _build_kapp_adapter(atm, state)


def main():
    atm = load_atm(atm_path)
    tables = load_atlas_tables()
    teff = float(atm.metadata.get("teff", 5500.0))
    adapter = build_adapter(atm)
    ifop = [1]*20
    tcst = init_tcorr(atm.layers)

    wave_nm, rco = build_waveset(teff)
    freq_hz = 2.99792458e17 / np.maximum(wave_nm, 1e-300)

    # Monkey-patch compute_kapp_continuum to add timing
    import atlas_py.physics.kapp_continuum as kc_mod
    original_fn = kc_mod.compute_kapp_continuum

    # Use Python's line profiler approach: instrument the frequency loops
    # by measuring start/end times at key points
    import cProfile

    # Actually, let's just time each call with selective ifop
    species_flags = {
        "HOP (H bound-free)": [0],
        "H2PLOP (H2+)": [1],
        "HMINOP (H-)": [2],
        "HRAYOP (H scat)": [3],
        "HE1OP (He I)": [4],
        "HE2OP (He II)": [5],
        "HEMIOP (He-)": [6],
        "C1OP/MG1OP/AL1OP/SI1OP/FE1OP (metals)": [8],
        "COOLOP (mol)": [17],
    }

    # Baseline: all species
    t0 = time.perf_counter()
    acont_ref, sigmac_ref, scont_ref = compute_kapp(
        adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop, tcst=tcst
    )
    dt_all = time.perf_counter() - t0
    print(f"All species: {dt_all:.3f}s")

    # None: all flags off
    ifop_none = [0]*20
    t0 = time.perf_counter()
    compute_kapp(adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop_none, tcst=tcst)
    dt_none = time.perf_counter() - t0
    print(f"No species (baseline overhead): {dt_none:.3f}s")

    # Each species individually
    for name, flags in species_flags.items():
        ifop_test = [0]*20
        for f in flags:
            ifop_test[f] = 1
        t0 = time.perf_counter()
        compute_kapp(adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop_test, tcst=tcst)
        dt = time.perf_counter() - t0
        print(f"  {name}: {dt - dt_none:.3f}s (incremental)")

    # Also test: HOTOP (ifop[9])
    for name, idx in [("HOTOP (hot stars)", 9), ("N1OP (nitrogen)", 10),
                       ("LUKEOP (other metals)", 11), ("XCONOP (ross table)", 18)]:
        ifop_test = [0]*20
        ifop_test[idx] = 1
        t0 = time.perf_counter()
        compute_kapp(adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop_test, tcst=tcst)
        dt = time.perf_counter() - t0
        print(f"  {name}: {dt - dt_none:.3f}s (incremental)")


if __name__ == "__main__":
    main()

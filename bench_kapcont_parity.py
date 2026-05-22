#!/usr/bin/env python3
"""Test parity and performance of vectorized kapp_continuum."""

import time
import numpy as np
from pathlib import Path

from atlas_py.io.atmosphere import load_atm
from atlas_py.physics.kapcont import build_waveset
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
    wave_nm, _ = build_waveset(teff)
    freq_hz = 2.99792458e17 / np.maximum(wave_nm, 1e-300)

    # Run 3 times, report best
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        acont, sigmac, scont = compute_kapp(
            adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop, tcst=tcst
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  Run {i+1}: {dt:.3f}s")

    print(f"\nBest: {min(times):.3f}s")
    print(f"acont range: [{acont.min():.6e}, {acont.max():.6e}]")
    print(f"sigmac range: [{sigmac.min():.6e}, {sigmac.max():.6e}]")

    # Per-species timing
    print("\n--- Per-species timing ---")
    ifop_none = [0]*20
    t0 = time.perf_counter()
    compute_kapp(adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop_none, tcst=tcst)
    dt_none = time.perf_counter() - t0
    print(f"Baseline (no species): {dt_none:.3f}s")

    for name, idx in [("HE1OP", 4), ("HE2OP", 5), ("Metals (C/Mg/Al/Si/Fe)", 8),
                       ("LUKEOP", 9), ("HOTOP", 10)]:
        ifop_test = [0]*20
        ifop_test[idx] = 1
        t0 = time.perf_counter()
        compute_kapp(adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop_test, tcst=tcst)
        dt = time.perf_counter() - t0
        print(f"  {name}: {dt - dt_none:.3f}s")


if __name__ == "__main__":
    main()

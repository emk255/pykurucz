#!/usr/bin/env python3
"""Profile kapcont_baseline to find exact bottlenecks in compute_kapp_continuum."""

import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("bench_kapcont")

# Minimal ATLAS setup to call kapcont_baseline
from atlas_py.io.atmosphere import load_atm
from atlas_py.physics.kapcont import kapcont_baseline, build_waveset
from atlas_py.physics.kapp import KappAtmosphereAdapter, compute_kapp
from atlas_py.physics.atlas_tables import load_atlas_tables
from atlas_py.physics.populations import pops
from atlas_py.physics.popsall import popsall
from atlas_py.physics.tcorr import init_tcorr

# Use the same atmosphere from the pipeline tests
from pathlib import Path
atm_path = Path("samples/at12_aaaaa_t05500g4.00.atm")


def build_adapter(atm):
    """Build KappAtmosphereAdapter from an atmosphere."""
    from atlas_py.engine.driver import _runtime_from_atm, _build_kapp_adapter
    state = _runtime_from_atm(atm)
    # Run POPS + POPSALL to populate state
    dummy = np.zeros((atm.layers, 1), dtype=np.float64)
    itemp_cache = {"pops_itemp": -1}
    pops(code=0.0, mode=1, out=dummy, ifmol=False, ifpres=True,
         temperature_k=atm.temperature, tk_erg=atm.tk, state=state,
         itemp=1, itemp_cache=itemp_cache)
    popsall(temperature_k=atm.temperature, tk_erg=atm.tk, state=state,
            ifmol=False, ifpres=True, itemp=1, itemp_cache=itemp_cache)
    return _build_kapp_adapter(atm, state)


def main():
    logger.info("Loading atmosphere...")
    atm = load_atm(atm_path)
    tables = load_atlas_tables()
    teff = float(atm.metadata.get("teff", 5500.0))

    logger.info("Building adapter (POPS + POPSALL)...")
    t0 = time.perf_counter()
    adapter = build_adapter(atm)
    logger.info("  Adapter built in %.3fs", time.perf_counter() - t0)

    ifop = [1]*20  # All opacity sources enabled
    tcst = init_tcorr(atm.layers)

    # Build waveset
    wave_nm, rco = build_waveset(teff)
    freq_hz = 2.99792458e17 / np.maximum(wave_nm, 1e-300)
    logger.info("  nfreq = %d, n_layers = %d", freq_hz.size, atm.layers)

    # Profile kapcont_baseline (which calls compute_kapp)
    logger.info("\n--- kapcont_baseline timing ---")
    t0 = time.perf_counter()
    wave_nm_out, rco_out, acont, sigmac, scont = kapcont_baseline(
        adapter=adapter, teff=teff, atlas_tables=tables, ifop=ifop, tcst=tcst
    )
    dt = time.perf_counter() - t0
    logger.info("kapcont_baseline total: %.3fs", dt)
    logger.info("  acont shape: %s, max: %.3e", acont.shape, np.max(acont))

    # Run again to check repeatability
    logger.info("\n--- Second run ---")
    t0 = time.perf_counter()
    wave_nm_out2, rco_out2, acont2, sigmac2, scont2 = kapcont_baseline(
        adapter=adapter, teff=teff, atlas_tables=tables, ifop=ifop, tcst=tcst
    )
    dt2 = time.perf_counter() - t0
    logger.info("kapcont_baseline total: %.3fs", dt2)

    # Now profile compute_kapp_continuum with cProfile
    logger.info("\n--- cProfile of compute_kapp ---")
    import cProfile
    import pstats
    from io import StringIO

    pr = cProfile.Profile()
    pr.enable()
    acont3, sigmac3, scont3 = compute_kapp(
        adapter=adapter, freq_hz=freq_hz, atlas_tables=tables, ifop=ifop, tcst=tcst
    )
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    logger.info("\n%s", s.getvalue())

    # Also time-based profiling
    s2 = StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(30)
    logger.info("\n%s", s2.getvalue())


if __name__ == "__main__":
    main()

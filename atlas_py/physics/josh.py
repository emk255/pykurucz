"""Radiative transfer solver entry (JOSH/BLOCKJ/BLOCKH).

Fortran reference:
- `atlas12.for` lines ~10335-10602 (`SUBROUTINE JOSH`)
- Uses pretabulated BLOCKJ/BLOCKH matrices extracted from `atlas12.for`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from .josh_math import _deriv, _integ, _map1
from .josh_tables_atlas12 import load_josh_tables

_TABLES_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None = None
_JOSH_DEBUG_XS: np.ndarray | None = None
_KNU_TRACE_COUNTER = 0


def _get_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    global _TABLES_CACHE
    if _TABLES_CACHE is None:
        t = load_josh_tables()
        # Fortran stores CK/COEFJ/COEFH/XTAU in REAL*4.
        ck = np.asarray(t["ck_weights"], dtype=np.float32)
        coefh = np.asarray(t["coefh_matrix"], dtype=np.float32)
        coefj = np.asarray(t["coefj_matrix"], dtype=np.float32)
        # XTAU8 is REAL*8 in atlas12.for JOSH.
        xtau = np.asarray(t["xtau_grid"], dtype=np.float64)
        _TABLES_CACHE = (ck, coefh, coefj, xtau, int(xtau.size))
    return _TABLES_CACHE


_EPS = 1.0e-38
_ITER_TOL = 1.0e-5


@dataclass
class JoshResult:
    """Depth profiles produced by one JOSH solve."""

    taunu: np.ndarray
    snu: np.ndarray
    hnu: np.ndarray
    jnu: np.ndarray
    jmins: np.ndarray
    abtot: np.ndarray
    alpha: np.ndarray
    knu_surface: float
    maxj: int


def josh_depth_profiles(
    *,
    ifscat: int,
    ifsurf: int,
    acont: np.ndarray,
    scont: np.ndarray,
    aline: np.ndarray,
    sline: np.ndarray,
    sigmac: np.ndarray,
    sigmal: np.ndarray,
    rhox: np.ndarray,
    bnu: np.ndarray,
    freq_hz: float = 0.0,
    wave_nm: float = 0.0,
) -> JoshResult:
    """Compute JOSH depth profiles for a single frequency.

    Currently supports the profile mode used by ATLAS iteration (`IFSURF=0`).
    """

    if ifsurf != 0:
        raise NotImplementedError("atlas_py JOSH currently supports only IFSURF=0")
    knu_log_path = os.getenv("ATLAS_KNU_LOG")
    knu_trace_target = 2214

    def _log_knu(branch: str, knu_surface: float, jnu0: float, maxj_val: int) -> int:
        global _KNU_TRACE_COUNTER
        _KNU_TRACE_COUNTER += 1
        if not knu_log_path:
            return _KNU_TRACE_COUNTER
        with open(knu_log_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"KNU,{branch},{float(wave_nm):.8e},{float(freq_hz):.8e},{float(knu_surface):.8e},"
                f"{float(jnu0):.8e},{int(maxj_val):d}\n"
            )
        return _KNU_TRACE_COUNTER
    ck_weights, coefh_matrix, coefj_matrix, xtau_grid, nxtau = _get_tables()

    acont = np.asarray(acont, dtype=np.float64)
    scont = np.asarray(scont, dtype=np.float64)
    aline = np.asarray(aline, dtype=np.float64)
    sline = np.asarray(sline, dtype=np.float64)
    sigmac = np.asarray(sigmac, dtype=np.float64)
    sigmal = np.asarray(sigmal, dtype=np.float64)
    rhox = np.asarray(rhox, dtype=np.float64)
    bnu = np.asarray(bnu, dtype=np.float64)

    n = rhox.size
    abtot = np.maximum(acont + aline + sigmac + sigmal, 1e-300)
    alpha = (sigmac + sigmal) / abtot
    den = acont + aline
    snubar = np.asarray(bnu, dtype=np.float64).copy()
    np.divide(
        acont * scont + aline * sline,
        den,
        out=snubar,
        where=den > 0.0,
    )
    taunu = _integ(rhox, abtot, float(abtot[0] * rhox[0]))

    snu = np.zeros(n, dtype=np.float64)
    hnu = np.zeros(n, dtype=np.float64)
    jnu = np.zeros(n, dtype=np.float64)
    jmins = np.zeros(n, dtype=np.float64)
    xsbar8 = np.full(nxtau, np.nan, dtype=np.float32)
    xalpha8 = np.full(nxtau, np.nan, dtype=np.float32)
    diag = np.full(nxtau, np.nan, dtype=np.float32)

    maxj = 0
    xs = np.zeros(nxtau, dtype=np.float32)
    if ifscat != 1:
        snu[:] = snubar
        xs8, maxj = _map1(taunu, snu, xtau_grid)
        xs[:] = np.asarray(xs8, dtype=np.float32)
        alpha[:] = 0.0
    else:
        if taunu[0] > xtau_grid[-1]:
            maxj = 1
        else:
            xsbar8, maxj = _map1(taunu, snubar, xtau_grid)
            xalpha8, maxj = _map1(taunu, alpha, xtau_grid)
            xalpha8 = np.maximum(np.asarray(xalpha8, dtype=np.float32), np.float32(0.0))
            xsbar8 = np.maximum(np.asarray(xsbar8, dtype=np.float32), np.float32(1.0e-38))
            mask = xtau_grid < taunu[0]
            if np.any(mask):
                xsbar8[mask] = max(snubar[0], 1.0e-38)
                xalpha8[mask] = max(alpha[0], 0.0)
            xs[:] = xsbar8
            one32 = np.float32(1.0)
            diag = one32 - xalpha8 * np.diag(coefj_matrix).astype(np.float32)
            xsbar_mod = (one32 - xalpha8) * xsbar8
            for _ in range(nxtau):
                iferr = 0
                for kk in range(nxtau):
                    k = nxtau - 1 - kk
                    # Keep update and stop criterion in float32 to mirror REAL*4 JOSH loop.
                    delxs = np.float32(np.dot(coefj_matrix[k, :], xs))
                    num = np.float32(delxs * xalpha8[k] + xsbar_mod[k] - xs[k])
                    den = np.float32(diag[k])
                    if abs(float(den)) < 1.0e-37:
                        den = np.float32(1.0e-37 if float(den) >= 0.0 else -1.0e-37)
                    delxs = np.float32(num / den)
                    xbase = np.float32(xs[k])
                    if abs(float(xbase)) < 1.0e-37:
                        xbase = np.float32(1.0e-37 if float(xbase) >= 0.0 else -1.0e-37)
                    errx = np.float32(abs(float(delxs / xbase)))
                    if errx > np.float32(_ITER_TOL):
                        iferr = 1
                    xs[k] = np.float32(max(float(np.float32(xs[k] + delxs)), 1.0e-37))
                if iferr == 0:
                    break
            xs8 = xs.astype(np.float64)
            snu_head, _ = _map1(xtau_grid, xs8, taunu[:maxj])
            snu[:maxj] = snu_head

    global _JOSH_DEBUG_XS
    _JOSH_DEBUG_XS = xs.copy()

    if maxj == n:
        xjs = -xs + coefj_matrix @ xs
        xh = coefh_matrix @ xs
        jmins[:maxj], _ = _map1(xtau_grid, xjs, taunu[:maxj])
        hnu[:maxj], _ = _map1(xtau_grid, xh, taunu[:maxj])
        jnu[:maxj] = np.maximum(jmins[:maxj] + snu[:maxj], 1.0e-38)
        knu_surface = float(np.dot(ck_weights, xs))
        trace_idx = _log_knu("XK", knu_surface, float(jnu[0]), int(maxj))
        if knu_log_path and trace_idx == knu_trace_target:
            with open(knu_log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"INP,{float(abtot[0]):.8e},{float(alpha[0]):.8e},{float(acont[0]):.8e},"
                    f"{float(aline[0]):.8e},{float(sigmac[0]):.8e},{float(sigmal[0]):.8e},"
                    f"{float(snubar[0]):.8e},{float(taunu[0]):.8e}\n"
                )
                for m in range(nxtau):
                    fh.write(
                        f"XSP,{m + 1:d},{float(xs[m]):.8e},{float(xsbar8[m]):.8e},"
                        f"{float(xalpha8[m]):.8e},{float(diag[m]):.8e}\n"
                    )
        return JoshResult(
            taunu=taunu,
            snu=snu,
            hnu=hnu,
            jnu=jnu,
            jmins=jmins,
            abtot=abtot,
            alpha=alpha,
            knu_surface=knu_surface,
            maxj=int(maxj),
        )

    # Fortran label 401 branch: solve outer layers on physical TAUNU grid.
    maxj1 = maxj + 1
    if maxj == 1:
        maxj1 = 1
    snu[maxj1 - 1 :] = snubar[maxj1 - 1 :]
    m = max(maxj - 1, 1)
    m0 = m - 1
    nmj0 = maxj - 1
    for _ in range(nxtau):
        error = 0.0
        ifneg = 0
        if np.any(snu[m0:] <= 0.0):
            ifneg = 1
            snubar[m0:] = bnu[m0:]
            snu[m0:] = bnu[m0:]
        hnu[m0:] = _deriv(taunu[m0:], snu[m0:]) / 3.0
        if np.any(hnu[m0:] <= 0.0):
            ifneg = 1
            snubar[m0:] = bnu[m0:]
            snu[m0:] = bnu[m0:]
            hnu[m0:] = _deriv(taunu[m0:], snu[m0:]) / 3.0
        jmins[nmj0:] = _deriv(taunu[nmj0:], hnu[nmj0:])
        for j in range(maxj1 - 1, n):
            if ifneg == 1:
                jmins[j] = 0.0
            jnu[j] = jmins[j] + snu[j]
            snew = (1.0 - alpha[j]) * snubar[j] + alpha[j] * jnu[j]
            error += abs(snew - snu[j]) / max(abs(snew), 1e-300)
            snu[j] = snew
        if error < _ITER_TOL:
            break

    if maxj == 1:
        knu = jnu[0] / 3.0
        _log_knu("J1", float(knu), float(jnu[0]), int(maxj))
        return JoshResult(
            taunu=taunu,
            snu=snu,
            hnu=hnu,
            jnu=jnu,
            jmins=jmins,
            abtot=abtot,
            alpha=alpha,
            knu_surface=float(knu),
            maxj=int(maxj),
        )

    xjs = np.asarray(-xs + coefj_matrix @ xs, dtype=np.float64)
    xh = np.asarray(coefh_matrix @ xs, dtype=np.float64)
    jmins[:maxj], _ = _map1(xtau_grid, xjs, taunu[:maxj])
    hnu[:maxj], _ = _map1(xtau_grid, xh, taunu[:maxj])
    jnu[:maxj] = np.maximum(jmins[:maxj] + np.maximum(snu[:maxj], _EPS), _EPS)
    knu_surface = float(np.dot(ck_weights, xs))
    trace_idx = _log_knu("XK", knu_surface, float(jnu[0]), int(maxj))
    if knu_log_path and trace_idx == knu_trace_target:
        with open(knu_log_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"INP,{float(abtot[0]):.8e},{float(alpha[0]):.8e},{float(acont[0]):.8e},"
                f"{float(aline[0]):.8e},{float(sigmac[0]):.8e},{float(sigmal[0]):.8e},"
                f"{float(snubar[0]):.8e},{float(taunu[0]):.8e}\n"
            )
            for m in range(nxtau):
                fh.write(
                    f"XSP,{m + 1:d},{float(xs[m]):.8e},{float(xsbar8[m]):.8e},"
                    f"{float(xalpha8[m]):.8e},{float(diag[m]):.8e}\n"
                )
    return JoshResult(
        taunu=taunu,
        snu=snu,
        hnu=hnu,
        jnu=jnu,
        jmins=jmins,
        abtot=abtot,
        alpha=alpha,
        knu_surface=knu_surface,
        maxj=int(maxj),
    )


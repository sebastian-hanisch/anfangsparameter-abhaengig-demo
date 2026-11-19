"""Live-Läufe: Phase-I-Stichproben aus dem SplitMix64-Strom erzeugen und die bedingte ARL₀ je Stichprobe rechnen (kleine Läufe, R <= 500).

Die Läufe sind klein und mit Zeitangabe (Spinner in der App); die schweren Tabellen sind gefroren (apb_tables.json, apb_reference.py)."""

import time
from dataclasses import dataclass

import apb_ar1 as A
import apb_constants as C
import apb_normal as N
import apb_t2 as T


class LiveLimitError(ValueError):
    """Der gewünschte Live-Lauf überschreitet die Obergrenze für Zufallszahlen (R · m · p <= 2 · 10^6)."""


def check_limit(R: int, m: int, p: int = 1) -> None:
    if R * m * p > C.LIVE_LIMIT:
        raise LiveLimitError(f"R · m · p = {R} · {m} · {p} = {R * m * p:,} liegt über der Obergrenze {C.LIVE_LIMIT:,} für Live-Läufe".replace(",", " "))


@dataclass(frozen=True)
class Ar1Live:
    phi: float
    m: int
    R: int
    karte: str
    x0: tuple            # die erste Phase-I-Stichprobe (zum Zeichnen)
    xbar0: float
    s0: float
    phi_hat0: float
    arl: tuple           # bedingte ARL₀ je Stichprobe
    rate: tuple          # Alarmrate je Beobachtung je Stichprobe
    known: float         # ARL₀ bei bekannten Parametern (Nyström)
    summary: dict
    mean_phi_hat: float
    seconds: float


def ar1_live(phi: float, m: int, R: int, karte: str, seed: int) -> Ar1Live:
    """R Phase-I-Stichproben (m Werte) der AR(1)-Reihe, die Karte `karte` (Schlüssel roh / resid_bekannt / resid_geschaetzt) und ihre bedingte ARL₀ je Stichprobe."""
    import numpy as np
    check_limit(R, m)
    t0 = time.perf_counter()
    x = A.ar1_samples(N.Stream(seed * 7 + 1), R, m, phi)
    xb, s, ph, _ = A.estimates(x)
    if karte == "roh":
        arl, rate = A.raw_chart_cond(x, phi)
    else:
        arl, rate, _ = A.resid_chart_cond(x, phi, karte == "resid_bekannt")
    return Ar1Live(phi, m, R, karte, tuple(float(v) for v in x[0]), float(xb[0]), float(s[0]), float(ph[0]),
                   tuple(float(v) for v in arl), tuple(float(v) for v in rate), A.known_arl0(phi), A.summarize(arl, rate),
                   float(np.mean(ph)), time.perf_counter() - t0)


@dataclass(frozen=True)
class T2Live:
    p: int
    m: int
    R: int
    rules: dict          # Regel -> dict(arl=tuple, rate=tuple, summary=dict)
    seconds: float


def t2_live(p: int, m: int, R: int, rules, seed: int) -> T2Live:
    """R Phase-I-Stichproben (m Vektoren mit p Kanälen) aus Rohnormalzahlen; je Regel bedingte ARL₀ je Stichprobe (Imhof-Integral)."""
    check_limit(R, m, p)
    t0 = time.perf_counter()
    cell = T.t2_cell(N.Stream(seed * 7 + 2), R, m, p, rules=tuple(rules))
    out = {rule: dict(arl=tuple(float(v) for v in c["arl"]), rate=tuple(float(v) for v in c["rate"]), summary=T.summarize_t2(c["arl"], c["rate"])) for rule, c in cell.items()}
    return T2Live(p, m, R, out, time.perf_counter() - t0)

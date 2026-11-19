"""Phase I mit geschätzter Kovarianz: Hotelling-T² (p Kanäle), χ²-Regel und F-Regel, bedingte Alarmwahrscheinlichkeit nach Imhof, Preis der F-Regel.

Modell: p Kanäle, Vektoren x ~ N(0, Σ) unabhängig; wegen der affinen Invarianz von T² genügt Σ = I. Phase I: m Vektoren ohne Änderung, Phase II: dieselbe Verteilung.
T² = (x - x̄)' S⁻¹ (x - x̄), S mit Nenner m - 1; Alarm bei T² > UCL.
χ²-Regel: UCL = χ²-Quantil zu 1/370,398 mit p Freiheitsgraden (die Regel für bekanntes Σ). F-Regel: UCL so, dass die UNBEDINGTE ARL₀ (Kehrwert der mittleren
Alarmrate über Phase I) 370,398 ist. Eigene Herleitung (Quelle nicht gelesen): T² eines neuen Vektors = p (m+1)(m-1)/(m (m-p)) · F(p, m-p);
die Tests prüfen sie an Rohnormalzahlen, an der t-Formel (p = 1) und an der geschlossenen Form (p = 2)."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2, f as f_dist, ncx2

from apb_ar1 import GAMMA0, HALF

P0 = 1.0 / GAMMA0
_GL16 = np.polynomial.legendre.leggauss(16)


# ------------------------------------------------------------------------------------------------------------------ Grenzen und unbedingte Rechnung
def ucl_chi2(p: int) -> float:
    """Grenze bei bekanntem Σ: 9,000 / 11,829 / 18,205 / 26,901 bei p = 1 / 2 / 5 / 10."""
    return float(chi2.isf(P0, p))


def f_factor(m: int, p: int) -> float:
    """T² (Phase II, neuer Vektor) = f_factor · F(p, m - p); nur für m > p."""
    if m <= p:
        raise ValueError(f"m = {m} muss größer sein als die Zahl der Kanäle p = {p} (sonst ist S singulär)")
    return p * (m + 1.0) * (m - 1.0) / (m * (m - p))


def ucl_f(m: int, p: int) -> float:
    """Grenze der F-Regel: unbedingte ARL₀ gleich 370,398."""
    return f_factor(m, p) * float(f_dist.isf(P0, p, m - p))


def ucl_rule(rule: str, m: int, p: int) -> float:
    if rule == "chi2":
        return ucl_chi2(p)
    if rule == "F":
        return ucl_f(m, p)
    raise ValueError(f"unbekannte Regel {rule!r}")


def alarm_prob_uncond(ucl: float, m: int, p: int) -> float:
    """Mittlere Alarmwahrscheinlichkeit je Phase-II-Vektor (über alle Phase-I-Stichproben, exakt über die F-Verteilung)."""
    return float(f_dist.sf(ucl / f_factor(m, p), p, m - p))


def arl0_uncond(rule: str, m: int, p: int) -> float:
    """Unbedingte ARL₀ = Kehrwert der mittleren Alarmrate (Vektoren bis zum Alarm)."""
    return 1.0 / alarm_prob_uncond(ucl_rule(rule, m, p), m, p)


def arl1_price(rule: str, m: int, p: int, delta: float = 1.0) -> float:
    """ARL₁ bei BEKANNTEN Parametern (Σ = I, ein Kanal springt um delta σ, Nichtzentralität delta²): 1 / P(T² > UCL), T² nichtzentral χ² mit p Freiheitsgraden."""
    return float(1.0 / ncx2.sf(ucl_rule(rule, m, p), p, delta * delta))


# ------------------------------------------------------------------------------------------------------------------ Imhof-Integral
def _imhof_nodes(c: float, U: float):
    """Zusammengesetzte Gauß-Legendre-Regel auf (0, U]; Panelbreite 0,7 mal der Periode der Schwingung sin(c u / 2)."""
    h = 0.7 * 4.0 * np.pi / c
    n = int(np.ceil(U / h))
    e = np.linspace(0.0, U, n + 1)
    mid = 0.5 * (e[1:] + e[:-1])
    half = 0.5 * (e[1:] - e[:-1])
    x, w = _GL16
    return (mid[:, None] + half[:, None] * x[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def imhof_sf(lam, a, c: float, U: float, chunk: int = 40) -> np.ndarray:
    """P(Q > c) mit Q = Σ_j λ_j (Z_j + a_j)², Z_j unabhängig N(0, 1), λ_j > 0 (Imhof 1961, Quadratur auf (0, U], mit Randkorrektur des abgeschnittenen
    oszillierenden Rests). Arrays lam, a der Form (B, p)."""
    lam = np.atleast_2d(np.asarray(lam, float))
    a = np.atleast_2d(np.asarray(a, float))
    u, w = _imhof_nodes(c, U)
    out = np.empty(lam.shape[0])
    for i in range(0, lam.shape[0], chunk):
        lm, am = lam[i:i + chunk], a[i:i + chunk]
        L = lm[:, :, None] * u[None, None, :]
        d2 = am[:, :, None] ** 2
        den = 1.0 + L * L
        theta = 0.5 * (np.arctan(L) + d2 * L / den).sum(axis=1) - 0.5 * c * u[None, :]
        logrho = 0.25 * np.log(den).sum(axis=1) + 0.5 * (d2 * L * L / den).sum(axis=1)
        val = 0.5 + (w[None, :] * np.sin(theta) / (u[None, :] * np.exp(logrho))).sum(axis=1) / np.pi
        lu, d2u = lm * U, am ** 2
        denu = 1.0 + lu * lu
        thU = 0.5 * (np.arctan(lu) + d2u * lu / denu).sum(axis=1) - 0.5 * c * U
        dthU = 0.5 * (lm / denu + d2u * lm * (1.0 - lu * lu) / denu ** 2).sum(axis=1) - 0.5 * c
        gU = 1.0 / (U * np.exp(0.25 * np.log(denu).sum(axis=1) + 0.5 * (d2u * lu * lu / denu).sum(axis=1)))
        out[i:i + chunk] = val + np.cos(thU) * gU / dthU / np.pi
    return out


def u_for(p: int) -> float:
    """Obergrenze U des Integrationsbereichs je Zahl der Kanäle (Imhof-Integral, Randkorrektur)."""
    return {1: 1000.0, 2: 250.0}.get(p, 40.0 if p <= 5 else 20.0)


# ------------------------------------------------------------------------------------------------------------------ Phase I und bedingte Alarmwahrscheinlichkeit
def phase1_from_raw(z: np.ndarray):
    """Aus Rohdaten z (R, m, p) der Phase I: Eigenwerte d von S (R, p), a = -Vᵀ x̄ (R, p), x̄ und S."""
    R, m, p = z.shape
    if m <= p:
        raise ValueError(f"m = {m} muss größer sein als p = {p}")
    xb = z.mean(axis=1)
    D = z - xb[:, None, :]
    S = np.einsum("rki,rkj->rij", D, D) / (m - 1)
    d, V = np.linalg.eigh(S)
    a = -np.einsum("rkj,rk->rj", V, xb)
    return d, a, xb, S


def cond_alarm_prob(d: np.ndarray, a: np.ndarray, ucl: float, p: int) -> np.ndarray:
    """Bedingte Alarmwahrscheinlichkeit je Phase-II-Vektor, gegeben (x̄, S); p = 1 geschlossen, sonst Imhof."""
    if np.any(d <= 0):
        raise ValueError("S ist singulär (ein Eigenwert <= 0)")
    if p == 1:
        s = np.sqrt(d[:, 0])
        return ndtr(-a[:, 0] - np.sqrt(ucl) * s) + ndtr(a[:, 0] - np.sqrt(ucl) * s)
    return imhof_sf(1.0 / d, a, ucl, u_for(p))


def t2_cell(stream, R: int, m: int, p: int, rules=("chi2", "F")) -> dict:
    """Phase I frisch aus Rohnormalzahlen des Stroms: je Regel die bedingte ARL₀ (1 / Alarmwahrscheinlichkeit) und die Alarmrate je Stichprobe."""
    z = stream.normal((R, m, p))
    d, a, _, _ = phase1_from_raw(z)
    out = {}
    for rule in rules:
        pal = cond_alarm_prob(d, a, ucl_rule(rule, m, p), p)
        out[rule] = dict(arl=1.0 / pal, rate=pal)
    return out


def summarize_t2(arl: np.ndarray, rate: np.ndarray, thr: float = HALF) -> dict:
    return dict(med=float(np.median(arl)), q05=float(np.quantile(arl, 0.05)), q95=float(np.quantile(arl, 0.95)),
                p_below=float((arl < thr).mean()), flotte=float(1.0 / rate.mean()))

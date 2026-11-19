"""Handbeispiele (H 7 und H 8 des Plans): Brüche, geschlossene Formeln, kleine Fälle zum Nachrechnen von Hand.

Jede Zahl wird von einem Rechenweg ohne den Code der Demo erzeugt (Brüche, Arkustangens, Quadratur, geschlossene F-Verteilung) und dann mit der Demo verglichen.
Die Phase-I-Stichproben mit m = 2, 3, 4 sind ausgedacht (Nachrechnen von Hand, keine Aussage über übliche Größen)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr
from scipy.stats import chi2, f as f_d, ncx2, norm, t as t_d

import apb_ar1 as A
import apb_t2 as T

P0 = 2 * ndtr(-3.0)


# ----------------------------------------------------------------------------------------------------------------------------------- H 7: AR(1)
def var_xbar_bruch(m, phi):
    return (1 + 2 * sum((1 - F(k, m)) * phi ** k for k in range(1, m))) / m


def es2_bruch(m, phi):
    return F(m, m - 1) * (1 - var_xbar_bruch(m, phi))


def es2_matrix(m, phi):
    """Zweiter Weg: Kovarianzmatrix C_ij = φ^|i-j|; E[Σ (x_i - x̄)²] = tr C - (Summe aller C_ij) / m."""
    tot = sum(phi ** abs(i - j) for i in range(m) for j in range(m))
    return (F(m) - tot / m) / (m - 1)


def test_arl0_der_3sigma_karte():
    assert abs(1 / P0 - 370.3983) < 1e-3 and abs(A.GAMMA0 - 370.3983) < 1e-3


def test_m2_e_s2_ist_eins_minus_phi():
    assert es2_bruch(2, F(1, 2)) == es2_matrix(2, F(1, 2)) == F(1, 2)
    assert abs(A.e_s2_closed(2, 0.5) - 0.5) < 1e-13 and abs(A.e_s2_closed(2, 0.3) - 0.7) < 1e-13


def test_m3_phi_halb_bruchwerte():
    assert var_xbar_bruch(3, F(1, 2)) == F(11, 18)
    assert es2_bruch(3, F(1, 2)) == es2_matrix(3, F(1, 2)) == F(7, 12)
    assert abs(A.var_xbar_closed(3, 0.5) - 11 / 18) < 1e-13 and abs(A.e_s2_closed(3, 0.5) - 7 / 12) < 1e-13


@pytest.mark.parametrize("m", [2, 3, 5, 8])
def test_phi_null_unabhaengig(m):
    assert es2_bruch(m, F(0)) == 1 and var_xbar_bruch(m, F(0)) == F(1, m)
    assert abs(A.e_s2_closed(m, 0.0) - 1) < 1e-13 and abs(A.var_xbar_closed(m, 0.0) - 1 / m) < 1e-13


def test_m100_phi_09_streuung_und_mitte():
    assert F(83, 100) < es2_bruch(100, F(9, 10)) < F(84, 100)
    assert abs(A.e_s2_closed(100, 0.9) - 0.836) < 5e-4
    assert abs(A.var_xbar_closed(100, 0.9) - 0.172) < 5e-4
    assert abs(float(es2_matrix(100, F(9, 10))) - A.e_s2_closed(100, 0.9)) < 1e-12


def test_m2_phi_dach_ist_immer_minus_halb():
    rng = np.random.default_rng(14001)
    x = rng.standard_normal((20, 2))
    assert np.allclose(A.estimates(x)[2], -0.5, atol=1e-12)


@pytest.mark.parametrize("phi,soll,kehr", [(0.0, 0.246752, 4.0527), (0.5, 0.354978, 2.8171), (0.9, 0.620102, 1.6126)])
def test_m2_alarmwahrscheinlichkeit_arkustangens(phi, soll, kehr):
    ep = A.mean_alarm_prob_m2(phi)
    assert abs(ep - soll) < 5e-6 and abs(1 / ep - kehr) < 5e-4


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_m2_arkustangens_gleich_quadratur(phi):
    """Zweiter Weg: Doppelintegral P(|W| > k |Z|) mit W ~ N(0, τ²), Z ~ N(0, 1), k = 3 sqrt(1 - φ)."""
    tau, k = math.sqrt((3 + phi) / 2), 3 * math.sqrt(1 - phi)
    val, _ = integrate.quad(lambda z: 2 * norm.pdf(z) * 2 * norm.sf(k * z / tau), 0, 12, epsabs=1e-13, epsrel=1e-12)
    assert abs(val - A.mean_alarm_prob_m2(phi)) < 1e-9


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_m2_arkustangens_gleich_simulation_aus_rohwerten(phi):
    """Dritter Weg: AR(1)-Paar plus ein stationärer Wert, 1 Mio. Stichproben (numpy, nur Gegenprobe), Band 4 Standardfehler."""
    rng = np.random.default_rng(14002 + int(phi * 10))
    n = 1_000_000
    s = math.sqrt(1 - phi ** 2)
    x1 = rng.standard_normal(n)
    x2 = phi * x1 + s * rng.standard_normal(n)
    xn = rng.standard_normal(n)
    xb, s2 = (x1 + x2) / 2, (x1 - x2) ** 2 / 2
    r = (np.abs(xn - xb) > 3 * np.sqrt(s2)).mean()
    assert abs(r - A.mean_alarm_prob_m2(phi)) < 4 * math.sqrt(r * (1 - r) / n)


def _nystroem80(a, b, phi):
    xg, wg = np.polynomial.legendre.leggauss(80)
    x, w = 0.5 * (a + b) + 0.5 * (b - a) * xg, 0.5 * (b - a) * wg
    s = math.sqrt(1 - phi ** 2)
    K = norm.pdf((x[None, :] - phi * x[:, None]) / s) / s * w[None, :]
    g = np.linalg.solve(np.eye(80) - K, np.ones(80))
    return 1 + float((w * norm.pdf(x) * g).sum())


def _zellen800(a, b, phi):
    n = 800
    e = np.linspace(a, b, n + 1)
    c = 0.5 * (e[:-1] + e[1:])
    s = math.sqrt(1 - phi ** 2)
    Q = np.diff(ndtr((e[None, :] - phi * c[:, None]) / s), axis=1)
    return 1 + float(np.diff(ndtr(e)) @ np.linalg.solve(np.eye(n) - Q, np.ones(n)))


@pytest.mark.parametrize("phi,soll", [(0.0, 370.398), (0.3, 376.383), (0.6, 419.377), (0.9, 831.783)])
def test_bekannte_grenzen_nystroem_gegen_zellenkette_gegen_plan_05(phi, soll):
    a1, a2, a3 = _nystroem80(-3.0, 3.0, phi), _zellen800(-3.0, 3.0, phi), A.known_arl0(phi)
    assert abs(a1 - soll) < 5e-3 and abs(a2 / a1 - 1) < 2e-4 and abs(a3 / a1 - 1) < 1e-9


def test_enge_grenzen_zwei_wege_gleich():
    assert abs(_nystroem80(-2.1, 2.7, 0.6) / _zellen800(-2.1, 2.7, 0.6) - 1) < 2e-4
    assert abs(A.raw_chart_arl([-2.1], [2.7], 0.6)[0] / _zellen800(-2.1, 2.7, 0.6) - 1) < 2e-4


# ----------------------------------------------------------------------------------------------------------------------------------- H 8: T²
def f_faktor(m, p):
    return p * (m + 1) * (m - 1) / (m * (m - p))


@pytest.mark.parametrize("m", [2, 10, 30, 100])
def test_p1_f_formel_ist_t_formel(m):
    if m == 2:
        assert abs(f_d.sf(9.0 / f_faktor(2, 1), 1, 1) - A.mean_alarm_prob_m2(0.0)) < 1e-12     # Arkustangens-Fall aus H 7 (φ = 0)
    assert abs(f_d.sf(9.0 / f_faktor(m, 1), 1, m - 1) - 2 * t_d.sf(3 / math.sqrt(1 + 1 / m), m - 1)) < 1e-12
    if m > 1 + 1:
        assert abs(T.alarm_prob_uncond(9.0, m, 1) - 2 * t_d.sf(3 / math.sqrt(1 + 1 / m), m - 1)) < 1e-12


def test_p2_grenze_ist_zwei_ln_370():
    ucl2 = 2 * math.log(1 / P0)
    assert abs(ucl2 - 11.8292) < 5e-5 and abs(math.exp(-ucl2 / 2) - P0) < 1e-12
    assert abs(ucl2 - T.ucl_chi2(2)) < 1e-9


@pytest.mark.parametrize("m", [4, 6, 12, 50, 400])
def test_p2_verteilungsfunktion_geschlossen(m):
    nu = m - 2
    ucl2 = 2 * math.log(1 / P0)
    y = ucl2 / f_faktor(m, 2)
    assert abs((1 + 2 * y / nu) ** (-nu / 2) - T.alarm_prob_uncond(ucl2, m, 2)) < 1e-12


def test_m4_p2_von_hand():
    """Faktor p (m+1)(m-1)/(m (m-p)) = 15/4, ν = 2, y = (4/15) · 2 ln 370,398 = 3,1544, mittlere Alarmrate 1/(1 + y), ARL₀ der χ²-Regel = 1 + y = 4,154."""
    assert f_faktor(4, 2) == 3.75 and abs(T.f_factor(4, 2) - 3.75) < 1e-15
    ucl2 = 2 * math.log(1 / P0)
    y = ucl2 / 3.75
    assert abs(y - 3.1544) < 5e-5
    assert abs(1 + y - 4.1544) < 5e-4 and abs(T.arl0_uncond("chi2", 4, 2) - (1 + y)) < 1e-9
    # F-Regel: y_F = γ - 1 (ν = 2), UCL_F = 3,75 · 369,398 = 1385,24; ARL₀ 370,398 nach Konstruktion
    assert abs(T.ucl_f(4, 2) - 3.75 * (1 / P0 - 1)) < 1e-6 and abs(T.ucl_f(4, 2) - 1385.24) < 0.05
    assert abs(T.arl0_uncond("F", 4, 2) - 1 / P0) < 1e-6


def test_m4_p2_gegenprobe_aus_rohnormalzahlen():
    """1,2 Mio. Phase-I-Stichproben (numpy, nur Gegenprobe) mit je einem frischen Phase-II-Vektor: Alarmanteil der χ²-Grenze gegen 1/(1 + y)."""
    rng = np.random.default_rng(14003)
    R, cnt = 1_200_000, 0
    ucl2 = T.ucl_chi2(2)
    for _ in range(3):
        X = rng.standard_normal((R // 3, 4, 2))
        xb = X.mean(axis=1)
        D = X - xb[:, None, :]
        S = np.einsum("rki,rkj->rij", D, D) / 3
        z = rng.standard_normal((R // 3, 2)) - xb
        t2 = np.einsum("ri,rij,rj->r", z, np.linalg.inv(S), z)
        cnt += int((t2 > ucl2).sum())
    r = cnt / R
    soll = 1 / (1 + ucl2 / 3.75)
    assert abs(r - soll) < 4 * math.sqrt(r * (1 - r) / R)


@pytest.mark.parametrize("p,m,a_chi,a_f", [(2, 100, 67.32, 99.20), (5, 100, 114.38, 290.89), (10, 100, 159.91, 1328.38), (2, 300, 67.32, 76.16), (5, 300, 114.38, 152.24), (10, 300, 159.91, 293.21)])
def test_preis_der_f_regel_arl1(p, m, a_chi, a_f):
    uc = float(chi2.isf(P0, p))
    uf = f_faktor(m, p) * float(f_d.isf(P0, p, m - p))
    assert abs(1 / ncx2.sf(uc, p, 1.0) - a_chi) < 0.006
    assert abs(1 / ncx2.sf(uf, p, 1.0) - a_f) < 0.006 * max(1.0, a_f / 100)
    assert abs(T.arl1_price("chi2", m, p) - a_chi) < 0.006 and abs(T.arl1_price("F", m, p) - a_f) < 0.006 * max(1.0, a_f / 100)


def test_p1_m100_f_regel_gleich_l_von_a():
    assert abs(math.sqrt(T.ucl_f(100, 1)) - 3.0929) < 5e-5


@pytest.mark.parametrize("p,soll", [(2, 239.6), (5, 154.7), (10, 75.0)])
def test_unbedingte_arl0_der_chi2_regel_m100(p, soll):
    assert abs(1 / float(f_d.sf(float(chi2.isf(P0, p)) / f_faktor(100, p), p, 100 - p)) - soll) < 0.06
    assert abs(T.arl0_uncond("chi2", 100, p) - soll) < 0.06

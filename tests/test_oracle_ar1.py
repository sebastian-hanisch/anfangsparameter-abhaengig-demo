"""Orakel für Phase I bei abhängigen Werten: anderer Rechenweg als der der Demo (ORACLE-PLAYBOOK §5, §6).

(a) Nyström (Demo) gegen Zellenkette (hier), gegen die Werte aus Plan 05 und gegen Simulation einer bedingten Karte (numpy).
(b) φ = 0: Anteil und Median gegen die EXAKTE iid-Verteilung (Doppelintegral über (x̄, s) mit der Wurzelsuche über s; die Tabellen-Erzeugung sucht über x̄).
(c) Residuenkarte mit φ̂ = φ, x̄ = 0, σ̂_e = σ_e: ARL₀ = 370,3983 exakt (Zeilensummen der Kette); Kette gegen Simulation.
(d) E[x̄²], E[s²] geschlossen gegen die Kovarianzmatrix in Brüchen und gegen Simulation; Verzerrung von φ̂.
Zufall der Gegenproben: numpy (nicht das Ergebnis der App)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import chi2, norm

import apb_ar1 as A
import apb_normal as N


# ----------------------------------------------------------------------------------------------------------------------------------- (a)
def zellenkette(a, b, phi, n=800):
    e = np.linspace(a, b, n + 1)
    c = 0.5 * (e[:-1] + e[1:])
    s = math.sqrt(1 - phi ** 2)
    Q = np.diff(ndtr((e[None, :] - phi * c[:, None]) / s), axis=1)
    return 1 + float(np.diff(ndtr(e)) @ np.linalg.solve(np.eye(n) - Q, np.ones(n)))


def sim_karte(a, b, phi, runs, seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(1 - phi ** 2)
    x = rng.standard_normal(runs)
    T = np.zeros(runs)
    alive = np.arange(runs)
    t = 0
    while alive.size:
        t += 1
        if t > 1:
            x[alive] = phi * x[alive] + s * rng.standard_normal(alive.size)
        al = (x[alive] < a) | (x[alive] > b)
        T[alive[al]] = t
        alive = alive[~al]
    return T.mean(), T.std(ddof=1) / math.sqrt(runs)


@pytest.mark.parametrize("a,b,phi", [(-3.0, 3.0, 0.9), (-2.1, 2.7, 0.6), (-1.0, 2.5, 0.9), (-2.7, 3.2, 0.6), (-2.0, 2.0, 0.3)])
def test_naeherung_nystroem_gegen_zellenkette(a, b, phi):
    assert abs(A.raw_chart_arl([a], [b], phi)[0] / zellenkette(a, b, phi) - 1) < 3e-4


@pytest.mark.parametrize("a,b,phi,runs", [(-2.7, 3.2, 0.6, 30_000), (-1.0, 2.5, 0.9, 30_000)])
def test_bedingte_karte_gegen_simulation(a, b, phi, runs):
    mu, se = sim_karte(a, b, phi, runs, 13500 + int(100 * phi))
    assert abs(mu - A.raw_chart_arl([a], [b], phi)[0]) <= 4 * se


def test_gitterfeinheit_n60_n100_n200_auf_1e_6():
    """Nyström n = 60 gegen n = 100 gegen n = 200 an 40 Phase-I-Stichproben (m = 100 und m = 30, φ = 0,9)."""
    for m in (100, 30):
        x = A.ar1_samples(N.Stream(4), 40, m, 0.9)
        xb, s, _, _ = A.estimates(x)
        a, b = xb - 3 * s, xb + 3 * s
        r60, r100, r200 = (A.raw_chart_arl(a, b, 0.9, n) for n in (60, 100, 200))
        assert np.max(np.abs(r60 / r200 - 1)) < 1e-6 and np.max(np.abs(r100 / r200 - 1)) < 1e-6


def test_alle_werte_gleich_gibt_arl_eins():
    """s = 0: die Grenzen fallen auf x̄ zusammen, jede Beobachtung außer x̄ alarmiert (ARL 1); keine Division durch null, kein Absturz."""
    x = np.full((3, 20), 0.4)
    arl, rate = A.raw_chart_cond(x, 0.9)
    assert np.allclose(arl, 1.0) and np.allclose(rate, 1.0)
    arl2, rate2, _ = A.resid_chart_cond(x, 0.9, False)
    assert np.allclose(arl2, 1.0, atol=1e-9)


# ----------------------------------------------------------------------------------------------------------------------------------- (b)
def iid_anteil_x_aussen(m, thr):
    """P(bedingte ARL₀ < thr) bei unabhängigen Werten. Für festes x̄ fällt die Alarmwahrscheinlichkeit p(x̄, s) = 1 - Φ(x̄ + 3s) + Φ(x̄ - 3s) in s; die Schwelle s*(x̄)
    mit p = 1/thr bestimmt die Wurzelsuche, P(s < s*) ist die χ-Verteilung; außen das Integral über x̄ ~ N(0, 1/m)."""
    target = 1.0 / thr

    def sstar(xb):
        return brentq(lambda s: 1 - ndtr(xb + 3 * s) + ndtr(xb - 3 * s) - target, 1e-9, 30.0, xtol=1e-14)

    dens = lambda xb: math.sqrt(m) * norm.pdf(math.sqrt(m) * xb) * chi2.cdf((m - 1) * sstar(xb) ** 2, m - 1)   # noqa: E731
    w = 9 / math.sqrt(m)
    return 2 * quad(dens, 0, w, limit=400, epsabs=1e-13, epsrel=1e-11)[0]


@pytest.mark.parametrize("m,soll", [(30, 0.35966), (100, 0.17920), (300, 0.04185), (1000, 0.00058)])
def test_iid_exakte_verteilung_aus_zweitem_weg(m, soll):
    assert abs(iid_anteil_x_aussen(m, A.HALF) - soll) < 1e-5


def test_iid_mindestzahl_274_exakt():
    """Kleinste ganze Zahl m mit P(bedingte ARL₀ < 185,2) <= 5 %: 274 (Plan 04, vom Prüfer unabhängig bestätigt); hier mit dem x̄-äußeren Doppelintegral."""
    assert iid_anteil_x_aussen(273, A.HALF) > 0.05 >= iid_anteil_x_aussen(274, A.HALF)


@pytest.mark.parametrize("m", [30, 100, 300])
def test_phi_null_live_gegen_exakte_iid_verteilung(m):
    R = 4000
    x = A.ar1_samples(N.Stream(77 + m), R, m, 0.0)
    arl, _ = A.raw_chart_cond(x, 0.0)
    exakt = iid_anteil_x_aussen(m, A.HALF)
    p = float((arl < A.HALF).mean())
    assert abs(p - exakt) <= 4 * math.sqrt(exakt * (1 - exakt) / R)


def test_phi_null_median_gegen_exakten_median():
    m, R = 100, 4000
    x = A.ar1_samples(N.Stream(99), R, m, 0.0)
    arl, _ = A.raw_chart_cond(x, 0.0)
    med = brentq(lambda t: iid_anteil_x_aussen(m, t) - 0.5, 100.0, 600.0, xtol=1e-4)
    assert abs(float(np.median(arl)) - med) < 0.06 * med and abs(med - 341.996) < 5e-3


# ----------------------------------------------------------------------------------------------------------------------------------- (c)
@pytest.mark.parametrize("phi", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("n", [80, 160, 320])
def test_residuenkarte_exakte_parameter_gibt_370_3983(phi, n):
    arl, rate = A.resid_chain_arl(phi, [phi], [0.0], [3.0 * math.sqrt(1 - phi ** 2)], n=n)
    assert abs(arl[0] - A.GAMMA0) < 1e-4 * A.GAMMA0 and abs(1 / rate[0] - A.GAMMA0) < 1e-3 * A.GAMMA0


def sim_resid(phi_t, phi_k, c, w, runs, seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(1 - phi_t ** 2)
    xprev = rng.standard_normal(runs)
    T = np.zeros(runs)
    alive = np.arange(runs)
    t = 0
    while alive.size:
        t += 1
        xn = phi_t * xprev[alive] + s * rng.standard_normal(alive.size)
        al = np.abs(xn - phi_k * xprev[alive] - c) > w
        xprev[alive] = xn
        T[alive[al]] = t
        alive = alive[~al]
    return T.mean(), T.std(ddof=1) / math.sqrt(runs)


@pytest.mark.parametrize("pt,pk,c,w,seed", [(0.7, 0.5, 0.1, 2.4, 1), (0.9, 0.8, -0.2, 1.0, 2)])
def test_zellenkette_gegen_simulation(pt, pk, c, w, seed):
    mu, se = sim_resid(pt, pk, c, w, 30_000, 13520 + seed)
    assert abs(mu - A.resid_chain_arl(pt, [pk], [c], [w], n=320)[0][0]) <= 4 * se


def test_richardson_naeher_an_feinem_gitter_als_n80():
    """Die Richardson-Fassung der Demo (n = 80 und 160) liegt näher an der Kette n = 320 als die rohe n = 80."""
    x = A.ar1_samples(N.Stream(21), 25, 100, 0.9)
    xb, _, ph, d = A.estimates(x)
    se = A.sigma_e_hat(d, ph)
    c, w = (1 - ph) * xb, 3 * se
    fein = A.resid_chain_arl(0.9, ph, c, w, n=320)[0]
    grob = A.resid_chain_arl(0.9, ph, c, w, n=80)[0]
    rich = A.resid_chart_cond(x, 0.9, False)[0]
    assert np.max(np.abs(rich / fein - 1)) < np.max(np.abs(grob / fein - 1)) and np.max(np.abs(rich / fein - 1)) < 3e-3


# ----------------------------------------------------------------------------------------------------------------------------------- (d)
def e_s2_matrix(m, phi):
    tot = sum(phi ** abs(i - j) for i in range(m) for j in range(m))
    return float((F(m) - tot / m) / (m - 1))


@pytest.mark.parametrize("m", [2, 3, 5, 8, 30])
def test_e_s2_geschlossen_gegen_kovarianzmatrix(m):
    assert abs(A.e_s2_closed(m, 0.5) - e_s2_matrix(m, F(1, 2))) < 1e-13


@pytest.mark.parametrize("phi,m", [(0.9, 30), (0.9, 100), (0.6, 100)])
def test_streuung_und_mitte_gegen_simulation_200000(phi, m):
    rng = np.random.default_rng(13500 + m)
    R = 200_000
    z = rng.standard_normal((R, m))
    s = math.sqrt(1 - phi ** 2)
    e = z.copy()
    e[:, 1:] *= s
    from scipy.signal import lfilter
    x = lfilter([1.0], [1.0, -phi], e, axis=1)
    xb, sd, ph, _ = A.estimates(x)
    for sim, soll, vals in ((float((xb ** 2).mean()), A.var_xbar_closed(m, phi), xb ** 2), (float((sd ** 2).mean()), A.e_s2_closed(m, phi), sd ** 2)):
        se = vals.std(ddof=1) / math.sqrt(R)
        assert abs(sim - soll) <= 3.5 * se


def test_verzerrung_phi_dach_m30_phi_09():
    """E[φ̂] - φ ≈ -0,18 bei m = 30 (200 000 Stichproben); die Näherung -(1 + 3φ)/m (nicht gelesen) unterschätzt sie um den Faktor 1,14 bis 1,46."""
    rng = np.random.default_rng(13501)
    R, m, phi = 200_000, 30, 0.9
    e = rng.standard_normal((R, m))
    e[:, 1:] *= math.sqrt(1 - phi ** 2)
    from scipy.signal import lfilter
    x = lfilter([1.0], [1.0, -phi], e, axis=1)
    ph = A.estimates(x)[2]
    bias = float(ph.mean()) - phi
    assert abs(bias + 0.180) < 0.01
    assert 1.1 <= bias / (-(1 + 3 * phi) / m) <= 1.6


# ----------------------------------------------------------------------------------------------------------------------------------- Eigenschaften
def test_spiegelung_aendert_die_bedingte_arl_nicht():
    """Phase I und wahre Reihe gemeinsam gespiegelt (x -> -x): die zweiseitige Rohdatenkarte hat dieselbe bedingte ARL₀."""
    x = A.ar1_samples(N.Stream(5), 30, 60, 0.6)
    assert np.allclose(A.raw_chart_cond(x, 0.6)[0], A.raw_chart_cond(-x, 0.6)[0], rtol=1e-10)
    assert np.allclose(A.resid_chart_cond(x[:10], 0.6, False)[0], A.resid_chart_cond(-x[:10], 0.6, False)[0], rtol=1e-10)


def test_verschiebung_der_phase_i_aendert_die_karte_nicht_aber_die_wahre_mitte_schon():
    """Die Schätzer x̄ und s sind verschiebungsinvariant: die Grenzen wandern mit; Streuung gleich."""
    x = A.ar1_samples(N.Stream(6), 5, 50, 0.6)
    xb1, s1, ph1, _ = A.estimates(x)
    xb2, s2, ph2, _ = A.estimates(x + 2.5)
    assert np.allclose(xb2, xb1 + 2.5) and np.allclose(s1, s2) and np.allclose(ph1, ph2)


def test_gleicher_zufalls_seed_gleiche_ausgabe_und_anderer_seed_andere():
    a = A.ar1_samples(N.Stream(8), 10, 40, 0.9)
    assert np.array_equal(a, A.ar1_samples(N.Stream(8), 10, 40, 0.9))
    assert not np.array_equal(a, A.ar1_samples(N.Stream(9), 10, 40, 0.9))


def test_rand_verteilung_der_reihe_ist_normal_fuer_jedes_phi():
    """Stationäre Randverteilung N(0, 1): Varianz und Lag-1-Korrelation der erzeugten Reihe (R = 4000, m = 50, Band)."""
    for phi in (0.3, 0.9):
        x = A.ar1_samples(N.Stream(31), 4000, 50, phi)
        assert abs(x[:, 25].var() - 1) < 0.08 and abs(np.mean(x[:, 24] * x[:, 25]) - phi) < 0.05 and abs(x[:, 0].var() - 1) < 0.08


def test_verteilungsfunktion_der_arl_ist_monoton_und_endet_bei_eins():
    x = A.ar1_samples(N.Stream(10), 300, 100, 0.9)
    arl, _ = A.raw_chart_cond(x, 0.9)
    v = np.sort(arl)
    y = (np.arange(v.size) + 1) / v.size
    assert np.all(np.diff(v) >= 0) and np.all(np.diff(y) > 0) and y[-1] == 1.0 and v[0] >= 1.0

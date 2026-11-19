"""Orakel für Phase I mit geschätzter Kovarianz (T²): anderer Rechenweg als der der Demo.

(e) Imhof-Integral gegen scipy `ncx2` (S = I), gegen den geschlossenen Fall p = 1 und gegen Simulation aus Rohnormalzahlen (numpy).
(f) F-Verteilung des Phase-II-T² gegen Rohnormalzahlen, gegen die t-Formel (p = 1) und die geschlossene Form (p = 2).
(g) Die mittlere Alarmrate der Live-Stichproben gegen die exakte F-Rechnung.
(h) Mindestzahl p = 1 gegen die exakte 274 (x̄-äußeres Doppelintegral aus tests/test_oracle_ar1.py).
Eigenschaften: Umnummerieren der Kanäle, orthogonale Drehung, F-Regel hat 370,398 für jedes (p, m), χ²-Regel wächst in m und fällt in p."""

import math

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import ncx2

import apb_normal as N
import apb_t2 as T

P0 = T.P0


def _rohstichproben(rng, m, p):
    X = rng.standard_normal((m, p))
    xb = X.mean(axis=0)
    D = X - xb
    return xb, D.T @ D / (m - 1)


# ----------------------------------------------------------------------------------------------------------------------------------- (e)
def test_imhof_gegen_geschlossenen_fall_p1():
    rng = np.random.default_rng(13700)
    xb = rng.standard_normal(30) / math.sqrt(20)
    s = np.sqrt(rng.chisquare(19, 30) / 19)
    ex = ndtr(-xb - 3 * s) + ndtr(xb - 3 * s)
    im = T.imhof_sf((1 / s ** 2)[:, None], (-xb)[:, None], 9.0, 1000.0)
    assert np.max(np.abs(im - ex)) < 5e-8


@pytest.mark.parametrize("p", [2, 5, 10])
def test_imhof_gegen_ncx2_fuer_s_gleich_i(p):
    rng = np.random.default_rng(13701 + p)
    xb = rng.standard_normal((20, p)) / math.sqrt(30)
    ucl = T.ucl_chi2(p)
    ex = ncx2.sf(ucl, p, (xb ** 2).sum(axis=1))
    im = T.imhof_sf(np.ones((20, p)), -xb, ucl, T.u_for(p))
    im15 = T.imhof_sf(np.ones((20, p)), -xb, ucl, 1.5 * T.u_for(p))
    assert np.max(np.abs(im - ex)) < 2e-9 and np.max(np.abs(im - im15)) < 2e-9
    assert abs(T.imhof_sf(np.ones((1, p)), np.zeros((1, p)), ucl, T.u_for(p))[0] - P0) < 2e-9      # x̄ = 0, S = I: 1/370,398


@pytest.mark.parametrize("p", [2, 5, 10])
def test_imhof_gegen_simulation_4_millionen_vektoren(p):
    """Eine Phase-I-Stichprobe (m = 30) aus Rohdaten: Alarmanteil aus 2 Mio. neuen Vektoren gegen das Imhof-Integral der Demo (4 Standardfehler)."""
    rng = np.random.default_rng(13710 + p)
    z0 = N.Stream(40 + p).normal((1, 30, p))
    d, a, xb, S = T.phase1_from_raw(z0)
    ucl = T.ucl_chi2(p)
    ex = float(T.cond_alarm_prob(d, a, ucl, p)[0])
    Sinv = np.linalg.inv(S[0])
    cnt, n = 0, 2_000_000
    for _ in range(2):
        z = rng.standard_normal((n // 2, p)) - xb[0][None, :]
        cnt += int((np.einsum("ij,jk,ik->i", z, Sinv, z) > ucl).sum())
    r = cnt / n
    assert abs(r - ex) <= 4 * math.sqrt(ex * (1 - ex) / n)


def test_bedingte_alarmwahrscheinlichkeit_aus_rohdaten_gegen_einzelrechnung():
    """Die gestapelte Demo-Rechnung (Eigenzerlegung, Imhof) gegen eine Einzelrechnung je Stichprobe mit numpy-Rohdaten, S direkt (nicht über V)."""
    z = N.Stream(2).normal((6, 40, 3))
    d, a, xb, S = T.phase1_from_raw(z)
    ucl = T.ucl_chi2(3)
    pal = T.cond_alarm_prob(d, a, ucl, 3)
    for r in range(6):
        lam, V = np.linalg.eigh(S[r])
        a1 = -(V.T @ xb[r])
        assert np.allclose(lam, d[r]) and np.allclose(np.abs(a1), np.abs(a[r]))
        assert abs(T.imhof_sf((1 / lam)[None, :], a1[None, :], ucl, T.u_for(3))[0] - pal[r]) < 1e-12


# ----------------------------------------------------------------------------------------------------------------------------------- (f)
@pytest.mark.parametrize("p,m", [(2, 30), (5, 30), (10, 100)])
def test_f_verteilung_des_phase_ii_t2_gegen_rohnormalzahlen(p, m):
    rng = np.random.default_rng(13720 + p)
    R, cnt_c, cnt_f = (600_000 if p * m <= 300 else 150_000), 0, 0
    for _ in range(3):
        Rr = R // 3
        X = rng.standard_normal((Rr, m, p))
        xb = X.mean(axis=1)
        D = X - xb[:, None, :]
        S = np.einsum("rki,rkj->rij", D, D) / (m - 1)
        z = rng.standard_normal((Rr, p)) - xb
        t2 = np.einsum("ri,rij,rj->r", z, np.linalg.inv(S), z)
        cnt_c += int((t2 > T.ucl_chi2(p)).sum())
        cnt_f += int((t2 > T.ucl_f(m, p)).sum())
    pc, pf = T.alarm_prob_uncond(T.ucl_chi2(p), m, p), 1 / T.GAMMA0
    assert abs(cnt_c / R - pc) <= 4 * math.sqrt(pc * (1 - pc) / R)
    assert abs(cnt_f / R - pf) <= 4 * math.sqrt(pf * (1 - pf) / R)


@pytest.mark.parametrize("p,m", [(2, 4), (2, 10), (2, 50)])
def test_p2_geschlossene_f_verteilungsfunktion(p, m):
    nu = m - 2.0
    y = T.ucl_chi2(2) / T.f_factor(m, 2)
    assert abs((1 + 2 * y / nu) ** (-nu / 2) - T.alarm_prob_uncond(T.ucl_chi2(2), m, 2)) < 1e-12


def test_m_kleiner_gleich_p_ist_eine_meldung():
    with pytest.raises(ValueError):
        T.f_factor(5, 5)
    with pytest.raises(ValueError):
        T.phase1_from_raw(np.zeros((2, 5, 5)))


# ----------------------------------------------------------------------------------------------------------------------------------- (g)
@pytest.mark.parametrize("p,m", [(2, 30), (5, 100), (10, 100)])
def test_mittlere_alarmrate_der_stichproben_gegen_exakte_f_rechnung(p, m):
    R = 500
    cell = T.t2_cell(N.Stream(11 * p + m), R, m, p)
    for rule in ("chi2", "F"):
        rate = cell[rule]["rate"]
        ex = T.alarm_prob_uncond(T.ucl_rule(rule, m, p), m, p)
        assert abs(rate.mean() - ex) <= 4 * rate.std(ddof=1) / math.sqrt(R)


# ----------------------------------------------------------------------------------------------------------------------------------- (h)
def test_p1_chi2_regel_ist_die_skalare_karte_exakt_274():
    """p = 1: die T²-Karte mit χ²-Grenze 9 ist die Karte x̄ ± 3 s (Stück 4). Die Mindestzahl 274 steht exakt aus dem Doppelintegral (tests/test_oracle_ar1.py)."""
    from test_oracle_ar1 import iid_anteil_x_aussen
    assert iid_anteil_x_aussen(273, T.HALF) > 0.05 >= iid_anteil_x_aussen(274, T.HALF)
    assert abs(T.ucl_chi2(1) - 9.0) < 1e-9


def test_p1_bedingte_alarmwahrscheinlichkeit_gleich_skalarer_karte():
    z = N.Stream(3).normal((50, 30, 1))
    d, a, xb, S = T.phase1_from_raw(z)
    pal = T.cond_alarm_prob(d, a, 9.0, 1)
    s = np.sqrt(S[:, 0, 0])
    karte = 1 - (ndtr(xb[:, 0] + 3 * s) - ndtr(xb[:, 0] - 3 * s))
    assert np.allclose(pal, karte, atol=1e-12)


# ----------------------------------------------------------------------------------------------------------------------------------- Eigenschaften
def test_umnummerieren_und_drehung_aendern_die_bedingte_arl_nicht():
    """Affine Invarianz: Kanäle umnummerieren und die Phase-I-Daten orthogonal drehen (die Phase-II-Verteilung N(0, I) ist drehinvariant)."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        p = int(rng.integers(2, 6))
        z = rng.standard_normal((1, 40, p))
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        perm = rng.permutation(p)
        base = T.cond_alarm_prob(*T.phase1_from_raw(z)[:2], T.ucl_chi2(p), p)
        drehung = T.cond_alarm_prob(*T.phase1_from_raw(z @ Q)[:2], T.ucl_chi2(p), p)
        umnum = T.cond_alarm_prob(*T.phase1_from_raw(z[:, :, perm])[:2], T.ucl_chi2(p), p)
        assert np.allclose(base, drehung, rtol=1e-6, atol=1e-10) and np.allclose(base, umnum, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("p", [1, 2, 5, 10])
@pytest.mark.parametrize("m", [12, 30, 100, 300, 1000])
def test_f_regel_hat_unbedingt_370_398(p, m):
    assert abs(T.arl0_uncond("F", m, p) - T.GAMMA0) < 1e-5


def test_chi2_regel_waechst_in_m_und_faellt_in_p():
    ms = (30, 100, 300, 1000)
    for p in (1, 2, 5, 10):
        v = [T.arl0_uncond("chi2", m, p) for m in ms]
        assert v == sorted(v) and v[-1] < T.GAMMA0
    for m in ms:
        v = [T.arl0_uncond("chi2", m, p) for p in (1, 2, 5, 10)]
        assert v == sorted(v, reverse=True)


def test_grenzen_der_chi2_regel():
    assert [round(T.ucl_chi2(p), 3) for p in (1, 2, 5, 10)] == [9.000, 11.829, 18.205, 26.901]


def test_f_grenze_liegt_ueber_der_chi2_grenze_und_naehert_sich_ihr():
    for p in (2, 5, 10):
        v = [T.ucl_f(m, p) / T.ucl_chi2(p) for m in (30, 100, 300, 1000)]
        assert all(x > 1 for x in v) and v == sorted(v, reverse=True)


def test_preis_der_f_regel_nimmt_mit_m_ab():
    for p in (2, 5, 10):
        v = [T.arl1_price("F", m, p) for m in (100, 300, 1000)]
        assert v == sorted(v, reverse=True) and v[-1] > T.arl1_price("chi2", 1000, p)


def test_bedingte_arl_verteilung_ist_monoton_und_endet_bei_eins():
    cell = T.t2_cell(N.Stream(1), 200, 100, 5)
    for rule in ("chi2", "F"):
        v = np.sort(cell[rule]["arl"])
        y = (np.arange(v.size) + 1) / v.size
        assert np.all(np.diff(v) >= 0) and y[-1] == 1.0 and v[0] > 1.0


def test_gleicher_zufalls_seed_gleiche_ausgabe():
    a = T.t2_cell(N.Stream(5), 50, 60, 3)
    b = T.t2_cell(N.Stream(5), 50, 60, 3)
    assert all(np.array_equal(a[r]["arl"], b[r]["arl"]) for r in ("chi2", "F"))


def test_singulaere_stichprobe_wird_gemeldet():
    z = np.zeros((1, 30, 3))
    with pytest.raises(ValueError):
        d, a, _, _ = T.phase1_from_raw(z)
        T.cond_alarm_prob(d, a, 10.0, 3)

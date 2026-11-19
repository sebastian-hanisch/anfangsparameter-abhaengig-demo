"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): m = 2, φ am Rand, R = 1, Obergrenze der Live-Läufe, m <= p, Zufalls-Seed 0, Werte außerhalb der Tabellen."""

import numpy as np
import pytest

import apb_ar1 as A
import apb_normal as N
import apb_reference as REF
import apb_scenario as SC
import apb_t2 as T


def test_m_gleich_2_rohdatenkarte_laeuft_und_phi_dach_ist_minus_halb():
    """Der kleinste Fall: zwei Phase-I-Werte (Zweige `m < 3` und Division durch m - 1 dürfen nicht greifen)."""
    x = A.ar1_samples(N.Stream(1), 20, 2, 0.9)
    xb, s, ph, _ = A.estimates(x)
    assert np.allclose(ph, -0.5, atol=1e-12)
    arl, rate = A.raw_chart_cond(x, 0.9)
    assert np.all(np.isfinite(arl)) and np.all(arl >= 1.0) and np.all((rate > 0) & (rate <= 1))


def test_m_gleich_2_mittlere_alarmrate_aus_den_stichproben_gegen_arkustangens():
    x = A.ar1_samples(N.Stream(2), 4000, 2, 0.5)
    _, rate = A.raw_chart_cond(x, 0.5)
    se = rate.std(ddof=1) / np.sqrt(rate.size)
    assert abs(rate.mean() - A.mean_alarm_prob_m2(0.5)) <= 4 * se


@pytest.mark.parametrize("phi", [0.0, 0.95, -0.6])
def test_phi_am_rand_und_negativ(phi):
    """φ = 0 (iid), φ = 0,95 (Regler-Obergrenze) und negatives φ (Nyström rechnet; nicht in der Messung): endliche bedingte ARL₀, bekannte ARL₀ >= 1."""
    x = A.ar1_samples(N.Stream(3), 30, 60, phi)
    arl, rate = A.raw_chart_cond(x, phi)
    assert np.all(np.isfinite(arl)) and np.all(arl >= 1.0)
    assert A.known_arl0(phi) > 300
    arl2, _, _ = A.resid_chart_cond(x[:5], phi, False)
    assert np.all(np.isfinite(arl2)) and np.all(arl2 >= 1.0 - 1e-9)


def test_phi_negativ_gleich_positiv_bei_bekannten_grenzen():
    assert abs(A.known_arl0(-0.6) - A.known_arl0(0.6)) < 1e-6


def test_eine_einzige_live_stichprobe_r_gleich_1():
    live = SC.ar1_live(0.9, 100, 1, "roh", 0)
    assert len(live.arl) == 1 and live.summary["p_below"] in (0.0, 1.0)
    t = SC.t2_live(5, 100, 1, ("chi2", "F"), 0)
    assert len(t.rules["chi2"]["arl"]) == 1


def test_zufalls_seed_null_ist_gueltig():
    a, b = SC.ar1_live(0.6, 50, 5, "roh", 0), SC.ar1_live(0.6, 50, 5, "roh", 0)
    assert a.arl == b.arl and a.x0 == b.x0
    assert SC.ar1_live(0.6, 50, 5, "roh", 1).arl != a.arl


def test_obergrenze_der_live_laeufe_ist_eine_meldung_keine_rechnung():
    with pytest.raises(SC.LiveLimitError):
        SC.t2_live(10, 1000, 300, ("chi2",), 1)
    with pytest.raises(SC.LiveLimitError):
        SC.ar1_live(0.9, 1000, 5000, "roh", 1)
    SC.check_limit(200, 1000, 10)          # genau 2 000 000: erlaubt
    with pytest.raises(SC.LiveLimitError):
        SC.check_limit(201, 1000, 10)


def test_m_kleiner_gleich_p_wird_gemeldet():
    with pytest.raises(ValueError):
        T.ucl_f(10, 10)
    with pytest.raises(ValueError):
        T.t2_cell(N.Stream(1), 5, 5, 5)


def test_alle_phase_i_werte_gleich_bei_t2_ist_singulaer():
    with pytest.raises(ValueError):
        d, a, _, _ = T.phase1_from_raw(np.ones((1, 20, 2)))
        T.cond_alarm_prob(d, a, 10.0, 2)


def test_tabellen_ausserhalb_der_messung_geben_none():
    assert REF.ar1_cell("roh", 0.45, 100) is None and REF.ar1_cell("roh", 0.9, 50) is None
    assert REF.ar1_cell("resid_bekannt", 0.0, 100) is None            # Residuenkarte erst ab φ = 0,3 gemessen
    assert REF.mindest_ar1(0.45) is None and REF.t2_cell("chi2", 3, 100) is None and REF.mindest_t2("F", 7) is None


def test_arl_der_rohdatenkarte_waechst_mit_s_und_faellt_mit_x_bar_abstand():
    """Randverhalten der bedingten Karte: breitere Grenzen -> größere ARL₀; verschobene Grenzen (x̄ ≠ 0) -> kleinere ARL₀ als die symmetrischen."""
    sym = [A.raw_chart_arl([-w], [w], 0.6)[0] for w in (1.5, 2.0, 3.0, 4.0)]
    assert sym == sorted(sym)
    assert A.raw_chart_arl([-2.0], [4.0], 0.6)[0] < A.raw_chart_arl([-3.0], [3.0], 0.6)[0]

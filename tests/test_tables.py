"""Gefrorene Tabellen (apb_tables.json): Aufbau, Herkunft, Beziehungen untereinander, Neurechnung im Band und Abgleich mit den Vormessungen V13b/V13c.

Neu gerechnet wird als Band (|z| <= 4, Standardfehler beider Seiten), nie bitgleich: die Normalzahlen kommen aus der Umkehrfunktion der C-Bibliothek,
Linux und Windows können in der letzten Binärstelle abweichen (DEMO-PLAYBOOK §5). Die Vormessungswerte (Plan 04 Befund I und J) sind Erwartungen, keine Ziele:
die Tabelle ist mit anderem Zufalls-Seed neu gemessen; die Abweichungen stehen in der README („Befunde und Korrekturen“)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import apb_reference as REF
import apb_t2 as T

ROOT = Path(__file__).resolve().parent.parent
T_ = REF.tables()


def _band(z_p, z_ref, R1=4000, R2=4000):
    return abs(z_p - z_ref) / math.sqrt(max(z_ref * (1 - z_ref), 1e-4) * (1 / R1 + 1 / R2))


def test_kopf_nennt_quelle_stand_und_zufall():
    m = REF.meta()
    assert m["stand"] == "2026-11-18" and "SplitMix64" in m["zufall"] and m["R"] == 4000 and "V13b" in m["vormessung"] and "Simulation" in m["hinweis"]


def test_jede_zelle_traegt_seed_und_r_und_quantile_sind_monoton():
    cells = list(T_["ar1"].values()) + list(T_["t2"].values())
    assert len(cells) == 16 + 24 + 24          # roh 16, Residuenkarte 2 x 12, T² 3 p x 4 m x 2 Regeln
    seeds = set()
    for c in cells:
        assert c["R"] == 4000 and isinstance(c["seed"], int)
        q = c["quantile"]
        assert len(q) == 99 and q == sorted(q) and q[0] >= 1.0
        assert c["q05"] <= c["med"] <= c["q95"]
        assert 0.0 <= c["p_below"] <= 1.0
        seeds.add(c["seed"])
    assert len(seeds) == 16 + 24 + 12            # beide T²-Regeln rechnen auf denselben Stichproben (gleicher Seed); alle anderen Zellen haben je einen eigenen


def test_anteil_stimmt_mit_den_gespeicherten_quantilen_ueberein():
    """p_below ist der Anteil unter 185,2; die 99 Quantile (1 % bis 99 %) müssen dazu passen (Interpolation, Band 0,02)."""
    for key, c in {**T_["ar1"], **T_["t2"]}.items():
        q = np.array(c["quantile"])
        p = np.interp(185.199, q, np.arange(1, 100) / 100, left=0.0, right=1.0)
        if 0.015 < c["p_below"] < 0.985:
            assert abs(p - c["p_below"]) < 0.02, key


def test_tabellen_enthalten_die_gemessenen_zellen_und_sonst_nichts():
    ar1 = set(T_["ar1"])
    assert ar1 == {f"roh|{ph}|{m}" for ph in ("0.0", "0.3", "0.6", "0.9") for m in (30, 100, 300, 1000)} | \
        {f"{k}|{ph}|{m}" for k in ("resid_bekannt", "resid_geschaetzt") for ph in ("0.3", "0.6", "0.9") for m in (30, 100, 300, 1000)}
    assert set(T_["t2"]) == {f"{r}|{p}|{m}" for r in ("chi2", "F") for p in (2, 5, 10) for m in (30, 100, 300, 1000)}


def test_anteil_faellt_mit_m_bis_auf_rauschen():
    for phi in ("0.0", "0.3", "0.6", "0.9"):
        v = [T_["ar1"][f"roh|{phi}|{m}"]["p_below"] for m in (30, 100, 300, 1000)]
        assert all(a >= b - 0.015 for a, b in zip(v, v[1:])), phi
    for p in (2, 5, 10):
        for rule in ("chi2", "F"):
            v = [T_["t2"][f"{rule}|{p}|{m}"]["p_below"] for m in (30, 100, 300, 1000)]
            assert all(a >= b - 0.015 for a, b in zip(v, v[1:])), (p, rule)


def test_roh_anteil_waechst_mit_phi_bei_m_30_bis_300():
    """„Die Abhängigkeit verschlechtert die Phase-I-Karte“ nur so: Rohdatenkarte, AR(1), m = 30 / 100 / 300, Anteil unter 185,2 wächst mit φ (Rauschen 0,015)."""
    for m in (30, 100, 300):
        v = [T_["ar1"][f"roh|{ph}|{m}"]["p_below"] for ph in ("0.0", "0.3", "0.6", "0.9")]
        assert all(a <= b + 0.015 for a, b in zip(v, v[1:])) and v[-1] > v[0] + 0.05, m
    assert all(T_["ar1"][f"roh|{ph}|1000"]["p_below"] < 0.01 for ph in ("0.0", "0.3", "0.6", "0.9"))      # bei m = 1000 alle Anteile unter 1 %


def test_flotte_der_t2_tabelle_gegen_exakte_f_rechnung():
    for p in (2, 5, 10):
        for m in (30, 100, 300, 1000):
            for rule in ("chi2", "F"):
                ex = T.arl0_uncond(rule, m, p)
                assert abs(T_["t2"][f"{rule}|{p}|{m}"]["flotte"] / ex - 1) < 0.08, (rule, p, m)


def test_exakte_iid_werte_gegen_zweiten_rechenweg_und_plan_04():
    from test_oracle_ar1 import iid_anteil_x_aussen
    for m, soll in ((30, 0.35966), (100, 0.17920), (300, 0.04185), (1000, 0.00058)):
        assert abs(REF.iid_exact(m)["p_below"] - soll) < 1e-4 and abs(iid_anteil_x_aussen(m, 185.199) - REF.iid_exact(m)["p_below"]) < 1e-5
    assert REF.iid_exact(100)["median"] == pytest.approx(341.996, abs=1e-3) and REF.iid_min_m() == 274


# ----------------------------------------------------------------------------------------------------------------------------------- Vormessung
V13 = {   # Plan 04 Befund I und J (V13b / V13c, numpy-Zufall, andere Zufalls-Seeds): Anteil unter 185,2
    "roh|0.9|30": 0.793, "roh|0.9|100": 0.440, "roh|0.9|300": 0.134, "roh|0.6|100": 0.298, "roh|0.3|100": 0.226, "roh|0.0|100": 0.186,
    "resid_bekannt|0.9|100": 0.188, "resid_geschaetzt|0.9|100": 0.297, "resid_geschaetzt|0.9|30": 0.676, "resid_bekannt|0.9|30": 0.375,
}
V13C = {"chi2|2|100": 0.248, "chi2|5|100": 0.557, "chi2|10|100": 0.974, "F|2|100": 0.074, "F|5|100": 0.062, "F|10|100": 0.059, "F|5|30": 0.139, "F|2|30": 0.139, "F|10|30": 0.133}


@pytest.mark.parametrize("key,soll", sorted(V13.items()))
def test_ar1_zellen_gegen_vormessung_v13b_im_band(key, soll):
    assert _band(T_["ar1"][key]["p_below"], soll) <= 4, key


@pytest.mark.parametrize("key,soll", sorted(V13C.items()))
def test_t2_zellen_gegen_vormessung_v13c_im_band(key, soll):
    assert _band(T_["t2"][key]["p_below"], soll) <= 4, key


def test_neurechnung_kleines_gitter_gegen_gespeicherte_tabelle():
    """tools/apb_make_tables.py --check: m = 100 und 300, R = 400, anderer Zufalls-Seed, Band 4 Standardfehler (rund 10 s)."""
    spec = importlib.util.spec_from_file_location("apb_make_tables", ROOT / "tools" / "apb_make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["apb_make_tables"] = mod
    spec.loader.exec_module(mod)
    assert mod.check() == 0


# ----------------------------------------------------------------------------------------------------------------------------------- Mindestzahlen
def test_mindestzahlen_aufbau_und_gruppenvergleich():
    for phi, rec in T_["mindest_ar1"].items():
        assert rec["mstar"] is not None and 30 <= rec["mstar"] <= 1600
        assert rec["p_at"]["m*"] <= 0.06 and rec["p_at"]["0,8 m*"] >= rec["p_at"]["1,25 m*"]
    m = [T_["mindest_ar1"][k]["mstar"] for k in ("0.0", "0.3", "0.6", "0.9")]
    assert m == sorted(m) and m[-1] > m[0] * 1.4
    for key, rec in T_["mindest_t2"].items():
        assert rec["mstar"] is not None and rec["p_at"]["m*"] <= 0.06 and rec["p_at"]["0,8 m*"] >= rec["p_at"]["1,25 m*"], key


def test_mindestzahl_f_regel_kleiner_als_chi2_regel_und_p1_nahe_der_exakten_274():
    for p in (1, 2, 5, 10):
        assert T_["mindest_t2"][f"F|{p}"]["mstar"] < T_["mindest_t2"][f"chi2|{p}"]["mstar"], p
    assert 225 <= T_["mindest_t2"]["chi2|1"]["mstar"] <= 330 and abs(T_["mindest_t2"]["chi2|1"]["mstar"] / 274 - 1) < 0.15


def test_mindestzahl_chi2_regel_waechst_mit_p_ab_p_2_und_f_regel_fast_nicht():
    c = [T_["mindest_t2"][f"chi2|{p}"]["mstar"] for p in (2, 5, 10)]
    f = [T_["mindest_t2"][f"F|{p}"]["mstar"] for p in (2, 5, 10)]
    assert c == sorted(c) and c[-1] > 1.4 * c[0]
    assert max(f) / min(f) < 1.4

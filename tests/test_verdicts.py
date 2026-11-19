"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds; die Tests prüfen die Grenzen (3 Standardfehler, 80 % von 370,4) und jede genannte Zahl."""

import math

import apb_evaluation as ev


def _se(p, R):
    return math.sqrt(p * (1 - p) / R)


# ----------------------------------------------------------------------------------------------------------------------------------- Format
def test_zahlenformat_mit_dezimalkomma():
    assert ev.de(3.14159, 2) == "3,14" and ev.pct(0.1792) == "17,9 %" and ev.de_s(-2.5, 1) == "-2,5" and ev.de_s(0.0, 1) == "+0,0"
    assert ev.pct(0.0006, 2) == "0,06 %"


def test_standardfehler_hat_untergrenze_fuer_nullen():
    assert ev.se_share(0.0, 200) == 0.5 / 200
    assert abs(ev.se_share(0.18, 100) - math.sqrt(0.18 * 0.82 / 100)) < 1e-15


# ----------------------------------------------------------------------------------------------------------------------------------- Experiment 1
def test_exp1_ueber_drei_standardfehlern_ist_warnung_und_nennt_jede_zahl():
    iid, R = 0.1792, 200
    p = iid + 3 * _se(iid, R) + 0.01
    kind, text = ev.verdict_exp1(0.9, 100, "Rohdatenkarte", p, 224.7, 831.8, R, iid)
    assert kind == "warning"
    for teil in ("φ = 0,90, m = 100, Rohdatenkarte", f"bei {R} Live-Phase-I-Stichproben haben {ev.pct(p)} eine bedingte ARL₀ unter 185,2", "exakt 17,9 %", "Median 225",
                 "Bekannte Parameter: ARL₀ 831,8", f"Das sind {ev.de(100 * (p - iid), 1)} Prozentpunkte mehr", "zero-state"):
        assert teil in text, teil


def test_exp1_genau_drei_standardfehler_ist_noch_info():
    iid, R = 0.1792, 200
    kind, text = ev.verdict_exp1(0.9, 100, "Rohdatenkarte", iid + 3 * _se(iid, R) - 1e-6, 300.0, 831.8, R, iid)
    assert kind == "info" and "innerhalb von 3 Standardfehlern" in text


def test_exp1_unabhaengig_meldet_keine_verschlechterung():
    """Das Preset „Unabhängig“ darf keine Wirkung der Abhängigkeit melden, auch wenn der Live-Anteil zufällig hoch liegt."""
    kind, text = ev.verdict_exp1(0.0, 100, "Rohdatenkarte", 0.30, 330.0, 370.4, 200, 0.1792)
    assert kind == "info" and "ändert hier nichts" in text and "Prozentpunkte mehr" not in text


def test_exp1_niedriger_anteil_ist_info():
    kind, _ = ev.verdict_exp1(0.9, 1000, "Rohdatenkarte", 0.004, 696.0, 831.8, 200, 0.0006)
    assert kind == "info"


# ----------------------------------------------------------------------------------------------------------------------------------- Experiment 2
def test_exp2_chi2_unter_80_prozent_ist_warnung():
    chi = dict(ucl=18.2051, arl0=154.7, p_live=0.557)
    f = dict(ucl=20.7681, arl0=370.398, p_live=0.062)
    kind, text = ev.verdict_exp2(5, 100, 200, chi, f, (114.4, 290.9), 1.0)
    assert kind == "warning"
    for teil in ("p = 5, m = 100", "Grenze 18,21", "unbedingte ARL₀ 154,7", "55,7 % der 200 Live-Stichproben unter 185,2", "Grenze 20,77", "unbedingte ARL₀ 370,4",
                 "6,2 % der 200", "ARL₁ (ein Kanal springt um 1,0 σ, bekannte Parameter, Σ = I) 114,4 → 290,9"):
        assert teil in text, teil


def test_exp2_schwelle_bei_80_prozent_von_370_4():
    chi = lambda a: dict(ucl=18.0, arl0=a, p_live=0.1)   # noqa: E731
    assert ev.verdict_exp2(2, 1000, 100, chi(0.8 * 370.398 - 0.1), None, (1.0, 2.0), 1.0)[0] == "warning"
    assert ev.verdict_exp2(2, 1000, 100, chi(0.8 * 370.398 + 0.1), None, (1.0, 2.0), 1.0)[0] == "info"


def test_exp2_nur_f_regel_nennt_keine_chi2_zeile():
    kind, text = ev.verdict_exp2(2, 100, 100, None, dict(ucl=12.8, arl0=370.398, p_live=0.07), (67.3, 99.2), 1.0)
    assert kind == "info" and "χ²-Regel" not in text and "F-Regel (Grenze 12,80)" in text


# ----------------------------------------------------------------------------------------------------------------------------------- Experiment 3
def test_exp3_ar1_nennt_mindestzahl_und_iid_zahl():
    kind, text = ev.verdict_exp3_ar1(0.9, dict(mstar=509), 274)
    assert kind == "info" and "φ = 0,90" in text and "etwa 509" in text and "274 (exakt)" in text and "10 bis 15 %" in text


def test_exp3_ar1_ausserhalb_der_messung_und_ohne_ueberquerung():
    assert "nicht zu den gemessenen Werten" in ev.verdict_exp3_ar1(0.45, None, 274)[1]
    assert ev.verdict_exp3_ar1(0.9, dict(mstar=None), 274)[0] == "warning"


def test_exp3_t2_nennt_beide_mindestzahlen_und_den_preis():
    kind, text = ev.verdict_exp3_t2(5, dict(mstar=315), dict(mstar=121), (150.2, 400.9))
    assert kind == "info" and "etwa 315 Vektoren mit der χ²-Regel" in text and "etwa 121 mit der F-Regel" in text and "m = 121) 150,2 → 400,9" in text
    assert ev.verdict_exp3_t2(5, dict(mstar=None), dict(mstar=121), None)[0] == "warning"


def test_settings_regeln():
    assert ev.Settings(regel="χ²-Regel").regeln() == ("chi2",)
    assert ev.Settings(regel="F-Regel").regeln() == ("F",)
    assert ev.Settings().regeln() == ("chi2", "F")

"""Claims: jede Zahl der README wird hier aus der Rechnung bzw. der gefrorenen Tabelle neu erzeugt und muss wörtlich in der README stehen.

Die Tabelle „Zahl → Test“ der README nennt je Zeile die Funktion. Zufallsabhängige Zahlen (Verteilungen über Phase-I-Stichproben, Mindestzahlen) stehen in
`apb_tables.json` und werden in tests/test_tables.py im Band neu gerechnet; exakte Zahlen (bekannte Parameter, unbedingte ARL₀, Preis) werden hier gerechnet.
Zahlen der Vormessung (Plan) stehen als Erwartung in der README; ist der Plan-Ordner erreichbar, prüft ein Test, dass sie dort so stehen."""

import math
import re
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter

import apb_ar1 as A
import apb_reference as REF
import apb_scenario as SC
import apb_t2 as T
import apb_texts as TX
from apb_evaluation import de, pct

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
TB = REF.tables()


def _slash(werte, f=str, sep=" / "):
    return sep.join(f(w) for w in werte)


def _im_readme(text):
    assert text in README, f"nicht in der README: {text!r}"


def _p(kind, phi, m):
    return TB["ar1"][f"{kind}|{phi}|{m}"]["p_below"]


def test_arl0_bekannte_parameter():
    soll = {0.0: 370.398, 0.3: 376.383, 0.6: 419.377, 0.9: 831.783}
    for phi, s in soll.items():
        assert abs(A.known_arl0(phi) - s) < 5e-4
    _im_readme(_slash(soll.values(), lambda v: de(v, 3)))


def test_rohdatenkarte_phi_09():
    _im_readme(_slash((_p("roh", "0.9", m) for m in (30, 100, 300, 1000)), pct))
    _im_readme(_slash((TB["ar1"][f"roh|0.9|{m}"]["med"] for m in (30, 100, 300, 1000)), lambda v: de(v, 0)))


def test_unabhaengige_werte_exakt():
    _im_readme(_slash((REF.iid_exact(m)["p_below"] for m in (30, 100, 300)), pct) + " / " + pct(REF.iid_exact(1000)["p_below"], 2) + "; Median " + de(REF.iid_exact(100)["median"], 0))


def test_rohdatenkarte_anteil_waechst_mit_phi():
    for m in (100, 30):
        _im_readme(f"m = {m}: " + _slash((_p("roh", ph, m) for ph in ("0.0", "0.3", "0.6", "0.9")), lambda v: de(100 * v, 1), " / ") + " %")
    v = [_p("roh", ph, 100) for ph in ("0.0", "0.3", "0.6", "0.9")]
    assert v == sorted(v) and v[-1] > 2 * v[0]


def test_residuenkarte():
    _im_readme(_slash((_p("resid_bekannt", ph, 100) for ph in ("0.3", "0.6", "0.9")), lambda v: de(100 * v, 1), " / ") + " %")
    _im_readme(_slash((_p("resid_geschaetzt", ph, 100) for ph in ("0.3", "0.6", "0.9")), lambda v: de(100 * v, 1), " / ") + " %")
    _im_readme(pct(_p("resid_bekannt", "0.9", 30)) + " / " + pct(_p("resid_geschaetzt", "0.9", 30)))
    mehr = max(_p("resid_geschaetzt", ph, m) - _p("resid_bekannt", ph, m) for ph in ("0.3", "0.6", "0.9") for m in (300, 1000))
    _im_readme(f"bis zu (φ = 0,3 / 0,6 / 0,9, m = 300 und 1000) | {de(100 * mehr, 1)} Prozentpunkte")
    # Zählung: bekanntes φ liegt in allen 3 φ bei m = 100 innerhalb von 3 Prozentpunkten um die exakte iid-Zahl; geschätztes φ in keinem darunter... (Band)
    iid = REF.iid_exact(100)["p_below"]
    assert all(abs(_p("resid_bekannt", ph, 100) - iid) < 0.03 for ph in ("0.3", "0.6", "0.9"))
    assert all(_p("resid_geschaetzt", ph, 100) > iid + 0.05 for ph in ("0.3", "0.6", "0.9"))
    _im_readme("(m = 100: 18,9 bis 20,0 % gegen 17,9 % exakt)")
    _im_readme("(24,1 bis 31,1 % bei m = 100)")


def test_streuung_und_mitte():
    assert abs(A.e_s2_closed(100, 0.9) - 0.836) < 5e-4 and abs(A.var_xbar_closed(100, 0.9) - 0.172) < 5e-4
    _im_readme("0,836 und 0,172")


def _bias(m, phi=0.9, R=200_000, seed=13501):
    rng = np.random.default_rng(seed + m)
    e = rng.standard_normal((R, m))
    e[:, 1:] *= math.sqrt(1 - phi ** 2)
    x = lfilter([1.0], [1.0, -phi], e, axis=1)
    return float(A.estimates(x)[2].mean()) - phi


def test_verzerrung_phi_dach():
    b30, b100 = _bias(30), _bias(100)
    assert abs(b30 + 0.180) < 0.006 and abs(b100 + 0.051) < 0.005
    _im_readme("−0,180 / −0,051")


def test_t2_unbedingt():
    for m, p_list in ((100, (2, 5, 10)), (30, (2, 5, 10))):
        _im_readme(_slash((T.arl0_uncond("chi2", m, p) for p in p_list), lambda v: de(v, 1)))
    assert [round(T.ucl_f(100, p), 2) for p in (2, 5, 10)] == [12.83, 20.77, 33.19]
    _im_readme("12,83 / 20,77 / 33,19; 370,398")
    _im_readme("11,83 / 18,21 / 26,90")
    assert all(abs(T.arl0_uncond("F", 100, p) - T.GAMMA0) < 1e-5 for p in (2, 5, 10))


def test_t2_bedingt():
    def t(rule, p, m):
        return TB["t2"][f"{rule}|{p}|{m}"]["p_below"]
    _im_readme(_slash((t("chi2", p, 100) for p in (2, 5, 10)), lambda v: de(100 * v, 1), " / ") + " %")
    _im_readme(_slash((t("F", p, 100) for p in (2, 5, 10)), lambda v: de(100 * v, 1), " / ") + " %")
    _im_readme(_slash((t("F", p, 30) for p in (2, 5, 10)), lambda v: de(100 * v, 1), " / ") + " %")
    # Zählung (Band): F-Regel bei m = 100 in allen drei p unter 10 %; χ²-Regel in allen drei über 20 %
    assert all(t("F", p, 100) < 0.10 and t("chi2", p, 100) > 0.20 for p in (2, 5, 10))


def test_preis_der_f_regel():
    teile = [f"{de(T.arl1_price('chi2', 100, p), 1)} → {de(T.arl1_price('F', 100, p), 1)}" for p in (2, 5, 10)]
    _im_readme(_slash(teile))


def test_mindestzahlen():
    ar = [TB["mindest_ar1"][f"{ph}"]["mstar"] for ph in ("0.0", "0.3", "0.6", "0.9")]
    _im_readme(_slash(ar))
    _im_readme(_slash((TB["mindest_t2"][f"chi2|{p}"]["mstar"] for p in (1, 2, 5, 10))))
    _im_readme(_slash((TB["mindest_t2"][f"F|{p}"]["mstar"] for p in (1, 2, 5, 10))))
    assert REF.iid_min_m() == 274
    _im_readme("unabhängig exakt 274")


def test_handbeispiele_in_der_readme():
    assert abs(A.e_s2_closed(3, 0.5) - 7 / 12) < 1e-13 and abs(A.var_xbar_closed(3, 0.5) - 11 / 18) < 1e-13
    ps = [A.mean_alarm_prob_m2(p) for p in (0.0, 0.5, 0.9)]
    assert [round(v, 4) for v in ps] == [0.2468, 0.3550, 0.6201]
    assert abs(T.arl0_uncond("chi2", 4, 2) - 4.1544) < 5e-4 and abs(T.ucl_f(4, 2) - 1385.2) < 0.1
    for teil in ("E[s²] = 7/12, Var(x̄) = 11/18", "φ̂ = −½", "0,2468 / 0,3550 / 0,6201", "ARL₀ der χ²-Regel 4,154", "Grenze der F-Regel 1385,2"):
        _im_readme(teil)


def test_ziel_und_halbes_ziel():
    assert abs(T.GAMMA0 - 1 / (2 * 0.0013498980316301)) < 1e-3 and abs(T.HALF - T.GAMMA0 / 2) < 1e-12
    assert abs(T.GAMMA0 - 370.398) < 5e-4 and abs(T.HALF - 185.2) < 5e-3
    _im_readme("185,2 = 370,398 / 2; 370,398 = 1/(2 Φ(−3))")


def test_worst_case_laufzeit():
    """Worst-Case am Regler-Maximum: Obergrenze 20 s (Timeout statt Zeit-Assertion; die README nennt die Messung dieser Maschine, unter 1 s)."""
    t0 = time.perf_counter()
    SC.ar1_live(0.95, 1000, 500, "resid_geschaetzt", 1)
    SC.ar1_live(0.95, 30, 500, "roh", 1)
    SC.t2_live(10, 1000, 200, ("chi2", "F"), 1)
    SC.t2_live(10, 100, 500, ("chi2", "F"), 1)
    assert time.perf_counter() - t0 < 20.0


def test_annahmen_text_steht_wortgleich_in_der_readme():
    assert TX.ASSUMPTIONS_MARKDOWN.strip() in README


# ----------------------------------------------------------------------------------------------------------------------------------- Plan-Erwartungen
def test_erwartungen_der_readme_stehen_so_im_plan_wenn_der_plan_erreichbar_ist():
    plan = Path(__file__).resolve().parents[2] / "aenderungspunkt-planung" / "PLAN-04b-phase1-abhaengig-kanaele.md"
    if not plan.exists():
        pytest.skip("Planordner liegt nicht neben der Demo")
    text = plan.read_text(encoding="utf-8")
    for teil in ("44,0 %", "79,3 %", "55,7 %", "266 / 315 / 480", "147 / 121 / 110", "509", "413"):
        assert teil in text, teil
    for teil in ("44,0 %", "79,3 % / 13,4 % / 0,4 %", "55,7 %", "280 / 328 / 413 / 509", "266 / 315 / 480", "147 / 121 / 110", "18,8 %", "29,7 %", "67,6 %", "7,4 / 6,2 / 5,9 %"):
        assert teil in README, teil


def test_keine_verbotenen_woerter_in_texten():
    from_app = " ".join([TX.INTRO, TX.SO_FUNKTIONIERT, TX.exp1_frage(), TX.exp1_negativ(), TX.exp1_falle(), TX.exp2_frage(), TX.exp2_negativ(), TX.exp2_falle(), TX.exp3_frage(), TX.exp3_negativ(),
                         TX.ASSUMPTIONS_MARKDOWN, TX.MATH_INTRO, TX.CONVENTIONS_MARKDOWN, TX.MATH_MARKDOWN, README])
    for wort in ("Saatwert", "trägt", "tragen", "Schwanz", "TODO", "geplant"):
        assert not re.search(rf"\b{wort}\b", from_app), wort
    assert "Zufalls-Seed" in from_app and "Zufallsseed" not in from_app

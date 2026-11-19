"""Je Schnellstart die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a, Punkt 7): ein Preset ohne Wirkung meldet keine Wirkung.

Live-Läufe mit dem Standard-Zufalls-Seed der App (R = 200); die Erwartungen sind die Art (warning / info) und der Wortlaut des ersten Satzteils."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import apb_constants as C

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _run(name):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _verdikt(at, anfang):
    treffer = [(w, w.value) for w in list(at.warning) + list(at.info) + list(at.success) if w.value.startswith(anfang)]
    assert len(treffer) == 1, (anfang, [w.value[:60] for w in list(at.warning) + list(at.info)])
    return treffer[0]


# (Preset, Anfang des Verdikts von Experiment 1, erwartete Art, Teil des Textes)
EXP1 = [
    ("Stark (φ = 0,9, m = 100)", "φ = 0,90, m = 100, Rohdatenkarte", "warning", "Prozentpunkte mehr"),
    ("Unabhängig (φ = 0, m = 100)", "φ = 0,00, m = 100, Rohdatenkarte", "info", "ändert hier nichts"),
    ("Mittel (φ = 0,6, m = 100)", "φ = 0,60, m = 100, Rohdatenkarte", "warning", "Prozentpunkte mehr"),
    ("Stark (φ = 0,9, m = 30)", "φ = 0,90, m = 30, Rohdatenkarte", "warning", "Prozentpunkte mehr"),
    ("Residuenkarte, φ bekannt", "φ = 0,90, m = 100, Residuenkarte, φ bekannt", "info", "innerhalb von 3 Standardfehlern"),
    ("Residuenkarte, φ geschätzt", "φ = 0,90, m = 100, Residuenkarte, φ geschätzt", "warning", "Prozentpunkte mehr"),
]


@pytest.mark.parametrize("name,anfang,kind,teil", EXP1)
def test_verdikt_experiment_1_je_preset(name, anfang, kind, teil):
    w, text = _verdikt(_run(name), anfang)
    assert type(w).__name__.lower() == kind and teil in text


@pytest.mark.parametrize("name,p,m", [("Stark (φ = 0,9, m = 100)", 5, 100), ("T²: p = 2, m = 30", 2, 30), ("T²: p = 10, m = 100", 10, 100)])
def test_verdikt_experiment_2_je_preset(name, p, m):
    w, text = _verdikt(_run(name), f"p = {p}, m = {m}: χ²-Regel")
    assert type(w).__name__.lower() == "warning" and "unbedingte ARL₀" in text and "F-Regel (Grenze" in text and "ARL₁ (ein Kanal springt um 1,0 σ" in text


def test_preset_mindestzahl_t2_nennt_beide_regeln():
    at = _run("Mindestzahl (T², p = 5)")
    w, text = _verdikt(at, "T², p = 5: Mindestzahl etwa")
    assert "Vektoren mit der χ²-Regel" in text and "mit der F-Regel" in text and "ARL₁ (ein Kanal springt um 1 σ" in text
    assert len(at.get("plotly_chart")) == 4


def test_jedes_preset_liefert_drei_verdikte_oder_eine_meldung():
    for name in C.PRESETS:
        at = _run(name)
        texte = [w.value for w in list(at.warning) + list(at.info) + list(at.success)]
        assert any("Live-Phase-I-Stichproben" in t for t in texte), name
        assert any("unbedingte ARL₀" in t for t in texte), name
        assert any("Mindestzahl" in t for t in texte), name

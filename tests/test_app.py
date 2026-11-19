"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Erweiterungen dieser Demo (der Anfang der Datei bleibt die Vorlage aus tools/templates/test_app_smoke.py) -----------------------------------
import pytest  # noqa: E402

import apb_constants as C  # noqa: E402
import apb_presets as P  # noqa: E402


def test_seite_hat_drei_experimente_und_titel():
    at = _fresh_app()
    assert at.title and len(at.subheader) >= 4


def _texte(at) -> str:
    return " ".join([w.value for w in at.success] + [w.value for w in at.info] + [w.value for w in at.warning])


def test_standard_zeigt_alle_drei_verdikte_und_die_diagramme():
    at = _fresh_app()
    text = _texte(at)
    assert "bei 200 Live-Phase-I-Stichproben" in text and "unbedingte ARL₀" in text and "Mindestzahl etwa" in text
    assert len(at.get("plotly_chart")) == 4          # Reihe, Verteilung (Exp. 1), Verteilung (Exp. 2), Mindestzahl


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_und_setzt_alle_regler(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, state_key in P.PRESET_KEYS.items():
        soll = C.PRESETS[name].get(key, P.SETTING_SPECS[state_key].default)
        ist = at.session_state[state_key]
        assert abs(ist - soll) < 1e-9 if isinstance(soll, float) else ist == soll, (name, state_key)


def test_zufalls_button_aendert_den_seed():
    at = _fresh_app()
    alt = at.session_state["seed_input"]
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["seed_input"] != alt


def test_every_option_of_every_selection_does_not_raise():
    """Jede Option jeder Auswahl (select_slider, radio) einzeln; die Werte stammen aus SETTING_SPECS (AppTest nennt die Optionen als Text)."""
    gefunden = 0
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is None:
            continue
        for option in spec.options:
            at = _fresh_app()
            try:
                w = at.sidebar.select_slider(key=state_key)
            except KeyError:
                w = at.sidebar.radio(key=state_key)
            w.set_value(option).run(timeout=TIMEOUT)
            gefunden += 1
            assert not at.exception, (state_key, option, [str(e) for e in at.exception])
            assert at.session_state[state_key] == option
    assert gefunden == 4 + 3 + 4 + 3 + 3 + 2        # m_ar, p, m_t2, Karte, Regel, Modell


def test_obergrenze_der_live_laeufe_zeigt_meldung_statt_rechnung():
    """p = 10, m = 1000, R = 500: R · m · p = 5 Mio. liegt über 2 Mio.: Warnung, kein Absturz, keine Verteilung für Experiment 2."""
    at = _fresh_app()
    at.sidebar.select_slider(key="p_select").set_value(10).run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="m_t2_select").set_value(1000).run(timeout=TIMEOUT)
    at.sidebar.slider(key="r_t2_slider").set_value(500).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Live-Lauf nicht gerechnet" in w.value for w in at.warning)


def test_phi_ausserhalb_der_messung_nennt_es_und_rechnet_live():
    at = _fresh_app()
    at.sidebar.slider(key="phi_slider").set_value(0.45).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("gehört nicht zu den gemessenen Werten" in i.value for i in at.info)
    assert "bei 200 Live-Phase-I-Stichproben" in _texte(at)

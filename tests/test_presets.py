"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset einzigartig; Wirkung je Preset (Verdikt-Zeile)."""

import apb_constants as C
import apb_evaluation as ev
import apb_presets as P
import apb_reference as REF


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def _wert(preset, key):
    return preset.get(key, P.SETTING_SPECS[P.PRESET_KEYS[key]].default)


def test_preset_values_are_inside_the_regler_bounds_and_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec, v = P.SETTING_SPECS[state_key], _wert(preset, key)
            if spec.options is not None:
                assert v in spec.options, (name, key)
            elif spec.lo is not None:
                assert spec.lo <= v <= spec.hi, (name, key)


def test_default_preset_is_the_first_and_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {} and ev.Settings() == ev.Settings(**{k: _wert(C.PRESETS[first], k) for k in P.PRESET_KEYS})


def test_every_preset_differs_from_every_other():
    """Zwei Presets mit denselben Reglerwerten wären nicht unterscheidbar (aktives_preset nähme immer das erste)."""
    stand = [tuple(_wert(p, k) for k in P.PRESET_KEYS) for p in C.PRESETS.values()]
    assert len(set(stand)) == len(stand)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_phi_presets_liegen_auf_der_gemessenen_gitterung_und_m_in_der_tabelle():
    for name, preset in C.PRESETS.items():
        phi, m = _wert(preset, "phi"), _wert(preset, "m_ar")
        assert any(abs(phi - g) < 1e-9 for g in C.PHI_GRID), name
        assert REF.ar1_cell("roh", phi, m) is not None, name
        assert REF.t2_cell("F", _wert(preset, "p"), _wert(preset, "m_t2")) is not None, name


def test_permalink_rastet_ein_und_begrenzt():
    spec = P.SETTING_SPECS["phi_slider"]
    assert P._snap(0.47, spec) == 0.45 and P._snap(2.0, spec) == 0.95 and P._snap(-1.0, spec) == 0.0
    m = P.SETTING_SPECS["m_ar_select"]
    assert P._snap(80, m) == 100 and P._snap(5000, m) == 1000
    k = P.SETTING_SPECS["karte_radio"]
    assert P._snap("Unsinn", k) is None and P._snap(C.KARTE_RESID_BEKANNT, k) == C.KARTE_RESID_BEKANNT
    r = P.SETTING_SPECS["r_ar_slider"]
    assert P._snap(260, r) == 300 and P._snap(9999, r) == 500

"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import apb_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # Auswahl (select_slider, radio): nur diese Werte sind gültig
    step: float | None = None         # Raster eines Zahlenreglers (Permalink rastet ein)


SETTING_SPECS = {
    "phi_slider": SettingSpec("phi", float, C.DEFAULT_PHI, C.PHI_MIN, C.PHI_MAX, step=C.PHI_STEP),
    "m_ar_select": SettingSpec("mar", int, C.DEFAULT_M_AR, C.M_AR_OPTIONS[0], C.M_AR_OPTIONS[-1], options=C.M_AR_OPTIONS),
    "karte_radio": SettingSpec("karte", str, C.DEFAULT_KARTE, options=C.KARTEN),
    "r_ar_slider": SettingSpec("rar", int, C.DEFAULT_R_AR, C.R_MIN, C.R_MAX, step=C.R_STEP),
    "p_select": SettingSpec("p", int, C.DEFAULT_P, C.P_OPTIONS[0], C.P_OPTIONS[-1], options=C.P_OPTIONS),
    "m_t2_select": SettingSpec("mt2", int, C.DEFAULT_M_T2, C.M_T2_OPTIONS[0], C.M_T2_OPTIONS[-1], options=C.M_T2_OPTIONS),
    "regel_radio": SettingSpec("regel", str, C.DEFAULT_REGEL, options=C.REGELN),
    "r_t2_slider": SettingSpec("rt2", int, C.DEFAULT_R_T2, C.R_MIN, C.R_MAX, step=C.R_STEP),
    "delta_slider": SettingSpec("delta", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX, step=C.DELTA_STEP),
    "modell_radio": SettingSpec("modell", str, C.DEFAULT_MODELL, options=C.MODELLE),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"phi": "phi_slider", "m_ar": "m_ar_select", "karte": "karte_radio", "r_ar": "r_ar_slider", "p": "p_select", "m_t2": "m_t2_select",
               "regel": "regel_radio", "r_t2": "r_t2_slider", "delta": "delta_slider", "modell": "modell_radio", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(value, spec: SettingSpec):
    """Permalink-Wert auf die Reglergrenzen begrenzen und auf das Raster bzw. die Auswahl einrasten; None, wenn nicht möglich."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else None
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.lo is not None:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(spec.hi, value)
        if spec.caster is float:
            value = round(value, 4)
    return spec.caster(value)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _snap(value, spec)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt ALLE Regler: was es nicht nennt, kehrt zum Standardwert zurück (sonst bliebe ein früheres Preset halb erhalten)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)

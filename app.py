"""Phase I unter Abhängigkeit und mit vielen Kanälen: dieselbe Karte, andere Annahme - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4b der Reihe „Änderungserkennung“: die Phase-I-Karte aus Stück 4, wenn die Werte abhängig sind (AR(1)) oder mehrere Kanäle zugleich laufen (Hotelling-T²).
Gemessen wird die Qualität der Karte, die Phase I geliefert hat: ihre bedingte ARL₀, exakt je Phase-I-Stichprobe.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import apb_constants as C
import apb_evaluation as EV
import apb_reference as REF
import apb_scenario as SC
import apb_t2 as T2
import apb_texts as TX
import apb_visualization as V
from apb_ar1 import known_arl0
from apb_layout import dunkles_design, kopf_css
from apb_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Phase I unter Abhängigkeit und mit vielen Kanälen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _ar1(phi, m, R, karte_key, seed):
    return SC.ar1_live(phi, m, R, karte_key, seed)


@st.cache_data(show_spinner=False)
def _t2(p, m, R, rules, seed):
    return SC.t2_live(p, m, R, rules, seed)


@st.cache_data(show_spinner=False)
def _known(phi):
    return known_arl0(phi)


de, pct, de_s = EV.de, EV.pct, EV.de_s

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in apb_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Phase I unter Abhängigkeit und mit vielen Kanälen: dieselbe Karte, andere Annahme")
st.markdown(TX.INTRO)

with st.expander("So funktioniert die Phase-I-Karte und wie sie hier geprüft wird", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Experiment 1: abhängige Werte")
    phi = st.slider("AR-Koeffizient φ", *bounds("phi_slider"), step=C.PHI_STEP, key="phi_slider", format="%.2f",
                    help="Autokorrelation zwischen Wert t und t + k ist φ hoch k. Gemessen (gefroren) sind φ = 0 / 0,3 / 0,6 / 0,9; dazwischen rechnet nur der Live-Lauf.")
    m_ar = st.select_slider("Phase-I-Umfang m (Werte)", options=C.M_AR_OPTIONS, key="m_ar_select", help="Zahl der Phase-I-Werte der Reihe.")
    karte = st.radio("Karte", C.KARTEN, key="karte_radio", help="Rohdatenkarte: Grenzen x̄ ± 3 s, als wären die Werte unabhängig. Residuenkarte: überwacht Residuen, mit bekanntem oder geschätztem φ.")
    r_ar = st.slider("Live-Stichproben R", *bounds("r_ar_slider"), step=C.R_STEP, key="r_ar_slider", help="Zahl der Phase-I-Stichproben, die live gerechnet werden.")
    st.header("⚙️ Experiment 2: viele Kanäle")
    p = st.select_slider("Zahl der Kanäle p", options=C.P_OPTIONS, key="p_select", help="Kanäle sind unabhängig und standardnormal (wegen der affinen Invarianz von T² genügt Σ = I).")
    m_t2 = st.select_slider("Phase-I-Umfang m (Vektoren)", options=C.M_T2_OPTIONS, key="m_t2_select", help="Zahl der Phase-I-Vektoren; m muss größer sein als p.")
    regel = st.radio("Grenze", C.REGELN, key="regel_radio", help="χ²-Regel: Grenze für bekanntes Σ. F-Regel: Grenze so, dass die unbedingte ARL₀ 370,398 ist.")
    r_t2 = st.slider("Live-Stichproben R (T²)", *bounds("r_t2_slider"), step=C.R_STEP, key="r_t2_slider", help="Zahl der Phase-I-Stichproben, die live gerechnet werden (R · m · p höchstens 2 000 000).")
    delta = st.slider("Sprung eines Kanals δ (σ)", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider", format="%.1f",
                      help="Für den Preis der F-Grenze: ein Kanal springt um δ Standardabweichungen (bekannte Parameter, nicht Phase-I-gemessen).")
    st.header("⚙️ Experiment 3: Mindestzahl")
    modell = st.radio("Modell", C.MODELLE, key="modell_radio", help="Die Mindestzahl steht in einer gefrorenen Tabelle (Simulation, gelb); φ und p kommen aus den Reglern der Experimente 1 und 2.")
    st.header("⚙️ Zufall")
    seed = st.number_input("Zufalls-Seed der Live-Läufe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Live-Stichproben", width="stretch", on_click=randomize_seed)

sync_query_params({"phi_slider": float(phi), "m_ar_select": int(m_ar), "karte_radio": karte, "r_ar_slider": int(r_ar), "p_select": int(p), "m_t2_select": int(m_t2),
                   "regel_radio": regel, "r_t2_slider": int(r_t2), "delta_slider": float(delta), "modell_radio": modell, "seed_input": int(seed)})
S = EV.Settings(round(float(phi), 4), int(m_ar), karte, int(r_ar), int(p), int(m_t2), regel, int(r_t2), float(delta), modell, int(seed))
karte_key = C.KARTE_KEY[S.karte]

# =====================================================================================================================================================
# Experiment 1
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 1: Phase I aus einer abhängigen Reihe")
st.markdown(TX.exp1_frage())
live1 = None
try:
    with st.spinner(f"Rechne {S.r_ar} Phase-I-Stichproben (m = {S.m_ar}) und ihre bedingte ARL₀ exakt …"):
        live1 = _ar1(S.phi, S.m_ar, S.r_ar, karte_key, S.seed)
except SC.LiveLimitError as e:
    st.warning(f"Live-Lauf nicht gerechnet: {e}.")
if live1 is not None:
    arl1 = np.array(live1.arl)
    iid = REF.iid_exact(S.m_ar)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anteil unter 185,2 (live)", pct(live1.summary["p_below"]), delta=f"{de_s(100 * (live1.summary['p_below'] - iid['p_below']), 1)} Prozentpunkte gegen unabhängig", delta_color="inverse")
    c2.metric("Median der bedingten ARL₀", de(live1.summary["med"], 0), delta=f"{de_s(live1.summary['med'] - iid['median'], 0)} gegen unabhängig ({de(iid['median'], 0)})", delta_color="normal")
    c3.metric("ARL₀ bei bekannten Parametern", de(live1.known, 1), delta="zum Vergleich", delta_color="off")
    c4.metric("Mittleres φ̂ der Stichproben", de(live1.mean_phi_hat, 2), delta=f"wahres φ = {de(S.phi, 2)}", delta_color="off")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(V.build_series(live1.x0, live1.xbar0, live1.s0, S.phi, live1.phi_hat0), width="stretch", key="series_chart")
        st.caption(f"Erste der Live-Stichproben: x̄ = {de(live1.xbar0, 2)}, s = {de(live1.s0, 2)}, φ̂ = {de(live1.phi_hat0, 2)} (wahr: φ = {de(S.phi, 2)}). Die gebauten Grenzen sehen nur diese Werte; die wahren Grenzen kennt die Karte nicht.")
    frozen_cell = REF.ar1_cell(karte_key, S.phi, S.m_ar)
    frozen_curves = {}
    if frozen_cell is not None:
        col = {C.KARTE_ROH: C.COLOR_ROH, C.KARTE_RESID_BEKANNT: C.COLOR_BEKANNT, C.KARTE_RESID_GESCHAETZT: C.COLOR_GESCHAETZT}[S.karte]
        frozen_curves = {S.karte: (frozen_cell["quantile"], C.COLOR_FROZEN)}
    with right:
        col = {C.KARTE_ROH: C.COLOR_ROH, C.KARTE_RESID_BEKANNT: C.COLOR_BEKANNT, C.KARTE_RESID_GESCHAETZT: C.COLOR_GESCHAETZT}[S.karte]
        st.plotly_chart(V.build_cdf({S.karte: (arl1, col)}, frozen_curves,
                                    [(C.HALF, "185,2", "#555"), (live1.known, f"bekannt: {de(live1.known, 1)}", C.COLOR_BEKANNT)],
                                    "Verteilung der bedingten ARL₀ (Phase-I-Stichproben)"), width="stretch", key="cdf1_chart")
        st.caption(f"Live gerechnet in {de(live1.seconds, 2)} s (Rechenzeit dieser Maschine, kein Befund). Werte außerhalb von 1 bis 100 000 stehen am Rand.")
    kind, text = EV.verdict_exp1(S.phi, S.m_ar, S.karte, live1.summary["p_below"], live1.summary["med"], live1.known, S.r_ar, iid["p_below"])
    getattr(st, kind)(text)
    if frozen_cell is None:
        st.info(f"φ = {de(S.phi, 2)} gehört nicht zu den gemessenen Werten (0 / 0,3 / 0,6 / 0,9; Residuenkarten nur ab φ = 0,3): die Live-Läufe rechnen es, die gefrorene Tabelle hat dafür keine Zeile.")

st.markdown("**Gefrorene Tabelle** (4000 Phase-I-Stichproben je Zeile, bedingte ARL₀ exakt je Stichprobe, Verteilung Simulation mit Standardfehler bis 0,008; Zähleinheit Phase-I-Stichproben):")
rows = []
for kn, kk in ((C.KARTE_ROH, "roh"), (C.KARTE_RESID_BEKANNT, "resid_bekannt"), (C.KARTE_RESID_GESCHAETZT, "resid_geschaetzt")):
    cell = REF.ar1_cell(kk, S.phi, S.m_ar)
    if cell is not None:
        rows.append({"Karte": kn, "Anteil < 185,2": pct(cell["p_below"]), "Median": de(cell["med"], 0), "5 %-Quantil": de(cell["q05"], 0), "95 %-Quantil": de(cell["q95"], 0),
                     "Kehrwert der mittleren Alarmrate": de(cell["flotte"], 0), "Zufalls-Seed / R": f"{cell['seed']} / {cell['R']}"})
iid = REF.iid_exact(S.m_ar)
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Unabhängige Werte (Stück 4, exakt, m = {S.m_ar}): Anteil unter 185,2 {pct(iid['p_below'])}, Median {de(iid['median'], 0)}. Herkunft: `apb_tables.json` aus `tools/apb_make_tables.py` "
               f"(SplitMix64, Stand {REF.meta()['stand']}).")
else:
    st.caption("Für diesen Wert von φ gibt es keine gefrorene Zeile.")
st.markdown(TX.exp1_negativ())
st.warning(TX.exp1_falle())

# =====================================================================================================================================================
# Experiment 2
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 2: Phase I mit geschätzter Kovarianz (Hotelling-T²)")
st.markdown(TX.exp2_frage())
live2 = None
try:
    with st.spinner(f"Rechne {S.r_t2} Phase-I-Stichproben (p = {S.p}, m = {S.m_t2}) mit Eigenzerlegung und Imhof-Integral …"):
        live2 = _t2(S.p, S.m_t2, S.r_t2, S.regeln(), S.seed)
except SC.LiveLimitError as e:
    st.warning(f"Live-Lauf nicht gerechnet: {e}.")
price = (T2.arl1_price("chi2", S.m_t2, S.p, S.delta), T2.arl1_price("F", S.m_t2, S.p, S.delta))
if live2 is not None:
    rule_info = {}
    for rule in S.regeln():
        rule_info[rule] = dict(ucl=T2.ucl_rule(rule, S.m_t2, S.p), arl0=T2.arl0_uncond(rule, S.m_t2, S.p), p_live=live2.rules[rule]["summary"]["p_below"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grenze χ²-Regel / F-Regel", f"{de(T2.ucl_chi2(S.p), 2)} / {de(T2.ucl_f(S.m_t2, S.p), 2)}", delta="Σ als bekannt gebaut / unbedingt eingestellt", delta_color="off")
    c2.metric("Unbedingte ARL₀ der χ²-Regel", de(T2.arl0_uncond("chi2", S.m_t2, S.p), 1), delta=f"{de_s(T2.arl0_uncond('chi2', S.m_t2, S.p) - C.GAMMA0, 1)} gegen 370,4", delta_color="normal")
    c3.metric("ARL₁ χ² → F (bekannte Parameter)", f"{de(price[0], 1)} → {de(price[1], 1)}", delta=f"ein Kanal um {de(S.delta, 1)} σ", delta_color="off")
    c4.metric("Live-Lauf", f"{de(live2.seconds, 2)} s", delta=f"{S.r_t2} Stichproben, p = {S.p}, m = {S.m_t2}", delta_color="off")
    live_curves, frozen_curves = {}, {}
    for rule, (nm, col) in {"chi2": ("χ²-Regel", C.COLOR_CHI2), "F": ("F-Regel", C.COLOR_F)}.items():
        if rule in live2.rules:
            live_curves[nm] = (np.array(live2.rules[rule]["arl"]), col)
            fc = REF.t2_cell(rule, S.p, S.m_t2)
            if fc is not None:
                frozen_curves[nm] = (fc["quantile"], C.COLOR_FROZEN)
    st.plotly_chart(V.build_cdf(live_curves, frozen_curves, [(C.HALF, "185,2", "#555"), (C.GAMMA0, "370,4 (bekanntes Σ)", C.COLOR_BEKANNT)],
                                "Verteilung der bedingten ARL₀ der T²-Karte"), width="stretch", key="cdf2_chart")
    st.caption(f"Live gerechnet in {de(live2.seconds, 2)} s (Rechenzeit dieser Maschine, kein Befund). Werte außerhalb von 1 bis 100 000 stehen am Rand.")
    kind, text = EV.verdict_exp2(S.p, S.m_t2, S.r_t2, rule_info.get("chi2"), rule_info.get("F"), price, S.delta)
    getattr(st, kind)(text)

st.markdown("**Grenzen, unbedingte ARL₀ (exakt, Kehrwert der mittleren Alarmrate) und Preis der F-Grenze für p = " + str(S.p) + "** (Phase-II-Vektoren bis zum Alarm; ARL₁ bei bekannten Parametern, Σ = I):")
rows = []
for m in C.M_T2_OPTIONS:
    rows.append({"m": m, "UCL χ²": de(T2.ucl_chi2(S.p), 2), "UCL F": de(T2.ucl_f(m, S.p), 2), "ARL₀ χ²-Regel": de(T2.arl0_uncond("chi2", m, S.p), 1), "ARL₀ F-Regel": de(T2.arl0_uncond("F", m, S.p), 1),
                 "ARL₁ χ²-Grenze": de(T2.arl1_price("chi2", m, S.p, S.delta), 1), "ARL₁ F-Grenze": de(T2.arl1_price("F", m, S.p, S.delta), 1)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown("**Gefrorene Tabelle** (4000 Phase-I-Stichproben je Zeile; bedingte ARL₀ exakt je Stichprobe (Imhof), Verteilung Simulation):")
rows = []
for rule, nm in (("chi2", "χ²-Regel"), ("F", "F-Regel")):
    cell = REF.t2_cell(rule, S.p, S.m_t2)
    if cell is not None:
        rows.append({"Regel": nm, "Anteil < 185,2": pct(cell["p_below"]), "Median": de(cell["med"], 0), "5 %-Quantil": de(cell["q05"], 0), "95 %-Quantil": de(cell["q95"], 0),
                     "Kehrwert der mittleren Alarmrate": de(cell["flotte"], 1), "Zufalls-Seed / R": f"{cell['seed']} / {cell['R']}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown(TX.exp2_negativ())
st.warning(TX.exp2_falle())

# =====================================================================================================================================================
# Experiment 3
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 3: Wie viele Phase-I-Werte braucht die Karte jetzt?")
st.markdown(TX.exp3_frage())
iid_min = REF.iid_min_m()
if S.modell == C.MODELL_AR:
    curves = []
    for ph, col in ((0.0, "rgba(90,90,90,0.9)"), (S.phi, C.COLOR_ROH)):
        rec = REF.mindest_ar1(ph)
        if rec is not None and not (ph == S.phi and S.phi == 0.0 and curves):
            curves.append((f"φ = {de(ph, 1)}", rec["raster"], rec.get("mstar"), col))
    if curves:
        st.plotly_chart(V.build_mindest(curves, iid_min), width="stretch", key="mindest_chart")
    kind, text = EV.verdict_exp3_ar1(S.phi, REF.mindest_ar1(S.phi), iid_min)
else:
    curves = []
    for rule, nm, col in (("chi2", "χ²-Regel", C.COLOR_CHI2), ("F", "F-Regel", C.COLOR_F)):
        rec = REF.mindest_t2(rule, S.p)
        curves.append((f"{nm}, p = {S.p}", rec["raster"], rec.get("mstar"), col))
    st.plotly_chart(V.build_mindest(curves, iid_min), width="stretch", key="mindest_chart")
    rf, rc = REF.mindest_t2("F", S.p), REF.mindest_t2("chi2", S.p)
    pr = None
    if rf.get("mstar"):
        pr = (T2.arl1_price("chi2", rf["mstar"], S.p), T2.arl1_price("F", rf["mstar"], S.p))
    kind, text = EV.verdict_exp3_t2(S.p, rc, rf, pr)
getattr(st, kind)(text)
st.caption("Kurven: Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2 je Umfang m (Simulation, 2000 Stichproben je Rasterpunkt, SplitMix64); m* aus Halbierung in log m, Prüfung mit 20 000 bzw. 8000 Stichproben. "
           "Auf etwa 10 bis 15 % genau (gelb). Gestrichelt: 5 %; gepunktet: unabhängige Werte (exakt).")

rows = []
for ph in C.PHI_GRID:
    rec = REF.mindest_ar1(ph)
    rows.append({"Modell": f"AR(1), Rohdatenkarte, φ = {de(ph, 1)}", "Regel": "x̄ ± 3 s", "Mindestzahl m* (Beobachtungen bzw. Vektoren)": rec["mstar"], "P bei 0,8 m* / m* / 1,25 m*": " / ".join(pct(rec["p_at"][k]) for k in ("0,8 m*", "m*", "1,25 m*"))})
for p_ in (1,) + C.P_OPTIONS:
    for rule, nm in (("chi2", "χ²-Regel"), ("F", "F-Regel")):
        rec = REF.mindest_t2(rule, p_)
        rows.append({"Modell": f"T², p = {p_}" + (" (= skalare Karte, Stück 4)" if p_ == 1 else ""), "Regel": nm, "Mindestzahl m* (Beobachtungen bzw. Vektoren)": rec["mstar"],
                     "P bei 0,8 m* / m* / 1,25 m*": " / ".join(pct(rec["p_at"][k]) for k in ("0,8 m*", "m*", "1,25 m*"))})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown(TX.exp3_negativ())

# =====================================================================================================================================================
# Grenzen und Formeln
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATH_INTRO)
    st.markdown(TX.CONVENTIONS_MARKDOWN)
    st.markdown(TX.MATH_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)

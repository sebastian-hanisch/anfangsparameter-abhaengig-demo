"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Experiment 1: autokorrelierte Phase I --------------------------------------------------------------------------------------------------
PHI_MIN, PHI_MAX, PHI_STEP, DEFAULT_PHI = 0.0, 0.95, 0.05, 0.9
PHI_GRID = (0.0, 0.3, 0.6, 0.9)                     # die gemessenen (gefrorenen) Werte von φ
M_AR_OPTIONS, DEFAULT_M_AR = (30, 100, 300, 1000), 100
KARTE_ROH, KARTE_RESID_BEKANNT, KARTE_RESID_GESCHAETZT = "Rohdatenkarte", "Residuenkarte, φ bekannt", "Residuenkarte, φ geschätzt"
KARTEN = (KARTE_ROH, KARTE_RESID_BEKANNT, KARTE_RESID_GESCHAETZT)
KARTE_KEY = {KARTE_ROH: "roh", KARTE_RESID_BEKANNT: "resid_bekannt", KARTE_RESID_GESCHAETZT: "resid_geschaetzt"}
DEFAULT_KARTE = KARTE_ROH
R_MIN, R_MAX, R_STEP = 100, 500, 100                  # Zahl der Live-Phase-I-Stichproben
DEFAULT_R_AR, DEFAULT_R_T2 = 200, 200

# --- Experiment 2: viele Kanäle -------------------------------------------------------------------------------------------------------------
P_OPTIONS, DEFAULT_P = (2, 5, 10), 5
M_T2_OPTIONS, DEFAULT_M_T2 = (30, 100, 300, 1000), 100
REGEL_CHI2, REGEL_F, REGEL_BEIDE = "χ²-Regel", "F-Regel", "beide Regeln"
REGELN = (REGEL_CHI2, REGEL_F, REGEL_BEIDE)
DEFAULT_REGEL = REGEL_BEIDE
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.5, 3.0, 0.5, 1.0   # Sprung eines Kanals in σ für den ARL₁-Preis

# --- Experiment 3: Mindestzahl --------------------------------------------------------------------------------------------------------------
MODELL_AR, MODELL_T2 = "AR(1)-Rohdatenkarte (φ aus Experiment 1)", "T²-Karte (p aus Experiment 2)"
MODELLE = (MODELL_AR, MODELL_T2)
DEFAULT_MODELL = MODELL_AR

SEED_MAX, DEFAULT_SEED = 999999, 35
LIVE_LIMIT = 2_000_000                               # Obergrenze für R · m · p (Zufallszahlen eines Live-Laufs); R · m bei AR(1)
ARL_PLOT_MIN, ARL_PLOT_MAX = 1.0, 1.0e5               # Achsenbereich der Verteilungsfunktionen (Werte außerhalb stehen am Rand)

GAMMA0 = 370.398                                     # 1 / (2 Φ(-3)): ARL₀-Ziel (nicht das gesetzte 370)
HALF = GAMMA0 / 2.0                                  # 185,2: die halbe nominelle ARL₀

COLOR_ROH = "#c0392b"
COLOR_BEKANNT = "#2F6B65"
COLOR_GESCHAETZT = "#d68910"
COLOR_FROZEN = "rgba(90,90,90,0.9)"
COLOR_CHI2 = "#c0392b"
COLOR_F = "#2F6B65"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Stark (φ = 0,9, m = 100)": {},
    "Unabhängig (φ = 0, m = 100)": {"phi": 0.0},
    "Mittel (φ = 0,6, m = 100)": {"phi": 0.6},
    "Stark (φ = 0,9, m = 30)": {"m_ar": 30},
    "Residuenkarte, φ bekannt": {"karte": KARTE_RESID_BEKANNT},
    "Residuenkarte, φ geschätzt": {"karte": KARTE_RESID_GESCHAETZT},
    "T²: p = 2, m = 30": {"p": 2, "m_t2": 30},
    "T²: p = 10, m = 100": {"p": 10},
    "Mindestzahl (T², p = 5)": {"modell": MODELL_T2},
}
PRESET_HELP = {
    "Stark (φ = 0,9, m = 100)": "Voreinstellung: Rohdatenkarte aus 100 Werten einer AR(1)-Reihe mit φ = 0,9, daneben T² mit p = 5 Kanälen und m = 100 Vektoren.",
    "Unabhängig (φ = 0, m = 100)": "Gleiche Karte bei unabhängigen Werten: hier ist die Annahme aus Stück 4 erfüllt, die Abhängigkeit ändert nichts.",
    "Mittel (φ = 0,6, m = 100)": "Mittlere Abhängigkeit (φ = 0,6), Rohdatenkarte, m = 100.",
    "Stark (φ = 0,9, m = 30)": "Starke Abhängigkeit bei kleinem m: die Rohdatenkarte schätzt Mitte und Streuung aus nur 30 Werten.",
    "Residuenkarte, φ bekannt": "Die Karte kennt φ und überwacht Residuen; Mitte und Streuung der Residuen bleiben geschätzt. Der Bestfall.",
    "Residuenkarte, φ geschätzt": "Die Karte schätzt auch φ aus Phase I (Lag-1-Schätzer, bei kleinem m nach unten verzerrt).",
    "T²: p = 2, m = 30": "Zwei Kanäle, Phase I aus nur 30 Vektoren (Experiment 2).",
    "T²: p = 10, m = 100": "Zehn Kanäle, Phase I aus 100 Vektoren: die χ²-Regel gilt dann nur auf dem Papier (Experiment 2).",
    "Mindestzahl (T², p = 5)": "Experiment 3 für T² mit p = 5 Kanälen: wie viele Vektoren braucht Phase I?",
}

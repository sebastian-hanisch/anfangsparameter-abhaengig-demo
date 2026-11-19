"""Einstellungen, Zahlenformat und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Maß in jedem Satz: bedingte ARL₀ = Beobachtungen (AR(1)) bzw. Vektoren (T²) bis zum Alarm der Karte, die Phase I geliefert hat, gegeben ihre Grenzen
(zero-state); Zähleinheit der Anteile: Phase-I-Stichproben; 185,2 = halbe nominelle ARL₀ (370,398 / 2)."""

import math
from dataclasses import dataclass

import apb_constants as C


@dataclass(frozen=True)
class Settings:
    phi: float = C.DEFAULT_PHI
    m_ar: int = C.DEFAULT_M_AR
    karte: str = C.DEFAULT_KARTE
    r_ar: int = C.DEFAULT_R_AR
    p: int = C.DEFAULT_P
    m_t2: int = C.DEFAULT_M_T2
    regel: str = C.DEFAULT_REGEL
    r_t2: int = C.DEFAULT_R_T2
    delta: float = C.DEFAULT_DELTA
    modell: str = C.DEFAULT_MODELL
    seed: int = C.DEFAULT_SEED

    def regeln(self) -> tuple[str, ...]:
        """Die gerechneten Regeln: Schlüssel chi2 / F."""
        return {C.REGEL_CHI2: ("chi2",), C.REGEL_F: ("F",), C.REGEL_BEIDE: ("chi2", "F")}[self.regel]


# ------------------------------------------------------------------------------------------------------------------ Zahlenformat (Dezimalkomma)
def de(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def de_s(x: float, nd: int = 1) -> str:
    """Mit Vorzeichen (ASCII-Minus, damit st.metric die Farbe des Deltas erkennt): mein Wert minus Referenz."""
    return ("+" if x >= 0 else "-") + de(abs(x), nd)


def pct(share: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma."""
    return de(100.0 * share, nd) + " %"


def se_share(share: float, R: int) -> float:
    """Standardfehler eines Anteils aus R Stichproben (mindestens der Wert für einen halben Treffer, damit Nullen keinen Standardfehler 0 liefern)."""
    return max(math.sqrt(max(share * (1 - share), 0.0) / R), 0.5 / R)


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 1
def verdict_exp1(phi: float, m: int, karte: str, p_live: float, med_live: float, known: float, R: int, iid_p: float) -> tuple[str, str]:
    """Anteil der Live-Phase-I-Stichproben mit bedingter ARL₀ unter 185,2 gegen den exakten Wert bei unabhängigen Werten (gleiches m)."""
    head = f"φ = {de(phi, 2)}, m = {m}, {karte}: "
    body = (f"bei {R} Live-Phase-I-Stichproben haben {pct(p_live)} eine bedingte ARL₀ unter 185,2 (unabhängige Werte: exakt {pct(iid_p)}), "
            f"Median {de(med_live, 0)}.")
    tail = f" Bekannte Parameter: ARL₀ {de(known, 1)}. Zähleinheit Phase-I-Stichproben, bedingte ARL₀ je Stichprobe exakt, zero-state."
    diff = p_live - iid_p
    se = se_share(iid_p, R)
    if phi == 0.0:
        return "info", head + body + " Die Werte sind unabhängig: die Annahme der Vorgänger-Demo (unabhängige Werte) gilt, die Abhängigkeit ändert hier nichts." + tail
    if diff > 3 * se:
        return "warning", head + body + f" Das sind {de(100 * diff, 1)} Prozentpunkte mehr als bei unabhängigen Werten (mehr als 3 Standardfehler der Live-Stichproben)." + tail
    return "info", head + body + " Der Unterschied zu unabhängigen Werten liegt bei dieser Zahl von Live-Stichproben innerhalb von 3 Standardfehlern." + tail


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 2
def verdict_exp2(p: int, m: int, R: int, chi2: dict | None, f: dict | None, price: tuple[float, float], delta: float) -> tuple[str, str]:
    """chi2 / f: dict(ucl, arl0, p_live) oder None (Regel nicht gewählt); price = (ARL₁ mit χ²-Grenze, ARL₁ mit F-Grenze) bei bekannten Parametern."""
    parts = []
    if chi2 is not None:
        parts.append(f"χ²-Regel (Σ als bekannt gebaut, Grenze {de(chi2['ucl'], 2)}): unbedingte ARL₀ {de(chi2['arl0'], 1)} (Kehrwert der mittleren Alarmrate, exakt), "
                     f"{pct(chi2['p_live'])} der {R} Live-Stichproben unter 185,2.")
    if f is not None:
        parts.append(f"F-Regel (Grenze {de(f['ucl'], 2)}): unbedingte ARL₀ {de(f['arl0'], 1)}, {pct(f['p_live'])} der {R} Live-Stichproben unter 185,2.")
    text = (f"p = {p}, m = {m}: " + " ".join(parts)
            + f" Preis der F-Grenze: ARL₁ ({'ein Kanal springt um ' + de(delta, 1) + ' σ'}, bekannte Parameter, Σ = I) {de(price[0], 1)} → {de(price[1], 1)}."
            + " Zähleinheit: Phase-II-Vektoren bis zum Alarm bzw. Phase-I-Stichproben; zero-state.")
    kind = "warning" if (chi2 is not None and chi2["arl0"] < 0.8 * C.GAMMA0) else "info"
    return kind, text


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 3
def verdict_exp3_ar1(phi: float, rec: dict | None, iid_min: int) -> tuple[str, str]:
    if rec is None:
        return "info", f"φ = {de(phi, 2)} gehört nicht zu den gemessenen Werten (0 / 0,3 / 0,6 / 0,9): für die Mindestzahl gibt es keine Tabelle, nur Live-Läufe in Experiment 1."
    if rec.get("mstar") is None:
        return "warning", f"φ = {de(phi, 2)}: im Raster (30 bis 1600) wird das Kriterium nicht erreicht."
    return "info", (f"AR(1), φ = {de(phi, 2)}: Mindestzahl etwa {rec['mstar']} Phase-I-Beobachtungen (Simulation, auf etwa 10 bis 15 % genau, gelb), "
                    f"bei unabhängigen Werten {iid_min} (exakt). Kriterium: kleinstes m, bei dem der Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2 etwa 5 % erreicht; Rohdatenkarte x̄ ± 3 s.")


def verdict_exp3_t2(p: int, rec_chi2: dict | None, rec_f: dict | None, price: tuple[float, float] | None) -> tuple[str, str]:
    if rec_chi2 is None or rec_f is None or rec_chi2.get("mstar") is None or rec_f.get("mstar") is None:
        return "warning", f"p = {p}: im Raster (30 bis 1600) wird das Kriterium nicht von beiden Regeln erreicht."
    text = (f"T², p = {p}: Mindestzahl etwa {rec_chi2['mstar']} Vektoren mit der χ²-Regel und etwa {rec_f['mstar']} mit der F-Regel (Simulation, auf etwa 10 bis 15 % genau, gelb). "
            "Kriterium: kleinstes m, bei dem der Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2 etwa 5 % erreicht.")
    if price is not None:
        text += (f" Die F-Regel kommt mit weniger Vektoren aus; ihre Grenze liegt weiter. Preis: ARL₁ (ein Kanal springt um 1 σ, bekannte Parameter, "
                 f"m = {rec_f['mstar']}) {de(price[0], 1)} → {de(price[1], 1)}.")
    return "info", text

"""Texte der App. Zahlen, die aus den gefrorenen Tabellen stammen, werden aus `apb_tables.json` eingesetzt (nie von Hand abgeschrieben);
`ASSUMPTIONS_MARKDOWN` steht wortgleich in der README (tests/test_claims.py vergleicht beide)."""

import apb_reference as REF
from apb_evaluation import de, pct

INTRO = """
Eine Alarmkarte meldet einen Wert, der außerhalb von **Mitte ± 3 Streuungen** liegt. Mitte und Streuung kennt sie nicht: sie **schätzt** beide in einer **Phase I**, aus Werten der Vergangenheit ohne Änderung. Die Vorgänger-Demo (`anfangsparameter-demo`, Parameter schätzen) rechnet das für **m unabhängige Einzelwerte einer Reihe** und misst die **Qualität der Karte, die Phase I geliefert hat**: ihre **bedingte ARL₀**, die Zahl der Beobachtungen bis zum ersten Fehlalarm, gegeben genau diese Grenzen.

Diese Demo hebt zwei Annahmen davon auf, mit **derselben Karte und demselben Maß**: Die Phase-I-Werte sind **abhängig** (eine AR(1)-Reihe; die Karte baut ihre Grenzen weiter, als wären die Werte unabhängig, oder überwacht Residuen), und es laufen **mehrere Kanäle** zugleich (Phase I schätzt auch die **Kovarianz**; Karte: Hotelling-T² mit χ²- oder F-Grenze). Gezählt werden **Phase-I-Stichproben**. Die Zahl „ARL₀ = 370,4“ hängt dann nicht nur von m ab, sondern von der **Annahme in Phase I**.

**Die Reihen sind ausgedacht:** ein AR(1)-Prozess mit normaler Randverteilung und standardnormale Messkanäle. In den gelesenen Quellen gibt es keine reale Reihe eines Terminals oder Lagers; wie sich geschätzte Anfangsparameter dort auswirken, wird hier nicht behauptet. Die Karten kennen Mittel und Streuung (bei T² die Kovarianz) nicht; die Residuenkarte gibt es mit bekanntem und mit geschätztem φ.

Weiter unten: drei Experimente, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung** (Symbole, Konventionen, Formeln).
"""

SO_FUNKTIONIERT = """
1. **Phase I:** Die Karte bekommt m Werte (bzw. m Vektoren mit p Kanälen) **ohne Änderung**, aus einer AR(1)-Reihe bzw. aus p unabhängigen Normalkanälen.
2. **Schätzen und Grenzen bauen:** Aus Phase I schätzt sie Mitte und Streuung (bei der Residuenkarte zusätzlich φ, bei T² den Mittelwertvektor und die Kovarianz) und baut daraus ihre Grenzen.
3. **Bedingte ARL₀ rechnen:** Wie lange läuft genau diese Karte in Phase II, wenn sich nichts ändert, bis zum Fehlalarm? Das wird **exakt je Phase-I-Stichprobe** gerechnet (Integralgleichung, Zellenkette bzw. Imhof-Integral), nicht simuliert.
4. **Verteilung über Phase-I-Stichproben:** Wie oft fällt die Karte schlecht aus? Als Maß dient der **Anteil der Stichproben mit bedingter ARL₀ unter 185,2**, der halben nominellen ARL₀ (370,398 / 2). Diese Verteilung ist Simulation (Standardfehler eines Anteils bis 0,008); die Rechnung je Stichprobe ist exakt.

*Zwei Arten von Zahlen:* **Live** gerechnete Läufe (kleine Zahl von Stichproben, mit Rechenzeit) und **gefrorene** Tabellen (4000 Stichproben je Zeile, einmal gerechnet und mit Zufalls-Seed und Stand abgelegt).
"""


def exp1_frage() -> str:
    return (
        "**Kernfrage 1: Was ändert die Abhängigkeit in Phase I an der gebauten Karte, und was heilt die Residuenkarte?** Die **Rohdatenkarte** baut x̄ ± 3 s aus m Phase-I-Werten einer AR(1)-Reihe "
        "(AR-Koeffizient φ: die Korrelation zwischen Wert t und Wert t + k ist φ hoch k). Die **Residuenkarte** überwacht x_t − φ̂·x_(t−1) − (1 − φ̂)·x̄ gegen 3 σ̂_e und kennt den vorigen Wert; "
        "es gibt sie mit **bekanntem** und mit **geschätztem** φ. Gemessen wird die bedingte ARL₀ der Karte, die Phase I geliefert hat, auf der wahren Reihe (zero-state: Phase II beginnt stationär und unabhängig von Phase I, Konvention dieser Demo; die ARL hängt sonst vom Startwert ab)."
    )


def exp1_negativ() -> str:
    t = REF.tables()["ar1"]
    diffs = [(t[f"resid_geschaetzt|{ph}|{m}"]["p_below"] - t[f"resid_bekannt|{ph}|{m}"]["p_below"], ph, m) for ph in ("0.3", "0.6", "0.9") for m in (300, 1000)]
    worst = max(diffs)
    return (
        f"**Ehrlicher Negativbefund.** Ab m = 300 kostet das Schätzen von φ gegenüber bekanntem φ in dieser Messung bis zu {de(100 * max(0.0, worst[0]), 1)} Prozentpunkte im Anteil unter 185,2 "
        "(gefrorene Tabelle, φ = 0,3 / 0,6 / 0,9, m = 300 und 1000; Standardfehler der Differenz etwa 0,5). Die Residuenkarte mit **bekanntem** φ ist der Bestfall: wer φ schätzen muss, hat ihn nicht. Die Schätzung φ̂ ist selbst verzerrt "
        "(E[φ̂] − φ ≈ −0,18 bei φ = 0,9 und m = 30, Simulation). **CUSUM, EWMA und Verzögerungen (ARL₁) mit geschätztem φ̂ sind nicht gemessen.**"
    )


def exp1_falle() -> str:
    t = REF.tables()["ar1"]
    roh = t["roh|0.9|100"]["p_below"]
    iid = REF.iid_exact(100)["p_below"]
    return (
        "**Falle:** Die Zahl für „Parameter geschätzt“ aus der Vorgänger-Demo (unabhängige Werte, eine Reihe) gilt **nicht** für abhängige Werte. "
        f"Bei φ = 0,9 und m = 100 haben {pct(roh)} der Phase-I-Stichproben der Rohdatenkarte eine bedingte ARL₀ unter 185,2, bei unabhängigen Werten {pct(iid)} (exakt)."
    )


def exp2_frage() -> str:
    return (
        "**Kernfrage 2: Was kostet die Schätzung der Kovarianz, und welche Grenze ist die richtige?** Die T²-Karte meldet einen Vektor x, wenn T² = (x − x̄)′ S⁻¹ (x − x̄) über der Grenze liegt (S: geschätzte Kovarianz, Nenner m − 1). "
        "Die **χ²-Regel** nimmt die Grenze für **bekanntes** Σ (das χ²-Quantil zu 1/370,398); die **F-Regel** legt die Grenze so, dass die **unbedingte** ARL₀ (Kehrwert der mittleren Alarmrate über Phase I, exakt aus der F-Verteilung) 370,398 ist. "
        "Die Kanäle sind unabhängig und standardnormal; wegen der affinen Invarianz von T² genügt Σ = I."
    )


def exp2_negativ() -> str:
    t = REF.tables()["t2"]
    f30 = [t[f"F|{p}|30"]["p_below"] for p in (2, 5, 10)]
    from apb_t2 import arl1_price
    return (
        f"**Ehrlicher Negativbefund.** Mit der F-Regel bleibt die bedingte ARL₀ bei m = 30 breit: {pct(min(f30))} bis {pct(max(f30))} der Phase-I-Stichproben liegen für p = 2 / 5 / 10 unter 185,2. "
        f"Die F-Regel macht die Mindestzahl kleiner **und** den ARL₁ größer (m = 100, p = 10, ein Kanal um 1 σ, bekannte Parameter: {de(arl1_price('chi2', 100, 10), 1)} → {de(arl1_price('F', 100, 10), 1)}): "
        "ein Gratis-Gewinn ist sie für die gemessenen p und m nicht."
    )


def exp2_falle() -> str:
    from apb_t2 import arl0_uncond
    return (
        "**Falle:** „ARL₀ 370,398 für jedes Σ“ gilt für **bekanntes** Σ. Wird Σ in Phase I geschätzt und die Grenze nach der χ²-Regel gebaut, liegt die unbedingte ARL₀ darunter "
        f"(m = 100: {de(arl0_uncond('chi2', 100, 2), 1)} / {de(arl0_uncond('chi2', 100, 5), 1)} / {de(arl0_uncond('chi2', 100, 10), 1)} bei p = 2 / 5 / 10, Kehrwert der mittleren Alarmrate, exakt)."
    )


def exp3_frage() -> str:
    return (
        "**Kernfrage 3: Wie viele Phase-I-Werte braucht die Karte jetzt?** Kriterium (fest, die „Mindestzahl“): kleinstes m, bei dem der Anteil der Phase-I-Stichproben mit bedingter ARL₀ **unter 185,2** etwa **5 %** erreicht."
        "Das ist das Kriterium der Vorgänger-Demo für die skalare Karte (dort exakt 274 Werte); andere Kriterien (¾ und 0,9 der nominellen ARL₀) und p > 10 sind hier nicht gemessen."
    )


def exp3_negativ() -> str:
    return (
        "**Ehrlicher Negativbefund.** Die F-Regel verlangt weniger Vektoren; ihre Grenze liegt weiter. Den Preis zeigt Experiment 2 (ARL₁ bei bekannten Parametern). "
        "Die Mindestzahlen sind Simulation und auf etwa 10 bis 15 % genau (Halbierung in log m, Rauschen der Anteile); sie sind **keine Garantie** für andere Verteilungen oder andere Formen der Abhängigkeit."
    )


ASSUMPTIONS_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Abhängigkeit ist AR(1)** mit normaler Randverteilung | AR(2), saisonale oder langgedächtnisreiche Reihen sind **nicht gemessen**; ebenso Σ̂ bei autokorrelierten Kanälen | offen |
| **Karte: Shewhart-Einzelwert bzw. T²** | CUSUM, EWMA und Verzögerungen (ARL₁) mit geschätztem φ̂ oder geschätzter Kovarianz sind **nicht gemessen**; gemessen ist nur der ARL₁-Preis der F-Grenze bei **bekannten** Parametern | offen |
| **Phase I in Kontrolle** (keine Änderung, keine Ausreißer) | T² mit verunreinigter Phase I und Šidák-getrennte Karten mit geschätzten Parametern sind nicht gemessen; für die skalare Karte steht die verunreinigte Phase I in der Vorgänger-Demo | Vorgänger-Demo / offen |
| **Ausgedachte Reihen**, p ≤ 10, Σ = I | Es gibt keine reale Reihe aus einem Terminal oder Lager; p > 10 ist nicht gemessen | offen |
| **Mindestzahlen aus Simulation** | Auf etwa 10 bis 15 % genau; nur das Kriterium „5 % der Stichproben unter der halben nominellen ARL₀“ ist gemessen | – |
| **Rechenumgebung** | Die gefrorenen Zahlen wurden unter Windows mit Python 3.14 gerechnet; Rechenzeiten auf Streamlit Cloud sind nicht gemessen; die Normalzahlen (Umkehrfunktion) sind über Plattformen nur bis etwa 1e-15 gleich, Tests prüfen Bänder | – |
| **Nutzen** | Der Nutzen einer Karte mit Phase-I-Schätzung in Logistik, Lager oder Fertigung ist **nicht belegt** | – |
"""

MATH_INTRO = r"""
**Lesehilfe.** *Gegeben:* eine Phase-I-Stichprobe aus m Werten einer AR(1)-Reihe bzw. aus m Vektoren mit p Kanälen, ohne Änderung. *Gesucht:* die bedingte ARL₀ der Karte, die daraus entsteht, und ihre Verteilung über Phase-I-Stichproben.

| Symbol | Bedeutung |
|---|---|
| $u_t$, $x_t$ | wahre AR(1)-Reihe, Beobachtung (Randverteilung N(0, 1)) |
| $\varphi$, $\hat\varphi$ | AR-Koeffizient, Lag-1-Schätzer aus Phase I |
| $m$ | Zahl der Phase-I-Werte (Vektoren) |
| $\bar x$, $s$ | Mittelwert, Standardabweichung (Nenner $m-1$) aus Phase I |
| $\hat\sigma_e$ | Streuung der Residuen: Quadratsumme der Residuen durch $m-1$, Wurzel (eigene Setzung) |
| $p$, $S$ | Zahl der Kanäle, geschätzte Kovarianz (Nenner $m-1$) |
| $T^2$ | Hotelling-Statistik eines Vektors |
| $\gamma$ | $1/(2\Phi(-3)) = 370{,}398$, die ARL₀-Zielgröße |
"""

CONVENTIONS_MARKDOWN = """
**Konventionen.** Die Tabelle nennt, was in jedem Satz der Demo gemeint ist.

| Begriff | Bedeutung |
|---|---|
| **bedingte ARL₀** | Beobachtungen (AR(1)) bzw. Vektoren (T²) bis zum Alarm in Phase II, **gegeben die Grenzen** der Karte; zero-state: Phase II beginnt stationär und unabhängig von Phase I (Konvention dieser Demo); die ARL hängt sonst vom Startwert ab |
| **Kehrwert der mittleren Alarmrate** („Flotte“) | 1 / E[Alarmwahrscheinlichkeit] über Phase-I-Stichproben; das einzige „mittlere“ Maß, das hier exakt vorkommt (T²: F-Verteilung). Das Mittel der bedingten ARL₀ ist bei kleinem m von Ausreißern bestimmt und nur ein Hinweis |
| **χ²-Regel** | Grenze = χ²-Quantil zu 1/370,398 mit p Freiheitsgraden (9,000 / 11,829 / 18,205 / 26,901 bei p = 1 / 2 / 5 / 10), Regel für **bekanntes** Σ |
| **F-Regel** | Grenze so, dass die **unbedingte** ARL₀ (Kehrwert der mittleren Alarmrate) 370,398 ist |
| **185,2** | die halbe nominelle ARL₀: 370,398 / 2 |
| **370 gegen 370,398** | gesetzt wird oft 370; hier gilt überall 370,398 = 1/(2 Φ(−3)), das Ziel der 3σ-Karte |
| **Zähleinheiten** | Phase-I-Stichproben (Verteilung der bedingten ARL₀), Phase-II-Beobachtungen bzw. -Vektoren (Lauflänge), Läufe nur bei Simulation einer Lauflänge |
| **Ampel** | grün: exakt oder durch zwei Rechenwege geprüft; gelb: Simulation mit Standardfehler (Verteilungen über Phase-I-Stichproben, Mindestzahlen) |
"""

MATH_MARKDOWN = r"""
**Modell (Reihe).** Stationäre AR(1)-Reihe, Randverteilung N(0, 1) für jedes $\varphi$:

$$u_t = \varphi\,u_{t-1} + \sqrt{1-\varphi^2}\;\varepsilon_t,\qquad \varepsilon_t \sim N(0,1)\ \text{unabhängig}$$

**Schätzer aus Phase I** (m Werte $x_1,\dots,x_m$):

$$\bar x = \tfrac1m\sum_t x_t,\qquad s^2 = \tfrac1{m-1}\sum_t (x_t-\bar x)^2,\qquad \hat\varphi = \frac{\sum_{t=2}^m (x_t-\bar x)(x_{t-1}-\bar x)}{\sum_{t=1}^m (x_t-\bar x)^2}$$

Unter AR(1) streut $\bar x$ weiter und $s^2$ ist im Mittel zu klein (geschlossen, auch über die Kovarianzmatrix in Brüchen geprüft):

$$\operatorname{Var}(\bar x)=\frac1m\Big(1+2\sum_{k=1}^{m-1}\big(1-\tfrac km\big)\varphi^k\Big),\qquad E[s^2]=\frac{m}{m-1}\big(1-\operatorname{Var}(\bar x)\big)$$

**Karten.** Rohdatenkarte (behandelt die Werte als unabhängig) und Residuenkarte:

$$\text{Alarm, wenn}\quad |x_t-\bar x| > 3s\qquad\text{bzw.}\qquad \big|x_t-\hat\varphi\,x_{t-1}-(1-\hat\varphi)\,\bar x\big| > 3\hat\sigma_e$$

**Bedingte ARL₀ der Rohdatenkarte** mit Grenzen $[a,b]$ auf der wahren AR(1)-Reihe (Integralgleichung, Nyström-Verfahren mit Gauß-Legendre-Knoten; $k$ ist die Dichte von $N(\varphi y,\,1-\varphi^2)$):

$$g(y)=1+\int_a^b k(y,z)\,g(z)\,dz,\qquad \mathrm{ARL}=1+\int_a^b \phi(y)\,g(y)\,dy$$

Die Residuenkarte rechnet dieselbe Aufgabe über eine Kette auf dem letzten Wert (Zellen, Richardson aus zwei Gitterfeinheiten), weil der Alarm am Paar $(x_{t-1},x_t)$ hängt.

**Viele Kanäle.** Für einen neuen Vektor, gegeben $(\bar x,S)$ aus Phase I:

$$T^2=(x-\bar x)'\,S^{-1}\,(x-\bar x)$$

Unbedingt (über Phase I gemittelt) gilt, als eigene Herleitung aus dem Standardsatz zu $z'S^{-1}z$ und an Rohnormalzahlen bestätigt:

$$T^2=\frac{p\,(m+1)(m-1)}{m\,(m-p)}\;F_{p,\,m-p}$$

Die χ²-Regel setzt die Grenze auf das χ²-Quantil, die F-Regel auf $\tfrac{p(m+1)(m-1)}{m(m-p)}$ mal das F-Quantil zu $1/\gamma$. Die bedingte Alarmwahrscheinlichkeit ist die Verteilungsfunktion einer gewichteten Summe nichtzentraler χ²-Größen (Eigenzerlegung $S=V\,\mathrm{diag}(d)\,V'$, $a=-V'\bar x$), gerechnet mit dem Integral von Imhof:

$$P(T^2>c)=P\Big(\sum_j \tfrac1{d_j}\,(Z_j+a_j)^2>c\Big),\qquad Z_j\sim N(0,1)\ \text{unabhängig}$$

**Preis der F-Grenze.** Springt ein Kanal um $\delta$ Standardabweichungen (bekannte Parameter, $\Sigma=I$), ist $T^2$ nichtzentral $\chi^2_p$ mit Nichtzentralität $\delta^2$ und $\mathrm{ARL}_1=1/P(\chi^2_p(\delta^2)>\mathrm{UCL})$.

**Was nicht aus einer gelesenen Quelle stammt.** Die Formel für $T^2$ mit $F$-Verteilung und die Verzerrung von $\hat\varphi$ sind eigene Nachrechnungen in diesem Modell (Quelle nicht gelesen); die Näherung $-(1+3\varphi)/m$ für die Verzerrung von $\hat\varphi$ ist nicht gelesen und wird nicht als Quelle zitiert. Vanhatalo und Kulahci (2015, gelesen, Abschnitte 1, 5 und 6) zeigen, dass die Residuenkarte bei **perfekt bekanntem** Modell die ARL₀ stabilisiert; ihre Aussage zu m ≥ 1000 bei zwei Variablen und T² hat ein anderes Maß und wird hier nicht verglichen. Zwetsloot, Jones-Farmer und Woodall (2024, gelesen): bei Autokorrelation ist das übliche Vorgehen, ein Modell anzupassen und die Residuen zu überwachen.
"""

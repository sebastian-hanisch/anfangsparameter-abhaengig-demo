# Phase I unter Abhängigkeit und mit vielen Kanälen: dieselbe Karte, andere Annahme – Streamlit-Demo

<!-- Nach dem Deploy direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-anfangsparameter-abhaengig-demo.streamlit.app/)** -->

Eine Demo-Seite der Reihe **Änderungserkennung** („Wie schnell merkt man, dass sich etwas geändert hat?“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vorgänger ist `anfangsparameter-demo` (Parameter schätzen: Phase I aus **m unabhängigen Einzelwerten einer Reihe**); die Kante ist ein `fix`: diese Demo hebt zwei Annahmen davon auf, mit **derselben Karte und demselben Maß**. Gelesen werden sollte sie nach den Demos zu abhängigen Werten und zu mehreren Kanälen (jeweils bei **bekannten** Parametern), denn sie verbindet beide Linien mit der Schätzung.

Vehikel A: **ausgedachte** Reihen, ein AR(1)-Prozess mit normaler Randverteilung und standardnormale Messkanäle (keine Betriebsdaten).

## Warum dieses Problem

Eine Karte „Alarm bei Mitte ± 3 Streuungen“ kennt Mitte und Streuung nicht; sie schätzt beide in einer **Phase I** aus Werten ohne Änderung. Die Vorgänger-Demo zeigt für unabhängige Werte: die Karte, die dabei entsteht, hat eine **bedingte ARL₀** (Beobachtungen bis zum ersten Fehlalarm, gegeben ihre Grenzen), und die streut von Stichprobe zu Stichprobe stark. Dieselbe Zahl hängt aber nicht nur von der Zahl m der Phase-I-Werte ab, sondern von der **Annahme in Phase I**. **Kernfrage:** Was ändert sich, wenn (1) die Phase-I-Werte **abhängig** sind (AR(1); die Karte baut ihre Grenzen weiter, als wären sie unabhängig, oder überwacht Residuen) oder wenn (2) **mehrere Kanäle** laufen und Phase I auch die **Kovarianz** schätzt (Hotelling-T², χ²-Regel gegen F-Regel)? Und: wie viele Phase-I-Werte braucht die Karte dann?

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen des Plans (Vormessungen V13b für AR(1), V13c für T²; numpy-Zufall) sind **Erwartungen**. Hier sind alle Verteilungen mit dem SplitMix64-Generator der Demo und anderen Zufalls-Seeds **neu gemessen** (`tools/apb_make_tables.py --voll`, 4000 Phase-I-Stichproben je Zeile). Abweichungen:

| Größe | Plan (Erwartung) | hier gemessen | Einordnung |
|---|---|---|---|
| Rohdatenkarte, φ = 0,9, m = 100: Anteil unter 185,2 | 44,0 % | 45,1 % | 1,0 Standardfehler der Differenz zweier Stichproben (je 4000); im Band |
| Rohdatenkarte, φ = 0,9, m = 30 / 300 / 1000 | 79,3 % / 13,4 % / 0,4 % | 79,1 % / 13,6 % / 0,5 % | im Band |
| Residuenkarte, φ bekannt, m = 100, φ = 0,9 | 18,8 % | 20,0 % | im Band (alle drei φ: 18,9 bis 20,0 %, Plan 18,8 bis 20,1 %) |
| Residuenkarte, φ geschätzt, m = 100, φ = 0,9 | 29,7 % | 31,1 % | im Band |
| Residuenkarte, φ geschätzt, m = 30, φ = 0,9 | 67,6 % | 68,9 % | im Band |
| T², χ²-Regel, m = 100, p = 5: Anteil unter 185,2 | 55,7 % | 56,6 % | im Band |
| T², F-Regel, m = 100, p = 2 / 5 / 10 | 7,4 / 6,2 / 5,9 % | 8,7 / 6,2 / 5,4 % | p = 2: 2,2 Standardfehler der Differenz; sonst im Band |
| Mindestzahl AR(1), φ = 0 / 0,3 / 0,6 / 0,9 | 280 / 328 / 413 / 509 | 281 / 328 / 418 / 473 | φ = 0,9: 7 % niedriger, im angegebenen Band von etwa 10 bis 15 % |
| Mindestzahl T², χ²-Regel, p = 2 / 5 / 10 | 266 / 315 / 480 | 275 / 323 / 460 | im Band |
| Mindestzahl T², F-Regel, p = 2 / 5 / 10 | 147 / 121 / 110 | 144 / 109 / 107 | p = 5: 10 % niedriger, am Rand des Bands |

Weitere Korrekturen beim Bau:

- **Zufallsquelle (Plan Z6):** Der Plan nahm `random()` mit `NormalDist().inv_cdf` (0,8 µs je Zahl). Gebaut ist stattdessen die **Stapel-Fassung desselben Generators** (`apb_normal.py`): SplitMix64 ist ein Zähler-Generator und lässt sich in numpy für ein ganzes Feld auf einmal rechnen, **Bit für Bit gleich** der Klasse in `apb_rng.py` (Test `tests/test_rng.py`). Normalzahlen über `scipy.special.ndtri`; bitgleich über Plattformen ist nur die Gleichverteilung, alles andere gilt als Band. Die Box-Muller-Funktion der Vorlage (70 µs je Zahl) wird nicht benutzt.
- **Gitter (Plan Z5 d, j):** Die Rohdatenkarte nutzt n = 100 Knoten bei m ≤ 50 und n = 60 sonst (n = 60 war in der Planung nur bei m ≥ 100 gemessen; bei m = 30 stimmen n = 100 und n = 200 auf 1,2e-9, Beobachtung beim Bau). Die Residuenkarte rechnet **auch live** nach Richardson aus den Gittern 80 und 160 (Fehler ~ 1/n²); die Zellenkette allein bei n = 80 wich in der Messung V13d der Planung bis 0,94 % von diesem Wert ab.
- **p = 1 in der Mindestzahl-Tabelle** ergänzt (T²-Karte gleich skalare Karte, χ²-Regel; Mindestzahl 269 gegen exakt 274 aus der Vorgänger-Demo; F-Regel 201), zur Kontrolle der Simulation.
- **Schnellstart:** neun statt zehn Einträge. „T²: p = 5, m = 100“ entfällt als eigener Eintrag, weil es der Voreinstellung (φ = 0,9, m = 100 und T² mit p = 5, m = 100 zugleich) gleicht; zwei Einträge mit denselben Reglerwerten wären nicht unterscheidbar.
- **Exakte iid-Werte:** Die erste Fassung der Doppelintegration in `tools/apb_make_tables.py` (feste Knotenzahl über einen weiten Bereich) gab 17,83 % statt 17,92 % bei m = 100; eine Simulation mit 2 Mio. Stichproben (17,94 ± 0,03 %, Messung beim Bau, nicht im Test) deckte es auf. Jetzt adaptive Quadratur; zweiter Weg in den Tests (äußeres Integral über x̄ statt über s).
- **Median bei unabhängigen Werten:** Der Plan nennt „Median 225 statt 336“ (Simulation); die Demo zeigt den exakten Median 342 und den gemessenen 218 für die Rohdatenkarte bei φ = 0,9.
- **φ als Regler:** Schritt 0,05 von 0 bis 0,95; die Residuenkarten-Tabelle hat keine Zeile bei φ = 0 (nur ab 0,3 gemessen).

## Modell

AR(1): u_t = φ u_(t−1) + √(1 − φ²) ε_t, Randverteilung N(0, 1) für jedes φ. **Phase I** besteht aus m Werten (bzw. m Vektoren) ohne Änderung, **Phase II** aus derselben Verteilung ohne Änderung: gemessen wird die **Qualität der Karte, die Phase I geliefert hat**, nicht ihre Verzögerung. Karten: **Rohdatenkarte** (Alarm bei |x − x̄| > 3 s, behandelt die Werte als unabhängig), **Residuenkarte** (Alarm bei |x_t − φ̂ x_(t−1) − (1 − φ̂) x̄| > 3 σ̂_e, die Karte kennt den vorigen Wert; mit bekanntem oder geschätztem φ), **Hotelling-T²** (T² = (x − x̄)′ S⁻¹ (x − x̄), χ²-Regel oder F-Regel). **Maß:** bedingte ARL₀; Zähleinheiten getrennt: Phase-I-Stichproben, Phase-II-Beobachtungen bzw. -Vektoren, Läufe nur bei Simulation einer Lauflänge. ARL₀-Ziel überall **370,398** (= 1/(2 Φ(−3)), nicht das gesetzte 370); **185,2** ist die halbe nominelle ARL₀. Zero-state: Phase II beginnt stationär und unabhängig von Phase I (Konvention dieser Demo); die ARL hängt sonst vom Startwert ab. Formeln, Symbole und Konventionen stehen im 📐-Expander der App.

## Methodik

- **Bedingte ARL₀ der Rohdatenkarte** auf der AR(1)-Reihe: Nyström-Integralgleichung mit Gauß-Legendre-Knoten, **exakt je Stichprobe**. Zweiter Rechenweg in den Tests: Zellenkette und Simulation einer bedingten Karte.
- **Residuenkarte:** Zellenkette auf dem letzten Wert, Richardson aus zwei Gitterfeinheiten; Orakel: Zeilensummen bei exakten Parametern (370,3983), Simulation.
- **T²:** (x̄, S) aus Rohnormalzahlen, Eigenzerlegung, bedingte Alarmwahrscheinlichkeit über das **Imhof-Integral**; Orakel: `ncx2` (S = I), geschlossener Fall p = 1, Simulation mit Millionen Vektoren. Die **unbedingte** ARL₀ (Kehrwert der mittleren Alarmrate) kommt exakt aus der F-Verteilung (eigene Herleitung, an Rohnormalzahlen, der t-Formel und der geschlossenen Form für p = 2 bestätigt).
- **Verteilung über Phase-I-Stichproben:** live R ≤ 500 Stichproben (R · m · p ≤ 2 000 000), gefroren je 4000 Stichproben (`apb_tables.json`, jede Zelle mit Zufalls-Seed und R, Kopf mit Quelle und Stand). Die Verteilung ist Simulation (**gelb**, Standardfehler eines Anteils bis 0,008), die Rechnung je Stichprobe exakt (**grün**).
- **Mindestzahl m:** kleinstes m, bei dem der Anteil der Phase-I-Stichproben unter 185,2 etwa 5 % erreicht: Raster (2000 Stichproben je Punkt), Halbierung in log m, Prüfung bei m*, 0,8 m* und 1,25 m*. Auf etwa 10 bis 15 % genau.
- **Live-Rechenzeiten** (diese Maschine, ein Thread, kein Befund): Rohdatenkarte mit 500 Stichproben bei m = 30 in 0,24 s und bei m = 1000 in 0,12 s; Residuenkarte mit 500 Stichproben (beide Gitter) in 0,9 s; T² mit 500 Stichproben und p = 10, m = 100 in 0,4 s, mit p = 10, m = 1000 und 200 Stichproben (2 · 10⁶ Zufallszahlen) in 0,25 s. Streamlit Cloud ist nicht gemessen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für: Shewhart-Einzelwert bzw. T², normale Randverteilung, AR(1) bzw. unabhängige Vektoren, Phase I in Kontrolle, ausgedachte Reihen, zero-state, Zähleinheit Phase-I-Stichproben (4000 je Zeile) bzw. Beobachtungen/Vektoren bis zum Alarm, ARL₀-Ziel 370,398. „Anteil“ heißt: Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2.

| Frage | Befund | Test (Datei, Funktion) |
|---|---|---|
| Bekannte Parameter, ARL₀ der Rohdatenkarte bei φ = 0 / 0,3 / 0,6 / 0,9 (Beobachtungen, exakt) | 370,398 / 376,383 / 419,377 / 831,783 | `test_claims.py::test_arl0_bekannte_parameter` |
| Rohdatenkarte, φ = 0,9: Anteil bei m = 30 / 100 / 300 / 1000 (4000 Stichproben) | 79,1 % / 45,1 % / 13,6 % / 0,5 % | `test_claims.py::test_rohdatenkarte_phi_09` |
| Median der bedingten ARL₀, φ = 0,9, m = 30 / 100 / 300 / 1000 | 51 / 218 / 492 / 696 | `test_claims.py::test_rohdatenkarte_phi_09` |
| Unabhängige Werte (exakt), Anteil bei m = 30 / 100 / 300 / 1000; Median bei m = 100 | 36,0 % / 17,9 % / 4,2 % / 0,06 %; Median 342 | `test_claims.py::test_unabhaengige_werte_exakt` |
| Rohdatenkarte, m = 100: Anteil bei φ = 0 / 0,3 / 0,6 / 0,9; m = 30: dieselben φ | m = 100: 18,2 / 22,7 / 30,5 / 45,1 %; m = 30: 36,0 / 43,1 / 55,0 / 79,1 % | `test_claims.py::test_rohdatenkarte_anteil_waechst_mit_phi` |
| Residuenkarte, φ bekannt, m = 100: Anteil bei φ = 0,3 / 0,6 / 0,9 | 18,9 / 19,6 / 20,0 % | `test_claims.py::test_residuenkarte` |
| Residuenkarte, φ geschätzt, m = 100: Anteil bei φ = 0,3 / 0,6 / 0,9 | 24,6 / 24,1 / 31,1 % | `test_claims.py::test_residuenkarte` |
| Residuenkarte, φ = 0,9, m = 30: Anteil bei bekanntem / geschätztem φ | 37,0 % / 68,9 % | `test_claims.py::test_residuenkarte` |
| Ab m = 300 kostet das Schätzen von φ in dieser Messung bis zu (φ = 0,3 / 0,6 / 0,9, m = 300 und 1000) | 1,9 Prozentpunkte (Standardfehler der Differenz etwa 0,5) | `test_claims.py::test_residuenkarte` |
| m = 100, φ = 0,9: E[s²], Var(x̄) (statt 1 und 0,01) | 0,836 und 0,172 | `test_claims.py::test_streuung_und_mitte` |
| Verzerrung E[φ̂] − φ bei φ = 0,9, m = 30 / 100 (200 000 Stichproben, Band) | −0,180 / −0,051 | `test_claims.py::test_verzerrung_phi_dach` |
| T², m = 100: unbedingte ARL₀ der χ²-Regel bei p = 2 / 5 / 10 (Kehrwert der mittleren Alarmrate, exakt) | 239,6 / 154,7 / 75,0 | `test_claims.py::test_t2_unbedingt` |
| T², m = 30: unbedingte ARL₀ der χ²-Regel bei p = 2 / 5 / 10 | 105,4 / 35,5 / 7,9 | `test_claims.py::test_t2_unbedingt` |
| T², F-Regel: Grenze bei p = 2 / 5 / 10, m = 100 (statt χ²: 11,83 / 18,21 / 26,90); unbedingte ARL₀ | 12,83 / 20,77 / 33,19; 370,398 | `test_claims.py::test_t2_unbedingt` |
| T², m = 100: Anteil der χ²-Regel bei p = 2 / 5 / 10 | 24,6 / 56,6 / 98,0 % | `test_claims.py::test_t2_bedingt` |
| T², m = 100: Anteil der F-Regel bei p = 2 / 5 / 10 | 8,7 / 6,2 / 5,4 % | `test_claims.py::test_t2_bedingt` |
| T², m = 30: Anteil der F-Regel bei p = 2 / 5 / 10 | 12,8 / 13,3 / 12,6 % | `test_claims.py::test_t2_bedingt` |
| Preis der F-Grenze: ARL₁ bei bekannten Parametern (ein Kanal um 1 σ, Σ = I), m = 100, p = 2 / 5 / 10, χ²-Grenze → F-Grenze | 67,3 → 99,2 / 114,4 → 290,9 / 159,9 → 1328,4 | `test_claims.py::test_preis_der_f_regel` |
| Mindestzahl m* (5 %, 185,2), AR(1)-Rohdatenkarte φ = 0 / 0,3 / 0,6 / 0,9; unabhängig exakt 274 | 281 / 328 / 418 / 473 | `test_claims.py::test_mindestzahlen` |
| Mindestzahl m*, T², χ²-Regel p = 1 (= skalare Karte) / 2 / 5 / 10 | 269 / 275 / 323 / 460 | `test_claims.py::test_mindestzahlen` |
| Mindestzahl m*, T², F-Regel p = 1 / 2 / 5 / 10 | 201 / 144 / 109 / 107 | `test_claims.py::test_mindestzahlen` |
| Handbeispiele: m = 3, φ = ½: E[s²] = 7/12, Var(x̄) = 11/18; m = 2: φ̂ = −½; m = 2 (φ = 0 / ½ / 0,9): mittlere Alarmrate 0,2468 / 0,3550 / 0,6201; m = 4, p = 2: ARL₀ der χ²-Regel 4,154, Grenze der F-Regel 1385,2 | exakt (Brüche, Arkustangens, geschlossene F-Verteilung) | `test_handbeispiele.py`, `test_claims.py::test_handbeispiele_in_der_readme` |
| 185,2 = 370,398 / 2; 370,398 = 1/(2 Φ(−3)) | exakt | `test_claims.py::test_ziel_und_halbes_ziel` |
| Live-Läufe unter 1 s (Messung dieser Maschine, kein Befund); der Test begrenzt nur nach oben (20 s) | Messung dieser README, nicht im Test | `test_claims.py::test_worst_case_laufzeit` |

**Gelesen als Zählung:** „Die Abhängigkeit verschlechtert die Phase-I-Karte“ gilt hier nur für die **Rohdatenkarte, AR(1), m = 30, 100 und 300** (Anteil wächst mit φ = 0 / 0,3 / 0,6 / 0,9 bis auf Rauschen von 1,5 Prozentpunkten; bei m = 1000 liegen alle Anteile unter 1 %). „Die Residuenkarte heilt es“ gilt nur **bei bekanntem φ** bis auf einen Rest von 1 bis 2 Prozentpunkten über dem Niveau unabhängiger Werte (m = 100: 18,9 bis 20,0 % gegen 17,9 % exakt); mit **geschätztem φ̂** bei m ≤ 100 nicht (24,1 bis 31,1 % bei m = 100). „Die F-Regel stellt 370,4 her“ gilt **unbedingt** (Kehrwert der mittleren Alarmrate), nicht für jede Stichprobe. „Die F-Regel macht die Mindestzahl kleiner“ gilt nur mit dem ARL₁-Preis daneben (bekannte Parameter, ein Kanal um 1 σ); ein Gratis-Gewinn ist sie für die gemessenen p und m nicht.

## Ehrliche Grenzen und 🚧 Nicht gemessen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Abhängigkeit ist AR(1)** mit normaler Randverteilung | AR(2), saisonale oder langgedächtnisreiche Reihen sind **nicht gemessen**; ebenso Σ̂ bei autokorrelierten Kanälen | offen |
| **Karte: Shewhart-Einzelwert bzw. T²** | CUSUM, EWMA und Verzögerungen (ARL₁) mit geschätztem φ̂ oder geschätzter Kovarianz sind **nicht gemessen**; gemessen ist nur der ARL₁-Preis der F-Grenze bei **bekannten** Parametern | offen |
| **Phase I in Kontrolle** (keine Änderung, keine Ausreißer) | T² mit verunreinigter Phase I und Šidák-getrennte Karten mit geschätzten Parametern sind nicht gemessen; für die skalare Karte steht die verunreinigte Phase I in der Vorgänger-Demo | Vorgänger-Demo / offen |
| **Ausgedachte Reihen**, p ≤ 10, Σ = I | Es gibt keine reale Reihe aus einem Terminal oder Lager; p > 10 ist nicht gemessen | offen |
| **Mindestzahlen aus Simulation** | Auf etwa 10 bis 15 % genau; nur das Kriterium „5 % der Stichproben unter der halben nominellen ARL₀“ ist gemessen | – |
| **Rechenumgebung** | Die gefrorenen Zahlen wurden unter Windows mit Python 3.14 gerechnet; Rechenzeiten auf Streamlit Cloud sind nicht gemessen; die Normalzahlen (Umkehrfunktion) sind über Plattformen nur bis etwa 1e-15 gleich, Tests prüfen Bänder | – |
| **Nutzen** | Der Nutzen einer Karte mit Phase-I-Schätzung in Logistik, Lager oder Fertigung ist **nicht belegt** | – |

Nicht gemessen sind außerdem: schiefe Phase I (Simulation der Planung, hier nicht nachgemessen); Mindestzahl für ¾ und 0,9 der nominellen ARL₀ (der Regler fehlt bewusst); der Gitterfehler der Residuenkarte getrennt vom Richardson-Gewinn (getestet ist nur der Abstand zum feineren Gitter).

**Quellen und Stufen.** Vanhatalo/Kulahci 2015 **gelesen (Stufe W)**, Abschnitte 1, 5, 6: die Residuenkarte stabilisiert die ARL₀ bei perfekt bekanntem Modell; ihre Aussage zu m ≥ 1000 bei zwei Variablen und T² hat ein anderes Maß und wird hier nicht verglichen. Zwetsloot/Jones-Farmer/Woodall 2024 **gelesen (W)**: bei Autokorrelation ist das übliche Vorgehen, ein Modell anzupassen und die Residuen zu überwachen. **Nicht gelesen** (nirgends als gelesen zitiert): Lehrbuchstoff zur T²-Karte mit F-Grenze (Hotelling 1947, Montgomery, Bersimis/Psarakis/Panaretos 2007), die Näherung für die Verzerrung des Lag-1-Schätzers (Kendall), Johnson/Bagshaw, Alwan/Roberts, Jensen u. a. 2006. **Keine Neuheit:** alles Lehrbuch oder Stand der Technik; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Orakeln. Was nicht aus einer gelesenen Quelle steht, heißt „eigene Nachrechnung in diesem Modell“: die T²-Formel mit F-Verteilung und die Verzerrung von φ̂.

## Tests

**288 Tests**, seriell rund 75 s (diese Maschine, Windows, Python 3.14, ein BLAS-Thread; `pytest tests/ -q`). Zusammensetzung:

- **Handbeispiele** (`test_handbeispiele.py`, 45): Brüche, Arkustangens, geschlossene F-Verteilung, m = 2, 3, 4 von Hand; jede Zahl von einem Rechenweg ohne den Code der Demo erzeugt und dann mit der Demo verglichen.
- **Orakel** (`test_oracle_ar1.py` 44, `test_oracle_t2.py` 48): Nyström gegen Zellenkette, gegen Plan-05-Werte und gegen Simulation; φ = 0 gegen die exakte iid-Verteilung (zweiter Rechenweg: Doppelintegral mit äußerem x̄-Integral) und die exakte Mindestzahl 274; Residuenkarte mit exakten Parametern (370,3983); E[s²] gegen die Kovarianzmatrix in Brüchen und Simulation; Imhof gegen `ncx2`, den geschlossenen Fall p = 1 und Simulation; die F-Verteilung des Phase-II-T² gegen Rohnormalzahlen; Eigenschaften (Spiegelung, Verschiebung, Umnummerieren und Drehung der Kanäle, Monotonie in m und p, F-Regel gleich 370,398 für jedes (p, m), gleicher Zufalls-Seed gleiche Ausgabe).
- **Claims** (`test_claims.py`, 17): jede Zahl der Tabelle oben, neu erzeugt und wörtlich in der README gesucht; Wortliste verbotener Wörter und Platzhalter.
- **Tabellen** (`test_tables.py`, 31): Aufbau, Zufalls-Seed und R je Zelle, Beziehungen, Abgleich mit den Vormessungen im Band, Neurechnung eines kleinen Gitters (`--check`).
- **Verdikte** (`test_verdicts.py` 13: reine Funktionen an von Hand gesetzten Zahlen, Schwellen; `test_preset_verdicts.py` 11: je Schnellstart die erwartete Verdikt-Zeile), **Randfälle** (13: m = 2, φ am Rand und negativ, R = 1, Zufalls-Seed 0, Obergrenze, m ≤ p, s = 0), **Schnellstarts** (7), **Layout** (24: Schnellstart in der Seitenleiste, fester Kopf, Hervorhebung, jeder Regler hebt sie auf), **App** (18, AppTest: jeder Button, jeder Regler an beiden Grenzen, jede Option jeder Auswahl), Footer (2), Laufzeit-Importe (1), Abbildungen (3: feste Achsen, Randklemmung), **Zufallsquelle** (11: Bitgleichheit der Stapel-Fassung mit der Klasse).
- **Plattform:** Neu-Rechnen-Tests sind Bänder (4 Standardfehler), nie bitgleich; Gleitkomma-Letztstellen nur mit Toleranz. Der Lauf unter Linux (`demo_linux_check.py`, Docker) wurde **nicht** ausgeführt.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | venv, Caches, Secrets; keine Nutzungsstatistik |
| `app.py` | Streamlit-App (drei Experimente, 🚧, 📐) |
| `apb_ar1.py` | AR(1): Erzeuger, Schätzer, Nyström (Rohdatenkarte), Zellenkette (Residuenkarte), geschlossene Formeln |
| `apb_t2.py` | T²: χ²- und F-Regel, unbedingte Rechnung, Imhof-Integral, Preis der F-Grenze |
| `apb_normal.py`, `apb_rng.py` | Stapel-Normalzahlen aus SplitMix64; Vorlage `apb_rng.py` unverändert (Test: Bitgleichheit) |
| `apb_scenario.py` | Live-Läufe (Erzeugen und bedingte ARL₀ je Stichprobe), Obergrenze |
| `apb_reference.py`, `apb_tables.json` | gefrorene Tabellen (Zufalls-Seed und R je Zelle), Zugriff |
| `apb_evaluation.py`, `apb_texts.py` | Einstellungen, Verdikte (reine Funktionen), Texte |
| `apb_constants.py`, `apb_presets.py`, `apb_layout.py`, `apb_visualization.py` | Regler, Schnellstart, fester Kopf, Plotly-Abbildungen |
| `tools/apb_make_tables.py` | erzeugt (`--voll`, rund 9 Minuten) und prüft (`--check`, rund 10 Sekunden) die Tabellen |
| `tests/` | Handbeispiele, Orakel (AR(1), T²), Claims, Verdikte, Randfälle, Tabellen, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Zufallsquelle |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, scipy, pandas, plotly), dazu pytest |

## Bewusst nicht umgesetzt

- **CUSUM, EWMA und Verzögerung (ARL₁) mit geschätzten Parametern** (nur der Preis der F-Grenze bei bekannten Parametern): eine eigene Frage, die Seite würde zu groß.
- **Andere Abhängigkeit als AR(1)**, Σ̂ bei autokorrelierten Kanälen, **T² mit verunreinigter Phase I**, Šidák-getrennte Karten, p > 10, Mindestzahl für ¾ und 0,9.
- **Kein Nachbau** von `niveausprung-demo` (Detektor im Prognosekontext), `output-analysis-demo` (Autokorrelation der Gate-Wartezeit, Batch Means), `elliptic-envelope-demo` (Mahalanobis-Abstand offline mit robuster Kovarianz, ohne Lauflänge), `multiple-testing-demo` (Familienfehler über Tests) und `forecast-interval-demo` (Quantil mit endlichem Fenster): andere Fragen, andere Zähleinheit.
- **Nutzen in Betrieb:** keine Aussage über Logistik, Lager oder Fertigung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/apb_make_tables.py --check     # gefrorene Tabellen gegen eine kleine Neurechnung
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).

"""Phase I bei abhängigen Werten: AR(1)-Reihe, Schätzer, bedingte ARL₀ der Roh- und der Residuenkarte (exakt je Phase-I-Stichprobe).

Modell: stationäre AR(1)-Reihe u_t = φ u_{t-1} + sqrt(1 - φ²) ε_t mit Randverteilung N(0, 1). Phase I: m Werte ohne Änderung. Phase II: dieselbe Reihe, stationär.
Bedingte ARL₀ = Beobachtungen bis zum Alarm der Karte, die Phase I geliefert hat, auf der wahren Reihe (zero-state: Phase II beginnt stationär und unabhängig von Phase I, Konvention dieser Demo;
die ARL hängt sonst vom Startwert ab). Die Residuenkarte kennt den vorigen Wert.
Zwei Rechenwege (Nyström-Integralgleichung für die Rohdatenkarte, Zellenkette für die Residuenkarte): die Tests rechnen jeden gegen den anderen Weg
und gegen Simulation (tests/test_oracle_ar1.py)."""

import math

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtr
from scipy.stats import norm

GAMMA0 = 1.0 / (2.0 * float(norm.sf(3.0)))        # 370,398: ARL₀ der 3σ-Karte bei bekannten Parametern und unabhängigen Werten
HALF = GAMMA0 / 2.0                               # 185,2: die halbe nominelle ARL₀
_GL = {}


def _gauss_legendre(n: int):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


# ------------------------------------------------------------------------------------------------------------------ Erzeugen und Schätzer
def ar1_samples(stream, R: int, m: int, phi: float) -> np.ndarray:
    """R unabhängige Phase-I-Stichproben (je m Werte) der stationären AR(1)-Reihe, Normalzahlen aus dem Strom `stream` (apb_normal.Stream)."""
    z = stream.normal((R, m))
    s = math.sqrt(1.0 - phi * phi)
    e = z.copy()
    e[:, 1:] *= s
    return lfilter([1.0], [1.0, -phi], e, axis=1)


def estimates(x: np.ndarray):
    """x̄, s (Nenner m - 1), φ̂ (Lag-1, Nenner Σ(x - x̄)²) und die zentrierten Werte je Stichprobe. Alle Werte gleich: s = 0 und φ̂ = 0."""
    m = x.shape[1]
    xb = x.mean(axis=1)
    d = x - xb[:, None]
    ss = (d ** 2).sum(axis=1)
    s = np.sqrt(ss / (m - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = np.where(ss > 0, (d[:, :-1] * d[:, 1:]).sum(axis=1) / np.where(ss > 0, ss, 1.0), 0.0)
    return xb, s, ph, d


def sigma_e_hat(d: np.ndarray, phi_vec: np.ndarray) -> np.ndarray:
    """σ̂_e: Quadratsumme der Residuen e_t = d_t - φ d_{t-1} (t = 2 ... m) durch (m - 1) (eigene Setzung), Wurzel."""
    m = d.shape[1]
    e = d[:, 1:] - phi_vec[:, None] * d[:, :-1]
    return np.sqrt((e ** 2).sum(axis=1) / (m - 1))


def var_xbar_closed(m: int, phi: float) -> float:
    """Var(x̄) = (1/m) (1 + 2 Σ_{k=1}^{m-1} (1 - k/m) φ^k) für die stationäre AR(1)-Reihe mit Varianz 1."""
    return (1.0 + 2.0 * sum((1.0 - k / m) * phi ** k for k in range(1, m))) / m


def e_s2_closed(m: int, phi: float) -> float:
    """E[s²] = m/(m - 1) (1 - Var(x̄))."""
    return m / (m - 1.0) * (1.0 - var_xbar_closed(m, phi))


def mean_alarm_prob_m2(phi: float) -> float:
    """m = 2: mittlere Alarmwahrscheinlichkeit der Karte x̄ ± 3 s auf der AR(1)-Reihe = 1 - (2/π) arctan(3 sqrt(1 - φ) / τ), τ² = (3 + φ)/2 (geschlossen)."""
    tau = math.sqrt((3.0 + phi) / 2.0)
    return 1.0 - 2.0 / math.pi * math.atan(3.0 * math.sqrt(1.0 - phi) / tau)


# ------------------------------------------------------------------------------------------------------------------ Rohdatenkarte: Nyström
def p_alarm_marginal(a, b):
    """Alarmwahrscheinlichkeit je Beobachtung der stationären Reihe (Randverteilung N(0,1)): 1 - (Φ(b) - Φ(a))."""
    return 1.0 - (ndtr(np.asarray(b, float)) - ndtr(np.asarray(a, float)))


def raw_chart_arl(a, b, phi: float, n: int = 100) -> np.ndarray:
    """Bedingte ARL₀ der Karte „Alarm, wenn x außerhalb [a, b]“ auf der AR(1)-Reihe (stationärer Start); Arrays a, b (Stapel).
    Integralgleichung g(y) = 1 + ∫_a^b k(y, z) g(z) dz mit k = N(φ y, 1 - φ²)-Dichte, ARL = 1 + ∫_a^b φ_N(y) g(y) dy; Nyström mit Gauß-Legendre (n Knoten).
    Ist b - a ≈ 0 (alle Phase-I-Werte gleich), alarmiert jede Beobachtung außer x̄ selbst: ARL = 1."""
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    out = np.ones(a.size)
    ok = (b - a) > 1e-9
    if not ok.any():
        return out
    xg, wg = _gauss_legendre(n)
    aa, bb = a[ok], b[ok]
    mid = 0.5 * (aa + bb)[:, None]
    half = 0.5 * (bb - aa)[:, None]
    x = mid + half * xg[None, :]
    w = half * wg[None, :]
    s = math.sqrt(1.0 - phi * phi)
    K = norm.pdf((x[:, None, :] - phi * x[:, :, None]) / s) / s * w[:, None, :]
    K[K < 1e-30] = 0.0                                   # Denormalzahlen verlangsamen LAPACK um das Hundertfache
    g = np.linalg.solve(np.eye(n)[None] - K, np.ones((aa.size, n, 1)))[..., 0]
    out[ok] = 1.0 + (w * norm.pdf(x) * g).sum(axis=1)
    return out


def raw_chart_cond(x: np.ndarray, phi: float, n: int | None = None):
    """Rohdatenkarte x̄ ± 3 s aus Phase-I-Stichproben x (R, m): (bedingte ARL₀ je Stichprobe, Alarmrate je Beobachtung). n = 100 Knoten bei m <= 50, sonst 60."""
    m = x.shape[1]
    if n is None:
        n = 100 if m <= 50 else 60
    xb, s, _, _ = estimates(x)
    a, b = xb - 3.0 * s, xb + 3.0 * s
    arl = np.concatenate([raw_chart_arl(a[i:i + 100], b[i:i + 100], phi, n) for i in range(0, a.size, 100)])
    return arl, p_alarm_marginal(a, b)


def known_arl0(phi: float, n: int = 100) -> float:
    """ARL₀ der Rohdatenkarte ±3 bei bekannten Parametern auf der AR(1)-Reihe: 370,398 / 376,383 / 419,377 / 831,783 bei φ = 0 / 0,3 / 0,6 / 0,9."""
    return float(raw_chart_arl([-3.0], [3.0], phi, n)[0])


# ------------------------------------------------------------------------------------------------------------------ Residuenkarte: Zellenkette
def resid_chain_arl(phi_true: float, phi_hat, c, w, n: int = 160, L: float = 7.0):
    """Residuenkarte auf der AR(1)-Reihe mit wahrem φ: Statistik r_t = x_t - φ̂ x_{t-1} - c, Alarm bei |r_t| > w (w = 3 σ̂_e, c = (1 - φ̂) x̄).
    Zustand y = x_{t-1} (n Zellen auf [-L, L], Vertreter = Mittelpunkt); die Überlebensmasse nach Zelle l ist die Masse von N(φ y, 1 - φ²) in Zelle l
    geschnitten mit dem Annahmebereich (je Zelle exakt, nur der Vertreter von y ist Näherung, Fehler O(h²)). Start x_0 ~ N(0, 1).
    Rückgabe (ARL, Alarmrate je Beobachtung im stationären Lauf). Arrays phi_hat, c, w (Stapel)."""
    ph = np.atleast_1d(np.asarray(phi_hat, float))
    c = np.atleast_1d(np.asarray(c, float))
    w = np.atleast_1d(np.asarray(w, float))
    B = ph.size
    e = np.linspace(-L, L, n + 1)
    y = 0.5 * (e[:-1] + e[1:])
    s = math.sqrt(1.0 - phi_true ** 2)
    mu = phi_true * y
    center = ph[:, None] * y[None, :] + c[:, None]
    lo = np.maximum(e[None, None, :-1], (center - w[:, None])[:, :, None])
    hi = np.minimum(e[None, None, 1:], (center + w[:, None])[:, :, None])
    mass = ndtr((hi - mu[None, :, None]) / s) - ndtr((lo - mu[None, :, None]) / s)
    Q = np.where(hi > lo, mass, 0.0)
    Q[Q < 1e-30] = 0.0
    pi = np.diff(ndtr(e))
    pi = pi / pi.sum()
    g = np.linalg.solve(np.eye(n)[None] - Q, np.ones((B, n, 1)))[..., 0]
    return g @ pi, 1.0 - Q.sum(axis=2) @ pi


def resid_chart_cond(x: np.ndarray, phi: float, phi_known: bool, n_fine: int = 160, n_coarse: int = 80):
    """Residuenkarte aus Phase-I-Stichproben x (R, m): (bedingte ARL₀ je Stichprobe nach Richardson aus n_coarse und n_fine (Fehler ~ 1/n²), Alarmrate).
    phi_known: die Karte kennt φ (x̄ und σ̂_e bleiben geschätzt); sonst φ̂ (Lag-1-Schätzer)."""
    xb, _, ph, d = estimates(x)
    pv = np.full(x.shape[0], float(phi)) if phi_known else ph
    c = (1.0 - pv) * xb
    w = 3.0 * sigma_e_hat(d, pv)
    A = {}
    for n in (n_coarse, n_fine):
        parts = [resid_chain_arl(phi, pv[i:i + 25], c[i:i + 25], w[i:i + 25], n=n) for i in range(0, x.shape[0], 25)]
        A[n] = (np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts]))
    arl = (4.0 * A[n_fine][0] - A[n_coarse][0]) / 3.0
    return arl, A[n_fine][1], pv


def summarize(arl: np.ndarray, rate: np.ndarray, thr: float = HALF) -> dict:
    """Median, 5-%- und 95-%-Quantil, Anteil unter thr, Mittel der bedingten ARL₀ (Hinweis, im Tail von Ausreißern bestimmt), Kehrwert der mittleren Alarmrate."""
    return dict(med=float(np.median(arl)), q05=float(np.quantile(arl, 0.05)), q95=float(np.quantile(arl, 0.95)),
                p_below=float((arl < thr).mean()), mean=float(arl.mean()), flotte=float(1.0 / rate.mean()))

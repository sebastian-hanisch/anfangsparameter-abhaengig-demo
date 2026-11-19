"""Erzeugt apb_tables.json (gefrorene Tabellen der App) und prüft sie.

    python tools/apb_make_tables.py --voll     rechnet alle Tabellen neu und schreibt apb_tables.json (rund 10 bis 15 Minuten, ein Thread)
    python tools/apb_make_tables.py --check    rechnet ein kleines Gitter (m = 100 und 300, R = 400, anderer Zufalls-Seed) und vergleicht es im Band
                                               (|z| <= 4) mit den gespeicherten Werten; Rückgabewert 1 bei Abweichung

Zufall: nur SplitMix64 (`apb_normal.Stream`, Zufalls-Seeds 44000 und folgende; die Vormessungen V13b/V13c nahmen numpy mit anderen Seeds).
Phase-I-Stichproben: R = 4000 je Zelle (AR(1)-Rohdatenkarte, Residuenkarte mit bekanntem und mit geschätztem φ, T² mit χ²- und F-Regel).
Mindestzahl m (Kriterium: höchstens 5 % der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2): Raster R = 2000, Halbierung in log m (7 Schritte, R = 5000,
fester Zufalls-Seed), Prüfung bei m*, 0,8 m* und 1,25 m* (R = 20 000 bzw. 8000). Die Mindestzahl ist Simulation und auf etwa 10 bis 15 % genau.
Exakte iid-Werte (Median, Anteil unter 185,2, Mindestzahl m = 274) aus der Doppelintegration über (x̄, s) mit Wurzelsuche (anderer Weg als in den Tests)."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from scipy.stats import chi2  # noqa: E402

import apb_ar1 as A  # noqa: E402
import apb_normal as N  # noqa: E402
import apb_t2 as T  # noqa: E402

PATH = ROOT / "apb_tables.json"
PHIS = (0.0, 0.3, 0.6, 0.9)
MS = (30, 100, 300, 1000)
PS = (2, 5, 10)
R_MAIN = 4000
SEED0 = 44000
RASTER_M = (30, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1100, 1600)
QUANTILE_P = [i / 100 for i in range(1, 100)]


def _sig(x, k=4):
    return float(f"{x:.{k}g}")


def cell_record(arl, rate, R, seed) -> dict:
    s = A.summarize(arl, rate)
    s = {k: _sig(v, 5) for k, v in s.items()}
    s["quantile"] = [_sig(v, 4) for v in np.quantile(arl, QUANTILE_P)]
    s["R"], s["seed"] = int(R), int(seed)
    return s


# ------------------------------------------------------------------------------------------------------------------ Zellen
def ar1_cells(phis, ms, R, seed_base, chunk=500) -> dict:
    out = {}
    idx = 0
    for karte in ("roh", "resid_bekannt", "resid_geschaetzt"):
        for phi in phis:
            if karte != "roh" and phi == 0.0:
                continue
            for m in ms:
                idx += 1
                seed = seed_base + idx
                st = N.Stream(seed)
                arls, rates = [], []
                for i in range(0, R, chunk):
                    x = A.ar1_samples(st, min(chunk, R - i), m, phi)
                    if karte == "roh":
                        arl, rate = A.raw_chart_cond(x, phi)
                    else:
                        arl, rate, _ = A.resid_chart_cond(x, phi, karte == "resid_bekannt")
                    arls.append(arl)
                    rates.append(rate)
                out[f"{karte}|{phi:.1f}|{m}"] = cell_record(np.concatenate(arls), np.concatenate(rates), R, seed)
    return out


def t2_cells(ps, ms, R, seed_base, chunk=250) -> dict:
    out = {}
    idx = 0
    for p in ps:
        for m in ms:
            idx += 1
            seed = seed_base + idx
            st = N.Stream(seed)
            parts = {"chi2": ([], []), "F": ([], [])}
            for i in range(0, R, chunk):
                cell = T.t2_cell(st, min(chunk, R - i), m, p)
                for rule in parts:
                    parts[rule][0].append(cell[rule]["arl"])
                    parts[rule][1].append(cell[rule]["rate"])
            for rule, (arls, rates) in parts.items():
                out[f"{rule}|{p}|{m}"] = cell_record(np.concatenate(arls), np.concatenate(rates), R, seed)
    return out


# ------------------------------------------------------------------------------------------------------------------ exakte iid-Werte
def iid_share_below(m: int, thr: float) -> float:
    """P(bedingte ARL₀ < thr) bei unabhängigen Werten: Alarmwahrscheinlichkeit p(x̄, s) = 1 - Φ(x̄ + 3s) + Φ(x̄ - 3s) > 1/thr.
    Äußeres Integral über s (χ-Verteilung), innen die Schwelle a(s) für |x̄| per Wurzelsuche: P(|x̄| > a) = 2 Φ(-sqrt(m) a)."""
    target = 1.0 / thr
    nu = m - 1

    def inner(s):
        pal = lambda xb: 1.0 - ndtr(xb + 3 * s) + ndtr(xb - 3 * s)  # noqa: E731
        if pal(0.0) > target:
            return 1.0
        a = brentq(lambda xb: pal(xb) - target, 0.0, 40.0, xtol=1e-14)
        return 2.0 * float(ndtr(-math.sqrt(m) * a))

    lo, hi = chi2.ppf(1e-13, nu), chi2.ppf(1 - 1e-13, nu)
    pts = [q for q in (0.6 * nu, 0.8 * nu, nu, 1.2 * nu, 1.5 * nu) if lo < q < hi]
    return float(quad(lambda q: chi2.pdf(q, nu) * inner(math.sqrt(q / nu)), lo, hi, points=pts, limit=800, epsabs=1e-13, epsrel=1e-11)[0])


def iid_exact(m: int) -> dict:
    med = brentq(lambda t: iid_share_below(m, t) - 0.5, 50.0, 2 * A.GAMMA0, xtol=1e-6)
    return dict(p_below=_sig(iid_share_below(m, A.HALF), 5), median=_sig(med, 6))


def iid_min_m() -> int:
    """Kleinste ganze Zahl m mit P(bedingte ARL₀ < 185,2) <= 5 % bei unabhängigen Werten (Plan 04: 274)."""
    lo, hi = 100, 600
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if iid_share_below(mid, A.HALF) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------------------------------------------------------ Mindestzahl
def _p_below_ar1(phi, m, R, seed):
    st = N.Stream(seed)
    arls = []
    for i in range(0, R, 500):
        x = A.ar1_samples(st, min(500, R - i), m, phi)
        arls.append(A.raw_chart_cond(x, phi)[0])
    return float((np.concatenate(arls) < A.HALF).mean())


def _p_below_t2(rule, p, m, R, seed):
    st = N.Stream(seed)
    cnt = 0
    for i in range(0, R, 250):
        cell = T.t2_cell(st, min(250, R - i), m, p, rules=(rule,))
        cnt += int((cell[rule]["arl"] < A.HALF).sum())
    return cnt / R


def mindest(fun, p_min_m, seed0) -> dict:
    """Raster, Halbierung in log m, Prüfung. `fun(m, R, seed)` liefert den Anteil unter 185,2."""
    ms = [m for m in RASTER_M if m > p_min_m]
    raster = {m: fun(m, 2000, seed0 + k) for k, m in enumerate(ms)}
    ps = [raster[m] for m in ms]
    k0 = next((k for k in range(len(ms) - 1) if ps[k] > 0.05 >= ps[k + 1]), None)
    rec = dict(raster={str(m): _sig(v, 4) for m, v in raster.items()}, R_raster=2000, seed=seed0)
    if k0 is None:
        rec.update(mstar=None, p_at=None)
        return rec
    lo, hi = ms[k0], ms[k0 + 1]
    for _ in range(7):
        mid = int(round(math.sqrt(lo * hi)))
        if mid in (lo, hi):
            break
        if fun(mid, 5000, seed0 + 500) > 0.05:
            lo = mid
        else:
            hi = mid
    mstar = hi
    conf = [fun(int(round(mstar * f)), 20000 if i == 0 else 8000, seed0 + 700 + i) for i, f in enumerate((1.0, 0.8, 1.25))]
    rec.update(mstar=int(mstar), p_at={"m*": _sig(conf[0], 4), "0,8 m*": _sig(conf[1], 4), "1,25 m*": _sig(conf[2], 4)}, R_pruefung="20000 bei m*, 8000 sonst")
    return rec


def mindest_cells(seed_base) -> tuple[dict, dict]:
    ar1 = {}
    for i, phi in enumerate(PHIS):
        ar1[f"{phi:.1f}"] = mindest(lambda m, R, s, phi=phi: _p_below_ar1(phi, m, R, s), 1, seed_base + 1000 * i)
    t2 = {}
    k = 0
    for p in (1,) + PS:
        for rule in ("chi2", "F"):
            k += 1
            t2[f"{rule}|{p}"] = mindest(lambda m, R, s, rule=rule, p=p: _p_below_t2(rule, p, m, R, s), p + 1, seed_base + 20000 + 1000 * k)
    return ar1, t2


# ------------------------------------------------------------------------------------------------------------------ Voll und Check
def build_all() -> dict:
    t0 = time.time()
    data = {"meta": {
        "stand": "2026-11-18",
        "erzeugt_von": "tools/apb_make_tables.py --voll",
        "zufall": "SplitMix64 (apb_normal.Stream), Zufalls-Seeds ab 44000; Normalzahlen über scipy.special.ndtri",
        "vormessung": "V13b (AR(1)) und V13c (T²) der Planung: Erwartungen, hier neu gemessen mit anderem Zufalls-Seed; Abweichungen stehen in der README",
        "R": R_MAIN,
        "python": sys.version.split()[0], "numpy": np.__version__, "plattform": sys.platform,
        "hinweis": "Verteilungen über Phase-I-Stichproben sind Simulation (Standardfehler eines Anteils bis 0,008); bedingte ARL₀ je Stichprobe exakt (Nyström, Zellenkette, Imhof)."}}
    print("iid exakt ...", flush=True)
    data["iid_exakt"] = {str(m): iid_exact(m) for m in MS}
    data["iid_exakt"]["mindestzahl"] = iid_min_m()
    print(f"  {time.time() - t0:.0f} s", flush=True)
    print("AR(1) ...", flush=True)
    data["ar1"] = ar1_cells(PHIS, MS, R_MAIN, SEED0)
    print(f"  {time.time() - t0:.0f} s", flush=True)
    print("T2 ...", flush=True)
    data["t2"] = t2_cells(PS, MS, R_MAIN, SEED0 + 100)
    print(f"  {time.time() - t0:.0f} s", flush=True)
    print("Mindestzahlen ...", flush=True)
    data["mindest_ar1"], data["mindest_t2"] = mindest_cells(SEED0 + 5000)
    print(f"  {time.time() - t0:.0f} s", flush=True)
    return data


def check() -> int:
    """Kleines Gitter, neuer Zufalls-Seed, Band |z| <= 4 gegen die gespeicherten Anteile (Standardfehler beider Seiten)."""
    stored = json.loads(PATH.read_text(encoding="utf-8"))
    R = 400
    bad = 0
    new = ar1_cells((0.3, 0.9), (100, 300), R, SEED0 + 90000)
    new.update(t2_cells((5, 10), (100, 300), R, SEED0 + 91000))
    for key, rec in new.items():
        old = stored["ar1"].get(key) or stored["t2"].get(key)
        p0 = old["p_below"]
        se = math.sqrt(max(p0 * (1 - p0), 1e-4) * (1 / R + 1 / old["R"]))
        z = (rec["p_below"] - p0) / se
        flag = "OK " if abs(z) <= 4 else "ABW"
        bad += abs(z) > 4
        print(f"{flag} {key:28s} neu {rec['p_below']:.3f} gespeichert {p0:.3f} z = {z:+.2f}")
    print("Abweichungen:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voll", action="store_true", help="alles neu rechnen und apb_tables.json schreiben")
    ap.add_argument("--check", action="store_true", help="kleines Gitter gegen apb_tables.json im Band prüfen")
    args = ap.parse_args()
    if args.voll:
        d = build_all()
        PATH.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("geschrieben:", PATH)
    elif args.check:
        sys.exit(check())
    else:
        ap.print_help()

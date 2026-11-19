"""Abbildungen: alle Achsen fest (fixedrange), logarithmische Achsen mit festem Bereich, Werte außerhalb des Bereichs stehen am Rand, Beschriftungen auf logarithmischen Achsen in log10."""

import math

import numpy as np

import apb_constants as C
import apb_visualization as V


def _fest(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_serie_hat_feste_achsen_und_vier_linien_plus_reihe():
    fig = V.build_series([0.1, -0.2, 0.3, 1.0], 0.2, 0.5, 0.9, 0.8)
    assert _fest(fig) and len(fig.data) == 5 and fig.layout.yaxis.range[1] > 4


def test_cdf_klemmt_werte_auf_den_achsenbereich_und_endet_bei_eins():
    live = {"Karte": (np.array([0.5, 20.0, 1e7]), "#000")}
    froz = {"Karte": (list(np.linspace(1, 1e6, 99)), "#999")}
    fig = V.build_cdf(live, froz, [(185.2, "185,2", "#555")], "Titel")
    assert _fest(fig) and fig.layout.xaxis.type == "log"
    xs = [t for tr in fig.data for t in tr.x if tr.mode == "lines"]
    assert min(xs) >= C.ARL_PLOT_MIN and max(xs) <= C.ARL_PLOT_MAX
    live_tr = next(t for t in fig.data if "live" in (t.name or ""))
    assert live_tr.y[-1] == 1.0 and list(live_tr.x) == sorted(live_tr.x)
    assert all(abs(a.x - math.log10(185.2)) < 1e-12 for a in fig.layout.annotations)     # auf logarithmischen Achsen steht der Anker in log10


def test_mindest_bereich_und_marker():
    raster = {"30": 0.5, "100": 0.2, "300": 0.04}
    fig = V.build_mindest([("Kurve", raster, 280, "#c00")], iid_min=274)
    assert _fest(fig) and fig.layout.xaxis.type == "log" and tuple(fig.layout.yaxis.range) == (0, 1.0)
    assert any(tr.mode == "markers+text" for tr in fig.data)
    assert list(fig.layout.xaxis.tickvals) == [30, 50, 100, 200, 500, 1000, 1600]

"""Plotly-Abbildungen (alle Achsen fest, fixedrange)."""

import math

import numpy as np
import plotly.graph_objects as go

import apb_constants as C

QP = [i / 100 for i in range(1, 100)]


def _layout(fig, title, xtitle, ytitle, height=380, logx=False, xrange=None, yrange=None):
    fig.update_layout(title=dict(text=title, font=dict(size=14), x=0.01, xanchor="left", y=0.97, yanchor="top"), height=height, margin=dict(l=55, r=15, t=85, b=50),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0), hovermode="closest")
    fig.update_xaxes(title=xtitle, fixedrange=True, type="log" if logx else "linear", range=xrange)
    fig.update_yaxes(title=ytitle, fixedrange=True, range=yrange)
    return fig


def build_series(x0, xbar, s, phi_true, phi_hat):
    """Eine Phase-I-Stichprobe: die Reihe, die gebauten Grenzen x̄ ± 3 s und die wahren Grenzen ± 3 (die Karte kennt nur die gebauten)."""
    m = len(x0)
    t = list(range(1, m + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=list(x0), mode="lines+markers", name="Phase-I-Werte", line=dict(color=C.COLOR_LINE, width=1.5), marker=dict(size=4)))
    for y, nm, col, dash in ((xbar + 3 * s, "gebaute Grenzen x̄ ± 3 s", C.COLOR_ROH, "solid"), (xbar - 3 * s, None, C.COLOR_ROH, "solid"),
                             (3.0, "wahre Grenzen ± 3", "rgba(90,90,90,0.9)", "dash"), (-3.0, None, "rgba(90,90,90,0.9)", "dash")):
        fig.add_trace(go.Scatter(x=[1, m], y=[y, y], mode="lines", name=nm or "", showlegend=nm is not None, line=dict(color=col, dash=dash, width=2), hoverinfo="skip"))
    lim = max(4.0, float(np.max(np.abs(x0))) + 0.5, abs(xbar) + 3 * s + 0.5) + 1.5
    return _layout(fig, "Eine Phase-I-Stichprobe und die gebauten Grenzen", "Zeitpunkt in Phase I", "Wert (in Einheiten der wahren Streuung)", height=340, yrange=[-lim, lim])


def build_cdf(live: dict, frozen: dict, refs: list, title: str):
    """Verteilungsfunktion der bedingten ARL₀ über Phase-I-Stichproben, logarithmische Achse. live: Name -> (Werte, Farbe); frozen: Name -> (99 Quantile, Farbe);
    refs: [(x, Text, Farbe)] senkrechte Linien (185,2 und bekannte Parameter). Werte außerhalb des Achsenbereichs stehen am Rand."""
    fig = go.Figure()
    lo, hi = C.ARL_PLOT_MIN, C.ARL_PLOT_MAX
    for name, (q, col) in frozen.items():
        fig.add_trace(go.Scatter(x=np.clip(q, lo, hi), y=QP, mode="lines", name=f"{name} (gefroren, 4000 Stichproben)", line=dict(color=col, width=2, dash="dot")))
    for name, (vals, col) in live.items():
        v = np.sort(np.clip(np.asarray(vals, float), lo, hi))
        fig.add_trace(go.Scatter(x=v, y=(np.arange(v.size) + 1) / v.size, mode="lines", name=f"{name} (live, {v.size} Stichproben)", line=dict(color=col, width=2.5, shape="hv")))
    for x, text, col in refs:
        fig.add_trace(go.Scatter(x=[x, x], y=[0, 1.05], mode="lines", showlegend=False, hoverinfo="skip", line=dict(color=col, dash="dash", width=1.5)))
        fig.add_annotation(x=math.log10(x), y=1.0, yref="paper", xref="x", text=text, showarrow=False, yanchor="bottom", font=dict(size=11, color=col), bgcolor="rgba(255,255,255,0.85)")
    return _layout(fig, title, "bedingte ARL₀ der gebauten Karte (logarithmisch)", "Anteil der Phase-I-Stichproben mit höchstens diesem Wert", height=420, logx=True,
                   xrange=[math.log10(lo), math.log10(hi)], yrange=[0, 1.05])


def build_mindest(curves: list, iid_min: int | None = None):
    """Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2 über m (Raster R = 2000, Simulation) und die Mindestzahl m* je Kurve.
    curves: [(Name, Raster {m: P}, m*, Farbe)]."""
    fig = go.Figure()
    for name, raster, mstar, col in curves:
        ms = sorted(int(k) for k in raster)
        fig.add_trace(go.Scatter(x=ms, y=[raster[str(k)] if str(k) in raster else raster[k] for k in ms], mode="lines+markers", name=name, line=dict(color=col, width=2), marker=dict(size=5)))
        if mstar:
            fig.add_trace(go.Scatter(x=[mstar], y=[0.05], mode="markers+text", text=[f"m* ≈ {mstar}"], textposition="top right", showlegend=False,
                                     marker=dict(color=col, size=11, symbol="diamond")))
    fig.add_hline(y=0.05, line=dict(color="rgba(90,90,90,0.9)", dash="dash", width=1.5))
    if iid_min:
        fig.add_trace(go.Scatter(x=[iid_min, iid_min], y=[0, 1.0], mode="lines", showlegend=False, hoverinfo="skip", line=dict(color="rgba(90,90,90,0.9)", dash="dot", width=1.5)))
        fig.add_annotation(x=math.log10(iid_min), y=1.0, yref="paper", xref="x", text=f"unabhängig: {iid_min} (exakt)", showarrow=False, yanchor="bottom", font=dict(size=11), bgcolor="rgba(255,255,255,0.85)")
    _layout(fig, "Anteil unter 185,2 je Phase-I-Umfang m", "Phase-I-Umfang m (logarithmisch)",
                   "Anteil mit bedingter ARL₀ < 185,2", height=420, logx=True, xrange=[math.log10(28), math.log10(1800)], yrange=[0, 1.0])
    fig.update_xaxes(tickvals=[30, 50, 100, 200, 500, 1000, 1600], ticktext=["30", "50", "100", "200", "500", "1000", "1600"])
    return fig

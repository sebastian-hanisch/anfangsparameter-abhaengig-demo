"""Gefrorene Tabellen (apb_tables.json, erzeugt von tools/apb_make_tables.py): Verteilungen der bedingten ARL₀ über 4000 Phase-I-Stichproben,
Mindestzahlen, exakte iid-Werte. Jede Zelle trägt Zufalls-Seed und R; der Kopf (`meta`) nennt Quelle und Stand."""

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parent / "apb_tables.json"


@lru_cache(maxsize=1)
def tables() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return tables()["meta"]


def ar1_cell(karte_key: str, phi: float, m: int):
    """Zelle (Karte, φ, m) der gefrorenen AR(1)-Tabelle oder None, wenn φ oder m nicht zu den gemessenen Werten gehört."""
    return tables()["ar1"].get(f"{karte_key}|{phi:.1f}|{m}") if abs(phi * 10 - round(phi * 10)) < 1e-9 else None


def t2_cell(rule: str, p: int, m: int):
    return tables()["t2"].get(f"{rule}|{p}|{m}")


def iid_exact(m: int):
    return tables()["iid_exakt"].get(str(m))


def iid_min_m() -> int:
    return int(tables()["iid_exakt"]["mindestzahl"])


def mindest_ar1(phi: float):
    return tables()["mindest_ar1"].get(f"{phi:.1f}") if abs(phi * 10 - round(phi * 10)) < 1e-9 else None


def mindest_t2(rule: str, p: int):
    return tables()["mindest_t2"].get(f"{rule}|{p}")

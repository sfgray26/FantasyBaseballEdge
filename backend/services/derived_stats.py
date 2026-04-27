"""
Pure null-safe derived-stat helpers for the partial MLB derived stats layer.

Scope: OPS, WHIP, ERA, AVG, ISO. NSB and advanced Statcast metrics are blocked
behind separate tasks (caught_stealing backfill, statcast_performances hardening).

All functions are pure: no I/O, no DB access, no logging side effects. Every
missing component returns None rather than a silent 0.0 — callers must decide
whether to skip the row, log a warning, or substitute a league average.

The helpers exist to give consumers (scoring engine, valuation worker, API
responses) one vetted entry point instead of re-implementing the math inline.
"""

from __future__ import annotations

from typing import Optional, Union

# Numeric input accepted from DB rows, Pydantic models, or raw JSON
Number = Union[int, float]


def _as_float(v: Optional[Number]) -> Optional[float]:
    """Coerce input to float, propagating None."""
    if v is None:
        return None
    return float(v)


def parse_innings_pitched(ip: Optional[Union[str, Number]]) -> Optional[float]:
    """
    Convert BDL-style '6.2' (6 innings + 2 outs) to decimal innings (6.667).

    Returns None when input is None, empty, non-numeric, or 0 outs with 0 innings.
    The zero case returns None because WHIP/ERA are mathematically undefined.
    """
    if ip is None:
        return None
    if isinstance(ip, (int, float)):
        return float(ip) if ip > 0 else None
    text = str(ip).strip()
    if not text:
        return None
    parts = text.split(".")
    try:
        innings = int(parts[0])
        outs = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except ValueError:
        return None
    if outs < 0 or outs > 2:
        # Baseball box-score convention: .0/.1/.2 only. Any other digit is junk.
        return None
    decimal = innings + (outs / 3.0)
    return decimal if decimal > 0 else None


def compute_ops(obp: Optional[Number], slg: Optional[Number]) -> Optional[float]:
    """OPS = OBP + SLG. Returns None if either component is missing."""
    a = _as_float(obp)
    b = _as_float(slg)
    if a is None or b is None:
        return None
    return a + b


def compute_avg(hits: Optional[Number], at_bats: Optional[Number]) -> Optional[float]:
    """AVG = H / AB. Returns None for missing components or AB == 0."""
    h = _as_float(hits)
    ab = _as_float(at_bats)
    if h is None or ab is None or ab == 0:
        return None
    return h / ab


def compute_iso(slg: Optional[Number], avg: Optional[Number]) -> Optional[float]:
    """ISO = SLG - AVG. Returns None if either component is missing."""
    a = _as_float(slg)
    b = _as_float(avg)
    if a is None or b is None:
        return None
    return a - b


def compute_whip(
    walks_allowed: Optional[Number],
    hits_allowed: Optional[Number],
    innings_pitched: Optional[Union[str, Number]],
) -> Optional[float]:
    """WHIP = (BB + H) / IP. Returns None for missing components or 0 IP."""
    bb = _as_float(walks_allowed)
    h = _as_float(hits_allowed)
    ip = parse_innings_pitched(innings_pitched)
    if bb is None or h is None or ip is None:
        return None
    return (bb + h) / ip


def compute_era(
    earned_runs: Optional[Number],
    innings_pitched: Optional[Union[str, Number]],
) -> Optional[float]:
    """ERA = (ER * 9) / IP. Returns None for missing components or 0 IP."""
    er = _as_float(earned_runs)
    ip = parse_innings_pitched(innings_pitched)
    if er is None or ip is None:
        return None
    return (er * 9.0) / ip


__all__ = [
    "parse_innings_pitched",
    "compute_ops",
    "compute_avg",
    "compute_iso",
    "compute_whip",
    "compute_era",
]

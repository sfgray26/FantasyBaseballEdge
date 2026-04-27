"""
pybaseball_loader.py — FanGraphs leaderboard integration for Statcast enrichment.

Pulls batting_stats() and pitching_stats() from pybaseball (FanGraphs), writes
24-hour JSON caches under data/cache/, and exposes:

  fetch_all_statcast_leaderboards(year, force_refresh) — network I/O, called once/day
  load_pybaseball_batters(year)    -> dict[str, StatcastBatter]
  load_pybaseball_pitchers(year)   -> dict[str, StatcastPitcher]
  match_yahoo_to_statcast(name, cache, position) -> Optional[str]
  log_statcast_coverage(yahoo_names, cache, label) -> float

All functions are safe to import even when pybaseball is not installed.
"""

import dataclasses
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

from backend.services.retry_logic import sync_retry

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
PYBASEBALL_CACHE_TTL = 24 * 3600  # seconds
_RATE_LIMIT_SLEEP = 2.0

_SUFFIX_STRIP = re.compile(r"\s+(Jr\.?|Sr\.?|II+|III+|IV|V)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def _strip_name(name: str) -> str:
    """Return canonical lowercase ASCII form: no accents, no Jr/Sr/II suffixes."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    stripped = _SUFFIX_STRIP.sub("", ascii_only.strip())
    return " ".join(stripped.lower().split())


# ---------------------------------------------------------------------------
# FanGraphs column remap guards (column names vary across pybaseball versions)
# ---------------------------------------------------------------------------

_BATTER_COL_REMAP = {
    "Barrel %": "Barrel%",
    "Hard Hit %": "HardHit%",
    "Sweet-Spot %": "Sweet-Spot%",
}

_PITCHER_COL_REMAP = {
    "Barrel %": "Barrel%",
    "Hard Hit %": "HardHit%",
    "vFA (pfx)": "FBv",
    "FBv (pfx)": "FBv",
}


def _apply_remap(df, remap: dict):
    """Rename columns in-place where the canonical name is missing."""
    rename = {}
    for old, new in remap.items():
        if old in df.columns and new not in df.columns:
            rename[old] = new
    if rename:
        df = df.rename(columns=rename)
    return df


# ---------------------------------------------------------------------------
# Float helper
# ---------------------------------------------------------------------------

def _float(val) -> float:
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# DataFrame -> dataclass dict converters
# ---------------------------------------------------------------------------

def _df_to_batter_dict(df) -> dict:
    """Convert a FanGraphs batting DataFrame to dict[canonical_name, StatcastBatter]."""
    from backend.fantasy_baseball.advanced_metrics import StatcastBatter

    df = _apply_remap(df, _BATTER_COL_REMAP)
    result = {}

    for _, row in df.iterrows():
        name = str(row.get("Name") or "").strip()
        if not name:
            continue

        team_abb = str(row.get("Team") or row.get("team") or "").strip()

        xwoba = _float(row.get("xwOBA"))
        woba = _float(row.get("wOBA"))
        xwoba_diff = xwoba - woba

        # wRC+ column name varies across pybaseball versions
        wrc_plus_raw = (
            row.get("wRC+") or row.get("wRCplus") or row.get("wRC_plus") or 100
        )
        wrc_plus_val = _float(wrc_plus_raw) if wrc_plus_raw else 100.0

        b = StatcastBatter(
            name=name,
            team=team_abb,
            xwoba=xwoba,
            xwoba_diff=xwoba_diff,
            barrel_pct=_float(row.get("Barrel%")),
            exit_velo_avg=_float(row.get("EV")),
            hard_hit_pct=_float(row.get("HardHit%")),
            sweet_spot_pct=_float(row.get("Sweet-Spot%")),
            o_swing_pct=_float(row.get("O-Swing%")),
            z_contact_pct=_float(row.get("Z-Contact%")),
            swstr_pct=_float(row.get("SwStr%")),
            gb_pct=_float(row.get("GB%")),
            fb_pct=_float(row.get("FB%")),
            ld_pct=_float(row.get("LD%")),
            pull_pct=_float(row.get("Pull%")),
            wrc_plus=wrc_plus_val if wrc_plus_val > 0 else 100.0,
            regression_up=xwoba_diff < -0.020,
            regression_down=xwoba_diff > 0.030,
        )
        result[_strip_name(name)] = b

    return result


def _df_to_pitcher_dict(df) -> dict:
    """Convert a FanGraphs pitching DataFrame to dict[canonical_name, StatcastPitcher]."""
    from backend.fantasy_baseball.advanced_metrics import StatcastPitcher

    df = _apply_remap(df, _PITCHER_COL_REMAP)
    result = {}

    for _, row in df.iterrows():
        name = str(row.get("Name") or "").strip()
        if not name:
            continue

        era = _float(row.get("ERA"))
        xera = _float(row.get("xERA"))
        xera_diff = xera - era

        p = StatcastPitcher(
            name=name,
            xera=xera,
            xera_diff=xera_diff,
            stuff_plus=_float(row.get("Stuff+") or row.get("Stuff+")),
            location_plus=_float(row.get("Location+") or row.get("Location+")),
            pitching_plus=_float(row.get("Pitching+") or row.get("Pitching+")),
            fb_velo_avg=_float(row.get("FBv")),
            spin_rate_fb=int(_float(row.get("FBSpin") or 0)),
            whiff_pct=_float(row.get("SwStr%")),
            chase_pct=_float(row.get("O-Swing%")),
            csw_pct=_float(row.get("CSW%")),
            barrel_allowed_pct=_float(row.get("Barrel%")),
            hard_hit_allowed_pct=_float(row.get("HardHit%")),
            xwoba_allowed=_float(row.get("xwOBA")),
            luck_regression=(xera_diff > 0.40),
        )
        result[_strip_name(name)] = p

    return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh(path: Path, now: float) -> bool:
    return path.exists() and (now - path.stat().st_mtime) < PYBASEBALL_CACHE_TTL


def _write_json_cache(path: Path, data: dict, fetched_at: float) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": fetched_at,
        "players": {k: dataclasses.asdict(v) for k, v in data.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Retry-protected pybaseball fetch functions
# ---------------------------------------------------------------------------

@sync_retry(max_retries=3, base_delay=2.0, max_delay=30.0)
def _fetch_batting_stats_with_retry(year: int, qual: int = 50):
    """Fetch batting stats from pybaseball with retry on 502 errors."""
    import pybaseball
    return pybaseball.batting_stats(year, qual=qual)


@sync_retry(max_retries=3, base_delay=2.0, max_delay=30.0)
def _fetch_pitching_stats_with_retry(year: int, qual: int = 25):
    """Fetch pitching stats from pybaseball with retry on 502 errors."""
    import pybaseball
    return pybaseball.pitching_stats(year, qual=qual)


@sync_retry(max_retries=3, base_delay=2.0, max_delay=30.0)
def _fetch_sprint_speed_with_retry(year: int):
    """Fetch sprint speed from pybaseball with retry on 502 errors."""
    import pybaseball
    return pybaseball.statcast_sprint_speed(year)


# ---------------------------------------------------------------------------
# Optional sprint speed enrichment
# ---------------------------------------------------------------------------

def _maybe_enrich_sprint_speed(year: int, batter_path: Path, now: float) -> None:
    """Attempt to add sprint speed from Statcast; silently skipped on any failure."""
    try:
        df_speed = _fetch_sprint_speed_with_retry(year)
        time.sleep(_RATE_LIMIT_SLEEP)

        if batter_path.exists():
            payload = json.loads(batter_path.read_text(encoding="utf-8"))
            players = payload.get("players", {})
            for _, row in df_speed.iterrows():
                name = str(row.get("name") or "").strip()
                key = _strip_name(name)
                if key in players:
                    players[key]["sprint_speed"] = _float(row.get("hp_to_1b") or row.get("sprint_speed"))
            payload["players"] = players
            batter_path.write_text(json.dumps(payload), encoding="utf-8")
            logger.info("pybaseball: sprint speed enrichment applied")
    except Exception as e:
        logger.debug("pybaseball sprint speed enrichment skipped: %s", e)


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_all_statcast_leaderboards(year: int = 2025, force_refresh: bool = False) -> None:
    """
    Fetch FanGraphs batting + pitching leaderboards via pybaseball and write
    24-hour JSON caches under data/cache/.

    Safe to call if pybaseball is not installed — logs a warning and returns.
    Idempotent within the TTL window unless force_refresh=True.
    """
    try:
        import pybaseball
    except ImportError:
        logger.warning("pybaseball not installed -- skipping Statcast leaderboard fetch")
        return

    batter_path = CACHE_DIR / f"pybaseball_batting_{year}.json"
    pitcher_path = CACHE_DIR / f"pybaseball_pitching_{year}.json"
    now = time.time()

    if not force_refresh and _cache_is_fresh(batter_path, now) and _cache_is_fresh(pitcher_path, now):
        logger.debug("pybaseball caches are fresh, skipping fetch")
        return

    try:
        pybaseball.cache.enable()
    except Exception:
        pass

    if force_refresh or not _cache_is_fresh(batter_path, now):
        try:
            df = _fetch_batting_stats_with_retry(year, qual=50)
            time.sleep(_RATE_LIMIT_SLEEP)
            data = _df_to_batter_dict(df)
            _write_json_cache(batter_path, data, now)
            logger.info("pybaseball: cached %d batters for %d", len(data), year)
        except Exception as e:
            logger.error("pybaseball batting fetch failed after retries: %s", e)

    if force_refresh or not _cache_is_fresh(pitcher_path, now):
        try:
            df = _fetch_pitching_stats_with_retry(year, qual=25)
            time.sleep(_RATE_LIMIT_SLEEP)
            data = _df_to_pitcher_dict(df)
            _write_json_cache(pitcher_path, data, now)
            logger.info("pybaseball: cached %d pitchers for %d", len(data), year)
        except Exception as e:
            logger.error("pybaseball pitching fetch failed after retries: %s", e)

    _maybe_enrich_sprint_speed(year, batter_path, now)


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------

def load_pybaseball_batters(year: int = 2025) -> dict:
    """Load pybaseball batter cache from disk. Returns empty dict on any failure."""
    from backend.fantasy_baseball.advanced_metrics import StatcastBatter

    path = CACHE_DIR / f"pybaseball_batting_{year}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = set(StatcastBatter.__dataclass_fields__)
        result = {}
        for key, f in payload.get("players", {}).items():
            try:
                result[key] = StatcastBatter(**{k: v for k, v in f.items() if k in fields})
            except TypeError:
                pass
        return result
    except Exception as e:
        logger.warning("pybaseball batter cache read failed: %s", e)
        return {}


def load_pybaseball_pitchers(year: int = 2025) -> dict:
    """Load pybaseball pitcher cache from disk. Returns empty dict on any failure."""
    from backend.fantasy_baseball.advanced_metrics import StatcastPitcher

    path = CACHE_DIR / f"pybaseball_pitching_{year}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = set(StatcastPitcher.__dataclass_fields__)
        result = {}
        for key, f in payload.get("players", {}).items():
            try:
                result[key] = StatcastPitcher(**{k: v for k, v in f.items() if k in fields})
            except TypeError:
                pass
        return result
    except Exception as e:
        logger.warning("pybaseball pitcher cache read failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Yahoo name -> cache key matching
# ---------------------------------------------------------------------------

def match_yahoo_to_statcast(yahoo_name: str, cache: dict, position: str = "") -> Optional[str]:
    """
    Tiered fuzzy match of a Yahoo display name against a pybaseball cache dict.

    Tier 1: exact canonical match (_strip_name)
    Tier 2: last-name + first-initial match
    Tier 3: last-name-only match if unique (exactly 1 result)
    Returns None if no match found. Never calls network.
    """
    if not yahoo_name:
        return None

    canonical = _strip_name(yahoo_name)

    # Tier 1: exact
    if canonical in cache:
        return canonical

    parts = canonical.split()
    if not parts:
        return None

    last = parts[-1]
    first_initial = parts[0][0] if len(parts) > 1 else ""

    # Tier 2: last-name + first-initial
    if first_initial:
        tier2 = [k for k in cache if k.split()[-1] == last and k.split()[0][0] == first_initial]
        if len(tier2) == 1:
            return tier2[0]

    # Tier 3: last-name unique
    tier3 = [k for k in cache if k.split()[-1] == last]
    if len(tier3) == 1:
        return tier3[0]

    return None


# ---------------------------------------------------------------------------
# Coverage logging
# ---------------------------------------------------------------------------

def log_statcast_coverage(yahoo_names: list, cache: dict, label: str = "waiver players") -> float:
    """
    Log and return the fraction of yahoo_names that have a Statcast cache hit.
    Returns 0.0 if yahoo_names is empty.
    """
    if not yahoo_names:
        return 0.0
    hits = sum(1 for n in yahoo_names if match_yahoo_to_statcast(n, cache) is not None)
    pct = hits / len(yahoo_names) * 100
    logger.info(
        "Statcast coverage %.0f%% (%d/%d %s)",
        pct, hits, len(yahoo_names), label,
    )
    return pct / 100

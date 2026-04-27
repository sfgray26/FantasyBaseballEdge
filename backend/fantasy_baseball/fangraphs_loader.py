"""
FanGraphs Rest-of-Season (RoS) Projection Loader

Fetches daily RoS projections from FanGraphs for four systems:
  ATC (30%), THE BAT (30%), Steamer (20%), ZiPS DC (20%)

Uses cloudscraper to bypass Cloudflare protection.
Handles "Last, First" name format for internal player_key compatibility.

Lock ID: 100_012 (reserved in daily_ingestion.py)
Cadence: Daily 3 AM ET

See reports/K25_FANGRAPHS_COLUMN_MAP.md for column spec.
"""

import logging
import time
from io import StringIO
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Projection system definitions
# ---------------------------------------------------------------------------

SYSTEMS = {
    "atc":      {"weight": 0.30, "type_param": "atc"},
    "thebat":   {"weight": 0.30, "type_param": "thebat"},
    "steamer":  {"weight": 0.20, "type_param": "steamerr"},   # Note: "steamerr" for RoS
    "zips":     {"weight": 0.20, "type_param": "zipsdc"},
}

_BASE_URL = "https://www.fangraphs.com/projections.aspx"

# K-25: All systems use "SO" for strikeouts (not "K")
# K-25: All systems use "Last, First" name format

# Batting columns we care about (intersection set)
_BAT_COLS = {"Name", "Team", "PA", "HR", "R", "RBI", "SB", "SO", "AVG", "OBP", "SLG", "OPS"}
# Pitching columns we care about
_PIT_COLS = {"Name", "Team", "IP", "W", "SV", "SO", "ERA", "WHIP", "GS", "BB", "K/9"}


def _normalize_name(name: str) -> str:
    """Convert 'Last, First' FanGraphs format to 'First Last'.

    >>> _normalize_name("Ohtani, Shohei")
    'Shohei Ohtani'
    >>> _normalize_name("Mike Trout")
    'Mike Trout'
    """
    if not name:
        return ""
    name = name.strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        return f"{parts[1]} {parts[0]}"
    return name


def _make_player_id(name: str) -> str:
    """Normalize player name to stable ASCII key — mirrors projections_loader._make_player_id."""
    import re
    name = _normalize_name(name)
    if not name:
        return ""
    # Strip generational suffixes
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\.?\s*$', '', name, flags=re.IGNORECASE).strip()
    # Normalize accented characters
    name = (name
            .replace("\xe9", "e").replace("\xe8", "e").replace("\xea", "e")
            .replace("\xe1", "a").replace("\xe0", "a").replace("\xe2", "a")
            .replace("\xf3", "o").replace("\xf2", "o").replace("\xf4", "o")
            .replace("\xfa", "u").replace("\xf9", "u").replace("\xfb", "u").replace("\xfc", "u")
            .replace("\xed", "i").replace("\xec", "i").replace("\xee", "i").replace("\xef", "i")
            .replace("\xf1", "n").replace("\xe7", "c"))
    return (name.lower()
            .replace(" ", "_").replace(".", "").replace("'", "")
            .replace(",", "").replace("-", "_"))


def _fetch_projection_html(system: str, stat_type: str) -> Optional[str]:
    """Fetch a single FanGraphs projection page via cloudscraper.

    Args:
        system: type param value (e.g. 'atc', 'thebat', 'steamerr', 'zipsdc')
        stat_type: 'bat' or 'pit'

    Returns:
        HTML string, or None on failure.
    """
    try:
        import cloudscraper
    except ImportError:
        logger.error("cloudscraper not installed — cannot fetch FanGraphs projections")
        return None

    url = (f"{_BASE_URL}?pos=all&stats={stat_type}&type={system}"
           f"&team=0&lg=all&players=0")

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )
    try:
        resp = scraper.get(url, timeout=45)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error("FanGraphs fetch failed for %s/%s: %s", system, stat_type, e)
        return None


def _parse_table(html: str) -> Optional[pd.DataFrame]:
    """Extract the main projection table from FanGraphs HTML."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 not installed — cannot parse FanGraphs HTML")
        return None

    soup = BeautifulSoup(html, "html.parser")

    # FanGraphs uses rgMasterTable for projection tables
    table = soup.find("table", {"class": "rgMasterTable"})
    if table is None:
        # Fallback: look for any large data table
        tables = soup.find_all("table")
        for t in tables:
            rows = t.find_all("tr")
            if len(rows) > 20:
                table = t
                break

    if table is None:
        logger.warning("No projection table found in FanGraphs HTML")
        return None

    try:
        dfs = pd.read_html(StringIO(str(table)))
        if dfs:
            return dfs[0]
    except Exception as e:
        logger.error("pd.read_html failed: %s", e)
    return None


def fetch_system_projections(
    system_key: str,
    stat_type: str = "bat",
) -> Optional[pd.DataFrame]:
    """Fetch and parse RoS projections for one system + stat type.

    Args:
        system_key: one of 'atc', 'thebat', 'steamer', 'zips'
        stat_type: 'bat' or 'pit'

    Returns:
        DataFrame with normalized 'Name' column (First Last) and 'player_id' key,
        or None on failure.
    """
    cfg = SYSTEMS.get(system_key)
    if not cfg:
        logger.error("Unknown projection system: %s", system_key)
        return None

    html = _fetch_projection_html(cfg["type_param"], stat_type)
    if not html:
        return None

    df = _parse_table(html)
    if df is None or df.empty:
        logger.warning("Empty projection table for %s/%s", system_key, stat_type)
        return None

    # Ensure Name column exists
    if "Name" not in df.columns:
        logger.warning("No 'Name' column in %s/%s table", system_key, stat_type)
        return None

    # Normalize names: "Last, First" -> "First Last"
    df["Name"] = df["Name"].apply(_normalize_name)
    df["player_id"] = df["Name"].apply(_make_player_id)
    df["system"] = system_key

    # Column-intersection: keep only columns we have, log missing ones
    expected = _BAT_COLS if stat_type == "bat" else _PIT_COLS
    available = set(df.columns)
    missing = expected - available
    if missing:
        logger.warning(
            "%s/%s missing columns %s — affected stats will be NaN",
            system_key, stat_type, sorted(missing),
        )

    logger.info(
        "Fetched %d %s projections from %s RoS",
        len(df), stat_type, system_key,
    )
    return df


def fetch_all_ros(
    stat_type: str = "bat",
    delay_seconds: float = 3.0,
) -> dict[str, pd.DataFrame]:
    """Fetch RoS projections from all four systems for one stat category.

    Args:
        stat_type: 'bat' or 'pit'
        delay_seconds: polite delay between requests to FanGraphs

    Returns:
        Dict mapping system_key -> DataFrame. Missing systems are omitted.
    """
    results: dict[str, pd.DataFrame] = {}
    for i, system_key in enumerate(SYSTEMS):
        if i > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        df = fetch_system_projections(system_key, stat_type)
        if df is not None and not df.empty:
            results[system_key] = df
    logger.info(
        "RoS %s fetch complete: %d/%d systems succeeded",
        stat_type, len(results), len(SYSTEMS),
    )
    return results


def compute_ensemble_blend(
    projections: dict[str, pd.DataFrame],
    stat_columns: list[str],
) -> Optional[pd.DataFrame]:
    """Compute weighted ensemble blend across available systems.

    Args:
        projections: dict from fetch_all_ros() — system_key -> DataFrame
        stat_columns: list of numeric column names to blend (e.g. ['HR', 'RBI', 'AVG'])

    Returns:
        DataFrame with player_id + blended stat columns, or None if no data.
    """
    if not projections:
        return None

    # Collect per-system data keyed by player_id
    all_players: dict[str, dict] = {}  # player_id -> {name, stats per system}

    for system_key, df in projections.items():
        weight = SYSTEMS[system_key]["weight"]
        for _, row in df.iterrows():
            pid = row.get("player_id", "")
            if not pid:
                continue
            if pid not in all_players:
                all_players[pid] = {
                    "player_id": pid,
                    "name": row.get("Name", ""),
                    "team": row.get("Team", ""),
                    "_weights": 0.0,
                }
                for col in stat_columns:
                    all_players[pid][col] = 0.0

            entry = all_players[pid]
            for col in stat_columns:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                if pd.isna(val):
                    val = 0.0
                entry[col] += val * weight
            entry["_weights"] += weight

    if not all_players:
        return None

    # Normalize by actual weight sum (handles missing systems gracefully)
    rows = []
    for entry in all_players.values():
        w = entry.pop("_weights", 1.0) or 1.0
        for col in stat_columns:
            entry[col] = round(entry[col] / w, 4)
        rows.append(entry)

    blend_df = pd.DataFrame(rows)
    logger.info(
        "Ensemble blend computed for %d players across %d stat columns",
        len(blend_df), len(stat_columns),
    )
    return blend_df


# ---------------------------------------------------------------------------
# Convenience: full daily pipeline
# ---------------------------------------------------------------------------

def run_daily_ros_pipeline() -> dict:
    """Execute the full daily RoS fetch + blend pipeline.

    Returns summary dict with counts and status.
    """
    summary = {"batting": {}, "pitching": {}, "status": "ok"}

    # Batting
    bat_raw = fetch_all_ros("bat", delay_seconds=3.0)
    if bat_raw:
        bat_blend = compute_ensemble_blend(
            bat_raw,
            stat_columns=["HR", "R", "RBI", "SB", "AVG", "OPS"],
        )
        summary["batting"] = {
            "systems_fetched": list(bat_raw.keys()),
            "blend_players": len(bat_blend) if bat_blend is not None else 0,
        }
    else:
        summary["batting"] = {"systems_fetched": [], "blend_players": 0}
        summary["status"] = "partial"

    # Pitching
    pit_raw = fetch_all_ros("pit", delay_seconds=3.0)
    if pit_raw:
        pit_blend = compute_ensemble_blend(
            pit_raw,
            stat_columns=["W", "SV", "SO", "ERA", "WHIP"],
        )
        summary["pitching"] = {
            "systems_fetched": list(pit_raw.keys()),
            "blend_players": len(pit_blend) if pit_blend is not None else 0,
        }
    else:
        summary["pitching"] = {"systems_fetched": [], "blend_players": 0}
        summary["status"] = "partial"

    if not bat_raw and not pit_raw:
        summary["status"] = "failed"

    return summary

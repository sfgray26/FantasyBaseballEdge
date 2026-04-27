"""
Projection System Loader — Steamer / ZiPS / ATC / FantasyPros ADP

Loads real projection CSV files (from FanGraphs or Kimi research output)
and converts them into the player_board dict format.

When real CSV files are available they REPLACE the hardcoded player_board.py
estimates. When not available the hardcoded board serves as fallback.

Expected file locations (drop into data/projections/):
  data/projections/steamer_batting_2026.csv   — FanGraphs Steamer batters
  data/projections/steamer_pitching_2026.csv  — FanGraphs Steamer pitchers
  data/projections/zips_batting_2026.csv      — ZiPS batters (optional)
  data/projections/zips_pitching_2026.csv     — ZiPS pitchers (optional)
  data/projections/adp_yahoo_2026.csv         — FantasyPros Yahoo 12-team ADP

── Steamer batting CSV columns (FanGraphs export) ──────────────────────────
  Name, Team, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, SO, HBP, SF, AVG,
  OBP, SLG, OPS, wOBA, wRC+, BsR, Off, Def, WAR

── Steamer pitching CSV columns (FanGraphs export) ─────────────────────────
  Name, Team, W, L, ERA, G, GS, IP, H, ER, HR, BB, SO, WHIP,
  K/9, BB/9, K/BB, H/9, HR/9, AVG, BABIP, LOB%, GB%, HR/FB, FIP, xFIP, WAR

── FantasyPros ADP CSV columns ──────────────────────────────────────────────
  PLAYER NAME, TEAM, POS, AVG, BEST, WORST, # TEAMS, STDEV

Run this module standalone to validate loaded data:
  python -m backend.fantasy_baseball.projections_loader
"""

import csv
import logging
import os
import re as _re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "projections"

# ---------------------------------------------------------------------------
# Yahoo position eligibility map — used when parsing position strings
# ---------------------------------------------------------------------------
YAHOO_POS_NORMALIZE = {
    "C": ["C"], "1B": ["1B"], "2B": ["2B"], "3B": ["3B"], "SS": ["SS"],
    "LF": ["LF", "OF"], "CF": ["CF", "OF"], "RF": ["RF", "OF"],
    "OF": ["OF"], "DH": ["DH"],
    "SP": ["SP"], "RP": ["RP"], "P": ["SP", "RP"],
    "C/1B": ["C", "1B"], "C/OF": ["C", "OF"],
    "1B/OF": ["1B", "OF"], "2B/SS": ["2B", "SS"],
    "2B/3B": ["2B", "3B"], "3B/SS": ["3B", "SS"],
    "SS/2B": ["SS", "2B"], "SS/3B": ["SS", "3B"],
    "OF/1B": ["OF", "1B"], "OF/DH": ["OF", "DH"],
    "SP/RP": ["SP", "RP"],
}


def _normalize_positions(pos_str: str) -> list[str]:
    """Convert FanGraphs position string to list of Yahoo-eligible positions."""
    if not pos_str:
        return ["Util"]
    pos_str = pos_str.strip()
    if pos_str in YAHOO_POS_NORMALIZE:
        return YAHOO_POS_NORMALIZE[pos_str]
    # Try splitting on /
    parts = [p.strip() for p in pos_str.replace(",", "/").split("/")]
    result = []
    for p in parts:
        if p in ("LF", "CF", "RF"):
            result.append(p)
            if "OF" not in result:
                result.append("OF")
        elif p in ("SP", "RP", "C", "1B", "2B", "3B", "SS", "DH", "OF"):
            result.append(p)
    return result if result else ["Util"]


# Suffixes to strip before normalizing (word-boundary, case-insensitive)
_SUFFIX_RE = _re.compile(r'\b(jr|sr|ii|iii|iv)\.?\s*$', _re.IGNORECASE)


def _make_player_id(name: str) -> str:
    """Normalize a player name to a stable ASCII identifier.

    Handles:
    - Accented characters (e, a, o, u, i, n, c variants)
    - Name suffixes (Jr., Sr., II, III, IV)
    - Last-name-first format ("Ohtani, Shohei" -> "shohei_ohtani")
    """
    if not name:
        return ""
    name = name.strip()

    # Flip last-name-first format
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"

    # Strip generational suffixes
    name = _SUFFIX_RE.sub("", name).strip()

    # Normalize accented characters
    name = (name
            .replace("e\u0301", "e").replace("e\u0300", "e")  # é, è (combining)
            .replace("\xe9", "e").replace("\xe8", "e").replace("\xea", "e")
            .replace("a\u0301", "a").replace("a\u0300", "a")
            .replace("\xe1", "a").replace("\xe0", "a").replace("\xe2", "a")
            .replace("o\u0301", "o").replace("o\u0300", "o")
            .replace("\xf3", "o").replace("\xf2", "o").replace("\xf4", "o")
            .replace("u\u0301", "u").replace("u\u0300", "u")
            .replace("\xfa", "u").replace("\xf9", "u").replace("\xfb", "u").replace("\xfc", "u")
            .replace("i\u0301", "i").replace("i\u0300", "i")
            .replace("\xed", "i").replace("\xec", "i").replace("\xee", "i").replace("\xef", "i")
            .replace("\xf1", "n")   # ñ
            .replace("\xe7", "c")   # ç
            )

    return (name.lower()
            .replace(" ", "_").replace(".", "").replace("'", "")
            .replace(",", "").replace("-", "_"))


# ---------------------------------------------------------------------------
# Steamer batting loader
# ---------------------------------------------------------------------------

def load_steamer_batting(path: Path) -> list[dict]:
    """
    Load FanGraphs Steamer batting projections CSV.
    Returns list of player dicts compatible with player_board format.
    """
    players = []
    if not path.exists():
        logger.warning(f"Steamer batting file not found: {path}")
        return players

    _BATTING_EXPECTED = {"Name", "PA", "HR", "RBI", "R", "AVG"}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # K-16: Warn loudly on missing critical columns — never reject the whole file.
        if reader.fieldnames:
            headers = {h.strip() for h in reader.fieldnames if h}
            missing = _BATTING_EXPECTED - headers
            if missing:
                logger.warning(
                    "Steamer batting CSV missing expected columns %s — "
                    "affected stats will default to 0. Check FanGraphs export format.",
                    sorted(missing),
                )
        for row in reader:
            try:
                name = row.get("Name", row.get("name", "")).strip()
                team = row.get("Team", row.get("team", "FA")).strip().upper()
                pos_str = row.get("POS", row.get("Pos", row.get("pos", "OF"))).strip()
                positions = _normalize_positions(pos_str)

                pa = float(row.get("PA", 0) or 0)
                r = float(row.get("R", 0) or 0)
                h = float(row.get("H", 0) or 0)
                hr = float(row.get("HR", 0) or 0)
                rbi = float(row.get("RBI", 0) or 0)
                sb = float(row.get("SB", 0) or 0)
                cs = float(row.get("CS", 0) or 0)
                so = float(row.get("SO", 0) or 0)
                avg = float(row.get("AVG", 0) or 0)
                ops = float(row.get("OPS", 0) or 0)
                slg = float(row.get("SLG", 0) or 0)

                # Compute derived stats
                nsb = sb - cs  # NSB can be negative for H2H One Win format (e.g., 0 SB - 1 CS = -1)
                tb = round(h * slg / max(avg, 0.001)) if avg > 0 else 0

                player = {
                    "id": _make_player_id(name),
                    "name": name,
                    "team": team,
                    "positions": positions,
                    "type": "batter",
                    "tier": 0,    # Will be assigned after z-score ranking
                    "adp": 999.0, # Will be filled from ADP file
                    "rank": 0,
                    "proj": {
                        "pa": pa, "r": r, "h": h, "hr": hr, "rbi": rbi,
                        "k_bat": so, "tb": tb, "avg": avg, "ops": ops,
                        "nsb": nsb, "slg": slg,
                    },
                    "z_score": 0.0,
                    "cat_scores": {},
                    "source": "steamer",
                }
                players.append(player)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping projection row %r: %s", row.get('Name', '?'), e)

    logger.info(f"Loaded {len(players)} Steamer batters from {path}")
    return players


# ---------------------------------------------------------------------------
# Steamer pitching loader
# ---------------------------------------------------------------------------

def load_steamer_pitching(path: Path) -> list[dict]:
    """
    Load FanGraphs Steamer pitching projections CSV.
    Separates SP (GS >= 10) from RP (GS < 5).
    """
    players = []
    if not path.exists():
        logger.warning(f"Steamer pitching file not found: {path}")
        return players

    _PITCHING_EXPECTED = {"Name", "IP", "ERA", "WHIP", "W", "SO"}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # K-16: Warn loudly on missing critical columns — never reject the whole file.
        if reader.fieldnames:
            headers = {h.strip() for h in reader.fieldnames if h}
            missing = _PITCHING_EXPECTED - headers
            if missing:
                logger.warning(
                    "Steamer pitching CSV missing expected columns %s — "
                    "affected stats will default to 0. Check FanGraphs export format.",
                    sorted(missing),
                )
        for row in reader:
            try:
                name = row.get("Name", row.get("name", "")).strip()
                team = row.get("Team", row.get("team", "FA")).strip().upper()

                ip = float(row.get("IP", 0) or 0)
                w = float(row.get("W", 0) or 0)
                l = float(row.get("L", 0) or 0)
                sv = float(row.get("SV", 0) or 0)
                bs = float(row.get("BS", 0) or 0)
                gs = float(row.get("GS", 0) or 0)
                k = float(row.get("SO", row.get("K", row.get("SO", 0))) or 0)
                era = float(row.get("ERA", 4.5) or 4.5)
                whip = float(row.get("WHIP", 1.3) or 1.3)
                hr_pit = float(row.get("HR", 0) or 0)
                bb = float(row.get("BB", 0) or 0)  # noqa: F841

                k9 = (k / ip * 9) if ip > 0 else 0.0
                qs = round(gs * 0.55) if gs >= 10 else 0
                nsv = max(0, sv - bs)

                # Determine position type
                if gs >= 10:
                    positions = ["SP"]
                elif sv > 5 or (sv > 0 and ip < 80):
                    positions = ["RP"]
                else:
                    positions = ["SP", "RP"]

                player = {
                    "id": _make_player_id(name),
                    "name": name,
                    "team": team,
                    "positions": positions,
                    "type": "pitcher",
                    "tier": 0,
                    "adp": 999.0,
                    "rank": 0,
                    "proj": {
                        "ip": ip, "w": w, "l": l, "sv": sv, "bs": bs,
                        "qs": qs, "k_pit": k, "era": era, "whip": whip,
                        "k9": k9, "hr_pit": hr_pit, "nsv": nsv,
                    },
                    "z_score": 0.0,
                    "cat_scores": {},
                    "source": "steamer",
                }
                players.append(player)
            except (ValueError, KeyError) as e:
                logger.warning("Skipping projection row %r: %s", row.get('Name', '?'), e)

    logger.info(f"Loaded {len(players)} Steamer pitchers from {path}")
    return players


# ---------------------------------------------------------------------------
# ADP loader (FantasyPros Yahoo 12-team format)
# ---------------------------------------------------------------------------

def load_adp(path: Path) -> dict[str, float]:
    """
    Load FantasyPros consensus ADP CSV.
    Returns dict mapping normalized player name → ADP float.
    """
    adp_map = {}
    if not path.exists():
        logger.warning(f"ADP file not found: {path}")
        return adp_map

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (
                row.get("PLAYER NAME", row.get("Name", row.get("name", "")))
                .strip()
            )
            try:
                adp_val = float(
                    row.get("AVG", row.get("ADP", row.get("adp", 999))) or 999
                )
                if name:
                    adp_map[_make_player_id(name)] = adp_val
            except (ValueError, TypeError):
                pass

    logger.info(f"Loaded {len(adp_map)} ADP entries from {path}")
    return adp_map


def _apply_adp(players: list[dict], adp_map: dict[str, float]) -> None:
    """Merge ADP data into player list in-place.

    Match strategy (in order):
    1. Exact normalized ID match
    2. Last name + first initial match (handles abbreviated first names)
    """
    # Pre-build first_initial_last_name -> adp_id map for fallback lookups
    initial_map: dict[str, str] = {}
    for adp_id in adp_map:
        parts = adp_id.split("_")
        if len(parts) >= 2:
            first_initial = parts[0][0] if parts[0] else ""
            last_name = parts[-1]
            key = f"{first_initial}_{last_name}"
            if key not in initial_map:
                initial_map[key] = adp_id
            else:
                logger.warning(
                    "ADP initial-fallback collision: key %r claimed by %r, ignoring %r -- "
                    "both players will attempt exact match only",
                    key, initial_map[key], adp_id,
                )

    matched_exact = 0
    matched_fallback = 0

    for p in players:
        pid = p["id"]

        # Pass 1: exact match
        if pid in adp_map:
            p["adp"] = adp_map[pid]
            matched_exact += 1
            continue

        # Pass 2: last name + first initial fallback
        parts = pid.split("_")
        if len(parts) >= 2:
            first_initial = parts[0][0] if parts[0] else ""
            last_name = parts[-1]
            key = f"{first_initial}_{last_name}"
            if key in initial_map:
                p["adp"] = adp_map[initial_map[key]]
                matched_fallback += 1

    total = matched_exact + matched_fallback
    logger.info(
        f"ADP matched {total}/{len(players)} players "
        f"(exact={matched_exact}, fallback={matched_fallback})"
    )


# ---------------------------------------------------------------------------
# Tier assignment based on z-score rank
# ---------------------------------------------------------------------------

def assign_tiers(players: list[dict]) -> None:
    """Assign tier 1-8 based on z-score rank within type, in-place."""
    batters = [p for p in players if p["type"] == "batter"]
    pitchers = [p for p in players if p["type"] == "pitcher"]

    tier_cutoffs_bat = [12, 36, 72, 108, 144, 180, 220, 999]
    tier_cutoffs_pit = [10, 30, 60, 90, 120, 160, 200, 999]

    batters.sort(key=lambda p: p.get("z_score", 0), reverse=True)
    pitchers.sort(key=lambda p: p.get("z_score", 0), reverse=True)

    for i, p in enumerate(batters):
        for tier, cutoff in enumerate(tier_cutoffs_bat, 1):
            if i < cutoff:
                p["tier"] = tier
                break

    for i, p in enumerate(pitchers):
        for tier, cutoff in enumerate(tier_cutoffs_pit, 1):
            if i < cutoff:
                p["tier"] = tier
                break


# ---------------------------------------------------------------------------
# Injury flags loader
# ---------------------------------------------------------------------------

def load_injury_flags(path: Path) -> dict[str, dict]:
    """
    Load injury_flags_2026.csv.
    Returns dict mapping player_id -> {"injury_risk": str, "injury_note": str, "avoid": bool}

    Columns: Name, Team, Status, Expected_PA_or_IP, Notes, Avoid_flag
    Status values: TJS_return, injury_risk, active
    """
    flags = {}
    if not path.exists():
        return flags

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            status = row.get("Status", "active").strip().lower()
            notes = row.get("Notes", "").strip()
            avoid = row.get("Avoid_flag", "no").strip().lower() == "yes"

            if not name:
                continue

            if avoid:
                risk = "extreme"
            elif status in ("tjs_return", "injury_risk"):
                risk = "high"
            else:
                risk = "low"

            flags[_make_player_id(name)] = {
                "injury_risk": risk,
                "injury_note": notes,
                "avoid": avoid,
            }

    logger.info(f"Loaded {len(flags)} injury flags from {path}")
    return flags


def _apply_injury_flags(players: list[dict], flags: dict[str, dict]) -> None:
    """Annotate players with injury risk in-place."""
    matched = 0
    for p in players:
        if p["id"] in flags:
            flag = flags[p["id"]]
            p["injury_risk"] = flag["injury_risk"]
            p["injury_note"] = flag["injury_note"]
            p["avoid"] = flag["avoid"]
            matched += 1
    logger.info(f"Injury flags applied to {matched}/{len(players)} players")


# ---------------------------------------------------------------------------
# Closer situations loader
# ---------------------------------------------------------------------------

def load_closer_situations(path: Path) -> dict[str, dict]:
    """
    Load closer_situations_2026.csv.
    Returns dict mapping team -> {"closer_id": str, "nsv_projection": float, "role": str}

    Columns: Team, Closer, Role, NSV_projection, Notes
    """
    closers = {}
    if not path.exists():
        return closers

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row.get("Team", "").strip().upper()
            closer_name = row.get("Closer", "").strip()
            role = row.get("Role", "unknown").strip().lower()
            try:
                nsv = float(row.get("NSV_projection", 0) or 0)
            except (ValueError, TypeError):
                nsv = 0.0

            if team and closer_name:
                closers[team] = {
                    "closer_id": _make_player_id(closer_name),
                    "nsv_projection": nsv,
                    "role": role,
                }

    logger.info(f"Loaded {len(closers)} closer situations from {path}")
    return closers


def _apply_closer_situations(players: list[dict], closers: dict[str, dict]) -> None:
    """
    Override nsv projections for confirmed closers in-place.
    Only updates when the closer's nsv projection is higher than Steamer's estimate.
    """
    # Build closer_id -> nsv map
    closer_nsv = {v["closer_id"]: (v["nsv_projection"], v["role"]) for v in closers.values()}

    updated = 0
    for p in players:
        if p["type"] != "pitcher":
            continue
        pid = p["id"]
        if pid in closer_nsv:
            nsv_proj, role = closer_nsv[pid]
            current_nsv = p["proj"].get("nsv", 0)
            # Only override if our projection is higher or it's a locked role
            if nsv_proj > current_nsv or role == "locked":
                p["proj"]["nsv"] = nsv_proj
                p["closer_role"] = role
                updated += 1

    logger.info(f"Closer NSV updated for {updated} pitchers")


# ---------------------------------------------------------------------------
# Position eligibility loader
# ---------------------------------------------------------------------------

def load_position_eligibility(path: Path) -> dict[str, list[str]]:
    """
    Load position_eligibility_2026.csv.
    Returns dict mapping player_id -> list of Yahoo positions.

    Columns: Name, Team, Yahoo_Positions_2026, Source_Note
    """
    eligibility = {}
    if not path.exists():
        return eligibility

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            pos_str = row.get("Yahoo_Positions_2026", "").strip()
            if not name or not pos_str:
                continue
            positions = [p.strip() for p in pos_str.split(",") if p.strip()]
            # Expand OF shorthand
            expanded = []
            for pos in positions:
                if pos == "OF":
                    expanded.extend(["LF", "CF", "RF", "OF"])
                else:
                    expanded.append(pos)
                    if pos in ("LF", "CF", "RF") and "OF" not in expanded:
                        expanded.append("OF")
            eligibility[_make_player_id(name)] = expanded

    logger.info(f"Loaded {len(eligibility)} position eligibility overrides from {path}")
    return eligibility


def _apply_position_eligibility(players: list[dict], eligibility: dict[str, list[str]]) -> None:
    """Override positions for players with verified multi-position eligibility in-place."""
    updated = 0
    for p in players:
        if p["id"] in eligibility:
            p["positions"] = eligibility[p["id"]]
            updated += 1
    logger.info(f"Position eligibility updated for {updated} players")


# ---------------------------------------------------------------------------
# Master loader — tries real CSVs, falls back to hardcoded board
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_full_board(data_dir: Optional[Path] = None) -> Optional[list[dict]]:
    """
    Attempt to load real projection data from CSV files.
    Returns None if no CSV files found (caller falls back to player_board.py).

    Priority:
    1. Steamer 2026 (most accurate, publicly available on FanGraphs)
    2. ZiPS 2026 (optional second source for averaging)
    3. ATC (average of all systems — if available)

    Note: passing a non-None data_dir bypasses and may evict the production
    cache (lru_cache is keyed on all args). Use non-None only in tests.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)

    bat_path = data_dir / "steamer_batting_2026.csv"
    pit_path = data_dir / "steamer_pitching_2026.csv"
    adp_path = data_dir / "adp_yahoo_2026.csv"
    injury_path = data_dir / "injury_flags_2026.csv"
    closer_path = data_dir / "closer_situations_2026.csv"
    eligibility_path = data_dir / "position_eligibility_2026.csv"

    if not bat_path.exists() and not pit_path.exists():
        logger.info("No Steamer CSV files found — using hardcoded player board")
        return None

    batters = load_steamer_batting(bat_path)
    pitchers = load_steamer_pitching(pit_path)

    if not batters and not pitchers:
        return None

    adp_map = load_adp(adp_path) if adp_path.exists() else {}

    # Compute z-scores using same logic as player_board.py
    from backend.fantasy_baseball.player_board import _compute_zscores
    _compute_zscores(batters, pitchers)

    all_players = batters + pitchers

    # Deduplicate by player ID — keeps first occurrence (batters take priority
    # over pitchers, so two-way players like Ohtani are counted as batters).
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for p in all_players:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            deduped.append(p)
    if len(deduped) < len(all_players):
        logger.info(
            "Removed %d duplicate player entries (two-way players in both CSVs)",
            len(all_players) - len(deduped),
        )
    all_players = deduped

    _apply_adp(all_players, adp_map)

    # Apply supplemental overlays (graceful — missing files are silently skipped)
    injury_flags = load_injury_flags(injury_path)
    if injury_flags:
        _apply_injury_flags(all_players, injury_flags)

    closer_situations = load_closer_situations(closer_path)
    if closer_situations:
        _apply_closer_situations(all_players, closer_situations)

    eligibility = load_position_eligibility(eligibility_path)
    if eligibility:
        _apply_position_eligibility(all_players, eligibility)

    assign_tiers(all_players)

    # Sort by ADP for final rank
    all_players.sort(key=lambda p: p["adp"])
    for i, p in enumerate(all_players, 1):
        p["rank"] = i

    logger.info(f"Loaded real projection board: {len(batters)} batters, {len(pitchers)} pitchers")
    return all_players


# ---------------------------------------------------------------------------
# CSV template generator — gives Kimi exact format to deliver data in
# ---------------------------------------------------------------------------

def write_csv_templates(data_dir: Optional[Path] = None) -> None:
    """
    Write empty CSV template files with correct column headers.
    Kimi or manual entry can populate these.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    bat_headers = ["Name", "Team", "POS", "G", "PA", "AB", "H", "2B", "3B",
                   "HR", "R", "RBI", "BB", "SO", "SB", "CS", "HBP", "SF",
                   "AVG", "OBP", "SLG", "OPS", "wOBA"]
    pit_headers = ["Name", "Team", "POS", "W", "L", "ERA", "G", "GS", "IP",
                   "H", "ER", "HR", "BB", "SO", "SV", "BS", "HLD", "WHIP",
                   "K/9", "BB/9", "FIP", "xFIP"]
    adp_headers = ["PLAYER NAME", "TEAM", "POS", "AVG", "BEST", "WORST",
                   "# TEAMS", "STDEV"]

    templates = [
        (data_dir / "steamer_batting_2026.csv", bat_headers),
        (data_dir / "steamer_pitching_2026.csv", pit_headers),
        (data_dir / "adp_yahoo_2026.csv", adp_headers),
    ]

    for path, headers in templates:
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"Created template: {path}")
        else:
            print(f"Already exists (not overwriting): {path}")


# ---------------------------------------------------------------------------
# Bridge: export FanGraphs RoS cache to Steamer-format CSVs
# ---------------------------------------------------------------------------

def export_ros_to_steamer_csvs(
    bat_raw: dict,  # {system_name: DataFrame} from fetch_all_ros("bat")
    pit_raw: dict,  # {system_name: DataFrame} from fetch_all_ros("pit")
    data_dir: Optional[Path] = None,
) -> dict:
    """Export FanGraphs RoS cache DataFrames to Steamer-format CSV files.

    Prefers the 'steamer' system if available, otherwise falls back to the
    first available system in the dict.  Writes CSVs that are directly
    loadable by ``load_steamer_batting()`` / ``load_steamer_pitching()``.

    Returns ``{"batting_rows": N, "pitching_rows": M}``.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    result = {"batting_rows": 0, "pitching_rows": 0}

    def _pick_df(raw: dict):
        """Return preferred DataFrame from the system dict, or None."""
        if not raw:
            return None
        if "steamer" in raw and not raw["steamer"].empty:
            return raw["steamer"]
        # Fallback: first non-empty system
        for df in raw.values():
            if not df.empty:
                return df
        return None

    # --- Batting ---
    bat_df = _pick_df(bat_raw)
    if bat_df is not None and len(bat_df) > 0:
        bat_out = bat_df.copy()
        if "POS" not in bat_out.columns:
            bat_out["POS"] = "DH"
        bat_path = data_dir / "steamer_batting_2026.csv"
        bat_out.to_csv(bat_path, index=False, encoding="utf-8-sig")
        result["batting_rows"] = len(bat_out)
        logger.info("Wrote %d batting rows to %s", len(bat_out), bat_path)

    # --- Pitching ---
    pit_df = _pick_df(pit_raw)
    if pit_df is not None and len(pit_df) > 0:
        pit_out = pit_df.copy()
        if "POS" not in pit_out.columns:
            if "GS" in pit_out.columns:
                pit_out["POS"] = pit_out["GS"].apply(
                    lambda gs: "SP" if float(gs or 0) >= 10 else "RP"
                )
            else:
                pit_out["POS"] = "SP"
        pit_path = data_dir / "steamer_pitching_2026.csv"
        pit_out.to_csv(pit_path, index=False, encoding="utf-8-sig")
        result["pitching_rows"] = len(pit_out)
        logger.info("Wrote %d pitching rows to %s", len(pit_out), pit_path)

    return result


if __name__ == "__main__":
    print("Writing CSV templates to data/projections/...")
    write_csv_templates()
    print()
    result = load_full_board()
    if result:
        print(f"Loaded {len(result)} players from real projections")
    else:
        print("No real projection CSVs found — templates created.")
        print("Drop Steamer CSV exports from FanGraphs into data/projections/")
        print("and re-run to activate real projection mode.")

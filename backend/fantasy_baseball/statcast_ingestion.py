"""
Statcast Daily Ingestion Pipeline

Pulls daily MLB data from Baseball Savant and updates player projections
using Bayesian inference with shrinkage priors.

Usage:
    from backend.fantasy_baseball.statcast_ingestion import StatcastIngestionAgent
    
    agent = StatcastIngestionAgent()
    agent.run_daily_ingestion()  # Pull yesterday's games and update projections

Schedule:
    Runs daily at 6:00 AM ET via scheduler in main.py
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.models import SessionLocal, PlayerProjection, StatcastPerformance
from backend.fantasy_baseball.player_board import get_or_create_projection

logger = logging.getLogger(__name__)


class PlayerIdResolver:
    """
    Cache-based player name → MLBAM ID resolver.

    Loads player_id_mapping table at import time and provides
    fast name-to-ID lookups for Statcast CSV rows that lack player_id.
    """
    def __init__(self):
        self._by_name: Dict[str, int] = {}  # full_name → mlbam_id
        self._by_normalized: Dict[str, int] = {}  # normalized_name → mlbam_id
        self._loaded = False

    def load(self, db: Session) -> None:
        """Load player_id_mapping into memory caches."""
        if self._loaded:
            return

        from backend.models import PlayerIDMapping

        mappings = db.query(PlayerIDMapping).all()
        for m in mappings:
            if m.mlbam_id:
                self._by_name[m.full_name.lower()] = m.mlbam_id
                if m.normalized_name:
                    self._by_normalized[m.normalized_name] = m.mlbam_id

        self._loaded = True
        logger.info("PlayerIdResolver loaded %d name→mlbam_id mappings", len(self._by_name))

    def resolve(self, player_name: str) -> str:
        """
        Resolve player_name to an ID string for StatcastPerformance.player_id.

        Returns mlbam_id as string if found in player_id_mapping,
        otherwise returns player_name as-is (fallback identifier).
        """
        if not player_name:
            return "unknown"

        # Try exact full_name match (case-insensitive)
        mlbam_id = self._by_name.get(player_name.lower())
        if mlbam_id:
            return str(mlbam_id)

        # Fallback: return player_name as identifier
        # Note: StatcastPerformance.player_id is String(50), not an FK, so names work
        return player_name


# Module-level resolver singleton (will be loaded on first agent use)
_player_id_resolver = PlayerIdResolver()


@dataclass
class PlayerDailyPerformance:
    """Single day performance from Statcast."""
    player_id: str
    player_name: str
    team: str
    game_date: date
    
    # Offense
    pa: int  # Plate appearances
    ab: int
    h: int
    doubles: int
    triples: int
    hr: int
    r: int
    rbi: int
    bb: int
    so: int
    hbp: int
    sb: int
    cs: int
    
    # Statcast Quality Metrics
    exit_velocity_avg: float
    launch_angle_avg: float
    hard_hit_pct: float  # 95+ mph exit velocity
    barrel_pct: float    # Ideal EV/LA combination
    xba: float           # Expected batting average
    xslg: float          # Expected slugging
    xwoba: float         # Expected weighted on-base average
    
    # Pitching (if applicable)
    ip: float = 0.0
    er: int = 0
    k_pit: int = 0
    bb_pit: int = 0
    pitches: int = 0

    # Metadata for type-scoped upserts
    is_pitcher: bool = False

    @property
    def avg(self) -> float:
        return self.h / self.ab if self.ab > 0 else 0.0
    
    @property
    def obp(self) -> float:
        pa_no_sf = self.pa  # Simplified
        return (self.h + self.bb + self.hbp) / pa_no_sf if pa_no_sf > 0 else 0.0
    
    @property
    def slg(self) -> float:
        tb = self.h + self.doubles + (2 * self.triples) + (3 * self.hr)
        return tb / self.ab if self.ab > 0 else 0.0
    
    @property
    def ops(self) -> float:
        return self.obp + self.slg
    
    @property
    def woba(self) -> float:
        """Calculate wOBA from components."""
        # Simplified wOBA calculation
        # 2024 weights: BB/HBP .69, 1B .89, 2B 1.27, 3B 1.62, HR 2.11
        singles = self.h - self.doubles - self.triples - self.hr
        numerator = (0.69 * (self.bb + self.hbp) + 
                     0.89 * singles + 
                     1.27 * self.doubles + 
                     1.62 * self.triples + 
                     2.11 * self.hr)
        return numerator / self.pa if self.pa > 0 else 0.0


@dataclass
class UpdatedProjection:
    """Result of Bayesian update."""
    player_id: str
    player_name: str
    
    # Prior (Steamer/ZiPS)
    prior_woba: float
    prior_variance: float
    
    # Likelihood (recent performance)
    sample_woba: float
    sample_variance: float
    sample_size: int  # PA
    
    # Posterior
    posterior_woba: float
    posterior_variance: float
    shrinkage: float  # 1.0 = trust prior fully, 0.0 = trust data fully
    
    # Additional stats
    updated_avg: float
    updated_obp: float
    updated_slg: float
    updated_ops: float
    updated_xwoba: float
    
    # Quality indicators
    data_quality_score: float  # 0-1 based on sample size, recency
    confidence_interval_95: Tuple[float, float]


class DataQualityChecker:
    """
    Validates ingested Statcast data for completeness and accuracy.
    """
    
    def __init__(self):
        self.issues: List[Dict] = []
    
    def validate_daily_pull(self, df: pd.DataFrame, target_date: date) -> bool:
        """
        Validate a daily Statcast pull.
        
        Checks:
        - Minimum games (should be ~15 for full slate)
        - Minimum players (should be ~300+)
        - Data completeness (% nulls)
        - Date range correctness
        """
        self.issues = []
        is_valid = True
        
        # Check 1: Minimum rows
        if len(df) < 200:
            self.issues.append({
                'severity': 'ERROR',
                'type': 'INSUFFICIENT_DATA',
                'message': f'Only {len(df)} player records, expected 300+',
                'target_date': target_date.isoformat()
            })
            is_valid = False
        
        # Check 2: Date range
        if 'game_date' in df.columns:
            dates = pd.to_datetime(df['game_date']).dt.date
            unique_dates = dates.unique()
            
            if target_date not in unique_dates:
                self.issues.append({
                    'severity': 'ERROR',
                    'type': 'WRONG_DATE',
                    'message': f'Target date {target_date} not in data. Found: {list(unique_dates)}',
                    'target_date': target_date.isoformat()
                })
                is_valid = False
            
            # Warn if multiple dates
            if len(unique_dates) > 1:
                self.issues.append({
                    'severity': 'WARNING',
                    'type': 'MULTIPLE_DATES',
                    'message': f'Data contains {len(unique_dates)} dates: {list(unique_dates)}',
                    'target_date': target_date.isoformat()
                })
        
        # Check 3: Critical columns present
        # Accept either raw Savant column names or cleaned aliases
        required_cols = ['player_name', 'team', 'game_date', 'pa']
        missing_cols = [c for c in required_cols if c not in df.columns]
        xwoba_present = (
            'estimated_woba_using_speedangle' in df.columns
            or 'xwoba' in df.columns
        )
        if not xwoba_present:
            missing_cols.append('xwoba (or estimated_woba_using_speedangle)')
        if missing_cols:
            self.issues.append({
                'severity': 'ERROR',
                'type': 'MISSING_COLUMNS',
                'message': f'Missing required columns: {missing_cols}',
                'target_date': target_date.isoformat()
            })
            is_valid = False
        
        # Check 4: Null rate
        if len(df) > 0:
            null_rates = df.isnull().mean()
            high_null_cols = null_rates[null_rates > 0.5].index.tolist()
            if high_null_cols:
                self.issues.append({
                    'severity': 'WARNING',
                    'type': 'HIGH_NULL_RATE',
                    'message': f'Columns with >50% nulls: {high_null_cols}',
                    'target_date': target_date.isoformat()
                })
        
        # Check 5: Value ranges
        if 'pa' in df.columns:
            invalid_pa = df[df['pa'] < 0]['pa'].count()
            if invalid_pa > 0:
                self.issues.append({
                    'severity': 'ERROR',
                    'type': 'INVALID_DATA',
                    'message': f'{invalid_pa} rows with negative PA',
                    'target_date': target_date.isoformat()
                })
                is_valid = False
        
        # Check 6: Reasonable xwoba range (0.000 to 0.600 is typical)
        xwoba_col = (
            'estimated_woba_using_speedangle' if 'estimated_woba_using_speedangle' in df.columns
            else 'xwoba'
        )
        if xwoba_col in df.columns:
            outlier_xwoba = df[(df[xwoba_col] < 0.000) | (df[xwoba_col] > 0.700)][xwoba_col].count()
            if outlier_xwoba > len(df) * 0.05:  # More than 5% outliers
                self.issues.append({
                    'severity': 'WARNING',
                    'type': 'DATA_ANOMALY',
                    'message': f'{outlier_xwoba} players with unusual xwoba values',
                    'target_date': target_date.isoformat()
                })
        
        return is_valid
    
    def get_validation_report(self) -> Dict:
        """Generate validation report for logging/monitoring."""
        errors = [i for i in self.issues if i['severity'] == 'ERROR']
        warnings = [i for i in self.issues if i['severity'] == 'WARNING']
        
        return {
            'is_valid': len(errors) == 0,
            'error_count': len(errors),
            'warning_count': len(warnings),
            'errors': errors,
            'warnings': warnings
        }


class StatcastIngestionAgent:
    """
    Agent responsible for daily Statcast data ingestion.
    
    Orchestrates:
    1. Fetching yesterday's data from Baseball Savant
    2. Validating data quality
    3. Running Bayesian projection updates
    4. Storing results in database
    5. Logging issues for monitoring
    """
    
    def __init__(self):
        self.base_url = "https://baseballsavant.mlb.com/statcast_search/csv"
        self.quality_checker = DataQualityChecker()
        self.db = SessionLocal()

        # Load player_id_mapping cache for name→ID resolution
        _player_id_resolver.load(self.db)
    
    def _fetch_by_player_type(self, target_date: date, player_type: str) -> Optional[pd.DataFrame]:
        """
        Fetch Statcast data for a specific date and player type ('batter' or 'pitcher').

        Uses Baseball Savant CSV export API with strict-inequality date range
        (Baseball Savant treats game_date_gt/lt as exclusive bounds).
        """
        logger.info("Fetching Statcast %s data for %s", player_type, target_date)

        params = {
            'all': 'true',
            'hfGT': 'R|',
            'hfSea': f'{target_date.year}|',
            'player_type': player_type,
            'game_date_gt': (target_date - timedelta(days=1)).isoformat(),
            'game_date_lt': (target_date + timedelta(days=1)).isoformat(),
            'group_by': 'name-date',
            'sort_col': 'pitches',
            'player_event_sort': 'api_p_release_speed',
            'sort_order': 'desc',
            # Note: omitting 'type': 'details' returns the leaderboard-aggregated CSV
            # (one row per player per game), which includes hardhit_percent,
            # barrels_per_pa_percent, xwoba, xba, xslg, pa, abs, hits, hrs, etc.
            # With 'type':'details' the API returns raw pitch events (13k+ rows/day)
            # with none of the aggregated count/quality columns.
        }

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=60,
            )

            if response.status_code != 200:
                logger.error("Statcast API returned %d for player_type=%s", response.status_code, player_type)
                return None

            from io import StringIO
            text_content = response.text
            df = pd.read_csv(StringIO(text_content))

            # Bug 4 fix: log actual columns on first real data so mapping issues surface early
            if len(df) > 0:
                logger.info(
                    "Statcast %s columns present: %s",
                    player_type, sorted(df.columns.tolist()),
                )

            logger.info(
                "Statcast %s: %d rows fetched (%d bytes)",
                player_type, len(df), len(text_content),
            )
            return df

        except Exception as e:
            logger.exception("Failed to fetch Statcast %s data: %s", player_type, e)
            return None

    def fetch_statcast_day(self, target_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch Statcast data for a specific date — both batters AND pitchers.

        Returns a combined DataFrame with a '_statcast_player_type' column
        ('batter' or 'pitcher') so transform_to_performance can route fields
        correctly. Returns None if both fetches fail or return empty results.
        """
        batter_df = self._fetch_by_player_type(target_date, 'batter')
        pitcher_df = self._fetch_by_player_type(target_date, 'pitcher')

        frames = []
        n_batters = 0
        n_pitchers = 0

        if batter_df is not None and len(batter_df) > 0:
            batter_df = batter_df.copy()
            batter_df['_statcast_player_type'] = 'batter'
            frames.append(batter_df)
            n_batters = len(batter_df)

        if pitcher_df is not None and len(pitcher_df) > 0:
            pitcher_df = pitcher_df.copy()
            pitcher_df['_statcast_player_type'] = 'pitcher'
            frames.append(pitcher_df)
            n_pitchers = len(pitcher_df)

        if not frames:
            logger.warning("Statcast: both batter and pitcher fetches returned no data for %s", target_date)
            return None

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Statcast combined: %d batters + %d pitchers = %d total rows for %s",
            n_batters, n_pitchers, len(combined), target_date,
        )

        # Pre-aggregate to daily granularity (resilient to per-pitch data)
        combined = self._aggregate_to_daily(combined)

        return combined
    
    @staticmethod
    def _icol(row: pd.Series, *names: str, default: int = 0) -> int:
        """Return first non-null int found among candidate column names."""
        for name in names:
            v = row.get(name)
            if v is not None and str(v).strip() not in ('', 'nan', 'NaN'):
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    continue
        return default

    @staticmethod
    def _fcol(row: pd.Series, *names: str, default: float = 0.0) -> float:
        """Return first non-null float found among candidate column names."""
        for name in names:
            v = row.get(name)
            if v is not None and str(v).strip() not in ('', 'nan', 'NaN'):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    continue
        return default

    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-aggregate per-pitch rows to daily granularity.

        Baseball Savant sometimes returns per-pitch rows instead of daily
        aggregates. This method groups by (player, game_date, player_type)
        and SUMs counting stats while AVERAGEing quality metrics.

        Short-circuits if max group size is 1 (leaderboard data passthrough).
        """
        if df is None or df.empty:
            return df

        # Determine group key columns
        group_cols = []
        if 'player_id' in df.columns:
            group_cols.append('player_id')
        elif 'player_name' in df.columns:
            group_cols.append('player_name')
        else:
            return df  # Cannot group without identifier

        if 'game_date' in df.columns:
            group_cols.append('game_date')
        if '_statcast_player_type' in df.columns:
            group_cols.append('_statcast_player_type')

        # Short-circuit: if max group size is 1, skip aggregation
        grouped = df.groupby(group_cols, dropna=False)
        if grouped.size().max() <= 1:
            return df

        # Columns to SUM (counting stats)
        sum_cols = [
            'pa', 'ab', 'abs', 'h', 'hits', 'hit', 'singles', 'single',
            'doubles', 'double', 'triples', 'triple', 'hr', 'hrs',
            'home_run', 'home_runs', 'r', 'run', 'runs', 'rbi',
            'bb', 'walk', 'walks', 'so', 'strikeout', 'strikeouts',
            'hbp', 'hit_by_pitch', 'sb', 'stolen_base', 'stolen_bases',
            'stolen_base_2b', 'cs', 'caught_stealing', 'caught_stealing_2b',
            'pitches', 'er', 'p_strikeout', 'p_walk', 'k', 'k_pit',
            'bb_pit', 'ip',
        ]

        # Columns to AVERAGE (quality metrics)
        mean_cols = [
            'launch_speed', 'exit_velocity_avg', 'launch_angle',
            'launch_angle_avg', 'hardhit_percent', 'hard_hit_percent',
            'hard_hit_pct', 'barrels_per_pa_percent',
            'barrels_per_bbe_percent', 'barrel_batted_rate', 'barrel_pct',
            'xba', 'estimated_ba_using_speedangle', 'xslg',
            'estimated_slg_using_speedangle', 'xwoba',
            'estimated_woba_using_speedangle', 'woba',
        ]

        # Identity columns to take FIRST value
        identity_cols = ['player_name', 'team', 'player_id']
        identity_cols = [c for c in identity_cols if c in df.columns and c not in group_cols]

        # Filter to columns actually present in the DataFrame
        present_sum = [c for c in sum_cols if c in df.columns]
        present_mean = [c for c in mean_cols if c in df.columns]

        # Coerce numeric before aggregation
        for c in present_sum + present_mean:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # Build aggregation dict
        agg_dict = {}
        for c in present_sum:
            agg_dict[c] = 'sum'
        for c in present_mean:
            agg_dict[c] = 'mean'
        for c in identity_cols:
            agg_dict[c] = 'first'

        result = grouped.agg(agg_dict).reset_index()

        logger.info(
            "Statcast pre-aggregation: %d rows -> %d daily rows (max group size was %d)",
            len(df), len(result), grouped.size().max(),
        )

        return result

    def transform_to_performance(self, df: pd.DataFrame) -> List[PlayerDailyPerformance]:
        """
        Transform Statcast DataFrame to PlayerDailyPerformance objects.

        Handles both batter rows and pitcher rows (from the two-pass fetch).
        For pitcher rows (_statcast_player_type == 'pitcher'):
          - Batting counting stats (pa, ab, h, hr, bb, so, ...) are set to 0.
          - k_pit / bb_pit / pitches come from the pitcher-perspective event counts.
          - Statcast quality metrics (exit velocity, xwOBA, etc.) represent
            pitch outcomes against this pitcher and are stored as-is.
        """
        performances = []

        for _, row in df.iterrows():
            try:
                # Log first row structure for diagnostics (only once per call)
                if not hasattr(self, '_diag_logged'):
                    self._diag_logged = True
                    logger.info(
                        "transform_to_performance: DataFrame has %d rows, columns: %s. player_id column present: %s",
                        len(df),
                        sorted(df.columns),
                        'player_id' in df.columns,
                    )

                # Statcast CSV with group_by='name-date' does NOT include player_id column.
                # Extract identifier: prefer player_id column if present, otherwise resolve player_name.
                pid_col = row.get('player_id')
                if pid_col is not None and str(pid_col).strip() not in ('', 'nan', 'NaN'):
                    player_id = str(pid_col)
                else:
                    # Fallback: resolve player_name to mlbam_id via player_id_mapping cache
                    player_name_raw = row.get('player_name', '')
                    if not player_name_raw or str(player_name_raw).strip() in ('', 'nan', 'NaN'):
                        continue  # Skip rows with no identifiable player
                    player_id = _player_id_resolver.resolve(str(player_name_raw))

                is_pitcher = str(row.get('_statcast_player_type', 'batter')) == 'pitcher'

                if is_pitcher:
                    # Pitcher row: batting stats are zeroed; Ks/BBs come from
                    # the event counts which represent outcomes against this pitcher.
                    perf = PlayerDailyPerformance(
                        player_id=player_id,
                        player_name=str(row.get('player_name', '')),
                        team=str(row.get('team', '')),
                        game_date=pd.to_datetime(row.get('game_date')).date(),
                        pa=0, ab=0, h=0, doubles=0, triples=0,
                        hr=0, r=0, rbi=0, bb=0, so=0, hbp=0, sb=0, cs=0,
                        # Savant column names first, then our clean aliases
                        exit_velocity_avg=self._fcol(row, 'launch_speed', 'exit_velocity_avg'),
                        launch_angle_avg=self._fcol(row, 'launch_angle', 'launch_angle_avg'),
                        # hardhit_percent = leaderboard name; hard_hit_percent = details name
                        hard_hit_pct=self._fcol(row, 'hardhit_percent', 'hard_hit_percent', 'hard_hit_pct') / 100,
                        # barrels_per_pa_percent = leaderboard; barrel_batted_rate = details
                        barrel_pct=self._fcol(row, 'barrels_per_pa_percent', 'barrels_per_bbe_percent', 'barrel_batted_rate', 'barrel_pct') / 100,
                        xba=self._fcol(row, 'xba', 'estimated_ba_using_speedangle'),
                        xslg=self._fcol(row, 'xslg', 'estimated_slg_using_speedangle'),
                        xwoba=self._fcol(row, 'xwoba', 'estimated_woba_using_speedangle'),
                        ip=self._fcol(row, 'ip'),
                        er=self._icol(row, 'er'),
                        # strikeout / walk columns = pitcher Ks / BBs in pitcher-type fetch
                        k_pit=self._icol(row, 'so', 'p_strikeout', 'strikeout', 'k'),
                        bb_pit=self._icol(row, 'bb', 'p_walk', 'walk'),
                        pitches=self._icol(row, 'pitches'),
                        is_pitcher=True,
                    )
                else:
                    # Bug 4 fix: accept alternate column names Baseball Savant may return.
                    # Primary names come from the grouped name-date CSV; alternatives handle
                    # any Baseball Savant endpoint variation.
                    perf = PlayerDailyPerformance(
                        player_id=player_id,
                        player_name=str(row.get('player_name', '')),
                        team=str(row.get('team', '')),
                        game_date=pd.to_datetime(row.get('game_date')).date(),
                        pa=self._icol(row, 'pa'),
                        # leaderboard uses 'abs' for at-bats; details uses 'ab'
                        ab=self._icol(row, 'abs', 'ab'),
                        h=self._icol(row, 'hits', 'hit', 'singles', 'h'),
                        doubles=self._icol(row, 'doubles', 'double'),
                        triples=self._icol(row, 'triples', 'triple'),
                        # leaderboard uses 'hrs'; details uses 'home_run'
                        hr=self._icol(row, 'hrs', 'home_run', 'home_runs', 'hr'),
                        r=self._icol(row, 'run', 'runs', 'r'),
                        rbi=self._icol(row, 'rbi'),
                        bb=self._icol(row, 'bb', 'walk', 'walks'),
                        so=self._icol(row, 'so', 'strikeout', 'strikeouts'),
                        hbp=self._icol(row, 'hbp', 'hit_by_pitch'),
                        sb=self._icol(row, 'stolen_base_2b', 'sb', 'stolen_base', 'stolen_bases'),
                        cs=self._icol(row, 'caught_stealing_2b', 'cs', 'caught_stealing'),
                        # leaderboard / details Savant column names, then our clean aliases
                        exit_velocity_avg=self._fcol(row, 'launch_speed', 'exit_velocity_avg'),
                        launch_angle_avg=self._fcol(row, 'launch_angle', 'launch_angle_avg'),
                        # hardhit_percent = leaderboard name; hard_hit_percent = details name
                        hard_hit_pct=self._fcol(row, 'hardhit_percent', 'hard_hit_percent', 'hard_hit_pct') / 100,
                        # barrels_per_pa_percent = leaderboard; barrel_batted_rate = details
                        barrel_pct=self._fcol(row, 'barrels_per_pa_percent', 'barrels_per_bbe_percent', 'barrel_batted_rate', 'barrel_pct') / 100,
                        xba=self._fcol(row, 'xba', 'estimated_ba_using_speedangle'),
                        xslg=self._fcol(row, 'xslg', 'estimated_slg_using_speedangle'),
                        xwoba=self._fcol(row, 'xwoba', 'estimated_woba_using_speedangle'),
                        ip=0.0, er=0, k_pit=0, bb_pit=0,
                        pitches=self._icol(row, 'pitches'),
                    )

                performances.append(perf)
            except Exception as e:
                logger.warning(
                    "Failed to parse row for %s: %s",
                    row.get('player_name', 'unknown'), e,
                )
                continue

        return performances
    
    def store_performances(self, performances: List[PlayerDailyPerformance]) -> int:
        """
        Upsert daily performances to statcast_performances.

        Uses INSERT ON CONFLICT DO UPDATE (constraint uq_player_date) so that
        re-running the pipeline corrects previously-stored bad data rather than
        silently skipping rows.

        Returns the number of rows upserted.
        """
        rows_upserted = 0
        now = datetime.now(ZoneInfo("America/New_York"))

        for perf in performances:
            try:
                stmt = pg_insert(StatcastPerformance.__table__).values(
                    player_id=perf.player_id,
                    player_name=perf.player_name,
                    team=perf.team,
                    game_date=perf.game_date,
                    pa=perf.pa,
                    ab=perf.ab,
                    h=perf.h,
                    doubles=perf.doubles,
                    triples=perf.triples,
                    hr=perf.hr,
                    r=perf.r,
                    rbi=perf.rbi,
                    bb=perf.bb,
                    so=perf.so,
                    hbp=perf.hbp,
                    sb=perf.sb,
                    cs=perf.cs,
                    exit_velocity_avg=perf.exit_velocity_avg,
                    launch_angle_avg=perf.launch_angle_avg,
                    hard_hit_pct=perf.hard_hit_pct,
                    barrel_pct=perf.barrel_pct,
                    xba=perf.xba,
                    xslg=perf.xslg,
                    xwoba=perf.xwoba,
                    woba=perf.woba,
                    avg=perf.avg,
                    obp=perf.obp,
                    slg=perf.slg,
                    ops=perf.ops,
                    ip=perf.ip,
                    er=perf.er,
                    k_pit=perf.k_pit,
                    bb_pit=perf.bb_pit,
                    pitches=perf.pitches,
                    created_at=now,
                )

                # Scope the ON CONFLICT UPDATE by player type to protect
                # two-way players (e.g. Ohtani).  Pitcher rows must NOT
                # overwrite batting counting stats with zeros.
                if perf.is_pitcher:
                    update_set = dict(
                        player_name=perf.player_name,
                        team=perf.team,
                        exit_velocity_avg=perf.exit_velocity_avg,
                        launch_angle_avg=perf.launch_angle_avg,
                        hard_hit_pct=perf.hard_hit_pct,
                        barrel_pct=perf.barrel_pct,
                        xba=perf.xba,
                        xslg=perf.xslg,
                        xwoba=perf.xwoba,
                        ip=perf.ip,
                        er=perf.er,
                        k_pit=perf.k_pit,
                        bb_pit=perf.bb_pit,
                        pitches=perf.pitches,
                    )
                else:
                    update_set = dict(
                        player_name=perf.player_name,
                        team=perf.team,
                        pa=perf.pa,
                        ab=perf.ab,
                        h=perf.h,
                        doubles=perf.doubles,
                        triples=perf.triples,
                        hr=perf.hr,
                        r=perf.r,
                        rbi=perf.rbi,
                        bb=perf.bb,
                        so=perf.so,
                        hbp=perf.hbp,
                        sb=perf.sb,
                        cs=perf.cs,
                        exit_velocity_avg=perf.exit_velocity_avg,
                        launch_angle_avg=perf.launch_angle_avg,
                        hard_hit_pct=perf.hard_hit_pct,
                        barrel_pct=perf.barrel_pct,
                        xba=perf.xba,
                        xslg=perf.xslg,
                        xwoba=perf.xwoba,
                        woba=perf.woba,
                        avg=perf.avg,
                        obp=perf.obp,
                        slg=perf.slg,
                        ops=perf.ops,
                        ip=perf.ip,
                        er=perf.er,
                        k_pit=perf.k_pit,
                        bb_pit=perf.bb_pit,
                        pitches=perf.pitches,
                    )

                stmt = stmt.on_conflict_do_update(
                    index_elements=['player_id', 'game_date'],
                    set_=update_set,
                )
                self.db.execute(stmt)
                rows_upserted += 1
            except Exception as e:
                logger.warning("Failed to upsert performance for %s on %s: %s", perf.player_name, perf.game_date, e)
                continue

        self.db.commit()
        logger.info("Statcast: %d rows upserted for %s", rows_upserted, performances[0].game_date if performances else 'n/a')
        return rows_upserted


class BayesianProjectionUpdater:
    """
    Updates player projections using Bayesian inference with shrinkage priors.
    
    Key insight: Early season data is noisy. Use shrinkage to balance prior
    (Steamer/ZiPS) with likelihood (actual performance).
    
    shrinkage = 1.0 → Trust prior fully (no season data)
    shrinkage = 0.0 → Trust data fully (large sample)
    """
    
    def __init__(self):
        self.db = SessionLocal()
    
    def get_prior_projection(self, player_id: str) -> Optional[Dict]:
        """Get Steamer/ZiPS prior projection for a player."""
        # Try database first
        db_proj = self.db.query(PlayerProjection).filter(
            PlayerProjection.player_id == player_id
        ).first()
        
        if db_proj:
            return {
                'player_id': player_id,
                'woba': db_proj.woba,
                'avg': db_proj.avg,
                'obp': db_proj.obp,
                'slg': db_proj.slg,
                'ops': db_proj.ops,
                'hr': db_proj.hr,
                'r': db_proj.r,
                'rbi': db_proj.rbi,
                'sb': db_proj.sb,
                'variance': 0.0025  # Assumed variance in prior
            }
        
        # Fall back to player_board
        yahoo_player = {'player_key': player_id}
        board_proj = get_or_create_projection(yahoo_player)
        
        if board_proj:
            return {
                'player_id': player_id,
                'woba': board_proj.get('woba', 0.320),
                'avg': board_proj.get('avg', 0.250),
                'obp': board_proj.get('obp', 0.320),
                'slg': board_proj.get('slg', 0.400),
                'ops': board_proj.get('ops', 0.720),
                'hr': board_proj.get('hr', 15),
                'r': board_proj.get('r', 65),
                'rbi': board_proj.get('rbi', 65),
                'sb': board_proj.get('sb', 5),
                'variance': 0.0025
            }
        
        return None
    
    def get_recent_performance(
        self, 
        player_id: str, 
        lookback_days: int = 14
    ) -> Optional[Dict]:
        """
        Get aggregated performance over last N days.
        
        Returns weighted performance (recent games weighted more heavily).
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        performances = self.db.query(StatcastPerformance).filter(
            StatcastPerformance.player_id == player_id,
            StatcastPerformance.game_date >= start_date,
            StatcastPerformance.game_date <= end_date
        ).all()
        
        if not performances:
            return None
        
        # Calculate weighted averages (exponential decay)
        total_pa = sum(p.pa for p in performances)
        if total_pa < 10:  # Need minimum sample
            return None
        
        # Simple unweighted for now (can add decay later)
        weighted_woba = sum(p.woba * p.pa for p in performances) / total_pa
        weighted_xwoba = sum(p.xwoba * p.pa for p in performances) / total_pa
        weighted_avg = sum(p.avg * p.pa for p in performances) / total_pa
        weighted_slg = sum(p.slg * p.pa for p in performances) / total_pa
        weighted_ops = sum(p.ops * p.pa for p in performances) / total_pa
        
        # Calculate sample variance (simplified)
        sample_variance = 0.01 / total_pa if total_pa > 0 else 1.0
        
        return {
            'woba': weighted_woba,
            'xwoba': weighted_xwoba,
            'avg': weighted_avg,
            'slg': weighted_slg,
            'ops': weighted_ops,
            'pa': total_pa,
            'games': len(performances),
            'variance': sample_variance
        }
    
    def bayesian_update(
        self,
        prior: Dict,
        likelihood: Dict
    ) -> UpdatedProjection:
        """
        Perform conjugate normal update.
        
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * sample_mean) / posterior_precision
        """
        player_id = prior['player_id']
        
        # Prior parameters
        prior_mean = prior['woba']
        prior_precision = 1 / prior['variance']
        
        # Likelihood parameters
        sample_mean = likelihood['woba']
        sample_variance = likelihood['variance']
        likelihood_precision = 1 / sample_variance if sample_variance > 0 else 0
        
        # Posterior calculation
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (
            (prior_precision * prior_mean) + 
            (likelihood_precision * sample_mean)
        ) / posterior_precision
        
        # Shrinkage factor
        shrinkage = prior_precision / posterior_precision
        
        # Confidence interval (95%)
        posterior_std = (1 / posterior_precision) ** 0.5
        ci_lower = posterior_mean - (1.96 * posterior_std)
        ci_upper = posterior_mean + (1.96 * posterior_std)
        
        # Data quality score based on sample size
        # 0 PA = 0.0 quality, 200+ PA = 1.0 quality
        data_quality = min(1.0, likelihood['pa'] / 200)
        
        return UpdatedProjection(
            player_id=player_id,
            player_name=prior.get('player_name', player_id),
            prior_woba=prior_mean,
            prior_variance=prior['variance'],
            sample_woba=sample_mean,
            sample_variance=sample_variance,
            sample_size=likelihood['pa'],
            posterior_woba=posterior_mean,
            posterior_variance=1 / posterior_precision,
            shrinkage=shrinkage,
            updated_avg=likelihood['avg'],  # Simplified: use recent avg
            updated_obp=prior['obp'],  # Would calculate properly
            updated_slg=likelihood['slg'],
            updated_ops=likelihood['ops'],
            updated_xwoba=likelihood['xwoba'],
            data_quality_score=data_quality,
            confidence_interval_95=(ci_lower, ci_upper)
        )
    
    def update_all_projections(self, min_pa: int = 20) -> List[UpdatedProjection]:
        """
        Run Bayesian update for all players with sufficient recent data.
        
        Args:
            min_pa: Minimum plate appearances to trigger update
        """
        logger.info(f"Running Bayesian projection updates (min {min_pa} PA)")
        
        updated_projections = []
        
        # Get all players with recent data
        recent = self.db.query(StatcastPerformance.player_id).distinct().all()
        player_ids = [p[0] for p in recent]
        
        logger.info(f"Found {len(player_ids)} players with recent Statcast data")
        
        for player_id in player_ids:
            try:
                # Get prior
                prior = self.get_prior_projection(player_id)
                if not prior:
                    logger.debug(f"No prior projection for {player_id}")
                    continue
                
                # Get recent performance
                likelihood = self.get_recent_performance(player_id, lookback_days=14)
                if not likelihood or likelihood['pa'] < min_pa:
                    continue
                
                # Run Bayesian update
                updated = self.bayesian_update(prior, likelihood)
                updated_projections.append(updated)
                
                # Store in database
                self._store_updated_projection(updated)
                
            except Exception as e:
                logger.warning(f"Failed to update projection for {player_id}: {e}")
                continue
        
        self.db.commit()
        logger.info(f"Updated {len(updated_projections)} player projections")
        
        return updated_projections
    
    def _store_updated_projection(self, updated: UpdatedProjection):
        """Store updated projection in database."""
        # Check for existing
        existing = self.db.query(PlayerProjection).filter(
            PlayerProjection.player_id == updated.player_id
        ).first()
        
        if existing:
            # Update existing
            existing.woba = updated.posterior_woba
            existing.avg = updated.updated_avg
            existing.obp = updated.updated_obp
            existing.slg = updated.updated_slg
            existing.ops = updated.updated_ops
            existing.xwoba = updated.updated_xwoba
            existing.shrinkage = updated.shrinkage
            existing.data_quality_score = updated.data_quality_score
            existing.sample_size = updated.sample_size
            existing.updated_at = datetime.now(ZoneInfo("America/New_York"))
            existing.update_method = 'bayesian'
        else:
            # Create new
            record = PlayerProjection(
                player_id=updated.player_id,
                player_name=updated.player_name,
                woba=updated.posterior_woba,
                avg=updated.updated_avg,
                obp=updated.updated_obp,
                slg=updated.updated_slg,
                ops=updated.updated_ops,
                xwoba=updated.updated_xwoba,
                shrinkage=updated.shrinkage,
                data_quality_score=updated.data_quality_score,
                sample_size=updated.sample_size,
                updated_at=datetime.now(ZoneInfo("America/New_York")),
                update_method='bayesian'
            )
            self.db.add(record)


def run_daily_ingestion(target_date: Optional[date] = None):
    """
    Main entry point for daily Statcast ingestion.
    
    This function is called by:
    - Scheduled job (6:00 AM ET daily)
    - Manual trigger from admin panel
    - Backfill script for historical data
    
    Args:
        target_date: Date to ingest (default: yesterday)
    """
    if target_date is None:
        # Anchor to ET — Railway runs UTC so date.today() can be wrong after midnight ET
        target_date = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date()
    
    logger.info("=" * 60)
    logger.info(f"Starting daily Statcast ingestion for {target_date}")
    logger.info("=" * 60)
    
    # Step 1: Ingest data
    agent = StatcastIngestionAgent()
    df = agent.fetch_statcast_day(target_date)
    
    if df is None or len(df) == 0:
        logger.error(f"Failed to fetch Statcast data for {target_date}")
        return {
            'success': False,
            'date': target_date.isoformat(),
            'error': 'Failed to fetch data',
            'records_processed': 0
        }
    
    # Step 2: Validate data quality
    is_valid = agent.quality_checker.validate_daily_pull(df, target_date)
    validation_report = agent.quality_checker.get_validation_report()
    
    logger.info(f"Data validation: {validation_report['error_count']} errors, "
                f"{validation_report['warning_count']} warnings")
    
    if not is_valid:
        logger.error("Data quality validation failed")
        for error in validation_report['errors']:
            logger.error(f"  - {error['type']}: {error['message']}")
    
    # Step 3: Transform and store — close agent DB session when done
    performances = agent.transform_to_performance(df)
    rows_stored = 0
    try:
        rows_stored = agent.store_performances(performances)
    finally:
        agent.db.close()

    # Step 4: Run Bayesian updates — close updater DB session when done
    updater = BayesianProjectionUpdater()
    try:
        updated_projections = updater.update_all_projections(min_pa=20)
    finally:
        updater.db.close()

    # Step 5: Generate summary
    high_confidence = [p for p in updated_projections if p.data_quality_score > 0.5]
    big_movers = [p for p in updated_projections
                  if abs(p.posterior_woba - p.prior_woba) > 0.020]

    logger.info("=" * 60)
    logger.info("Daily ingestion complete for %s", target_date)
    logger.info("  Records processed: %d", rows_stored)
    logger.info("  Projections updated: %d", len(updated_projections))
    logger.info("  High confidence (>50% quality): %d", len(high_confidence))
    logger.info("  Big movers (>20 wOBA points): %d", len(big_movers))
    logger.info("=" * 60)

    return {
        'success': True,
        'date': target_date.isoformat(),
        'records_processed': rows_stored,
        'projections_updated': len(updated_projections),
        'high_confidence_updates': len(high_confidence),
        'big_movers': len(big_movers),
        'validation': validation_report,
        'big_mover_details': [
            {
                'name': p.player_name,
                'prior': round(p.prior_woba, 3),
                'posterior': round(p.posterior_woba, 3),
                'delta': round(p.posterior_woba - p.prior_woba, 3),
                'shrinkage': round(p.shrinkage, 3),
            }
            for p in big_movers[:10]
        ],
    }


if __name__ == "__main__":
    # Run for yesterday when called directly
    import sys
    
    if len(sys.argv) > 1:
        # Parse date from command line: YYYY-MM-DD
        target = date.fromisoformat(sys.argv[1])
    else:
        target = None
    
    result = run_daily_ingestion(target)
    print(result)

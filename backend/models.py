"""
Database models for CBB Edge Analyzer
SQLAlchemy ORM with PostgreSQL
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    DateTime,
    Boolean,
    JSON,
    Text,
    ForeignKey,
    Date,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, date
from zoneinfo import ZoneInfo
import os
import time

# Try to load dotenv, but don't fail if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Railway provides env vars directly

# 2. Sync URL — used by background scripts, migrations, and legacy sync code.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/cbb_edge")

# 3. Async URL — swaps psycopg2 driver for asyncpg.
#    Falls back gracefully if DATABASE_URL is not set (e.g. test environments).
_ASYNC_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://", "postgresql+asyncpg://"
).replace(
    "postgresql+psycopg2://", "postgresql+asyncpg://"
)

# ── Sync engine (keep for all existing sync paths) ──────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Async engine (used by nightly analysis hot path and APScheduler coroutines)
# Wrapped in try/except so the server still starts if asyncpg is not installed.
# Install asyncpg to enable the async hot path: pip install asyncpg==0.29.0
try:
    async_engine = create_async_engine(
        _ASYNC_DATABASE_URL,
        pool_pre_ping=False,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
except Exception as _async_engine_exc:  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "asyncpg not available — async DB engine disabled (%s). "
        "Install asyncpg to enable the async hot path.",
        _async_engine_exc,
    )
    async_engine = None  # type: ignore[assignment]
    AsyncSessionLocal = None  # type: ignore[assignment]

Base = declarative_base()


# ── Session dependencies ─────────────────────────────────────────────────────

def get_db():
    """Sync session dependency with retry on transient connection failures."""
    db = None
    for attempt in range(3):
        try:
            db = SessionLocal()
            break
        except Exception as e:
            if attempt == 2:
                raise
            error_str = str(e).lower()
            if any(k in error_str for k in ("connection", "timeout", "ssl")):
                time.sleep(0.1 * (2 ** attempt))
            else:
                raise
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_db():
    """Async session dependency for FastAPI async routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

# ... (rest of your classes like Game, Prediction, etc. remain exactly the same)

class Game(Base):
    """CBB game with teams and basic info"""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True)  # From odds API
    game_date = Column(DateTime, nullable=False, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    venue = Column(String)
    is_neutral = Column(Boolean, default=False)
    
    # Actual results (filled after game)
    home_score = Column(Integer)
    away_score = Column(Integer)
    completed = Column(Boolean, default=False)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="game")
    bet_logs = relationship("BetLog", back_populates="game")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    """Model predictions for each game"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    
    # Model metadata
    model_version = Column(String, default="v7.0")
    prediction_date = Column(Date, nullable=False, index=True, default=date.today)
    run_tier = Column(String, default="nightly", nullable=False)  # "opener" | "nightly" | "closing"
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Ratings used (for auditing)
    kenpom_home = Column(Float)
    kenpom_away = Column(Float)
    barttorvik_home = Column(Float)
    barttorvik_away = Column(Float)
    evanmiya_home = Column(Float)
    evanmiya_away = Column(Float)
    
    # Model outputs
    projected_margin = Column(Float)  # Positive = home favored
    adjusted_sd = Column(Float)
    point_prob = Column(Float)  # Point estimate probability
    lower_ci_prob = Column(Float)  # Lower 95% CI
    upper_ci_prob = Column(Float)  # Upper 95% CI

    # Actual outcome — populated by update_completed_games() after game finishes.
    # NULL while game is pending.  home_score - away_score (positive = home won).
    actual_margin = Column(Float)

    # Edge calculations
    edge_point = Column(Float)  # Point estimate edge
    edge_conservative = Column(Float)  # Lower CI edge (decision threshold)
    kelly_full = Column(Float)
    kelly_fractional = Column(Float)
    recommended_units = Column(Float)

    # V9 SNR & Integrity
    snr = Column(Float)
    snr_kelly_scalar = Column(Float)
    integrity_verdict = Column(String)
    
    # Verdict
    verdict = Column(String, nullable=False, index=True)  # "PASS" or "Bet X units..."
    pass_reason = Column(String)  # If PASS, why?
    
    # Full analysis (for debugging)
    full_analysis = Column(JSON)
    
    # Data quality
    data_freshness_tier = Column(String)  # Tier 1/2/3
    penalties_applied = Column(JSON)  # Dict of penalty types & values

    # K-14: which simulation engine produced this prediction
    pricing_engine = Column(String(20))  # 'markov' | 'gaussian' | None

    # K-15: Oracle Validation — divergence from rating-system consensus
    oracle_flag = Column(Boolean)         # True when z-score ≥ time-weighted threshold
    oracle_result = Column(JSON)          # OracleResult.to_dict() snapshot

    # Relationship
    game = relationship("Game", back_populates="predictions")

    __table_args__ = (UniqueConstraint('game_id', 'prediction_date', 'run_tier', name='_game_prediction_date_tier_uc'),)


class BetLog(Base):
    """Actual bets placed (manual entry or tracking)"""

    __tablename__ = "bet_logs"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    
    # Bet details
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    pick = Column(String, nullable=False)  # "Duke -4.5" or "UNC/Duke U145.5"
    bet_type = Column(String)  # "spread", "total", "moneyline"
    odds_taken = Column(Float, nullable=False)  # American odds
    
    # Sizing
    bankroll_at_bet = Column(Float)
    kelly_full = Column(Float)
    kelly_fractional = Column(Float)
    bet_size_pct = Column(Float)  # % of bankroll
    bet_size_units = Column(Float)  # In "units" (1 unit = 1% starting bankroll)
    bet_size_dollars = Column(Float)  # Actual $ amount
    
    # Model at time of bet
    model_prob = Column(Float)
    lower_ci_prob = Column(Float)
    point_edge = Column(Float)
    conservative_edge = Column(Float)
    
    # Outcome (filled after game)
    outcome = Column(Integer)  # 1=win, 0=loss, null=pending
    profit_loss_units = Column(Float)
    profit_loss_dollars = Column(Float)
    
    # CLV tracking
    closing_line = Column(Float)  # American odds at close
    clv_points = Column(Float)  # Points gained vs close
    clv_prob = Column(Float)  # Probability edge vs close
    
    # Flags
    is_backfill = Column(Boolean, default=False)  # Historical simulation
    is_paper_trade = Column(Boolean, default=False)  # Not real money
    executed = Column(Boolean, default=False)  # Actually placed
    
    # Notes
    notes = Column(Text)
    
    # Relationship
    game = relationship("Game", back_populates="bet_logs")


class ModelParameter(Base):
    """Tracking of model parameter changes over time"""

    __tablename__ = "model_parameters"

    id = Column(Integer, primary_key=True, index=True)
    effective_date = Column(DateTime, default=datetime.utcnow, index=True)
    parameter_name = Column(String, nullable=False)
    parameter_value = Column(Float)
    parameter_value_json = Column(JSON)  # For complex params like weights
    reason = Column(String)  # "quarterly_recalibration", "manual_adjustment", etc.
    changed_by = Column(String)  # "auto" or user identifier
    
    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceSnapshot(Base):
    """Daily/weekly/monthly performance summaries"""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(DateTime, default=datetime.utcnow, index=True)
    period_type = Column(String)  # "daily", "weekly", "monthly", "quarterly"
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    # Aggregate stats
    total_bets = Column(Integer)
    total_wins = Column(Integer)
    total_losses = Column(Integer)
    win_rate = Column(Float)
    
    # Financial
    total_risked = Column(Float)
    total_profit_loss = Column(Float)
    roi = Column(Float)
    
    # CLV
    mean_clv = Column(Float)
    median_clv = Column(Float)
    
    # Calibration
    calibration_error = Column(Float)  # MAE between predicted prob and actual
    calibration_bins = Column(JSON)  # {bin: {predicted: X, actual: Y, count: N}}
    
    # Model performance
    mean_edge = Column(Float)
    bets_recommended = Column(Integer)
    pass_rate = Column(Float)
    
    # By system
    kenpom_mae = Column(Float)
    barttorvik_mae = Column(Float)
    evanmiya_mae = Column(Float)


class DataFetch(Base):
    """Track data fetches for monitoring scraper health"""

    __tablename__ = "data_fetches"

    id = Column(Integer, primary_key=True, index=True)
    fetch_time = Column(DateTime, default=datetime.utcnow, index=True)
    data_source = Column(String, nullable=False, index=True)  # "kenpom", "odds_api", etc.
    success = Column(Boolean, nullable=False)
    records_fetched = Column(Integer)
    error_message = Column(Text)
    response_time_ms = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)


class ClosingLine(Base):
    """Closing lines captured near game time for CLV calculation."""

    __tablename__ = "closing_lines"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    captured_at = Column(DateTime, default=datetime.utcnow, index=True)

    spread = Column(Float)        # Home team spread (negative = home favourite)
    spread_odds = Column(Integer)  # American odds for the home spread
    total = Column(Float)
    total_odds = Column(Integer)
    moneyline_home = Column(Integer)
    moneyline_away = Column(Integer)

    game = relationship("Game")


class TeamProfile(Base):
    """
    Per-team offensive and defensive four-factor stats persisted from BartTorvik.

    Columns mirror the TeamSimProfile / TeamPlayStyle dataclasses so that the
    Markov simulator and matchup engine can load real per-team defensive data
    from the database instead of D1-average defaults.
    """

    __tablename__ = "team_profiles"

    id = Column(Integer, primary_key=True, index=True)
    team_name = Column(String, nullable=False, index=True)
    season_year = Column(Integer, nullable=False, index=True)
    source = Column(String, nullable=False, default="barttorvik")  # "barttorvik" | "kenpom"

    # Efficiency margins (KenPom / BartTorvik AdjEM scale, ≈ -30 to +30)
    adj_oe = Column(Float)
    adj_de = Column(Float)
    adj_em = Column(Float)

    # Offensive four factors
    pace = Column(Float)          # Possessions per 40 min
    efg_pct = Column(Float)       # Effective FG% (offensive)
    to_pct = Column(Float)        # Turnover rate (offensive, lower is better)
    ft_rate = Column(Float)       # FT attempts / FGA (offensive)
    three_par = Column(Float)     # 3PA / FGA (offensive)

    # Defensive four factors — the data the Markov engine was previously blind to
    def_efg_pct = Column(Float)   # Opponent eFG% allowed
    def_to_pct = Column(Float)    # Opponent TO rate forced
    def_ft_rate = Column(Float)   # Opponent FT rate allowed
    def_three_par = Column(Float) # Opponent 3PA rate allowed

    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "team_name", "season_year", "source",
            name="_team_season_source_uc",
        ),
    )


class DBAlert(Base):
    """Persisted performance alerts surfaced in the dashboard."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)   # INFO | WARNING | CRITICAL
    message = Column(Text, nullable=False)
    threshold = Column(Float)
    current_value = Column(Float)
    acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_at = Column(DateTime)


class FantasyDraftSession(Base):
    """Tracks a single fantasy draft session state."""

    __tablename__ = "fantasy_draft_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_key = Column(String(50), unique=True, nullable=False, index=True)
    my_draft_position = Column(Integer, nullable=False)
    num_teams = Column(Integer, nullable=False, default=12)
    num_rounds = Column(Integer, nullable=False, default=23)
    current_pick = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    picks = relationship("FantasyDraftPick", back_populates="session",
                         cascade="all, delete-orphan")


class FantasyDraftPick(Base):
    """Records each pick made during a fantasy draft."""

    __tablename__ = "fantasy_draft_picks"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("fantasy_draft_sessions.id"),
                        nullable=False, index=True)
    pick_number = Column(Integer, nullable=False)
    round_number = Column(Integer, nullable=False)
    drafter_position = Column(Integer, nullable=False)  # 1-12
    is_my_pick = Column(Boolean, nullable=False, default=False)
    player_id = Column(String(100), nullable=False)
    player_name = Column(String(100), nullable=False)
    player_team = Column(String(10))
    player_positions = Column(JSON)  # list of position strings
    player_tier = Column(Integer)
    player_adp = Column(Float)
    player_z_score = Column(Float)
    picked_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("FantasyDraftSession", back_populates="picks")

    __table_args__ = (
        UniqueConstraint("session_id", "player_id", name="_session_player_uc"),
    )


class FantasyLineup(Base):
    """Saved daily lineup for fantasy baseball."""

    __tablename__ = "fantasy_lineups"

    id = Column(Integer, primary_key=True, index=True)
    lineup_date = Column(Date, nullable=False, index=True)
    platform = Column(String(30), nullable=False, default="yahoo")
    positions = Column(JSON, nullable=False)   # {"C": "player_id", "1B": "player_id", ...}
    projected_points = Column(Float)
    actual_points = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("lineup_date", "platform", name="_lineup_date_platform_uc"),
    )


class PlayerDailyMetric(Base):
    """
    Sparse time-series of per-player analytics (EMAC-077 EPIC-1).
    One row per (player_id, metric_date, sport). NULL fields are not computed yet.
    """

    __tablename__ = "player_daily_metrics"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=False)
    metric_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)  # 'mlb' | 'cbb'

    # Core value metrics
    vorp_7d = Column(Float)
    vorp_30d = Column(Float)
    z_score_total = Column(Float)
    z_score_recent = Column(Float)

    # Statcast 2.0 (MLB only — always NULL for CBB rows)
    blast_pct = Column(Float)
    bat_speed = Column(Float)
    squared_up_pct = Column(Float)
    swing_length = Column(Float)
    stuff_plus = Column(Float)
    plv = Column(Float)

    # Flexible rolling windows: {"7d": {"avg": 0.310, ...}, "30d": {...}}
    rolling_window = Column(JSONB, nullable=False, default=dict)

    # Ensemble RoS blend columns (Phase 2.2 — ATC/BAT/Steamer/ZiPS weighted average)
    blend_hr = Column(Float)
    blend_rbi = Column(Float)
    blend_avg = Column(Float)
    blend_era = Column(Float)
    blend_whip = Column(Float)

    data_source = Column(String(50))
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_id", "metric_date", "sport",
                         name="_pdm_player_date_sport_uc"),
    )


class ProjectionSnapshot(Base):
    """
    Delta-compressed audit trail of projection changes (EMAC-077 EPIC-1).
    One row per (snapshot_date, sport). Only stores changed projections.
    """

    __tablename__ = "projection_snapshots"

    id = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)  # 'mlb' | 'cbb'

    # {player_id: {"old": {...}, "new": {...}, "delta_reason": "..."}}
    player_changes = Column(JSONB, nullable=False, default=dict)

    total_players = Column(Integer)
    significant_changes = Column(Integer)   # rows where |delta| > threshold
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ProjectionCacheEntry(Base):
    """Durable JSON cache for projection pipeline handoff data."""

    __tablename__ = "projection_cache_entries"

    id = Column(Integer, primary_key=True)
    cache_key = Column(String(100), nullable=False, unique=True, index=True)
    payload = Column(JSONB, nullable=False, default=dict)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class PlayerValuationCache(Base):
    """
    Pre-computed PlayerValuationReport per player per day (ARCH-001 Phase 2).
    Worker writes at 6 AM ET; API reads. Soft-delete via invalidated_at.
    """

    __tablename__ = "player_valuation_cache"

    id = Column(String(36), primary_key=True)  # UUID stored as string
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=False)
    target_date = Column(Date, nullable=False)
    league_key = Column(String(100), nullable=False)
    report = Column(JSONB, nullable=False)
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    invalidated_at = Column(DateTime, nullable=True)
    data_as_of = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "player_id", "target_date", "league_key",
            name="_pvc_player_date_league_uc"
        ),
    )


# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


if __name__ == "__main__":
    init_db()



# ═════════════════════════════════════════════════════════════════════════════
# FANTASY BASEBALL — LIVE DATA MODELS (Added March 26, 2026)
# ═════════════════════════════════════════════════════════════════════════════

class StatcastPerformance(Base):
    """
    Daily Statcast performance data from Baseball Savant.
    
    Stores granular hitting/pitching metrics for each player each day.
    Used for Bayesian projection updates and pattern detection.
    """
    
    __tablename__ = "statcast_performances"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Player identification
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=False)
    team = Column(String(10), nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    
    # Plate appearances and counting stats
    pa = Column(Integer, default=0)  # Plate appearances
    ab = Column(Integer, default=0)  # At-bats
    h = Column(Integer, default=0)   # Hits
    doubles = Column(Integer, default=0)
    triples = Column(Integer, default=0)
    hr = Column(Integer, default=0)  # Home runs
    r = Column(Integer, default=0)   # Runs
    rbi = Column(Integer, default=0) # RBIs
    bb = Column(Integer, default=0)  # Walks
    so = Column(Integer, default=0)  # Strikeouts
    hbp = Column(Integer, default=0) # Hit by pitch
    sb = Column(Integer, default=0)  # Stolen bases
    cs = Column(Integer, default=0)  # Caught stealing
    
    # Statcast quality metrics
    exit_velocity_avg = Column(Float, default=0.0)
    launch_angle_avg = Column(Float, default=0.0)
    hard_hit_pct = Column(Float, default=0.0)  # 95+ mph
    barrel_pct = Column(Float, default=0.0)    # Ideal combination
    
    # Expected stats
    xba = Column(Float, default=0.0)   # Expected batting average
    xslg = Column(Float, default=0.0)  # Expected slugging
    xwoba = Column(Float, default=0.0) # Expected weighted on-base average
    
    # Calculated traditional stats
    avg = Column(Float, default=0.0)   # Batting average
    obp = Column(Float, default=0.0)   # On-base percentage
    slg = Column(Float, default=0.0)   # Slugging percentage
    ops = Column(Float, default=0.0)   # On-base plus slugging
    woba = Column(Float, default=0.0)  # Weighted on-base average
    
    # Pitching stats (if applicable)
    ip = Column(Float, default=0.0)    # Innings pitched
    er = Column(Integer, default=0)    # Earned runs
    k_pit = Column(Integer, default=0) # Strikeouts (pitching)
    bb_pit = Column(Integer, default=0) # Walks (pitching)
    pitches = Column(Integer, default=0) # Total pitches thrown
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Unique constraint: one record per player per day
    __table_args__ = (
        UniqueConstraint('player_id', 'game_date', name='uq_player_date'),
    )


class PlayerProjection(Base):
    """
    Live-updated player projections using Bayesian inference.
    
    Combines prior (Steamer/ZiPS) with likelihood (recent performance)
    using shrinkage priors. Early season = trust prior more.
    Late season = trust recent data more.
    """
    
    __tablename__ = "player_projections"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Player identification
    player_id = Column(String(50), nullable=False, unique=True, index=True)
    player_name = Column(String(100), nullable=False)
    team = Column(String(10))
    positions = Column(JSON)  # List of eligible positions
    
    # Core projection stats (updated via Bayesian inference)
    woba = Column(Float, default=0.320)   # Weighted on-base average
    avg = Column(Float, default=0.250)    # Batting average
    obp = Column(Float, default=0.320)    # On-base percentage
    slg = Column(Float, default=0.400)    # Slugging percentage
    ops = Column(Float, default=0.720)    # On-base plus slugging
    xwoba = Column(Float, default=0.320)  # Expected wOBA
    
    # Counting stats (rate stats × projected PA)
    hr = Column(Integer, default=15)
    r = Column(Integer, default=65)
    rbi = Column(Integer, default=65)
    sb = Column(Integer, default=5)
    
    # Pitching stats
    era = Column(Float, default=4.00)
    whip = Column(Float, default=1.30)
    k_per_nine = Column(Float, default=8.5)
    bb_per_nine = Column(Float, default=3.0)
    
    # Bayesian metadata
    shrinkage = Column(Float, default=1.0)  # 1.0 = trust prior fully
    data_quality_score = Column(Float, default=0.0)  # 0-1 based on sample size
    sample_size = Column(Integer, default=0)  # PA in recent sample
    prior_source = Column(String(50), default='steamer')  # steamer, zips, thebat
    update_method = Column(String(50), default='prior')  # prior, bayesian, manual
    
    # Category scores (for H2H leagues)
    cat_scores = Column(JSONB, default=dict)  # Dict of category -> z-score
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PatternDetectionAlert(Base):
    """
    MLB-specific pattern detection alerts from OpenClaw.
    
    Detects: Pitcher fatigue, bullpen overuse, platoon splits,
    travel fatigue, weather impacts, etc.
    """
    
    __tablename__ = "pattern_detection_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Alert classification
    pattern_type = Column(String(50), nullable=False, index=True)
    # pitcher_fatigue, bullpen_overuse, platoon_split, travel_fatigue,
    # weather_impact, lineup_rest, etc.
    
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    confidence = Column(Float, default=0.5)  # 0.0 to 1.0
    
    # Affected entities
    player_id = Column(String(50), nullable=True, index=True)
    player_name = Column(String(100))
    team = Column(String(10), nullable=True, index=True)
    game_date = Column(Date, nullable=False, index=True)
    
    # Alert details
    title = Column(String(200), nullable=False)
    description = Column(Text)
    betting_implication = Column(Text)  # How to exploit this edge
    
    # Detection metadata
    detection_data = Column(JSONB, default=dict)  # Raw data that triggered alert
    data_sources = Column(JSON, default=list)  # List of sources
    
    # Status
    is_active = Column(Boolean, default=True)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    alerted_at = Column(DateTime, nullable=True)  # When Discord alert sent


class DataIngestionLog(Base):
    """
    Audit log for all data ingestion operations.
    
    Tracks: Statcast pulls, projection updates, pattern detection runs.
    Used for monitoring, debugging, and performance analysis.
    """
    
    __tablename__ = "data_ingestion_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Job classification
    job_type = Column(String(50), nullable=False, index=True)
    # statcast_daily, bayesian_update, pattern_detection, etc.
    
    target_date = Column(Date, nullable=False, index=True)
    
    # Status
    status = Column(String(20), nullable=False)  # SUCCESS, PARTIAL, FAILED
    
    # Metrics
    records_processed = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    processing_time_seconds = Column(Float)
    
    # Quality metrics
    validation_errors = Column(Integer, default=0)
    validation_warnings = Column(Integer, default=0)
    data_quality_score = Column(Float)  # 0-1 overall quality
    
    # Details
    error_details = Column(JSONB, default=list)  # List of error dicts
    warning_details = Column(JSONB, default=list)  # List of warning dicts
    summary_stats = Column(JSONB, default=dict)  # Job-specific stats
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)


class UserPreferences(Base):
    """
    User-customizable settings for the fantasy baseball dashboard.
    
    Stores notification preferences, dashboard layout configuration,
    and projection blending weights.
    """
    
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User identification (Yahoo user ID or internal user ID)
    user_id = Column(String(100), nullable=False, unique=True, index=True)
    user_email = Column(String(255), nullable=True)
    
    # Notification settings
    notifications = Column(JSONB, nullable=False, default=lambda: {
        "lineup_deadline": True,
        "injury_alerts": True,
        "waiver_suggestions": True,
        "trade_offers": False,
        "hot_streak_alerts": True,
        "channels": ["discord"],  # Options: email, discord, push
        "discord_user_id": None,
        "email_enabled": False,
    })
    
    # Dashboard layout configuration
    dashboard_layout = Column(JSONB, nullable=False, default=lambda: {
        "panels": [
            {"id": "lineup_gaps", "position": "top-left", "size": "medium", "enabled": True},
            {"id": "hot_cold_streaks", "position": "top-right", "size": "medium", "enabled": True},
            {"id": "waiver_targets", "position": "middle-left", "size": "medium", "enabled": True},
            {"id": "injury_flags", "position": "middle-right", "size": "small", "enabled": True},
            {"id": "matchup_preview", "position": "bottom-left", "size": "medium", "enabled": True},
            {"id": "probable_pitchers", "position": "bottom-right", "size": "small", "enabled": True},
        ],
        "refresh_interval_seconds": 300,  # 5 minutes
        "theme": "dark",  # dark, light, system
    })
    
    # Projection blending weights (must sum to 1.0)
    projection_weights = Column(JSONB, nullable=False, default=lambda: {
        "steamer": 0.30,
        "zips": 0.25,
        "depth_charts": 0.20,
        "atc": 0.15,
        "the_bat": 0.10,
    })
    
    # Streak calculation preferences
    streak_settings = Column(JSONB, nullable=False, default=lambda: {
        "hot_threshold": 0.5,  # z-score threshold for "hot"
        "cold_threshold": -0.5,  # z-score threshold for "cold"
        "min_sample_days": 7,  # Minimum days for streak calculation
        "rolling_windows": [7, 14, 30],  # Days to calculate trends
    })
    
    # Waiver wire preferences
    waiver_preferences = Column(JSONB, nullable=False, default=lambda: {
        "min_percent_owned": 0,  # Show players with >X% ownership
        "max_percent_owned": 100,  # Hide players owned by >X% (default: show all free agents)
        "positions_of_need": [],  # Auto-detect if empty
        "priority_categories": [],  # Auto-detect if empty
        "hide_injured": True,
        "streamer_threshold": 0.3,  # z-score threshold for streamer suggestions
    })

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# MLB Phase 2 — Game Log Ingestion (P7)
# Three-table schema: dimension (mlb_team) -> fact (mlb_game_log) -> snapshot (mlb_odds_snapshot)
# All tables use natural-key upserts (idempotent), dual-write raw_payload JSONB.
# ---------------------------------------------------------------------------

class MLBTeam(Base):
    """
    MLB team dimension table. Seeded from MLBTeam objects embedded in every
    BDL game response. Upserted on team_id before mlb_game_log writes.

    Contract source: backend/data_contracts/mlb_team.py
    All fields observed non-null across 19-game live sample (2026-04-05).
    """

    __tablename__ = "mlb_team"

    team_id      = Column(Integer, primary_key=True)          # BDL MLBTeam.id
    abbreviation = Column(String(10), nullable=False)         # "LAA", "NYY"
    name         = Column(String(100), nullable=False)        # "Angels"
    display_name = Column(String(150), nullable=False)        # "Los Angeles Angels"
    short_name   = Column(String(50), nullable=False)         # "Angels"
    location     = Column(String(100), nullable=False)        # "Los Angeles"
    slug         = Column(String(50), nullable=False)         # "los-angeles-angels"
    league       = Column(String(10), nullable=False)         # "National" | "American"
    division     = Column(String(10), nullable=False)         # "East" | "Central" | "West"
    ingested_at  = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Relationships
    home_games = relationship("MLBGameLog", foreign_keys="MLBGameLog.home_team_id", back_populates="home_team_obj")
    away_games = relationship("MLBGameLog", foreign_keys="MLBGameLog.away_team_id", back_populates="away_team_obj")


class MLBGameLog(Base):
    """
    MLB game fact table. One row per game (BDL game_id is stable and unique).
    Upserted on game_id -- status/scores updated as game progresses through
    STATUS_SCHEDULED -> STATUS_IN_PROGRESS -> STATUS_FINAL.

    Scores come from MLBTeamGameData.runs (not a flat score field).
    game_date stored as ET date (converted from MLBGame.date UTC ISO 8601 timestamp).

    Contract source: backend/data_contracts/mlb_game.py
    """

    __tablename__ = "mlb_game_log"

    game_id      = Column(Integer, primary_key=True)          # BDL MLBGame.id
    game_date    = Column(Date, nullable=False, index=True)   # ET date
    season       = Column(Integer, nullable=False)            # e.g. 2026
    season_type  = Column(String(20), nullable=False)         # "regular" | "postseason" | "preseason"
    status       = Column(String(30), nullable=False, index=True)  # STATUS_FINAL etc.
    home_team_id = Column(Integer, ForeignKey("mlb_team.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("mlb_team.team_id"), nullable=False)
    home_runs    = Column(Integer)                            # NULL pre-game; MLBTeamGameData.runs
    away_runs    = Column(Integer)
    home_hits    = Column(Integer)                            # MLBTeamGameData.hits
    away_hits    = Column(Integer)
    home_errors  = Column(Integer)                            # MLBTeamGameData.errors
    away_errors  = Column(Integer)
    venue        = Column(String(200))
    attendance   = Column(Integer)                            # NULL pre-game; 0 in API pre-game
    period       = Column(Integer)                            # Current/final inning
    raw_payload  = Column(JSONB, nullable=False)              # Full BDL MLBGame dict (dual-write)
    ingested_at  = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at   = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow,
                          onupdate=datetime.utcnow)

    # Relationships
    home_team_obj  = relationship("MLBTeam", foreign_keys=[home_team_id], back_populates="home_games")
    away_team_obj  = relationship("MLBTeam", foreign_keys=[away_team_id], back_populates="away_games")
    odds_snapshots = relationship("MLBOddsSnapshot", back_populates="game")

    __table_args__ = (
        Index("idx_mlb_game_log_season_date", "season", "game_date"),
    )


class MLBOddsSnapshot(Base):
    """
    MLB odds line-movement history. One row per (game_id, vendor, snapshot_window).
    snapshot_window is the poll timestamp rounded to the 30-minute bucket -- matches
    the _poll_mlb_odds job cadence and makes upserts idempotent.

    spread/total values stored as VARCHAR strings to match the BDL API contract
    (MLBBettingOdd.spread_home_value etc. -- the API sends strings, not floats).

    Contract source: backend/data_contracts/mlb_odds.py
    """

    __tablename__ = "mlb_odds_snapshot"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    odds_id          = Column(Integer, nullable=False)         # BDL MLBBettingOdd.id
    game_id          = Column(Integer, ForeignKey("mlb_game_log.game_id"), nullable=False, index=True)
    vendor           = Column(String(50), nullable=False)      # "draftkings", "fanduel", etc.
    snapshot_window  = Column(DateTime(timezone=True), nullable=False)  # rounded to 30-min bucket
    spread_home      = Column(String(10), nullable=False)      # str per contract e.g. "1.5"
    spread_away      = Column(String(10), nullable=False)      # str per contract e.g. "-1.5"
    spread_home_odds = Column(Integer, nullable=False)
    spread_away_odds = Column(Integer, nullable=False)
    ml_home_odds     = Column(Integer, nullable=False)
    ml_away_odds     = Column(Integer, nullable=False)
    total            = Column(String(10), nullable=False)      # str per contract e.g. "8.5"
    total_over_odds  = Column(Integer, nullable=False)
    total_under_odds = Column(Integer, nullable=False)
    raw_payload      = Column(JSONB, nullable=False)           # Full MLBBettingOdd dict (dual-write)

    # Relationship
    game = relationship("MLBGameLog", back_populates="odds_snapshots")

    __table_args__ = (
        UniqueConstraint("game_id", "vendor", "snapshot_window", name="_mlb_odds_game_vendor_window_uc"),
        Index("idx_mlb_odds_vendor_window", "vendor", "snapshot_window"),
    )


class MLBPlayerStats(Base):
    """
    MLB per-player per-game box stats (P11 -- Phase 2 stats ingestion).

    Natural key: (bdl_player_id, game_id) -- unique constraint enforces idempotent upserts.
    Dual-write: raw_payload stores the full BDL API dict alongside normalized columns.

    Column name mapping (API -> DB):
      r          -> runs          (avoids Python keyword clash)
      h          -> hits          (avoids ambiguity with home_hits in game_log)
      double     -> doubles       (avoids Python builtin keyword)
      hr         -> home_runs
      bb         -> walks
      so         -> strikeouts_bat
      sb         -> stolen_bases
      cs         -> caught_stealing
      h_allowed  -> hits_allowed
      r_allowed  -> runs_allowed
      er         -> earned_runs
      bb_allowed -> walks_allowed
      k          -> strikeouts_pit
      ip         -> innings_pitched  (string, e.g. "6.2")

    Contract source: backend/data_contracts/mlb_player_stats.py
    """

    __tablename__ = "mlb_player_stats"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    bdl_stat_id     = Column(Integer, nullable=True)           # BDL stats record id
    bdl_player_id   = Column(Integer, nullable=False)          # player.id from BDL
    game_id         = Column(Integer, ForeignKey("mlb_game_log.game_id"), nullable=True)
    game_date       = Column(Date, nullable=False)
    season          = Column(Integer, nullable=False, default=2026)

    # Batting stats (null for pure pitchers)
    ab              = Column(Integer, nullable=True)
    runs            = Column(Integer, nullable=True)           # 'r' from API
    hits            = Column(Integer, nullable=True)           # 'h' from API
    doubles         = Column(Integer, nullable=True)           # 'double' from API
    triples         = Column(Integer, nullable=True)
    home_runs       = Column(Integer, nullable=True)           # 'hr' from API
    rbi             = Column(Integer, nullable=True)
    walks           = Column(Integer, nullable=True)           # 'bb' from API
    strikeouts_bat  = Column(Integer, nullable=True)           # 'so' from API
    stolen_bases    = Column(Integer, nullable=True)           # 'sb' from API
    caught_stealing = Column(Integer, nullable=True)           # 'cs' from API
    avg             = Column(Float, nullable=True)
    obp             = Column(Float, nullable=True)
    slg             = Column(Float, nullable=True)
    ops             = Column(Float, nullable=True)

    # Pitching stats (null for pure hitters)
    innings_pitched = Column(String(10), nullable=True)        # 'ip' e.g. "6.2"
    hits_allowed    = Column(Integer, nullable=True)           # 'h_allowed' from API
    runs_allowed    = Column(Integer, nullable=True)           # 'r_allowed' from API
    earned_runs     = Column(Integer, nullable=True)           # 'er' from API
    walks_allowed   = Column(Integer, nullable=True)           # 'bb_allowed' from API
    strikeouts_pit  = Column(Integer, nullable=True)           # 'k' from API
    whip            = Column(Float, nullable=True)
    era             = Column(Float, nullable=True)

    # Audit columns
    raw_payload     = Column(JSON, nullable=False)             # Full BDL dict (dual-write)
    ingested_at     = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("bdl_player_id", "game_id", name="_mps_player_game_uc"),
        Index("idx_mps_player_date", "bdl_player_id", "game_date"),
        Index("idx_mps_game", "game_id"),
        Index("idx_mps_date", "game_date"),
    )


class PlayerIDMapping(Base):
    """
    Cross-system player identity mapping table (P10 — Phase 2 identity resolution).

    Maps Yahoo player keys, BDL player IDs, and mlbam IDs to a single canonical
    row per player. Seeded via pybaseball.playerid_lookup() + manual overrides.

    Resolution priority:
      1. Cache hit on yahoo_key or bdl_id -> return mlbam_id immediately
      2. pybaseball name lookup -> cache result (source='pybaseball')
      3. Manual override (source='manual') takes precedence in conflicts

    Key design decisions (from K-B spec):
      - yahoo_key is "469.p.7590" format — game_id.p.yahoo_id
      - yahoo_id "7590" is proprietary — NOT mlbam_id
      - bdl_id is BDL internal integer — NOT mlbam_id
      - mlbam_id is the canonical cross-platform identifier
      - normalized_name enables fuzzy matching across systems (Unicode-normalized)
    """

    __tablename__ = "player_id_mapping"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    yahoo_key            = Column(String(50), nullable=True)    # "469.p.7590" — unique per player
    yahoo_id             = Column(String(20), nullable=True)    # "7590" — proprietary Yahoo ID
    mlbam_id             = Column(Integer, nullable=True)       # MLB Advanced Media canonical ID
    bdl_id               = Column(Integer, nullable=True)       # BDL player.id internal
    full_name            = Column(String(150), nullable=False)
    normalized_name      = Column(String(150), nullable=False)  # lowercase, no accents
    source               = Column(String(20), nullable=False, default="manual")  # pybaseball|manual|api
    resolution_confidence = Column(Float, nullable=True)        # 0.0-1.0 for fuzzy matches
    created_at           = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(ZoneInfo("America/New_York"))
    )
    updated_at           = Column(
        DateTime(timezone=True), nullable=True,
        default=lambda: datetime.now(ZoneInfo("America/New_York")),
        onupdate=lambda: datetime.now(ZoneInfo("America/New_York"))
    )
    last_verified        = Column(Date, nullable=True)

    __table_args__ = (
        # Partial unique indexes — each external ID is unique where present
        UniqueConstraint("yahoo_key", name="_pim_yahoo_key_uc"),
        UniqueConstraint("bdl_id", name="_pim_bdl_id_uc"),
        Index("idx_pim_mlbam",       "mlbam_id"),
        Index("idx_pim_bdl",         "bdl_id"),
        Index("idx_pim_normalized",  "normalized_name"),
        Index("idx_pim_yahoo_id",    "yahoo_id"),
    )


class PlayerRollingStats(Base):
    """
    Decay-weighted rolling window metrics per player per date per window size (P13).

    Computed daily by _compute_rolling_windows() (lock 100_018, 3 AM ET).
    Exponential decay: weight = 0.95 ** days_back.
    Window sizes: 7, 14, 30 days.

    Batting fields are NULL for pure pitchers (no at-bats in window).
    Pitching fields are NULL for pure hitters (no innings pitched in window).
    Two-way players (e.g. Ohtani) have both batting and pitching fields populated.

    w_ip stores decimal innings (e.g. 6.667 for "6.2" BDL notation), not the raw string.
    Derived rates (w_avg, w_era, etc.) are recomputed from weighted sums -- not stored
    from the per-game rate stats which have no decay weighting.

    Natural key: (bdl_player_id, as_of_date, window_days).
    """

    __tablename__ = "player_rolling_stats"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id   = Column(Integer, nullable=False)
    as_of_date      = Column(Date, nullable=False)
    window_days     = Column(Integer, nullable=False)    # 7, 14, or 30
    games_in_window = Column(Integer, nullable=False)
    w_games         = Column(Float, nullable=True)       # M3: sum of decay weights

    # Batting weighted sums
    w_ab            = Column(Float, nullable=True)
    w_hits          = Column(Float, nullable=True)
    w_doubles       = Column(Float, nullable=True)
    w_triples       = Column(Float, nullable=True)
    w_home_runs     = Column(Float, nullable=True)
    w_rbi           = Column(Float, nullable=True)
    w_walks         = Column(Float, nullable=True)
    w_strikeouts_bat = Column(Float, nullable=True)
    w_stolen_bases  = Column(Float, nullable=True)
    w_caught_stealing  = Column(Float, nullable=True)  # P27 NSB support
    w_net_stolen_bases = Column(Float, nullable=True)  # P27 w_stolen_bases - w_caught_stealing

    # Batting derived rates (computed from weighted sums)
    w_avg           = Column(Float, nullable=True)   # w_hits / w_ab
    w_obp           = Column(Float, nullable=True)   # (w_hits + w_walks) / (w_ab + w_walks)
    w_slg           = Column(Float, nullable=True)   # weighted TB / w_ab
    w_ops           = Column(Float, nullable=True)   # w_obp + w_slg

    # Pitching weighted sums
    w_ip            = Column(Float, nullable=True)   # decimal IP (not "6.2" string)
    w_earned_runs   = Column(Float, nullable=True)
    w_hits_allowed  = Column(Float, nullable=True)
    w_walks_allowed = Column(Float, nullable=True)
    w_strikeouts_pit = Column(Float, nullable=True)

    # Pitching derived rates
    w_era           = Column(Float, nullable=True)   # 9 * w_earned_runs / w_ip
    w_whip          = Column(Float, nullable=True)   # (w_hits_allowed + w_walks_allowed) / w_ip
    w_k_per_9       = Column(Float, nullable=True)   # 9 * w_strikeouts_pit / w_ip

    # Statcast advanced metrics (P28 Phase 1)
    w_exit_velocity_avg = Column(Float, nullable=True)  # Avg exit velocity (mph)
    w_launch_angle_avg  = Column(Float, nullable=True)  # Avg launch angle (degrees)
    w_hard_hit_pct      = Column(Float, nullable=True)  # % of batted balls >= 95 mph
    w_barrel_pct        = Column(Float, nullable=True)  # % ideal EV + LA combinations
    w_xwoba             = Column(Float, nullable=True)  # Expected wOBA
    w_xba               = Column(Float, nullable=True)  # Expected batting average
    w_xslg              = Column(Float, nullable=True)  # Expected slugging
    w_xwoba_minus_woba  = Column(Float, nullable=True)  # Luck differential (xwOBA - wOBA)

    computed_at     = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "bdl_player_id", "as_of_date", "window_days",
            name="_prs_player_date_window_uc",
        ),
        Index("idx_prs_player_date", "bdl_player_id", "as_of_date"),
        Index("idx_prs_date_window", "as_of_date", "window_days"),
    )


class PlayerScore(Base):
    """
    P14 League Z-scores + composite score per player per date per window size.

    Computed daily by _compute_player_scores() (lock 100_019, 4 AM ET).
    Input: player_rolling_stats. Output: league Z-scores + 0-100 percentile rank.

    Z-score methodology:
      - Population std (ddof=0) across all players with non-null value.
      - MIN_SAMPLE = 5 players required before computing Z for a category.
      - Lower-is-better categories (ERA, WHIP): Z is negated so higher Z = better.
      - Z capped at +/-3.0 to reduce outlier distortion.
      - composite_z = mean of all applicable non-None per-category Z-scores.
      - score_0_100 = percentile rank (0-100) within player_type cohort.

    Hitter categories: z_hr, z_rbi, z_nsb (composite), z_sb (legacy), z_avg, z_obp.
    Pitcher categories: z_era, z_whip, z_k_per_9.
    Two-way players: all categories (Ohtani-style).

    NSB (Net Stolen Bases = SB - CS) is the canonical H2H One Win basestealing
    category. z_nsb replaces z_sb in the composite; z_sb is retained for
    backward compatibility with explainability narratives but excluded from
    composite_z to avoid double-counting (SB and NSB correlate >0.95).
    """

    __tablename__ = "player_scores"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id   = Column(Integer, nullable=False)
    as_of_date      = Column(Date, nullable=False)
    window_days     = Column(Integer, nullable=False)     # 7, 14, or 30
    player_type     = Column(String(10), nullable=False)  # "hitter" | "pitcher" | "two_way"
    games_in_window = Column(Integer, nullable=False)

    # Per-category Z-scores (NULL if not applicable or < MIN_SAMPLE)
    z_hr        = Column(Float, nullable=True)
    z_rbi       = Column(Float, nullable=True)
    z_sb        = Column(Float, nullable=True)   # legacy -- still computed, excluded from composite
    z_nsb       = Column(Float, nullable=True)   # P27 Net SB (SB - CS) -- drives composite
    z_avg       = Column(Float, nullable=True)
    z_obp       = Column(Float, nullable=True)
    z_era       = Column(Float, nullable=True)
    z_whip      = Column(Float, nullable=True)
    z_k_per_9   = Column(Float, nullable=True)

    composite_z = Column(Float, nullable=False, default=0.0)
    score_0_100 = Column(Float, nullable=False, default=50.0)
    confidence  = Column(Float, nullable=False, default=0.0)

    computed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "bdl_player_id", "as_of_date", "window_days",
            name="_ps_player_date_window_uc",
        ),
        Index("idx_ps_date_window", "as_of_date", "window_days"),
        Index("idx_ps_player_date", "bdl_player_id", "as_of_date"),
        Index("idx_ps_score", "as_of_date", "window_days", "score_0_100"),
    )


class PlayerMomentum(Base):
    """
    P15 Momentum layer -- delta-Z signals derived from 14d vs 30d player_scores.

    Computed daily by _compute_player_momentum() (lock 100_020, 5 AM ET).
    Input: player_scores (P14). Output: SURGING / HOT / STABLE / COLD / COLLAPSING.

    Signal thresholds:
      delta_z >  0.5  -> SURGING
      delta_z >= 0.2  -> HOT
      delta_z >  -0.2 -> STABLE
      delta_z >= -0.5 -> COLD
      else            -> COLLAPSING
    """

    __tablename__ = "player_momentum"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id   = Column(Integer, nullable=False)
    as_of_date      = Column(Date, nullable=False)
    player_type     = Column(String(10), nullable=False)
    delta_z         = Column(Float, nullable=False)
    signal          = Column(String(12), nullable=False)
    composite_z_14d = Column(Float, nullable=False)
    composite_z_30d = Column(Float, nullable=False)
    score_14d       = Column(Float, nullable=False)
    score_30d       = Column(Float, nullable=False)
    confidence_14d  = Column(Float, nullable=False)
    confidence_30d  = Column(Float, nullable=False)
    confidence      = Column(Float, nullable=False)
    computed_at     = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "bdl_player_id", "as_of_date",
            name="_pm_player_date_uc",
        ),
        Index("idx_pm_date_signal", "as_of_date", "signal"),
        Index("idx_pm_player_date", "bdl_player_id", "as_of_date"),
    )


class SimulationResult(Base):
    """
    P16 Rest-of-Season Monte Carlo simulation results per player per date.

    Computed daily by _run_ros_simulation() (lock 100_021, 6 AM ET).
    Input: player_rolling_stats (14d window -- current form baseline).
    Algorithm: N=1000 simulations, CV=0.35 game-to-game variation,
               remaining_games=130 (mid-April 2026 default).

    Hitter percentiles: proj_hr, proj_rbi, proj_sb, proj_avg (P10/25/50/75/90).
    Pitcher percentiles: proj_k, proj_era, proj_whip (P10/25/50/75/90).
    Two-way players: all fields populated (Ohtani-style).

    Risk metrics require league_means/league_stds from player_scores.
    If player_scores unavailable for the date, risk fields are NULL.

    Natural key: (bdl_player_id, as_of_date) -- one simulation snapshot per player per day.
    Downstream: P17 lineup/waiver decision engines.
    """

    __tablename__ = "simulation_results"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id   = Column(Integer, nullable=False)
    as_of_date      = Column(Date, nullable=False)
    window_days     = Column(Integer, nullable=False, default=14)
    remaining_games = Column(Integer, nullable=False)
    n_simulations   = Column(Integer, nullable=False)
    player_type     = Column(String(10), nullable=False)

    # Hitter projection percentiles (NULL for pure pitchers)
    proj_hr_p10  = Column(Float, nullable=True)
    proj_hr_p25  = Column(Float, nullable=True)
    proj_hr_p50  = Column(Float, nullable=True)
    proj_hr_p75  = Column(Float, nullable=True)
    proj_hr_p90  = Column(Float, nullable=True)

    proj_rbi_p10 = Column(Float, nullable=True)
    proj_rbi_p25 = Column(Float, nullable=True)
    proj_rbi_p50 = Column(Float, nullable=True)
    proj_rbi_p75 = Column(Float, nullable=True)
    proj_rbi_p90 = Column(Float, nullable=True)

    proj_sb_p10  = Column(Float, nullable=True)
    proj_sb_p25  = Column(Float, nullable=True)
    proj_sb_p50  = Column(Float, nullable=True)
    proj_sb_p75  = Column(Float, nullable=True)
    proj_sb_p90  = Column(Float, nullable=True)

    proj_avg_p10 = Column(Float, nullable=True)
    proj_avg_p25 = Column(Float, nullable=True)
    proj_avg_p50 = Column(Float, nullable=True)
    proj_avg_p75 = Column(Float, nullable=True)
    proj_avg_p90 = Column(Float, nullable=True)

    # Pitcher projection percentiles (NULL for pure hitters)
    proj_k_p10   = Column(Float, nullable=True)
    proj_k_p25   = Column(Float, nullable=True)
    proj_k_p50   = Column(Float, nullable=True)
    proj_k_p75   = Column(Float, nullable=True)
    proj_k_p90   = Column(Float, nullable=True)

    proj_era_p10  = Column(Float, nullable=True)
    proj_era_p25  = Column(Float, nullable=True)
    proj_era_p50  = Column(Float, nullable=True)
    proj_era_p75  = Column(Float, nullable=True)
    proj_era_p90  = Column(Float, nullable=True)

    proj_whip_p10 = Column(Float, nullable=True)
    proj_whip_p25 = Column(Float, nullable=True)
    proj_whip_p50 = Column(Float, nullable=True)
    proj_whip_p75 = Column(Float, nullable=True)
    proj_whip_p90 = Column(Float, nullable=True)

    # Risk metrics (NULL when league_means/stds unavailable for the date)
    composite_variance  = Column(Float, nullable=True)
    downside_p25        = Column(Float, nullable=True)
    upside_p75          = Column(Float, nullable=True)
    prob_above_median   = Column(Float, nullable=True)

    computed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "bdl_player_id", "as_of_date",
            name="_sr_player_date_uc",
        ),
        Index("idx_sr_date", "as_of_date"),
        Index("idx_sr_player_date", "bdl_player_id", "as_of_date"),
    )


class DecisionResult(Base):
    """
    P17 Decision Engine results -- lineup and waiver optimization outputs.

    Computed daily by _run_decision_optimization() (lock 100_022, 7 AM ET).
    Input: player_scores (P14) + player_momentum (P15) + simulation_results (P16).
    Decision types: "lineup" (slot assignment) | "waiver" (add/drop recommendation).

    Natural key: (as_of_date, decision_type, bdl_player_id).
    """

    __tablename__ = "decision_results"

    id             = Column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date     = Column(Date, nullable=False)
    decision_type  = Column(String(10), nullable=False)   # "lineup" | "waiver"
    bdl_player_id  = Column(Integer, nullable=False)
    target_slot    = Column(String(10), nullable=True)    # e.g. "OF", "SP"
    drop_player_id = Column(Integer, nullable=True)       # waiver drop target
    lineup_score   = Column(Float, nullable=True)
    value_gain     = Column(Float, nullable=True)
    confidence     = Column(Float, nullable=False)
    reasoning      = Column(String(500), nullable=True)
    computed_at    = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "as_of_date", "decision_type", "bdl_player_id",
            name="_dr_date_type_player_uc",
        ),
        Index("idx_dr_date_type",    "as_of_date", "decision_type"),
        Index("idx_dr_player_date",  "bdl_player_id", "as_of_date"),
    )


class BacktestResult(Base):
    """
    P18 Backtesting Harness results -- per-player forecast accuracy metrics.

    Computed daily by _run_backtesting() (lock 100_023, 8 AM ET).
    Input: simulation_results (P16 projections) vs mlb_player_stats (actuals).
    Compares proj_p50 against actual stats over a rolling 14-day window.

    Natural key: (bdl_player_id, as_of_date).
    Downstream: P19 Explainability Layer.
    """

    __tablename__ = "backtest_results"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id   = Column(Integer, nullable=False)
    as_of_date      = Column(Date, nullable=False)
    player_type     = Column(String(10), nullable=False)
    games_played    = Column(Integer, nullable=False)

    # Per-stat MAE (None when projection or actual unavailable)
    mae_hr          = Column(Float, nullable=True)
    rmse_hr         = Column(Float, nullable=True)
    mae_rbi         = Column(Float, nullable=True)
    rmse_rbi        = Column(Float, nullable=True)
    mae_sb          = Column(Float, nullable=True)
    rmse_sb         = Column(Float, nullable=True)
    mae_avg         = Column(Float, nullable=True)
    rmse_avg        = Column(Float, nullable=True)
    mae_k           = Column(Float, nullable=True)
    rmse_k          = Column(Float, nullable=True)
    mae_era         = Column(Float, nullable=True)
    rmse_era        = Column(Float, nullable=True)
    mae_whip        = Column(Float, nullable=True)
    rmse_whip       = Column(Float, nullable=True)

    composite_mae     = Column(Float, nullable=True)
    direction_correct = Column(Boolean, nullable=True)

    computed_at     = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("bdl_player_id", "as_of_date",
                         name="_br_player_date_uc"),
        Index("idx_br_date", "as_of_date"),
        Index("idx_br_player_date", "bdl_player_id", "as_of_date"),
    )


class DecisionExplanation(Base):
    """
    P19 Explainability Layer -- human-readable decision traces.

    Computed daily by _run_explainability() (lock 100_024, 9 AM ET).
    Input: decision_results (P17) + player_scores (P14) + player_momentum (P15)
           + simulation_results (P16) + backtest_results (P18).
    One row per decision_results row (1:1 relationship).

    Natural key: (decision_id,) -- one explanation per decision.
    Downstream: P20 Integration (UI display, API endpoint /admin/explanations/{id}).
    """

    __tablename__ = "decision_explanations"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    decision_id     = Column(BigInteger, nullable=False, unique=True)  # FK to decision_results.id
    bdl_player_id   = Column(Integer, nullable=False)
    as_of_date      = Column(Date, nullable=False)
    decision_type   = Column(String(10), nullable=False)

    summary         = Column(String(500), nullable=False)
    # factors stored as JSON array: [{name, value, label, weight, narrative}, ...]
    factors_json    = Column(JSON, nullable=False)
    confidence_narrative  = Column(String(200), nullable=True)
    risk_narrative        = Column(String(200), nullable=True)
    track_record_narrative = Column(String(200), nullable=True)

    computed_at     = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_de_date", "as_of_date"),
        Index("idx_de_player_date", "bdl_player_id", "as_of_date"),
        Index("idx_de_decision_id", "decision_id"),
    )


class DailySnapshot(Base):
    """
    P20 Daily Snapshot -- end-of-pipeline state capture.

    Computed daily by _run_snapshot() (lock 100_025, 10 AM ET).
    Runs after all 9 prior phases complete. One row per day.
    Captures counts, health status, top players, regression flag.

    Natural key: (as_of_date,) -- one snapshot per calendar day.
    Downstream: GET /admin/snapshot/latest and GET /admin/snapshot/{date} API endpoints.
    """

    __tablename__ = "daily_snapshots"

    id                    = Column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date            = Column(Date, nullable=False, unique=True)

    n_players_scored      = Column(Integer, nullable=False, default=0)
    n_momentum_records    = Column(Integer, nullable=False, default=0)
    n_simulation_records  = Column(Integer, nullable=False, default=0)
    n_decisions           = Column(Integer, nullable=False, default=0)
    n_explanations        = Column(Integer, nullable=False, default=0)
    n_backtest_records    = Column(Integer, nullable=False, default=0)

    mean_composite_mae    = Column(Float, nullable=True)
    regression_detected   = Column(Boolean, nullable=False, default=False)

    top_lineup_player_ids = Column(JSON, nullable=True)   # list of up to 5 bdl_player_ids
    top_waiver_player_ids = Column(JSON, nullable=True)   # list of up to 3 bdl_player_ids
    pipeline_jobs_run     = Column(JSON, nullable=True)   # list of job name strings

    pipeline_health       = Column(String(10), nullable=False, default="UNKNOWN")  # HEALTHY/DEGRADED/FAILED
    health_reasons        = Column(JSON, nullable=True)   # list of reason strings
    summary               = Column(String(500), nullable=True)

    computed_at           = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_ds_date", "as_of_date"),
    )


# ═════════════════════════════════════════════════════════════════════════════════════════════
# H2H ONE WIN UI DATA LAYER (Phase 1 — S27, Apr 8, 2026)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Note: PositionEligibility table requires separate ingestion from Yahoo Fantasy API.
# The 'eligible_positions' field in Yahoo responses includes LF/CF/RF but needs
# explicit parsing and persistence for scarcity calculations.
# ════════════════════════════════════════════════════════════════════════════════════════════════════


class PositionEligibility(Base):
    """
    Multi-position eligibility for Yahoo Fantasy H2H One Win format.

    ONE ROW PER PLAYER with boolean flags for ALL eligible positions.
    Keyed on yahoo_player_key (unique per player in Yahoo ecosystem).
    bdl_player_id is populated later via PlayerIDMapping.

    CF scarcity is significantly higher than LF/RF in H2H One Win —
    this table enables scarcity index calculations with OF sub-position granularity.

    Seeded from Yahoo Fantasy API 'eligible_positions' field via backfill_positions.py.
    Updated daily by _sync_position_eligibility() in daily_ingestion.py.
    """

    __tablename__ = "position_eligibility"

    id = Column(Integer, primary_key=True, index=True)
    yahoo_player_key = Column(String(50), nullable=False, index=True)  # e.g. "469.p.12345"
    bdl_player_id = Column(Integer, nullable=True, index=True)  # FK to MLBPlayerStats.bdl_player_id (populated via mapping)

    # Player identity
    player_name = Column(String(100))
    first_name = Column(String(50))
    last_name = Column(String(50))

    # Position-specific flags — ALL positions in ONE row
    can_play_c = Column(Boolean, nullable=False, default=False)
    can_play_1b = Column(Boolean, nullable=False, default=False)
    can_play_2b = Column(Boolean, nullable=False, default=False)
    can_play_3b = Column(Boolean, nullable=False, default=False)
    can_play_ss = Column(Boolean, nullable=False, default=False)
    can_play_lf = Column(Boolean, nullable=False, default=False)
    can_play_cf = Column(Boolean, nullable=False, default=False)
    can_play_rf = Column(Boolean, nullable=False, default=False)
    can_play_of = Column(Boolean, nullable=False, default=False)  # True if LF/CF/RF/OF
    can_play_dh = Column(Boolean, nullable=False, default=False)
    can_play_util = Column(Boolean, nullable=False, default=False)
    can_play_sp = Column(Boolean, nullable=False, default=False)
    can_play_rp = Column(Boolean, nullable=False, default=False)

    # Primary position for categorization
    primary_position = Column(String(10))  # "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "DH", "SP", "RP"
    player_type = Column(String(10), nullable=False, default="batter")  # "batter" | "pitcher" | "two_way"

    # Scarcity metrics (computed daily)
    scarcity_rank = Column(Integer)  # 1-100 within position group (1 = most scarce)
    league_rostered_pct = Column(Float)  # % of 10-team leagues rostering this player
    multi_eligibility_count = Column(Integer, nullable=False, default=0)  # Count of positions eligible

    # Audit
    fetched_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("yahoo_player_key", name="_pe_yahoo_uc"),
        Index("idx_pe_primary_position", "primary_position"),
        Index("idx_pe_bdl_player_id", "bdl_player_id"),
    )


class ProbablePitcherSnapshot(Base):
    """
    Daily probable pitchers from MLB Stats API.

    The DailyLineupOptimizer already fetches probable pitchers via MLB Stats API
    but does not persist them. This table enables historical tracking and frontend
    consumption for Two-Start Command Center UI.

    Data source: MLB Stats API /api/v1/schedule/games endpoint (probablePitchers field).
    Refresh cadence: Job 100_014 (6 AM ET daily) + game-day updates at 12 PM ET.

    is_confirmed flag:
      - True = Team officially announced starter (lineup card released)
      - False = Probable per MLB.com (subject to change)
    """

    __tablename__ = "probable_pitchers"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    game_date = Column(Date, nullable=False, index=True)
    team = Column(String(10), nullable=False)  # Team abbreviation (e.g., "NYY", "LAA")
    opponent = Column(String(10), nullable=True)  # Opponent for matchup context
    is_home = Column(Boolean, nullable=True)  # Home/away flag

    pitcher_name = Column(String(100), nullable=True)  # Full name
    bdl_player_id = Column(Integer, nullable=True, index=True)  # FK to player_id_mapping
    mlbam_id = Column(Integer, nullable=True)  # MLBAM ID for cross-reference
    is_confirmed = Column(Boolean, nullable=False, default=False)

    game_time_et = Column(String(10), nullable=True)  # "7:05 PM" format
    park_factor = Column(Float, nullable=True)  # Park factor for matchup quality
    quality_score = Column(Float, nullable=True)  # Precomputed matchup rating (-2.0 to +2.0)

    # Audit
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_date", "team", name="_pp_date_team_uc"),
        Index("idx_pp_date", "game_date"),
        Index("idx_pp_pitcher", "bdl_player_id"),
    )


class WeatherForecast(Base):
    """
    Canonical weather forecast for MLB games.

    Persists weather data for historical tracking and context enrichment.
    Source: OpenWeatherMap API via WeatherFetcher.
    """
    __tablename__ = "weather_forecasts"

    id = Column(Integer, primary_key=True)
    game_date = Column(Date, nullable=False, index=True)
    park_name = Column(String(100), nullable=False)
    forecast_date = Column(Date, nullable=False, default=datetime.utcnow)

    temperature_high = Column(Float)  # Celsius
    temperature_low = Column(Float)
    humidity = Column(Integer)  # Percentage
    wind_speed = Column(Float)  # km/h
    wind_direction = Column(String(10))  # N, NE, E, SE, S, SW, W, NW
    precipitation_probability = Column(Integer)  # Percentage
    conditions = Column(String(100))  # Rain, Cloudy, Sunny, etc.

    fetched_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game_date", "park_name", "forecast_date", name="_wf_game_park_date_uc"),
        Index("idx_weather_game_date", "game_date"),
    )


class ParkFactor(Base):
    """
    Canonical park factors for MLB stadiums.

    Park factors adjust player projections based on stadium characteristics.
    Values > 1.0 favor hitters, < 1.0 favor pitchers.
    """
    __tablename__ = "park_factors"

    id = Column(Integer, primary_key=True)
    park_name = Column(String(100), nullable=False, unique=True)

    # Factors: 1.0 = neutral, > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly
    hr_factor = Column(Float, nullable=False, default=1.0)
    run_factor = Column(Float, nullable=False, default=1.0)
    hits_factor = Column(Float, nullable=False, default=1.0)
    era_factor = Column(Float, nullable=False, default=1.0)
    whip_factor = Column(Float, nullable=False, default=1.0)

    data_source = Column(String(50))  # fangraphs, baseball-reference, etc.
    season = Column(Integer)

    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class DeploymentVersion(Base):
    """
    Deployment fingerprint for /admin/version endpoint.

    Stores the git commit SHA and build timestamp for deployment verification.
    """
    __tablename__ = "deployment_version"

    id = Column(Integer, primary_key=True)
    git_commit_sha = Column(String(100), nullable=False, unique=True)
    git_commit_date = Column(String(50))
    build_timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    app_version = Column(String(50), default="dev")
    deployed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

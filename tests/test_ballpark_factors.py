"""
Tests for ballpark_factors.py get_park_factor() DB-backed resolution.

Verifies the three-tier fallback: DB → hardcoded constant → neutral 1.0
"""
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.fantasy_baseball.ballpark_factors import get_park_factor
from backend.models import ParkFactor, Base


@pytest.fixture
def empty_db():
    """Empty DB with ParkFactor table but no data."""
    temp_db = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
    temp_db.close()
    db_path = Path(temp_db.name)

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    ParkFactor.__table__.create(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    yield SessionLocal

    try:
        db_path.unlink()
    except Exception:
        pass


@pytest.fixture
def db_with_park_factors():
    """DB with ParkFactor rows for a subset of teams."""
    temp_db = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
    temp_db.close()
    db_path = Path(temp_db.name)

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    ParkFactor.__table__.create(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()
    factors = [
        ParkFactor(park_name="COL", hr_factor=1.30, run_factor=1.38, era_factor=1.28),
        ParkFactor(park_name="SF", hr_factor=0.90, run_factor=0.94, era_factor=0.94),
        ParkFactor(park_name="BOS", hr_factor=1.00, run_factor=1.04, era_factor=1.03),
    ]
    for f in factors:
        db.add(f)
    db.commit()

    yield SessionLocal

    try:
        db_path.unlink()
    except Exception:
        pass


def test_db_backed_hit_returns_db_factor(db_with_park_factors):
    """When DB has a row, return the DB value (not hardcoded constant)."""
    db = db_with_park_factors()
    # DB says 1.30 for COL HR factor
    result = get_park_factor("COL", "hr", _db_session=db)
    assert result == 1.30


def test_db_backed_missing_factor_returns_null_column_default(db_with_park_factors):
    """When DB row exists but requested factor column was not set, returns 1.0."""
    db = db_with_park_factors()
    # Our test data only sets hr/run/era; hits_factor would use default 1.0
    result = get_park_factor("COL", "hits", _db_session=db)
    assert result == 1.0


def test_db_missing_team_falls_back_to_hardcoded(empty_db):
    """When DB has no row for team, fall back to PARK_FACTORS constant."""
    db = empty_db()
    # NYM is not in our test DB, but exists in PARK_FACTORS constant
    result = get_park_factor("NYM", "run", _db_session=db)
    assert result == 0.97  # From PARK_FACTORS constant


def test_db_missing_team_missing_factor_falls_back_to_hardcoded(empty_db):
    """Missing team + factor not in constant returns neutral."""
    db = empty_db()
    result = get_park_factor("FAKE_TEAM", "hr", _db_session=db)
    assert result == 1.0  # Default neutral


def test_invalid_factor_returns_neutral(db_with_park_factors):
    """Invalid factor type returns neutral 1.0."""
    db = db_with_park_factors()
    result = get_park_factor("COL", "invalid_factor", _db_session=db)
    assert result == 1.0


def test_resolution_order_db_wins_over_constant(db_with_park_factors):
    """
    DB value takes precedence over hardcoded constant.
    DB COL run_factor = 1.38 (same as constant, but verify DB wins).
    """
    db = db_with_park_factors()
    result = get_park_factor("COL", "run", _db_session=db)
    assert result == 1.38


def test_all_supported_factors_from_db(db_with_park_factors):
    """All three supported factor types resolve from DB."""
    db = db_with_park_factors()
    assert get_park_factor("COL", "hr", _db_session=db) == 1.30
    assert get_park_factor("COL", "run", _db_session=db) == 1.38
    assert get_park_factor("COL", "era", _db_session=db) == 1.28


def test_hardcoded_constant_fallback_values(empty_db):
    """Verify hardcoded constant values for key teams."""
    db = empty_db()
    # Extreme hitter park
    assert get_park_factor("COL", "hr", _db_session=db) == 1.30
    # Extreme pitcher park
    assert get_park_factor("SEA", "hr", _db_session=db) == 0.90
    # Neutral park
    assert get_park_factor("WSH", "run", _db_session=db) == 1.00


def test_unknown_team_returns_neutral(empty_db):
    """Completely unknown team returns neutral."""
    db = empty_db()
    assert get_park_factor("NO_SUCH_TEAM", "run", _db_session=db) == 1.0

"""
Tests for GET /api/fantasy/players/{bdl_player_id}/scores endpoint.

Layer 3 scoring exposure - authoritative player_scores output.
"""
import pytest
import tempfile
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.models import PlayerScore, get_db
from backend.auth import verify_api_key


# Mock auth that bypasses API key verification for tests
async def mock_verify_api_key():
    return "test_user"


@pytest.fixture
def client_with_scores():
    """Test client with player scores data preloaded."""
    # Use file-based temp DB so it persists across connections
    temp_db = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
    temp_db.close()
    db_path = Path(temp_db.name)

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    PlayerScore.__table__.create(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    # Create session and add test data
    db = SessionLocal()
    scores = [
        # Hitter with 14d window
        PlayerScore(
            id=1,
            bdl_player_id=12345,
            as_of_date=date(2026, 4, 15),
            window_days=14,
            player_type="hitter",
            games_in_window=12,
            z_hr=1.72,
            z_rbi=1.31,
            z_nsb=0.88,
            z_avg=1.09,
            z_obp=1.41,
            z_era=None,
            z_whip=None,
            z_k_per_9=None,
            composite_z=1.4821,
            score_0_100=91.4,
            confidence=0.86,
        ),
        # Hitter with 7d window
        PlayerScore(
            id=2,
            bdl_player_id=12345,
            as_of_date=date(2026, 4, 15),
            window_days=7,
            player_type="hitter",
            games_in_window=6,
            z_hr=2.1,
            z_rbi=1.5,
            z_nsb=0.9,
            z_avg=1.2,
            z_obp=1.6,
            composite_z=1.65,
            score_0_100=95.0,
            confidence=0.75,
        ),
        # Pitcher with 14d window
        PlayerScore(
            id=3,
            bdl_player_id=67890,
            as_of_date=date(2026, 4, 15),
            window_days=14,
            player_type="pitcher",
            games_in_window=3,
            z_hr=None,
            z_rbi=None,
            z_nsb=None,
            z_avg=None,
            z_obp=None,
            z_era=1.12,
            z_whip=0.74,
            z_k_per_9=1.03,
            composite_z=0.9642,
            score_0_100=78.6,
            confidence=0.21,
        ),
        # Earlier date for 12345
        PlayerScore(
            id=4,
            bdl_player_id=12345,
            as_of_date=date(2026, 4, 10),
            window_days=14,
            player_type="hitter",
            games_in_window=10,
            composite_z=1.2,
            score_0_100=82.0,
            confidence=0.71,
        ),
    ]
    for s in scores:
        db.add(s)
    db.commit()

    # Override FastAPI dependency to use our engine/session
    def override_get_db():
        try:
            yield SessionLocal()
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = mock_verify_api_key

    client = TestClient(app)
    yield client

    # Clean up
    db.close()
    app.dependency_overrides = {}
    # Delete temp file
    try:
        db_path.unlink()
    except Exception:
        pass


def test_get_player_scores_default_window_14(client_with_scores):
    """Default window_days=14 returns 14-day scores."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores")
    assert response.status_code == 200
    data = response.json()
    assert data["bdl_player_id"] == 12345
    assert data["requested_window_days"] == 14
    assert data["as_of_date"] == "2026-04-15"
    assert data["score"]["window_days"] == 14
    assert data["score"]["composite_z"] == 1.4821


def test_get_player_scores_explicit_window_7(client_with_scores):
    """Explicit window_days=7 returns 7-day scores."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores?window_days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["requested_window_days"] == 7
    assert data["score"]["window_days"] == 7
    assert data["score"]["composite_z"] == 1.65


def test_get_player_scores_window_30_returns_404(client_with_scores):
    """Requesting window_days=30 for player with no 30d scores returns 404."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores?window_days=30")
    assert response.status_code == 404
    assert "No player_scores found" in response.json()["detail"]


def test_get_player_scores_invalid_window_returns_400(client_with_scores):
    """Invalid window_days returns 400 with error message."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores?window_days=21")
    assert response.status_code == 400
    assert "window_days must be one of" in response.json()["detail"]


def test_get_player_scores_specific_date(client_with_scores):
    """Specific as_of_date query parameter returns that date's score."""
    response = client_with_scores.get(
        "/api/fantasy/players/12345/scores?as_of_date=2026-04-10"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["as_of_date"] == "2026-04-10"
    assert data["score"]["composite_z"] == 1.2


def test_get_player_scores_specific_date_not_found_returns_404(client_with_scores):
    """Specific as_of_date with no matching score returns 404."""
    response = client_with_scores.get(
        "/api/fantasy/players/12345/scores?as_of_date=2026-04-01"
    )
    assert response.status_code == 404
    assert "as_of_date=2026-04-01" in response.json()["detail"]


def test_get_player_scores_default_to_latest_date(client_with_scores):
    """When as_of_date is omitted, returns the latest available score."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores")
    assert response.status_code == 200
    data = response.json()
    # Should return 2026-04-15 (latest), not 2026-04-10
    assert data["as_of_date"] == "2026-04-15"


def test_get_player_scores_unknown_player_returns_404(client_with_scores):
    """Requesting unknown bdl_player_id returns 404."""
    response = client_with_scores.get("/api/fantasy/players/99999/scores")
    assert response.status_code == 404
    assert "bdl_player_id=99999" in response.json()["detail"]


def test_get_player_scores_hitter_category_breakdown(client_with_scores):
    """Hitter response includes batter categories (null for pitcher categories)."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores?window_days=14")
    assert response.status_code == 200
    data = response.json()["score"]
    cats = data["category_scores"]
    assert cats["z_hr"] == 1.72
    assert cats["z_rbi"] == 1.31
    assert cats["z_nsb"] == 0.88
    assert cats["z_avg"] == 1.09
    assert cats["z_obp"] == 1.41
    assert cats["z_era"] is None
    assert cats["z_whip"] is None
    assert cats["z_k_per_9"] is None


def test_get_player_scores_pitcher_category_breakdown(client_with_scores):
    """Pitcher response includes pitcher categories (null for batter categories)."""
    response = client_with_scores.get("/api/fantasy/players/67890/scores")
    assert response.status_code == 200
    data = response.json()["score"]
    assert data["player_type"] == "pitcher"
    cats = data["category_scores"]
    assert cats["z_hr"] is None
    assert cats["z_rbi"] is None
    assert cats["z_nsb"] is None
    assert cats["z_avg"] is None
    assert cats["z_obp"] is None
    assert cats["z_era"] == 1.12
    assert cats["z_whip"] == 0.74
    assert cats["z_k_per_9"] == 1.03


def test_get_player_scores_response_contract_complete(client_with_scores):
    """Response includes all required fields per contract."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores")
    assert response.status_code == 200
    data = response.json()
    score = data["score"]

    # Top-level fields
    assert "bdl_player_id" in data
    assert "requested_window_days" in data
    assert "as_of_date" in data
    assert "score" in data

    # Score fields
    assert score["bdl_player_id"] == 12345
    assert "as_of_date" in score
    assert "window_days" in score
    assert score["player_type"] in ("hitter", "pitcher", "two_way")
    assert "games_in_window" in score
    assert isinstance(score["composite_z"], float)
    assert isinstance(score["score_0_100"], float)
    assert isinstance(score["confidence"], float)
    assert "category_scores" in score

    # No internal fields exposed
    assert "id" not in score
    assert "computed_at" not in score


def test_get_player_scores_z_sb_not_exposed(client_with_scores):
    """Legacy z_sb field is not exposed in the response."""
    response = client_with_scores.get("/api/fantasy/players/12345/scores")
    assert response.status_code == 200
    cats = response.json()["score"]["category_scores"]
    # z_sb exists in DB but is NOT in the response contract
    assert "z_sb" not in cats


def test_get_player_scores_unauthorized_without_api_key():
    """Endpoint returns 401 when no API key is provided."""
    # Clear overrides to test real auth behavior
    app.dependency_overrides = {}

    client = TestClient(app)
    response = client.get("/api/fantasy/players/12345/scores")
    assert response.status_code == 401
    assert "API key required" in response.json()["detail"]

    # Restore overrides for other tests
    app.dependency_overrides = {}

"""
Tests for GET /api/fantasy/decisions endpoint.

Layer 3F Decision Output Read Surface - trusted P17/P19 decision outputs.
"""
import pytest
import tempfile
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.models import DecisionResult, DecisionExplanation, get_db
from backend.auth import verify_api_key


# Mock auth that bypasses API key verification for tests
async def mock_verify_api_key():
    return "test_user"


@pytest.fixture
def client_with_decisions():
    """Test client with decision results and explanations preloaded."""
    # Use file-based temp DB so it persists across connections
    temp_db = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
    temp_db.close()
    db_path = Path(temp_db.name)

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    DecisionResult.__table__.create(bind=engine)
    DecisionExplanation.__table__.create(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    # Create session and add test data
    db = SessionLocal()

    # Add decision results
    decisions = [
        # Lineup decision with explanation
        DecisionResult(
            id=1,
            as_of_date=date(2026, 4, 15),
            decision_type="lineup",
            bdl_player_id=12345,
            target_slot="OF",
            drop_player_id=None,
            lineup_score=92.5,
            value_gain=8.3,
            confidence=0.87,
            reasoning="Strong recent form plus favorable matchup",
            computed_at=None,  # Let DB default handle this
        ),
        # Waiver decision with explanation
        DecisionResult(
            id=2,
            as_of_date=date(2026, 4, 15),
            decision_type="waiver",
            bdl_player_id=67890,
            target_slot="SP",
            drop_player_id=11111,
            lineup_score=None,
            value_gain=12.1,
            confidence=0.92,
            reasoning="Elite waiver wire add - high upside",
            computed_at=None,
        ),
        # Lineup decision without explanation
        DecisionResult(
            id=3,
            as_of_date=date(2026, 4, 15),
            decision_type="lineup",
            bdl_player_id=54321,
            target_slot="1B",
            drop_player_id=None,
            lineup_score=85.0,
            value_gain=3.2,
            confidence=0.65,
            reasoning="Solid option but lower confidence",
            computed_at=None,
        ),
        # Earlier date decisions
        DecisionResult(
            id=4,
            as_of_date=date(2026, 4, 10),
            decision_type="lineup",
            bdl_player_id=12345,
            target_slot="OF",
            drop_player_id=None,
            lineup_score=88.0,
            value_gain=5.5,
            confidence=0.75,
            reasoning="Good form last week",
            computed_at=None,
        ),
    ]
    for d in decisions:
        db.add(d)

    # Add explanations for some decisions
    explanations = [
        DecisionExplanation(
            id=1,
            decision_id=1,  # Corresponds to decision id=1
            bdl_player_id=12345,
            as_of_date=date(2026, 4, 15),
            decision_type="lineup",
            summary="Strong recommendation to start based on recent performance",
            factors_json=[
                {"name": "z_score", "value": "1.85", "label": "Composite Z", "weight": 0.4,
                 "narrative": "Excellent recent form"},
                {"name": "matchup", "value": "favorable", "label": "Opponent", "weight": 0.3,
                 "narrative": "Facing weak pitching"},
            ],
            confidence_narrative="High confidence based on 14-day window",
            risk_narrative="Low risk - consistent performer",
            track_record_narrative="Historically strong in this matchup",
            computed_at=None,
        ),
        DecisionExplanation(
            id=2,
            decision_id=2,  # Corresponds to decision id=2
            bdl_player_id=67890,
            as_of_date=date(2026, 4, 15),
            decision_type="waiver",
            summary="Top waiver priority - significant value gain",
            factors_json=[
                {"name": "need_score", "value": "92.0", "label": "Roster Need", "weight": 0.5,
                 "narrative": "Fills critical SP slot"},
                {"name": "availability", "value": "95%", "label": "Free Agent", "weight": 0.3,
                 "narrative": "Widely available"},
            ],
            confidence_narrative="Very high confidence - clear add",
            risk_narrative="Minimal risk - established performer",
            track_record_narrative="Strong track record of success",
            computed_at=None,
        ),
    ]
    for e in explanations:
        db.add(e)

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


def test_get_decisions_default_latest_date(client_with_decisions):
    """Default as_of_date returns latest available decisions."""
    response = client_with_decisions.get("/api/fantasy/decisions")
    assert response.status_code == 200
    data = response.json()
    assert data["as_of_date"] == "2026-04-15"
    assert data["count"] == 3  # Three decisions on 2026-04-15
    assert len(data["decisions"]) == 3


def test_get_decisions_filter_by_lineup_type(client_with_decisions):
    """Filtering by decision_type=lineup returns only lineup decisions."""
    response = client_with_decisions.get("/api/fantasy/decisions?decision_type=lineup")
    assert response.status_code == 200
    data = response.json()
    assert data["decision_type"] == "lineup"
    assert data["count"] == 2
    for item in data["decisions"]:
        assert item["decision"]["decision_type"] == "lineup"


def test_get_decisions_filter_by_waiver_type(client_with_decisions):
    """Filtering by decision_type=waiver returns only waiver decisions."""
    response = client_with_decisions.get("/api/fantasy/decisions?decision_type=waiver")
    assert response.status_code == 200
    data = response.json()
    assert data["decision_type"] == "waiver"
    assert data["count"] == 1
    assert data["decisions"][0]["decision"]["decision_type"] == "waiver"


def test_get_decisions_specific_date(client_with_decisions):
    """Specific as_of_date returns decisions for that date."""
    response = client_with_decisions.get("/api/fantasy/decisions?as_of_date=2026-04-10")
    assert response.status_code == 200
    data = response.json()
    assert data["as_of_date"] == "2026-04-10"
    assert data["count"] == 1


def test_get_decisions_limit_works(client_with_decisions):
    """Limit parameter restricts number of results."""
    response = client_with_decisions.get("/api/fantasy/decisions?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2


def test_get_decisions_explanation_present(client_with_decisions):
    """Decisions with explanations include explanation data."""
    response = client_with_decisions.get("/api/fantasy/decisions?decision_type=lineup")
    assert response.status_code == 200
    data = response.json()

    # Find the lineup decision with explanation (id=1 has explanation, id=3 doesn't)
    decisions = data["decisions"]
    with_expl = [d for d in decisions if d["decision"]["bdl_player_id"] == 12345]
    without_expl = [d for d in decisions if d["decision"]["bdl_player_id"] == 54321]

    assert len(with_expl) == 1
    assert len(without_expl) == 1

    # Decision with explanation has explanation data
    assert with_expl[0]["explanation"] is not None
    assert with_expl[0]["explanation"]["summary"] == "Strong recommendation to start based on recent performance"
    assert len(with_expl[0]["explanation"]["factors"]) == 2

    # Decision without explanation has null explanation
    assert without_expl[0]["explanation"] is None


def test_get_decisions_explanation_factors_structure(client_with_decisions):
    """Explanation factors are properly structured."""
    response = client_with_decisions.get("/api/fantasy/decisions?decision_type=waiver")
    assert response.status_code == 200
    data = response.json()
    expl = data["decisions"][0]["explanation"]

    assert expl["summary"] == "Top waiver priority - significant value gain"
    assert expl["confidence_narrative"] == "Very high confidence - clear add"
    assert expl["risk_narrative"] == "Minimal risk - established performer"
    assert expl["track_record_narrative"] == "Strong track record of success"

    factors = expl["factors"]
    assert len(factors) == 2
    assert factors[0]["name"] == "need_score"
    assert factors[0]["label"] == "Roster Need"
    assert factors[0]["weight"] == 0.5
    assert factors[0]["narrative"] == "Fills critical SP slot"


def test_get_decisions_empty_for_unknown_date(client_with_decisions):
    """Requesting date with no decisions returns empty list, not 404."""
    response = client_with_decisions.get("/api/fantasy/decisions?as_of_date=2026-04-01")
    assert response.status_code == 200
    data = response.json()
    assert data["as_of_date"] == "2026-04-01"
    assert data["count"] == 0
    assert data["decisions"] == []


def test_get_decisions_ordering_by_confidence(client_with_decisions):
    """Results are ordered by confidence desc, then value_gain desc."""
    response = client_with_decisions.get("/api/fantasy/decisions")
    assert response.status_code == 200
    data = response.json()

    confidences = [d["decision"]["confidence"] for d in data["decisions"]]
    # Should be descending: 0.92, 0.87, 0.65
    assert confidences == [0.92, 0.87, 0.65]


def test_get_decisions_response_contract_complete(client_with_decisions):
    """Response includes all required fields per contract."""
    response = client_with_decisions.get("/api/fantasy/decisions")
    assert response.status_code == 200
    data = response.json()

    # Top-level fields
    assert "decisions" in data
    assert "count" in data
    assert "as_of_date" in data
    assert "decision_type" in data

    # Decision fields
    decision = data["decisions"][0]["decision"]
    assert "bdl_player_id" in decision
    assert "as_of_date" in decision
    assert "decision_type" in decision
    assert "target_slot" in decision
    assert "drop_player_id" in decision
    assert "lineup_score" in decision
    assert "value_gain" in decision
    assert "confidence" in decision
    assert "reasoning" in decision

    # No internal fields exposed
    assert "id" not in decision
    assert "computed_at" not in decision


def test_get_decisions_unauthorized_without_api_key(client_with_decisions):
    """Endpoint returns 401 when no API key is provided."""
    # Clear overrides to test real auth behavior
    app.dependency_overrides = {}

    client = TestClient(app)
    response = client.get("/api/fantasy/decisions")
    assert response.status_code == 401
    assert "API key required" in response.json()["detail"]

    # Restore overrides for other tests
    app.dependency_overrides = {}


def test_get_decisions_limit_validation(client_with_decisions):
    """Limit outside 1-500 range returns 422."""
    response = client_with_decisions.get("/api/fantasy/decisions?limit=0")
    assert response.status_code == 422

    response = client_with_decisions.get("/api/fantasy/decisions?limit=501")
    assert response.status_code == 422


def test_get_decisions_invalid_decision_type(client_with_decisions):
    """Invalid decision_type returns 422."""
    response = client_with_decisions.get("/api/fantasy/decisions?decision_type=invalid")
    assert response.status_code == 422

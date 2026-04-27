# FantasyBaseballEdge

Isolated MLB Fantasy Baseball engine with Statcast advanced metrics integration.

**Status:** Kimi-led engineering experiment — separate from CBB_Betting mainline.

---

## What This Is

A clean, focused MLB fantasy baseball intelligence engine that ingests Statcast data and uses it to find elite edges in waiver wire pickups, lineup optimization, and player valuation.

**Core principle:** Advanced batted-ball metrics (barrel%, xwOBA, exit velocity) predict future performance better than surface stats alone. This engine makes those signals actionable.

---

## Architecture

```
backend/
├── fantasy_baseball/          # Core baseball intelligence modules
│   ├── statcast_ingestion.py      # Baseball Savant data pipeline
│   ├── daily_lineup_optimizer.py  # Daily lineup + streaming recommendations
│   ├── momentum_engine.py         # Hot/cold/surging/collapsing classifications
│   ├── matchup_engine.py          # Head-to-head matchup analysis
│   ├── player_board.py            # Player valuation and projections
│   ├── simulation_engine.py       # Monte Carlo season projections
│   ├── ballpark_factors.py        # Park-adjusted scoring
│   └── ...
├── services/                  # Business logic layer
│   ├── scoring_engine.py          # Z-score computation (the heart)
│   ├── decision_engine.py         # Lineup + waiver optimization
│   ├── rolling_window_engine.py   # 7/14/30-day rolling stats
│   ├── derived_stats.py           # Feature engineering
│   └── daily_ingestion.py         # Orchestrated daily data pipeline
├── routers/                   # API endpoints
│   └── fantasy.py                 # /api/fantasy/* routes
├── core/                      # Pure math utilities
├── utils/                     # Shared helpers
├── models.py                  # SQLAlchemy ORM models
└── admin_*.py                 # Diagnostics endpoints

tests/                         # Test suite (only passing tests from CBB_Betting)
scripts/                       # Utility scripts
reports/                       # Analysis outputs
docs/                          # Documentation
```

---

## Key Differentiators

### Statcast Integration (Active Development)

Unlike traditional fantasy platforms that rely on counting stats (HR, RBI, AVG), this engine uses:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **xwOBA** | Expected weighted on-base average | Predicts future wOBA better than current wOBA |
| **Barrel %** | Ideal exit velocity + launch angle combinations | Best single predictor of power breakout |
| **Hard Hit %** | Batted balls ≥95 mph | Contact quality indicator |
| **Exit Velocity** | Average launch speed | Power + contact quality |
| **xwOBA - wOBA** | Luck differential | Buy-low / sell-high signal |

### Current Scoring Categories

**Hitters:**
- `z_hr` — Home runs
- `z_rbi` — Runs batted in
- `z_nsb` — Net stolen bases (SB - CS)
- `z_avg` — Batting average
- `z_obp` — On-base percentage

**Pitchers:**
- `z_era` — Earned run average
- `z_whip` — Walks + hits per inning
- `z_k_per_9` — Strikeouts per 9 innings

**Coming soon:** `z_power_quality` (barrel% + hard hit% + exit velocity composite)

---

## Development

### Setup

```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Running Tests

Core test suite (all passing):
```bash
pytest tests/test_scoring_engine.py -v        # 48 tests — Z-score computation
pytest tests/test_data_contracts.py -v        # 36 tests — Data validation
pytest tests/test_nsb_pipeline.py -v          # 15 tests — Net stolen bases
pytest tests/test_simulation_engine.py -v     # 28 tests — Monte Carlo
pytest tests/test_momentum_engine.py -v       # 18 tests — Momentum signals
pytest tests/test_matchup_engine.py -v        # 8 tests — Matchup analysis
pytest tests/test_db.py -v                    # 4 tests — Database layer
pytest tests/test_ballpark_factors.py -v      # 9 tests — Park factors
```

---

## Roadmap

### Phase 1: Statcast Rolling Windows
- Add `w_xwoba`, `w_barrel_pct`, `w_hard_hit_pct`, `w_exit_velocity_avg` to `player_rolling_stats`
- Pull from `statcast_performances` into rolling window computation

### Phase 2: Power Quality Scoring
- Create `z_power_quality` composite Z-score
- Overlay on existing scoring (not replacement)

### Phase 3: Decision Engine Integration
- "Statcast buy-low" waiver flag: `xwOBA > wOBA + 2σ`
- Barrel boost in daily lineup optimizer
- Pitcher batted-ball quality for ERA regression signals

---

## Origin

Extracted from [CBB_Betting](https://github.com/sfgray26/CBB_Betting) — a larger platform that included college basketball betting. This repo isolates the MLB fantasy baseball engine for focused development.

**Lead Engineer:** Kimi Claw
**Audit Report:** `reports/2026-04-27-statcast-usage-analysis.md`

---

*Built for finding the edge that surface stats miss.*

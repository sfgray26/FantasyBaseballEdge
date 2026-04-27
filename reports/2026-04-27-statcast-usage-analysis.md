# Statcast Advanced Statistics Usage Analysis
> **Date:** April 27, 2026  
> **Auditor:** Kimi Claw  
> **Scope:** How Statcast data drives (or doesn't drive) prediction models

---

## Executive Summary

### 🔴 CRITICAL FINDING: Statcast Data is Being Collected But NOT Used in Predictions

The platform ingests rich Statcast metrics daily (xwOBA, barrel%, hard hit%, exit velocity) but **these advanced statistics do NOT flow into the scoring engine, decision engine, or lineup optimizer**. They exist as **diagnostic/observational data** — valuable for manual analysis but completely absent from the automated prediction pipeline.

**This is your biggest untapped edge.**

---

## 1. What Statcast Data is Being Collected

### 1.1 Daily Ingestion Pipeline (`backend/fantasy_baseball/statcast_ingestion.py`)

The `StatcastIngestionAgent` fetches from Baseball Savant daily at 6:00 AM ET:

| Metric | Description | Stored In |
|--------|-------------|-----------|
| **xwOBA** | Expected weighted on-base average | `statcast_performances.xwoba` |
| **xBA** | Expected batting average | `statcast_performances.xba` |
| **xSLG** | Expected slugging | `statcast_performances.xslg` |
| **Barrel %** | Batted balls with ideal EV/LA combo | `statcast_performances.barrel_pct` |
| **Hard Hit %** | Batted balls ≥95 mph | `statcast_performances.hard_hit_pct` |
| **Exit Velocity** | Average launch speed | `statcast_performances.exit_velocity_avg` |
| **Launch Angle** | Average launch angle | `statcast_performances.launch_angle_avg` |
| **wOBA** | Actual weighted on-base average | `statcast_performances.woba` |

### 1.2 Data Quality Validation

The pipeline includes a `DataQualityChecker` that validates:
- Minimum rows (200+ expected)
- Date range correctness
- Column completeness
- Null rate thresholds
- Value range sanity (xwOBA 0.000–0.700)
- Pre-aggregation from per-pitch to per-game rows

### 1.3 Storage Volume

Per production data (April 16, 2026):
- `statcast_performances`: **7,408 rows**
- Zero-quality-metric rate: **~5%** (down from 42% after April 14 fix)
- Covers: exit velocity, xwOBA, hard hit%, barrel% for most players

---

## 2. How Statcast Data Flows (The Pipeline)

```
Baseball Savant API
       ↓
StatcastIngestionAgent.fetch_statcast_day()
       ↓
Pre-aggregation (per-pitch → per-game)
       ↓
DataQualityChecker.validate_daily_pull()
       ↓
Transform → PlayerDailyPerformance objects
       ↓
Upsert to statcast_performances table
       ↓
BayesianProjectionUpdater (OPTIONAL — see §3)
       ↓
...STOPS HERE...
```

### 2.1 What Happens Next

After `statcast_performances` is populated, the **daily ingestion pipeline** continues with:
- `mlb_player_stats` (BDL API — traditional counting stats)
- `player_rolling_stats` (7/14/30-day windows from BDL data)
- `player_scores` (Z-score computation from rolling stats)

**The Statcast table is NEVER JOINED into this flow.**

---

## 3. Where Statcast is NOT Used (The Gaps)

### 3.1 Scoring Engine (`backend/services/scoring_engine.py`)

The scoring engine computes Z-scores for:
- Hitters: HR, RBI, Net SB, AVG, OBP
- Pitchers: ERA, WHIP, K/9

**Statcast metrics referenced:** **ZERO**

The scoring engine only reads from `PlayerRollingStats` which sources from `mlb_player_stats` (BDL). No xwOBA, no barrel%, no exit velocity enters the composite Z-score.

### 3.2 Decision Engine (`backend/services/decision_engine.py`)

The decision engine uses:
- `score_0_100` (from scoring engine)
- `composite_z` (from scoring engine)
- `momentum_signal` (SURGING/HOT/STABLE/COLD/COLLAPSING)
- `proj_*_p50` (from simulation engine)

**Statcast metrics referenced:** **ZERO**

Lineup optimization and waiver recommendations are based entirely on rolling-window Z-scores and momentum signals derived from traditional stats.

### 3.3 Daily Lineup Optimizer (`backend/fantasy_baseball/daily_lineup_optimizer.py`)

The daily lineup optimizer uses:
- Sportsbook implied run totals
- Park factors (hardcoded)
- Injury status
- Two-start pitcher detection

**Statcast metrics referenced:** **ZERO**

No barrel rate, no exit velocity, no xwOBA enters the daily batter rankings or pitcher streaming decisions.

### 3.4 Simulation Engine (`backend/fantasy_baseball/simulation_engine.py`)

The simulation engine runs Monte Carlo projections using:
- Rolling window rate stats
- League means/stds
- Remaining games estimates

**Statcast metrics referenced:** **ZERO**

Simulations project HR, RBI, AVG, ERA, WHIP, K — but NOT xwOBA, barrel%, or any batted-ball quality metrics.

---

## 4. Where Statcast IS Used

### 4.1 Bayesian Projection Updater (Partial)

`BayesianProjectionUpdater` in `statcast_ingestion.py` **does** use Statcast data:
- Computes `sample_woba` from recent Statcast `woba` values
- Runs conjugate normal update against prior (Steamer/ZiPS)
- Stores posterior `woba`, `xwoba` in `PlayerProjection`

**BUT:** These updated projections are **not consumed** by the scoring or decision engines. The `PlayerProjection` table exists but the decision engine uses `simulation_engine` outputs instead.

### 4.2 Admin Diagnostics (Read-Only)

`admin_statcast_diagnostics.py` provides endpoints:
- `/admin/diagnose-statcast/summary`
- `/admin/diagnose-statcast/leaderboard`
- `/admin/diagnose-statcast/player`
- `/admin/diagnose-statcast/sanity-check`

These are **observational only** — useful for human analysts but not wired into automated decisions.

### 4.3 Data Reliability Engine

`data_reliability_engine.py` validates Statcast data freshness and range checks:
- Exit velocity >120 mph flagged as error
- Exit velocity <50 mph with PA>10 flagged as warning
- Barrel% >30% flagged as warning

This is **quality control**, not predictive input.

---

## 5. The Missed Edge: What Statcast Could Unlock

### 5.1 xwOBA vs wOBA Divergence

**The Signal:** Players with `xwOBA > wOBA` are hitting the ball hard but getting unlucky (defense, park, weather). These players are **buy-low opportunities**.

**Current State:** Not computed. The platform stores both values but never compares them.

**Implementation Path:**
```python
luck_factor = statcast_xwoba / actual_woba
if luck_factor > 1.15 and statcast_barrel_pct > 0.08:
    flag_as_buy_low()
```

### 5.2 Barrel Rate as Power Predictor

**The Signal:** Barrel% correlates more strongly with future HR production than current HR totals, especially in small samples (first 2 weeks of season).

**Current State:** Barrel% is stored but not used in `z_hr` computation.

**Implementation Path:**
- Add `w_barrel_pct` column to `PlayerRollingStats`
- Include in composite scoring for hitters (or as a separate power-quality Z)
- Weight heavily early season when counting stats are noisy

### 5.3 Exit Velocity as Contact Quality

**The Signal:** Sustained high exit velocity (>90th percentile) predicts positive AVG regression when current AVG is low.

**Current State:** Exit velocity stored but never compared to batting average.

### 5.4 Hard Hit% + Launch Angle Combo

**The Signal:** Hard hit% >40% + launch angle 10-25° = optimal GB/FB mix for extra-base hits.

**Current State:** Both metrics stored separately, never combined.

### 5.5 Pitcher Batted-Ball Profile

**The Signal:** Pitchers allowing high barrel% or hard hit% are due for ERA regression even if current ERA looks good.

**Current State:** Pitcher Statcast data is ingested (opponent quality metrics) but not used in ERA/WHIP scoring.

---

## 6. Recommended Integration Path

### Phase 1: Add Statcast Features to Rolling Windows (1–2 weeks)

Add computed columns to `player_rolling_stats`:
- `w_xwoba` — expected wOBA
- `w_barrel_pct` — barrel rate
- `w_hard_hit_pct` — hard hit rate
- `w_exit_velocity_avg` — exit velocity
- `w_xwoba_minus_woba` — luck differential

These become available for Z-score computation automatically.

### Phase 2: Create Statcast-Enhanced Scoring (2–3 weeks)

Add optional Statcast-augmented categories:
- `z_power_quality` = composite of barrel%, hard hit%, exit velocity
- `z_luck_factor` = xwOBA / wOBA ratio (normalized)
- Use as **overlay** on existing scoring, not replacement

### Phase 3: Waiver/Lineup Integration (3–4 weeks)

- `daily_lineup_optimizer.py`: rank batters by `implied_runs * park_factor * barrel_boost`
- `decision_engine.py`: add "Statcast buy-low" waiver flag when `xwOBA > wOBA + 2σ`
- Simulation engine: weight barrel% into HR projection variance

---

## 7. Bottom Line

| Question | Answer |
|----------|--------|
| Is Statcast data being collected? | ✅ Yes — 7,400+ rows, daily ingestion |
| Is it used in scoring? | ❌ No — scoring uses BDL counting stats only |
| Is it used in decisions? | ❌ No — decisions use Z-scores, not batted-ball quality |
| Is it used in lineup optimization? | ❌ No — optimizer uses odds/park factors only |
| Is it used in simulations? | ❌ No — projections use rate stats, not exit velocity |
| Is ANY advanced metric in the prediction path? | ⚠️ Only `z_nsb` (Net SB) — a counting stat derivative |

**The platform has built a Ferrari (Statcast ingestion) but is driving it like a Honda Civic (traditional stats for predictions).**

Your elite edge is sitting in `statcast_performances.xwoba`, `barrel_pct`, and `exit_velocity_avg` — **completely unused by the automated pipeline.**

---

## 8. Immediate Action Items

1. **Verify production Statcast freshness** — hit `/admin/diagnose-statcast/summary`
2. **Pick one metric to pilot** — recommend `xwOBA - wOBA` divergence as buy-low signal
3. **Add `w_xwoba` to rolling window computation** — requires `rolling_window_engine.py` update
4. **Create A/B test** — score_0_100 vs. score_0_100_with_statcast for waiver recommendations
5. **Monitor conversion** — track if Statcast-augmented picks outperform pure Z-score picks

---

*Analysis based on codebase inspection of statcast_ingestion.py, scoring_engine.py, decision_engine.py, daily_lineup_optimizer.py, simulation_engine.py, and admin diagnostics.*

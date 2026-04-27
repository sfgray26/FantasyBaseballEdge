"""
Orphaned Position Eligibility Linking Utility

Fuzzy name matching to link orphaned position_eligibility records
to player_id_mapping via difflib.SequenceMatcher.

Author: Task 21 Implementation
Date: 2026-04-10
"""

import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Optional, Tuple
import unicodedata
import re

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import PlayerIDMapping, PositionEligibility

logger = logging.getLogger(__name__)

# Configuration
SIMILARITY_THRESHOLD = 0.85  # 85% similarity required for fuzzy match


def normalize_name_for_matching(name: str) -> str:
    """
    Normalize player name for fuzzy matching.

    Steps:
    1. Convert to lowercase
    2. Remove accents/Unicode
    3. Remove extra whitespace
    4. Remove common suffixes (Jr., Sr., II, III, IV)
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower()

    # Remove accents
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])

    # Remove common suffixes
    name = re.sub(r'\s+jr\.?\s*$', '', name)
    name = re.sub(r'\s+sr\.?\s*$', '', name)
    name = re.sub(r'\s+ii\s*$', '', name)
    name = re.sub(r'\s+iii\s*$', '', name)
    name = re.sub(r'\s+iv\s*$', '', name)

    # Remove extra whitespace
    name = ' '.join(name.split())

    return name


def calculate_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names using SequenceMatcher.

    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    norm1 = normalize_name_for_matching(name1)
    norm2 = normalize_name_for_matching(name2)

    if not norm1 or not norm2:
        return 0.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def _build_last_name_index(candidates: List[PlayerIDMapping]) -> dict:
    """Build a dict mapping normalized last name -> list of candidates for fast filtering."""
    index: dict[str, list[PlayerIDMapping]] = {}
    for c in candidates:
        if not c.full_name:
            continue
        last = normalize_name_for_matching(c.full_name.split()[-1]) if c.full_name.strip() else ""
        if last:
            index.setdefault(last, []).append(c)
    return index


def find_best_match(
    orphan: PositionEligibility,
    mapping_candidates: List[PlayerIDMapping],
    verbose: bool = False,
    last_name_index: Optional[dict] = None,
) -> Tuple[Optional[PlayerIDMapping], float]:
    """
    Find best matching player_id_mapping record for an orphaned position_eligibility.

    Uses a two-pass strategy:
      1. Narrow search to candidates sharing the same last name (fast).
      2. Fall back to full scan only if narrow search finds nothing above threshold.

    Args:
        orphan: Orphaned PositionEligibility record
        mapping_candidates: List of PlayerIDMapping candidates (fallback pool)
        verbose: Enable verbose logging
        last_name_index: Pre-built {last_name: [candidates]} index for fast lookup

    Returns:
        Tuple of (best matching PlayerIDMapping or None, similarity score)
    """
    orphan_name = orphan.player_name or ""
    orphan_last = orphan.last_name or ""

    # Determine narrow candidate pool via last-name index
    narrow_pool = None
    if last_name_index and orphan_name:
        orphan_last_norm = normalize_name_for_matching(
            orphan_name.split()[-1] if orphan_name.strip() else ""
        )
        if orphan_last_norm:
            narrow_pool = last_name_index.get(orphan_last_norm)

    def _score_pool(pool: List[PlayerIDMapping]) -> Tuple[Optional[PlayerIDMapping], float]:
        best_match = None
        best_score = 0.0
        for candidate in pool:
            full_name_score = calculate_similarity(orphan_name, candidate.full_name)
            last_name_score = 0.0
            if orphan_last:
                candidate_last = candidate.full_name.split()[-1] if candidate.full_name else ""
                last_name_score = calculate_similarity(orphan_last, candidate_last)
            score = max(full_name_score, last_name_score)
            if score > best_score:
                best_score = score
                best_match = candidate
            if verbose and score > 0.7:
                logger.info(
                    f"  Similarity: {orphan_name} vs {candidate.full_name} = "
                    f"{score:.3f} (full: {full_name_score:.3f}, last: {last_name_score:.3f})"
                )
        return best_match, best_score

    # Pass 1: narrow pool (same last name)
    if narrow_pool:
        best_match, best_score = _score_pool(narrow_pool)
        if best_score >= SIMILARITY_THRESHOLD:
            if verbose:
                logger.info(
                    f"  MATCH FOUND (narrow): {orphan_name} -> {best_match.full_name} "
                    f"(score: {best_score:.3f})"
                )
            return best_match, best_score

    # Pass 2: full scan fallback
    best_match, best_score = _score_pool(mapping_candidates)

    if best_score >= SIMILARITY_THRESHOLD:
        if verbose:
            logger.info(
                f"  MATCH FOUND (full): {orphan_name} -> {best_match.full_name} "
                f"(score: {best_score:.3f})"
            )
        return best_match, best_score

    return None, best_score


def link_orphaned_records(
    db: Session,
    dry_run: bool = True,
    verbose: bool = False
) -> dict:
    """
    Link orphaned position_eligibility records to player_id_mapping.

    Args:
        db: Database session
        dry_run: If True, don't commit changes
        verbose: Enable verbose logging

    Returns:
        dict with status, linked_count, remaining_count, success_rate, elapsed_ms
    """
    t0 = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("Starting orphaned position_eligibility linking")
    logger.info("=" * 60)

    try:
        # Count orphaned records properly using LEFT JOIN
        total_elig = db.query(func.count(PositionEligibility.id)).scalar()

        # Count properly linked records (where bdl_player_id IS NOT NULL)
        properly_linked = db.query(func.count(PositionEligibility.id)).filter(
            PositionEligibility.bdl_player_id.isnot(None)
        ).scalar()

        # Orphans are records where bdl_player_id IS NULL
        orphan_count = total_elig - properly_linked

        logger.info(f"Total position_eligibility records: {total_elig}")
        logger.info(f"Properly linked (bdl_player_id IS NOT NULL): {properly_linked}")
        logger.info(f"Orphaned records: {orphan_count}")

        if orphan_count == 0:
            logger.info("No orphaned records found - exiting")
            return {
                "status": "success",
                "linked_count": 0,
                "remaining_count": 0,
                "success_rate": 100.0,
                "elapsed_ms": (datetime.now(timezone.utc) - t0).total_seconds() * 1000,
                "sample_unmatched": []
            }

        # Fetch all candidates from player_id_mapping
        logger.info("Fetching player_id_mapping candidates...")
        all_candidates = db.query(PlayerIDMapping).all()
        logger.info(f"Loaded {len(all_candidates)} candidates")

        # Build last-name index for fast narrow-pass matching
        ln_index = _build_last_name_index(all_candidates)
        logger.info(f"Built last-name index with {len(ln_index)} unique last names")

        # Fetch orphaned records (where bdl_player_id IS NULL)
        logger.info("Fetching orphaned position_eligibility records...")
        orphans = db.query(PositionEligibility).filter(
            PositionEligibility.bdl_player_id.is_(None)
        ).all()

        logger.info(f"Processing {len(orphans)} orphaned records...")

        # Link orphans to candidates
        linked_count = 0
        failed_count = 0
        sample_unmatched = []

        for i, orphan in enumerate(orphans, 1):
            if verbose:
                logger.info(f"\n[{i}/{len(orphans)}] Processing: {orphan.player_name}")

            best_match, score = find_best_match(orphan, all_candidates, verbose, last_name_index=ln_index)

            if best_match:
                # Update the orphan record
                orphan.bdl_player_id = best_match.bdl_id
                linked_count += 1

                if verbose:
                    logger.info(
                        f"  LINKED: {orphan.player_name} -> "
                        f"{best_match.full_name} (bdl_id: {best_match.bdl_id}, score: {score:.3f})"
                    )
            else:
                failed_count += 1
                # Collect sample of unmatched records
                if len(sample_unmatched) < 10:
                    sample_unmatched.append({
                        "player_name": orphan.player_name,
                        "yahoo_key": orphan.yahoo_player_key,
                        "best_score": score
                    })

                if verbose:
                    logger.warning(f"  NO MATCH: {orphan.player_name} (best score: {score:.3f})")

            # Progress update every 50 records
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(orphans)} processed ({linked_count} linked)")

        # Commit changes
        if not dry_run:
            logger.info("\nCommitting changes to database...")
            db.commit()
            logger.info("Changes committed successfully")
        else:
            logger.info("\nDRY RUN - rolling back changes")
            db.rollback()

        # Verify results
        remaining_orphans = orphan_count - linked_count
        success_rate = (linked_count / orphan_count * 100) if orphan_count > 0 else 0.0

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS:")
        logger.info(f"  Linked: {linked_count}")
        logger.info(f"  Remaining orphans: {remaining_orphans}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info("=" * 60)

        elapsed = (datetime.now(timezone.utc) - t0).total_seconds() * 1000

        return {
            "status": "success",
            "linked_count": linked_count,
            "remaining_count": remaining_orphans,
            "success_rate": success_rate,
            "elapsed_ms": elapsed,
            "sample_unmatched": sample_unmatched
        }

    except Exception as e:
        logger.error(f"Error during linking: {e}")
        db.rollback()
        raise

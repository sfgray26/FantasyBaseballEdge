"""
Qwen Local Model Advisor — Real-time draft/lineup advice via OpenClaw/Ollama.

Uses the Ollama OpenAI-compatible API endpoint (default: http://localhost:11434).
No rate limits, sub-5-second responses, runs entirely on your hardware.

Config via .env:
  OLLAMA_BASE_URL=http://localhost:11434   (default)
  OLLAMA_MODEL=qwen2.5:latest              (or qwen:7b, qwen2:7b, etc.)

Falls back gracefully if Ollama is not running — returns a pre-computed rationale.
"""

import logging
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
TIMEOUT_SECS = 12  # Hard cutoff — must respond within draft clock


def is_ollama_available() -> bool:
    """Check if Ollama is running and the model is loaded."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(OLLAMA_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


def ask_qwen(prompt: str, system: Optional[str] = None, timeout: int = TIMEOUT_SECS) -> str:
    """
    Send a prompt to Qwen via Ollama and return the response text.
    Returns empty string if Ollama is unavailable or times out.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,    # Low temp for consistent picks
            "num_predict": 200,    # Short responses needed for 90-sec clock
        },
    }
    if system:
        payload["system"] = system

    try:
        start = time.time()
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - start
        if resp.status_code != 200:
            logger.warning(f"Ollama returned {resp.status_code}")
            return ""
        text = resp.json().get("response", "").strip()
        logger.debug(f"Qwen responded in {elapsed:.1f}s: {text[:80]}")
        return text
    except requests.Timeout:
        logger.warning(f"Qwen timed out after {timeout}s — using pre-computed rationale")
        return ""
    except Exception as e:
        logger.warning(f"Qwen unavailable: {e}")
        return ""


DRAFT_SYSTEM_PROMPT = """You are a fantasy baseball expert assistant.
League format: 12-team Yahoo H2H One Win, 18 categories.
Batting cats: R, H, HR, RBI, K(negative), TB, AVG, OPS, NSB
Pitching cats: W, L(negative), HR(negative), K, ERA, WHIP, K/9, QS, NSV
Key rules:
- Batter K is NEGATIVE — strikeouts hurt your team
- Pitcher losses (L) are NEGATIVE — penalize SPs with bad run support
- NSB = net stolen bases (SB minus CS) — very scarce, high value
- NSV = net saves (SV minus BS) — closers are scarce and valuable

Be concise — the user has a 90-second clock. 2-3 sentences max."""


def draft_pick_rationale(
    pick_name: str,
    pick_positions: list[str],
    pick_type: str,
    round_num: int,
    roster_needs: dict,
    top_categories: list[str],
) -> str:
    """
    Generate a short pick rationale for the live draft.
    Falls back to template if Ollama is not running.
    """
    if not is_ollama_available():
        return _fallback_rationale(pick_name, pick_positions, pick_type, round_num, top_categories)

    needs_str = ", ".join(f"{k}({v})" for k, v in roster_needs.items()) or "none critical"
    cats_str = ", ".join(top_categories[:3]) if top_categories else "multi-category"

    prompt = (
        f"Round {round_num} pick: {pick_name} ({'/'.join(pick_positions[:2])}, {pick_type}). "
        f"Roster needs: {needs_str}. "
        f"Top category contributions: {cats_str}. "
        f"In 2-3 sentences, explain why this is a good pick at this draft position "
        f"for H2H One Win fantasy baseball."
    )

    response = ask_qwen(prompt, system=DRAFT_SYSTEM_PROMPT)
    return response if response else _fallback_rationale(
        pick_name, pick_positions, pick_type, round_num, top_categories
    )


def waiver_quick_check(
    player_name: str,
    player_type: str,
    positions: list[str],
    current_cats_weak: list[str],
    drop_candidate: str,
) -> str:
    """Quick yes/no waiver recommendation (for daily use)."""
    if not is_ollama_available():
        return f"Add {player_name}? Check their recent stats and if they address your weak categories: {', '.join(current_cats_weak[:3])}."

    prompt = (
        f"Should I add {player_name} ({'/'.join(positions[:2])}, {player_type}) "
        f"from waivers and drop {drop_candidate}? "
        f"My weak categories are: {', '.join(current_cats_weak[:4])}. "
        f"Answer yes or no with a 2-sentence explanation."
    )
    return ask_qwen(prompt, system=DRAFT_SYSTEM_PROMPT) or f"Check {player_name}'s recent stats and matchups."


def _fallback_rationale(
    name: str,
    positions: list[str],
    player_type: str,
    round_num: int,
    top_categories: list[str],
) -> str:
    """Pre-computed rationale when Qwen is unavailable."""
    pos_str = "/".join(positions[:2])
    cats_str = ", ".join(top_categories[:3]) if top_categories else "multiple categories"
    return (
        f"Round {round_num}: {name} ({pos_str}) contributes to {cats_str}. "
        f"Good value at this draft position — adding now prevents a mid-round reach later."
    )

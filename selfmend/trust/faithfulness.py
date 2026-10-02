"""
Custom faithfulness scorer for SelfMend.

RAGAS 0.4.x is incompatible with modern langchain-community (removes
langchain_community.chat_models.vertexai, see GitHub issue #2753).
We implement an equivalent faithfulness metric directly using Groq's
structured-output API.

Faithfulness measures whether the claims in the LLM's diagnosis are
supported by the retrieved evidence (incidents). Score ∈ [0, 1].

Algorithm (mirrors RAGAS faithfulness):
  1. Extract atomic claims from the diagnosis text.
  2. For each claim, ask the LLM: "Is this claim supported by the
     provided evidence? Answer YES or NO with a brief reason."
  3. faithfulness = supported_claims / total_claims

If no claims are found or the LLM is unavailable (offline mode), a
conservative fallback score is returned with a warning.

Public API:
    score_faithfulness(diagnosis_text, evidence_texts, llm) -> float
    score_faithfulness_mock(diagnosis_text, evidence_texts) -> float  # for tests
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# ── Offline / mock mode ───────────────────────────────────────────────────────
# When GROQ_API_KEY is not set, the mock scorer is used.
# It returns a deterministic score based on keyword overlap.
OFFLINE_FALLBACK_SCORE = 0.5


def _get_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
    """Lazy-import and instantiate Groq LLM. Returns None if key not set."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("gsk_REPLACE"):
        log.warning("GROQ_API_KEY not set — faithfulness will use mock scorer.")
        return None
    try:
        from langchain_groq import ChatGroq  # noqa: PLC0415
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_retries=2,
        )
    except Exception as exc:
        log.error("Failed to create Groq LLM: %s", exc)
        return None


def _extract_claims(text: str) -> list[str]:
    """
    Split diagnosis text into atomic claims (sentence-level).
    Filters out very short fragments.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]
    return claims[:10]   # cap at 10 claims to keep prompt size manageable


def _mock_faithfulness(diagnosis_text: str, evidence_texts: list[str]) -> float:
    """
    Offline fallback: keyword-overlap heuristic.
    Returns fraction of diagnosis words that appear in any evidence document.
    Always returns a value in [0.2, 0.95] to avoid extreme scores.
    """
    diag_words = set(re.findall(r"\b\w{4,}\b", diagnosis_text.lower()))
    ev_words   = set(
        w for ev in evidence_texts
        for w in re.findall(r"\b\w{4,}\b", ev.lower())
    )
    if not diag_words:
        return OFFLINE_FALLBACK_SCORE
    overlap = len(diag_words & ev_words) / len(diag_words)
    return round(min(0.95, max(0.20, overlap)), 4)


def score_faithfulness(
    diagnosis_text: str,
    evidence_texts: list[str],
    llm: Any = None,
    model: str = "llama-3.3-70b-versatile",
) -> float:
    """
    Score faithfulness of `diagnosis_text` against `evidence_texts`.

    Args:
        diagnosis_text : The LLM's diagnosis output as a string.
        evidence_texts : List of retrieved incident document strings.
        llm            : Pre-built LLM instance (optional). If None, one is created.
        model          : Groq model name (used if llm is None).

    Returns:
        Faithfulness score ∈ [0, 1].
    """
    if not evidence_texts:
        log.warning("No evidence texts provided — returning 0.0 faithfulness.")
        return 0.0

    if llm is None:
        llm = _get_groq_llm(model=model)

    if llm is None:
        score = _mock_faithfulness(diagnosis_text, evidence_texts)
        log.info("Faithfulness (mock/offline): %.3f", score)
        return score

    return _score_with_llm(diagnosis_text, evidence_texts, llm)


def _score_with_llm(
    diagnosis_text: str,
    evidence_texts: list[str],
    llm: Any,
) -> float:
    """
    Use the Groq LLM to assess each claim against the evidence.
    Returns supported_claims / total_claims.
    """
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    claims = _extract_claims(diagnosis_text)
    if not claims:
        log.warning("No extractable claims in diagnosis — returning 0.5.")
        return 0.5

    evidence_block = "\n\n".join(
        f"[Evidence {i+1}]: {ev}" for i, ev in enumerate(evidence_texts)
    )

    supported = 0
    for claim in claims:
        prompt = (
            f"Evidence:\n{evidence_block}\n\n"
            f"Claim: {claim}\n\n"
            f"Is this claim fully supported by the evidence above? "
            f"Answer with exactly YES or NO."
        )
        try:
            msgs = [
                SystemMessage(content="You are a faithful evidence checker. Answer only YES or NO."),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(msgs)
            answer = response.content.strip().upper()
            if answer.startswith("YES"):
                supported += 1
        except Exception as exc:
            log.error("Faithfulness LLM call failed for claim '%s': %s", claim[:50], exc)
            # Conservative: don't count as supported
            continue

    score = round(supported / len(claims), 4) if claims else 0.5
    log.info("Faithfulness (LLM): %.3f  (%d/%d claims supported)", score, supported, len(claims))
    return score


def score_faithfulness_mock(
    diagnosis_text: str,
    evidence_texts: list[str],
    forced_score: float | None = None,
) -> float:
    """
    Mock scorer for use in tests. If forced_score is set, always returns that.
    Otherwise uses the keyword-overlap heuristic.
    """
    if forced_score is not None:
        return forced_score
    return _mock_faithfulness(diagnosis_text, evidence_texts)

"""Query router.

Decides whether a user query is about a person, a place, or both. The router
is intentionally simple, in line with the assignment's "keyword based or rule
based approaches are acceptable" guidance:

1. Direct entity-name match against ``config.PEOPLE`` / ``config.PLACES``.
2. Cue-word match (e.g. "where", "located", "monument" -> place; "who",
   "discovered", "born" -> person).
3. Fallback: type=None (search across the whole collection).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from . import config

QueryType = Literal["person", "place", "both", "unknown"]


_PERSON_CUES = {
    "who", "whose", "person", "people", "man", "woman", "born", "died",
    "biography", "biographer", "actor", "actress", "scientist", "musician",
    "physicist", "mathematician", "painter", "writer", "poet", "footballer",
    "soccer", "singer", "philosopher", "leader", "president", "queen",
    "king", "inventor", "engineer", "discovered", "invented", "wrote",
    "painted", "composed", "performed", "starred", "famous person",
}

_PLACE_CUES = {
    "where", "located", "located in", "country", "city", "river", "mountain",
    "monument", "statue", "tower", "wall", "wonder", "ruin", "ruins",
    "temple", "tomb", "pyramid", "canyon", "desert", "island", "lake",
    "ocean", "continent", "stadium", "structure", "tourist", "landmark",
    "site", "valley", "park", "altitude", "elevation", "tall", "tallest",
    "highest", "famous place",
}

_COMPARE_CUES = {
    "compare", "vs", "versus", "difference", "differences", "similar",
    "similarities", "between", "and",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _find_entities(query: str, candidates: list[str]) -> list[str]:
    """Return the subset of ``candidates`` that appear in ``query``.

    Match is case-insensitive and respects word boundaries on the first and
    last character so we don't match "Mes" inside "Messi" inside another word.
    """
    q = _normalize(query)
    matches: list[str] = []
    for cand in candidates:
        cand_norm = _normalize(cand)
        if re.search(rf"(?<!\w){re.escape(cand_norm)}(?!\w)", q):
            matches.append(cand)
            continue
        # Also accept the last token of multi-word names — "Einstein" alone is
        # a strong signal for "Albert Einstein". This biases toward higher
        # recall on the cost of occasional false positives, which is fine
        # because retrieval still ranks within the chosen partition.
        last_token = cand_norm.split()[-1]
        if (
            len(last_token) >= 5
            and re.search(rf"(?<!\w){re.escape(last_token)}(?!\w)", q)
        ):
            matches.append(cand)
    return matches


def _cue_score(query: str, cues: set[str]) -> int:
    q = _normalize(query)
    return sum(1 for cue in cues if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", q))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class Routing:
    type: QueryType                 # which partition(s) to query
    matched_people: list[str]
    matched_places: list[str]
    is_comparison: bool             # heuristic: does this look like compare X and Y?
    rationale: str                  # human-readable explanation for the UI


def route(query: str) -> Routing:
    """Classify a user query."""

    matched_people = _find_entities(query, config.PEOPLE)
    matched_places = _find_entities(query, config.PLACES)

    person_score = _cue_score(query, _PERSON_CUES) + len(matched_people)
    place_score = _cue_score(query, _PLACE_CUES) + len(matched_places)
    is_comparison = _cue_score(query, _COMPARE_CUES) >= 1 and (
        len(matched_people) + len(matched_places) >= 2
    )

    if matched_people and matched_places:
        return Routing(
            type="both",
            matched_people=matched_people,
            matched_places=matched_places,
            is_comparison=is_comparison,
            rationale=(
                f"Mixed query — recognised people {matched_people} and "
                f"places {matched_places}."
            ),
        )

    if person_score > 0 and place_score == 0:
        return Routing(
            type="person",
            matched_people=matched_people,
            matched_places=[],
            is_comparison=is_comparison,
            rationale=(
                f"Person query — matches={matched_people or 'none'}, "
                f"person cues score={person_score}."
            ),
        )

    if place_score > 0 and person_score == 0:
        return Routing(
            type="place",
            matched_people=[],
            matched_places=matched_places,
            is_comparison=is_comparison,
            rationale=(
                f"Place query — matches={matched_places or 'none'}, "
                f"place cues score={place_score}."
            ),
        )

    if person_score > 0 and place_score > 0:
        # Both kinds of cues — search across both partitions.
        return Routing(
            type="both",
            matched_people=matched_people,
            matched_places=matched_places,
            is_comparison=is_comparison,
            rationale=(
                f"Ambiguous query — both person ({person_score}) and place "
                f"({place_score}) cues fire; searching both partitions."
            ),
        )

    return Routing(
        type="unknown",
        matched_people=[],
        matched_places=[],
        is_comparison=False,
        rationale="No strong cues — falling back to a corpus-wide search.",
    )

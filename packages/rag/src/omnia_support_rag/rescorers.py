"""Concrete rescorer implementations for post-retrieval quality boosting."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from omnia_support_models import ResolutionStatus

if TYPE_CHECKING:
    from omnia_support_store import ScoredChunk


class SatisfactionRescorer:
    """Rescorer that multiplies distance by a satisfaction-derived factor.

    Uses a centered scale where 3.0 is neutral: scores above 3 amplify the
    distance, scores below 3 reduce it. Chunks missing customer_satisfaction
    metadata are returned unchanged.

    The weight parameter controls the maximum adjustment magnitude. With the
    default of 0.2, a 5-star chunk is multiplied by 1.2 and a 1-star chunk
    by 0.8.
    """

    def __init__(self, weight: float = 0.2) -> None:
        """Create a satisfaction rescorer with the given weight."""
        self._weight = weight

    def rescore(self, chunk: ScoredChunk) -> ScoredChunk:  # noqa: D102
        satisfaction = chunk.metadata.get("customer_satisfaction")
        if satisfaction is None:
            return chunk
        multiplier = 1.0 + self._weight * (satisfaction - 3.0) / 2.0
        return replace(chunk, distance=chunk.distance * multiplier)


class ResolutionStatusRescorer:
    """Rescorer that multiplies distance by a per-status weight.

    Chunks missing resolution_status metadata are returned unchanged.
    """

    def __init__(
        self,
        *,
        resolved_weight: float = 1.2,
        active_weight: float = 1.0,
        pending_weight: float = 0.8,
    ) -> None:
        """Create a resolution status rescorer with per-status weights."""
        self._weights: dict[str, float] = {
            ResolutionStatus.RESOLVED: resolved_weight,
            ResolutionStatus.ACTIVE: active_weight,
            ResolutionStatus.PENDING: pending_weight,
        }

    def rescore(self, chunk: ScoredChunk) -> ScoredChunk:  # noqa: D102
        status = chunk.metadata.get("resolution_status")
        if status is None:
            return chunk
        return replace(chunk, distance=chunk.distance * self._weights.get(status, 1.0))

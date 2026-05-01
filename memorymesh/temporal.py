"""Temporal layer — confidence decay, spaced repetition, and time-travel.

This module implements the Ebbinghaus-inspired memory decay model and
the spaced repetition reinforcement system.  Both are computed lazily
on read to avoid write amplification (TMP-006).

Key formulas:
    Decay:    R(t) = R₀ × exp(−λ × Δt)     where Δt is hours since creation
    Boost:    R_boosted = min(R × 1.15, 1.0) when access_rate > threshold

PRD Requirements covered:
    TMP-005  Ebbinghaus confidence decay formula
    TMP-006  Lazy decay on read (no background jobs)
    TMP-007  Spaced repetition boost for frequently accessed memories
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Decay configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecayConfig:
    """Configuration for confidence decay behavior.

    Attributes
    ----------
    lambda_rate : float
        Decay rate per hour.  Higher = faster forgetting.
        Default 0.01/hour means ~63% retention after 46 hours.
    boost_factor : float
        Multiplicative boost for frequently accessed memories.
        Default 1.15 = 15% boost per reinforcement.
    boost_threshold : float
        Minimum reads per hour to trigger a boost.
        Default 3.0 reads/hour.
    boost_cap : float
        Maximum confidence after boosting.  Hard-capped at 1.0.
    """
    lambda_rate: float = 0.01
    boost_factor: float = 1.15
    boost_threshold: float = 3.0
    boost_cap: float = 1.0


# Default configuration — matches PRD specification exactly
DEFAULT_DECAY_CONFIG = DecayConfig()


# ---------------------------------------------------------------------------
# Access tracker — records when nodes are read
# ---------------------------------------------------------------------------

class AccessTracker:
    """Tracks access patterns for spaced repetition calculations.

    Records every ``get_node()`` access with a timestamp.  Used to
    compute access rate (reads/hour) for boost eligibility.

    This is intentionally kept separate from the WAL — access patterns
    are ephemeral metadata, not part of the durable CRDT state.
    """

    def __init__(self) -> None:
        # node_id → list of ISO 8601 access timestamps
        self._accesses: Dict[str, List[str]] = defaultdict(list)

    def record_access(
        self,
        node_id: str,
        timestamp_utc: Optional[str] = None,
    ) -> None:
        """Record a read access for a memory node."""
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
        self._accesses[node_id].append(ts)

    def access_count(self, node_id: str) -> int:
        """Total number of accesses for a node."""
        return len(self._accesses.get(node_id, []))

    def access_rate_per_hour(
        self,
        node_id: str,
        now_utc: Optional[str] = None,
        window_hours: float = 1.0,
    ) -> float:
        """Compute recent access rate (reads/hour) within a time window.

        Parameters
        ----------
        node_id : str
            Node to check.
        now_utc : str, optional
            Current time (ISO 8601).  Defaults to now.
        window_hours : float
            Look-back window in hours.

        Returns
        -------
        float
            Access rate in reads per hour within the window.
        """
        accesses = self._accesses.get(node_id, [])
        if not accesses:
            return 0.0

        now = _parse_iso(now_utc) if now_utc else datetime.now(timezone.utc)
        cutoff_hours = window_hours

        count = 0
        for ts_str in accesses:
            ts = _parse_iso(ts_str)
            delta_hours = (now - ts).total_seconds() / 3600.0
            if delta_hours <= cutoff_hours:
                count += 1

        if window_hours <= 0:
            return 0.0
        return count / window_hours

    def get_access_history(self, node_id: str) -> List[str]:
        """Return the full access history for a node."""
        return list(self._accesses.get(node_id, []))

    def clear(self) -> None:
        """Clear all access records."""
        self._accesses.clear()


# ---------------------------------------------------------------------------
# Confidence decay computation (TMP-005, TMP-006)
# ---------------------------------------------------------------------------

def compute_decayed_confidence(
    initial_confidence: float,
    created_utc: str,
    now_utc: Optional[str] = None,
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
) -> float:
    """Compute time-decayed confidence using Ebbinghaus curve (TMP-005).

    Formula: R(t) = R₀ × exp(−λ × Δt)

    Parameters
    ----------
    initial_confidence : float
        R₀ — the original confidence score at creation time.
    created_utc : str
        ISO 8601 UTC timestamp of node creation.
    now_utc : str, optional
        Current time.  Defaults to now.
    config : DecayConfig
        Decay parameters.

    Returns
    -------
    float
        Decayed confidence R(t), guaranteed ∈ [0, 1].

    Notes
    -----
    This is a pure function — no state is mutated.  The decay is
    computed fresh on every call (lazy evaluation, TMP-006).
    """
    now = _parse_iso(now_utc) if now_utc else datetime.now(timezone.utc)
    created = _parse_iso(created_utc)

    delta_hours = max(0.0, (now - created).total_seconds() / 3600.0)

    decayed = initial_confidence * math.exp(-config.lambda_rate * delta_hours)

    return max(0.0, min(1.0, decayed))


def compute_boosted_confidence(
    current_confidence: float,
    access_rate: float,
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
) -> Tuple[float, bool]:
    """Apply spaced repetition boost if access rate exceeds threshold (TMP-007).

    Formula: R_boosted = min(R_current × boost_factor, boost_cap)

    Parameters
    ----------
    current_confidence : float
        The confidence after decay has been applied.
    access_rate : float
        Reads per hour for this node.
    config : DecayConfig
        Boost parameters.

    Returns
    -------
    tuple[float, bool]
        (boosted_confidence, was_boosted) — the second element
        indicates whether a boost was actually applied.
    """
    if access_rate >= config.boost_threshold:
        boosted = min(current_confidence * config.boost_factor, config.boost_cap)
        return boosted, True
    return current_confidence, False


def compute_effective_confidence(
    initial_confidence: float,
    created_utc: str,
    access_rate: float = 0.0,
    now_utc: Optional[str] = None,
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
) -> Tuple[float, bool]:
    """Compute the full effective confidence: decay then optional boost.

    This is the single entry point for confidence computation on read.

    Parameters
    ----------
    initial_confidence : float
        R₀ at creation time.
    created_utc : str
        ISO 8601 UTC creation timestamp.
    access_rate : float
        Current reads/hour for the node.
    now_utc : str, optional
        Current time.
    config : DecayConfig
        Full decay/boost configuration.

    Returns
    -------
    tuple[float, bool]
        (effective_confidence, was_boosted)
    """
    decayed = compute_decayed_confidence(
        initial_confidence, created_utc, now_utc, config
    )
    return compute_boosted_confidence(decayed, access_rate, config)


# ---------------------------------------------------------------------------
# ISO 8601 parser helper
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a timezone-aware datetime.

    Handles both formats:
        - 2026-04-03T14:32:00+00:00
        - 2026-04-03T14:32:00Z
    """
    # Replace 'Z' suffix with '+00:00' for fromisoformat compatibility
    normalized = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

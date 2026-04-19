"""MemoryMesh type definitions — edge labels and custom exceptions.

This module defines the causal edge vocabulary and error types used
throughout the MemoryMesh substrate.  Edge labels are a closed set:
only these five relationship types are permitted in v1.0.
"""

from __future__ import annotations

from enum import Enum


class EdgeLabel(str, Enum):
    """Causal edge labels for the memory graph.

    Each label represents a specific epistemic relationship between
    two memory nodes.  Using ``str, Enum`` lets instances compare
    directly with plain strings (e.g. ``EdgeLabel.SUPPORTS == "supports"``).
    """

    CAUSED_BY = "caused-by"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived-from"


# ---------------------------------------------------------------------------
# Convenience set for O(1) membership checks (faster than Enum lookup)
# ---------------------------------------------------------------------------
ALLOWED_EDGE_LABELS: frozenset[str] = frozenset(label.value for label in EdgeLabel)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class CycleDetectedError(ValueError):
    """Raised when adding a causal edge would introduce a cycle.

    The memory graph MUST remain a DAG — cycles violate causal
    consistency (SYN-006).
    """

    def __init__(self, source_id: str, target_id: str, label: str) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.label = label
        super().__init__(
            f"CYCLE_DETECTED: edge ({source_id}) -[{label}]-> ({target_id}) "
            f"would create a cycle in the causal graph"
        )


class NodeNotFoundError(KeyError):
    """Raised when a requested memory node does not exist in the graph."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Memory node not found: {node_id}")


class NamespaceMismatchError(ValueError):
    """Raised when attempting to merge replicas from different namespaces."""

    def __init__(self, ns_a: str, ns_b: str) -> None:
        self.ns_a = ns_a
        self.ns_b = ns_b
        super().__init__(
            f"Cannot merge replicas from different namespaces: "
            f"'{ns_a}' vs '{ns_b}'"
        )

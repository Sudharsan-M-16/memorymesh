"""MemoryMesh — The Persistent Cognitive Substrate for the Agentic Internet.

Public API surface for v0.1.0 (Brain Stem / Week 1).

Core exports:
    MemoryNode          — 7-tuple cognitive unit (content-addressed)
    MemoryMeshCore      — 2P2P-Graph CRDT engine
    BayesianTrustEngine — Per-agent Beta-Binomial trust model

Types:
    EdgeLabel           — Causal edge label enum
    CycleDetectedError  — Raised on DAG cycle violation
    NodeNotFoundError   — Raised on missing node access
    NamespaceMismatchError — Raised on cross-namespace merge
"""

from .memory_node import MemoryNode, content_address, zero_embedding
from .mesh_core import CRDTState, MemoryMeshCore
from .trust_engine import AuditEntry, BayesianTrustEngine
from .types import (
    ALLOWED_EDGE_LABELS,
    CycleDetectedError,
    EdgeLabel,
    NamespaceMismatchError,
    NodeNotFoundError,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "MemoryNode",
    "MemoryMeshCore",
    "BayesianTrustEngine",
    # Types
    "EdgeLabel",
    "CRDTState",
    "AuditEntry",
    # Exceptions
    "CycleDetectedError",
    "NodeNotFoundError",
    "NamespaceMismatchError",
    # Helpers
    "ALLOWED_EDGE_LABELS",
    "content_address",
    "zero_embedding",
    # Meta
    "__version__",
]

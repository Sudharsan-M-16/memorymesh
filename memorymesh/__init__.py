"""MemoryMesh — The Persistent Cognitive Substrate for the Agentic Internet.

Public API surface for v0.2.0 (Brain Stem + Temporal Layer).

Core exports:
    MemoryNode          — 7-tuple cognitive unit (content-addressed)
    MemoryMeshCore      — 2P2P-Graph CRDT engine with WAL persistence
    BayesianTrustEngine — Per-agent Beta-Binomial trust model

Temporal:
    WriteAheadLog       — Append-only WAL for crash recovery + time-travel
    WALOp, WALEntry     — WAL operation types and entries
    DecayConfig         — Ebbinghaus decay + spaced repetition config
    AccessTracker       — Read-frequency tracking for boost eligibility

Types:
    EdgeLabel           — Causal edge label enum
    CycleDetectedError  — Raised on DAG cycle violation
    NodeNotFoundError   — Raised on missing node access
    NamespaceMismatchError — Raised on cross-namespace merge
"""

from .memory_node import MemoryNode, content_address, zero_embedding
from .mesh_core import CRDTState, MemoryMeshCore
from .temporal import (
    AccessTracker,
    DecayConfig,
    DEFAULT_DECAY_CONFIG,
    compute_decayed_confidence,
    compute_effective_confidence,
)
from .trust_engine import AuditEntry, BayesianTrustEngine
from .types import (
    ALLOWED_EDGE_LABELS,
    CycleDetectedError,
    EdgeLabel,
    NamespaceMismatchError,
    NodeNotFoundError,
)
from .wal import WALEntry, WALOp, WriteAheadLog

__version__ = "0.2.0"

__all__ = [
    # Core
    "MemoryNode",
    "MemoryMeshCore",
    "BayesianTrustEngine",
    # WAL
    "WriteAheadLog",
    "WALOp",
    "WALEntry",
    # Temporal
    "DecayConfig",
    "DEFAULT_DECAY_CONFIG",
    "AccessTracker",
    "compute_decayed_confidence",
    "compute_effective_confidence",
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

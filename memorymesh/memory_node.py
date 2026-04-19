"""MemoryNode — the 7-tuple cognitive unit.

Each node represents a single unit of agent cognition:

    m = (id, e, τ, C, R, T, κ)

where:
    id  — SHA-256 content-addressed identifier (deterministic across agents)
    e   — embedding vector (384-dim float32 numpy array, CUDA-ready)
    τ   — Lamport logical-clock vector {agent_id: int}
    C   — causal edge set (frozenset for CRDT immutability)
    R   — confidence score ∈ [0, 1]
    T   — trust provenance metadata
    κ   — raw content payload
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

import numpy as np

from .types import ALLOWED_EDGE_LABELS


# ---------------------------------------------------------------------------
# Content-addressing helpers
# ---------------------------------------------------------------------------

def canonical_content(content: str) -> str:
    """Normalize content for deterministic hashing.

    Strips whitespace and lowercases — ensures two agents writing
    semantically identical strings produce the same node ID (SYN-004).
    """
    return content.strip().lower()


def content_address(content: str) -> str:
    """Compute SHA-256 of canonical content string → node ID."""
    return hashlib.sha256(
        canonical_content(content).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def zero_embedding() -> np.ndarray:
    """Return a contiguous float32 zero vector (CUDA-transfer-ready)."""
    return np.zeros(_EMBEDDING_DIM, dtype=np.float32)


def validate_embedding(embedding: np.ndarray) -> np.ndarray:
    """Ensure the embedding is a contiguous float32 array of correct dim."""
    arr = np.ascontiguousarray(embedding, dtype=np.float32)
    if arr.shape != (_EMBEDDING_DIM,):
        raise ValueError(
            f"Embedding must be {_EMBEDDING_DIM}-dimensional, "
            f"got shape {arr.shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# Lamport vector helpers
# ---------------------------------------------------------------------------

def merge_lamport_vectors(
    a: Dict[str, int],
    b: Dict[str, int],
) -> Dict[str, int]:
    """Element-wise max merge of two Lamport vectors (TMP-002)."""
    merged: Dict[str, int] = {}
    for agent in set(a) | set(b):
        merged[agent] = max(a.get(agent, 0), b.get(agent, 0))
    return merged


# ---------------------------------------------------------------------------
# MemoryNode dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryNode:
    """Immutable 7-tuple memory unit.

    Frozen dataclass ensures nodes are hashable and safe for
    inclusion in CRDT grow-only sets.

    Attributes
    ----------
    id : str
        SHA-256 content hash — deterministic, cross-agent dedup key.
    embedding : np.ndarray
        384-dim float32 vector.  Zero-vector if not yet computed.
    lamport_vector : dict[str, int]
        Logical clock keyed by agent_id.
    causal_edges : frozenset[tuple[str, str, str]]
        (source_id, target_id, label) — immutable for CRDT safety.
    confidence : float
        Initial R₀ ∈ [0.0, 1.0].
    trust_provenance : dict
        {agent_id, wall_clock_utc, namespace, …}
    content : str
        Raw payload κ (stored after canonical normalization).
    """

    id: str
    embedding: np.ndarray
    lamport_vector: Dict[str, int]
    causal_edges: FrozenSet[Tuple[str, str, str]]
    confidence: float
    trust_provenance: Dict[str, Any]
    content: str

    # Allow numpy arrays in frozen dataclass (hash by id only)
    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryNode):
            return NotImplemented
        return self.id == other.id

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------
    @staticmethod
    def create(
        content: str,
        agent_id: str,
        confidence: float = 0.5,
        namespace: str = "default",
        embedding: Optional[np.ndarray] = None,
        causal_edges: Optional[Iterable[Tuple[str, str, str]]] = None,
        lamport_vector: Optional[Dict[str, int]] = None,
        extra_provenance: Optional[Dict[str, Any]] = None,
    ) -> "MemoryNode":
        """Build a MemoryNode with full validation.

        Parameters
        ----------
        content : str
            Raw cognitive payload.
        agent_id : str
            Writing agent identifier.
        confidence : float
            Initial confidence R₀ ∈ [0, 1].
        namespace : str
            Memory namespace scope.
        embedding : np.ndarray, optional
            384-d float32 vector.  Defaults to zero vector.
        causal_edges : iterable of (src, dst, label), optional
            Causal relationships to encode.
        lamport_vector : dict, optional
            Lamport clock state.  Defaults to {agent_id: 1}.
        extra_provenance : dict, optional
            Additional trust metadata fields.
        """
        # --- ID ---
        node_id = content_address(content)

        # --- Embedding ---
        if embedding is not None:
            emb = validate_embedding(embedding)
        else:
            emb = zero_embedding()

        # --- Lamport vector ---
        tau = dict(lamport_vector) if lamport_vector else {agent_id: 1}

        # --- Causal edges (validate labels) ---
        edges: set[Tuple[str, str, str]] = set()
        if causal_edges:
            for src, dst, label in causal_edges:
                if label not in ALLOWED_EDGE_LABELS:
                    raise ValueError(f"Invalid edge label: {label}")
                edges.add((src, dst, label))

        # --- Confidence clamping ---
        conf = float(max(0.0, min(1.0, confidence)))

        # --- Trust provenance ---
        now_utc = datetime.now(timezone.utc).isoformat()
        provenance: Dict[str, Any] = {
            "agent_id": agent_id,
            "wall_clock_utc": now_utc,
            "namespace": namespace,
        }
        if extra_provenance:
            provenance.update(extra_provenance)

        return MemoryNode(
            id=node_id,
            embedding=emb,
            lamport_vector=tau,
            causal_edges=frozenset(edges),
            confidence=conf,
            trust_provenance=provenance,
            content=canonical_content(content),
        )

    # -----------------------------------------------------------------
    # CRDT merge (for deduplication)
    # -----------------------------------------------------------------
    def merge_with(self, other: "MemoryNode") -> "MemoryNode":
        """Merge two nodes with the same content-address ID.

        Uses element-wise max for Lamport vectors, max for confidence,
        and union for causal edges — all monotonic operations that
        preserve CRDT convergence.
        """
        if self.id != other.id:
            raise ValueError(
                f"Cannot merge nodes with different IDs: "
                f"{self.id[:12]}… vs {other.id[:12]}…"
            )

        merged_tau = merge_lamport_vectors(
            self.lamport_vector, other.lamport_vector
        )
        merged_edges = self.causal_edges | other.causal_edges
        merged_conf = max(self.confidence, other.confidence)

        # Prefer non-zero embedding
        if np.any(self.embedding):
            merged_emb = self.embedding
        else:
            merged_emb = other.embedding

        # Merge provenance — keep both agents' info
        merged_prov = dict(self.trust_provenance)
        for k, v in other.trust_provenance.items():
            if k not in merged_prov:
                merged_prov[k] = v
            elif k == "agent_id" and merged_prov[k] != v:
                # Collect all contributing agent IDs
                existing = merged_prov[k]
                if not isinstance(existing, list):
                    existing = [existing]
                if v not in existing:
                    existing.append(v)
                merged_prov[k] = existing

        return MemoryNode(
            id=self.id,
            embedding=merged_emb,
            lamport_vector=merged_tau,
            causal_edges=merged_edges,
            confidence=merged_conf,
            trust_provenance=merged_prov,
            content=self.content,
        )

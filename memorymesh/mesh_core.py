"""MemoryMeshCore — the 2P2P-Graph CRDT engine.

This is the central nervous system of MemoryMesh.  It manages:

1. A 2P2P-Graph CRDT with four grow-only sets:
       (A_nodes, R_nodes, A_edges, R_edges)
   Observed state = A \\ R for both nodes and edges.

2. A NetworkX DiGraph for causal traversal and cycle detection.

3. Content-addressed deduplication via SHA-256 node IDs.

4. Lamport vector clocks for causal ordering.

5. Optional WAL persistence for crash recovery and time-travel.

6. Lazy confidence decay (Ebbinghaus) and spaced repetition boost.

PRD Requirements covered:
    SYN-001  2P2P-Graph CRDT state model
    SYN-002  Merge satisfies commutativity, associativity, idempotency
    SYN-003  Set-union merge + semantic conflict hook
    SYN-004  Content-addressed dedup via SHA-256
    SYN-005  O(N+E) merge complexity
    SYN-006  DFS cycle detection on edge insertion
    TMP-001  Wall-clock + Lamport timestamps on every write
    TMP-002  Lamport merge via element-wise max
    TMP-003  Point-in-time graph queries via query_at()
    TMP-004  Append-only WAL replay for state reconstruction
    TMP-005  Ebbinghaus confidence decay
    TMP-006  Lazy decay on read
    TMP-007  Spaced repetition boost
    TMP-008  causal_chain(node_id, depth) DAG traversal
    TMP-009  WAL compaction via snapshot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .memory_node import MemoryNode, content_address, merge_lamport_vectors
from .temporal import (
    AccessTracker,
    DecayConfig,
    DEFAULT_DECAY_CONFIG,
    compute_effective_confidence,
)
from .types import (
    ALLOWED_EDGE_LABELS,
    CycleDetectedError,
    NamespaceMismatchError,
    NodeNotFoundError,
)
from .wal import WALOp, WriteAheadLog


# ---------------------------------------------------------------------------
# CRDT State Container
# ---------------------------------------------------------------------------

@dataclass
class CRDTState:
    """Raw 2P2P-Graph CRDT quadruple.

    Used for serialization and snapshot comparison.
    All four sets are grow-only; observed state = add - remove.
    """

    add_nodes: frozenset[str] = field(default_factory=frozenset)
    rem_nodes: frozenset[str] = field(default_factory=frozenset)
    add_edges: frozenset[Tuple[str, str, str]] = field(default_factory=frozenset)
    rem_edges: frozenset[Tuple[str, str, str]] = field(default_factory=frozenset)

    def observed_nodes(self) -> frozenset[str]:
        return self.add_nodes - self.rem_nodes

    def observed_edges(self) -> frozenset[Tuple[str, str, str]]:
        return self.add_edges - self.rem_edges


# ---------------------------------------------------------------------------
# MemoryMeshCore
# ---------------------------------------------------------------------------

class MemoryMeshCore:
    """2P2P-Graph CRDT with causal integrity enforcement.

    This class is the primary API for all memory operations.  It wraps
    a NetworkX DiGraph for O(V+E) traversal and maintains the four
    CRDT grow-only sets for distributed merge correctness.

    Parameters
    ----------
    namespace : str
        Namespace scope.  CRDT merge is only permitted within the
        same namespace (SYN-007).
    wal_path : str or Path, optional
        If provided, enables WAL persistence.  All mutations are
        recorded to this file before being committed.
        If None, the mesh operates in-memory only (Week 1 compat).
    decay_config : DecayConfig, optional
        Configuration for Ebbinghaus confidence decay and spaced
        repetition boost.  Uses PRD defaults if not specified.
    replay_wal : bool
        If True and wal_path exists, replay WAL on initialization
        to reconstruct state.  Default True.
    """

    def __init__(
        self,
        namespace: str = "default",
        wal_path: Optional[str | Path] = None,
        decay_config: Optional[DecayConfig] = None,
        replay_wal: bool = True,
    ) -> None:
        self.namespace = namespace

        # --- CRDT sets (mutable during local ops, frozen on snapshot) ---
        self._add_nodes: set[str] = set()
        self._rem_nodes: set[str] = set()
        self._add_edges: set[Tuple[str, str, str]] = set()
        self._rem_edges: set[Tuple[str, str, str]] = set()

        # --- Node storage ---
        self._nodes: Dict[str, MemoryNode] = {}

        # --- Causal graph (NetworkX for traversal + cycle detection) ---
        self._graph: nx.DiGraph = nx.DiGraph()

        # --- Global Lamport vector (incremented per agent per write) ---
        self._lamport: Dict[str, int] = {}

        # --- WAL persistence (TMP-004) ---
        self._wal: Optional[WriteAheadLog] = None
        if wal_path is not None:
            self._wal = WriteAheadLog(wal_path)

        # --- Temporal layer (TMP-005/006/007) ---
        self._decay_config = decay_config or DEFAULT_DECAY_CONFIG
        self._access_tracker = AccessTracker()

        # --- Replay WAL to reconstruct state ---
        if replay_wal and self._wal is not None:
            self._replay_from_wal()

    # =================================================================
    # Properties
    # =================================================================

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX DiGraph (read-only intent)."""
        return self._graph

    @property
    def node_count(self) -> int:
        """Number of observed (non-removed) nodes."""
        return len(self.observed_nodes())

    @property
    def edge_count(self) -> int:
        """Number of observed (non-removed) edges."""
        return len(self.observed_edges())

    # =================================================================
    # CRDT observed state (SYN-001)
    # =================================================================

    def observed_nodes(self) -> set[str]:
        """Return IDs of all non-removed nodes."""
        return self._add_nodes - self._rem_nodes

    def observed_edges(self) -> set[Tuple[str, str, str]]:
        """Return all non-removed (src, dst, label) edge tuples."""
        return self._add_edges - self._rem_edges

    # =================================================================
    # Write operations
    # =================================================================

    def _advance_lamport(self, agent_id: str) -> Dict[str, int]:
        """Increment this agent's Lamport component and return the vector."""
        self._lamport[agent_id] = self._lamport.get(agent_id, 0) + 1
        return dict(self._lamport)

    def write_memory(
        self,
        content: str,
        agent_id: str,
        confidence: float = 0.5,
        embedding: Optional[np.ndarray] = None,
        causal_edges: Optional[list[Tuple[str, str, str]]] = None,
        extra_provenance: Optional[Dict[str, Any]] = None,
    ) -> MemoryNode:
        """Write a memory node into the mesh (SYN-004, TMP-001).

        If a node with the same content hash already exists, the two
        nodes are merged idempotently (Lamport max, confidence max,
        edge union).

        Parameters
        ----------
        content : str
            The cognitive payload to store.
        agent_id : str
            Identifier of the writing agent.
        confidence : float
            Initial confidence R₀ ∈ [0, 1].
        embedding : np.ndarray, optional
            384-dim float32 embedding vector.
        causal_edges : list of (src, dst, label), optional
            Causal relationships to encode as edges.
        extra_provenance : dict, optional
            Additional trust metadata.

        Returns
        -------
        MemoryNode
            The (possibly merged) node now stored in the mesh.
        """
        # Advance Lamport clock
        tau = self._advance_lamport(agent_id)

        # Build node
        node = MemoryNode.create(
            content=content,
            agent_id=agent_id,
            confidence=confidence,
            namespace=self.namespace,
            embedding=embedding,
            lamport_vector=tau,
            extra_provenance=extra_provenance,
        )

        # Check 2P-set constraint: removed nodes cannot be re-added
        if node.id in self._rem_nodes:
            raise ValueError(
                "2P-set constraint: removed nodes cannot be re-added"
            )

        # --- WAL: record before commit (TMP-004) ---
        if self._wal is not None:
            self._wal.append(
                op=WALOp.WRITE_NODE,
                agent_id=agent_id,
                namespace=self.namespace,
                payload={
                    "content": content,
                    "confidence": confidence,
                    "lamport_vector": tau,
                    "node_id": node.id,
                },
            )

        # Dedup via content-addressed merge (SYN-004)
        existing = self._nodes.get(node.id)
        if existing is not None:
            node = existing.merge_with(node)

        # Store
        self._nodes[node.id] = node
        self._add_nodes.add(node.id)
        if node.id not in self._graph:
            self._graph.add_node(node.id)

        # Add causal edges if provided
        if causal_edges:
            for src, dst, label in causal_edges:
                try:
                    self.add_causal_edge(src, dst, label)
                except (CycleDetectedError, ValueError):
                    # Edge rejected but node write succeeds
                    pass

        return node

    def remove_node(self, node_id: str) -> None:
        """Tombstone a node (2P-set remove).

        The node is added to R_nodes; it remains in A_nodes for CRDT
        consistency but is excluded from observed state.
        """
        if node_id not in self._add_nodes:
            raise NodeNotFoundError(node_id)

        # --- WAL: record before commit ---
        if self._wal is not None:
            self._wal.append(
                op=WALOp.REMOVE_NODE,
                agent_id="__system__",
                namespace=self.namespace,
                payload={"node_id": node_id},
            )

        self._rem_nodes.add(node_id)
        if node_id in self._graph:
            self._graph.remove_node(node_id)

    # =================================================================
    # Causal edge operations (SYN-006)
    # =================================================================

    def add_causal_edge(
        self,
        source_id: str,
        target_id: str,
        label: str,
    ) -> None:
        """Add a labeled causal edge with cycle detection (SYN-006).

        The edge is tentatively inserted, then DFS checks whether
        the target can reach the source.  If so, the edge is rejected
        with a CycleDetectedError.

        Parameters
        ----------
        source_id : str
            Source node ID.
        target_id : str
            Destination node ID.
        label : str
            One of the allowed EdgeLabel values.

        Raises
        ------
        CycleDetectedError
            If the edge would create a cycle.
        NodeNotFoundError
            If either node is not in the observed set.
        ValueError
            If the label is not in ALLOWED_EDGE_LABELS.
        """
        # Validate label
        if label not in ALLOWED_EDGE_LABELS:
            raise ValueError(f"Invalid edge label: {label}")

        # Both nodes must be observed (not removed)
        obs = self.observed_nodes()
        if source_id not in obs:
            raise NodeNotFoundError(source_id)
        if target_id not in obs:
            raise NodeNotFoundError(target_id)

        # Same-node self-loop is always a cycle
        if source_id == target_id:
            raise CycleDetectedError(source_id, target_id, label)

        # DFS cycle check: can target already reach source?
        if self._graph.has_node(target_id) and self._graph.has_node(source_id):
            if nx.has_path(self._graph, target_id, source_id):
                raise CycleDetectedError(source_id, target_id, label)

        # --- WAL: record before commit ---
        if self._wal is not None:
            self._wal.append(
                op=WALOp.ADD_EDGE,
                agent_id="__system__",
                namespace=self.namespace,
                payload={
                    "source_id": source_id,
                    "target_id": target_id,
                    "label": label,
                },
            )

        # Commit edge
        edge_tuple = (source_id, target_id, label)
        self._add_edges.add(edge_tuple)
        self._graph.add_edge(source_id, target_id, label=label)

        # Update the source node's causal_edges frozenset
        src_node = self._nodes[source_id]
        updated = MemoryNode(
            id=src_node.id,
            embedding=src_node.embedding,
            lamport_vector=src_node.lamport_vector,
            causal_edges=src_node.causal_edges | {edge_tuple},
            confidence=src_node.confidence,
            trust_provenance=src_node.trust_provenance,
            content=src_node.content,
        )
        self._nodes[source_id] = updated

    def remove_causal_edge(
        self,
        source_id: str,
        target_id: str,
        label: str,
    ) -> None:
        """Tombstone a causal edge (2P-set remove)."""
        edge_tuple = (source_id, target_id, label)
        if edge_tuple not in self._add_edges:
            raise ValueError(f"Edge not found: {edge_tuple}")

        # --- WAL: record before commit ---
        if self._wal is not None:
            self._wal.append(
                op=WALOp.REMOVE_EDGE,
                agent_id="__system__",
                namespace=self.namespace,
                payload={
                    "source_id": source_id,
                    "target_id": target_id,
                    "label": label,
                },
            )

        self._rem_edges.add(edge_tuple)
        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)

    # =================================================================
    # Causal chain traversal (TMP-008)
    # =================================================================

    def causal_chain(
        self,
        node_id: str,
        depth: int = 3,
    ) -> nx.DiGraph:
        """Return the causal DAG rooted at node_id up to N hops (TMP-008).

        Performs BFS up to ``depth`` hops in both upstream (predecessors)
        and downstream (successors) directions.

        Parameters
        ----------
        node_id : str
            Root node for traversal.
        depth : int
            Maximum number of hops.

        Returns
        -------
        nx.DiGraph
            Subgraph containing all causally connected nodes within
            the specified depth, with edge labels and confidence scores.
        """
        if node_id not in self.observed_nodes():
            raise NodeNotFoundError(node_id)

        subgraph = nx.DiGraph()
        visited: set[str] = set()
        frontier: list[Tuple[str, int]] = [(node_id, 0)]

        while frontier:
            current, d = frontier.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)

            # Add node with confidence metadata
            node = self._nodes.get(current)
            conf = node.confidence if node else 0.0
            subgraph.add_node(current, confidence=conf)

            if d < depth:
                # Downstream (successors)
                for succ in self._graph.successors(current):
                    if succ in self.observed_nodes():
                        edge_data = self._graph.get_edge_data(current, succ)
                        lbl = edge_data.get("label", "") if edge_data else ""
                        subgraph.add_edge(current, succ, label=lbl)
                        frontier.append((succ, d + 1))

                # Upstream (predecessors)
                for pred in self._graph.predecessors(current):
                    if pred in self.observed_nodes():
                        edge_data = self._graph.get_edge_data(pred, current)
                        lbl = edge_data.get("label", "") if edge_data else ""
                        subgraph.add_edge(pred, current, label=lbl)
                        frontier.append((pred, d + 1))

        return subgraph

    # =================================================================
    # Node access
    # =================================================================

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory node by ID, or None if removed/missing.

        Note: this returns the raw node without decay applied.
        Use ``get_node_with_decay()`` for time-adjusted confidence.
        """
        if node_id in self._rem_nodes:
            return None
        node = self._nodes.get(node_id)
        if node is not None:
            self._access_tracker.record_access(node_id)
        return node

    def get_node_with_decay(
        self,
        node_id: str,
        now_utc: Optional[str] = None,
    ) -> Optional[Tuple[MemoryNode, float, bool]]:
        """Retrieve a node with time-decayed confidence (TMP-005/006/007).

        Parameters
        ----------
        node_id : str
            Node to retrieve.
        now_utc : str, optional
            Current time for decay calculation.

        Returns
        -------
        tuple[MemoryNode, float, bool] or None
            (node, effective_confidence, was_boosted)
            Returns None if node is removed or missing.
        """
        if node_id in self._rem_nodes:
            return None
        node = self._nodes.get(node_id)
        if node is None:
            return None

        # Record access (for spaced repetition tracking)
        self._access_tracker.record_access(node_id, now_utc)

        # Get creation timestamp from provenance
        created_utc = node.trust_provenance.get("wall_clock_utc")
        if created_utc is None:
            return (node, node.confidence, False)

        # Compute access rate for boost eligibility
        access_rate = self._access_tracker.access_rate_per_hour(
            node_id, now_utc
        )

        # Lazy decay + boost (TMP-006)
        effective_conf, was_boosted = compute_effective_confidence(
            initial_confidence=node.confidence,
            created_utc=created_utc,
            access_rate=access_rate,
            now_utc=now_utc,
            config=self._decay_config,
        )

        return (node, effective_conf, was_boosted)

    def get_all_nodes(self) -> list[MemoryNode]:
        """Return all observed (non-removed) memory nodes."""
        return [
            self._nodes[nid]
            for nid in self.observed_nodes()
            if nid in self._nodes
        ]

    @property
    def access_tracker(self) -> AccessTracker:
        """Access the read-frequency tracker (for testing/inspection)."""
        return self._access_tracker

    # =================================================================
    # CRDT merge (SYN-002, SYN-003, SYN-005)
    # =================================================================

    def merge_replicas(self, other: "MemoryMeshCore") -> "MemoryMeshCore":
        """Merge two CRDT replicas into a new instance (SYN-002).

        Guarantees:
            - Commutativity:  merge(A, B) == merge(B, A)
            - Associativity:  merge(A, merge(B, C)) == merge(merge(A, B), C)
            - Idempotency:    merge(A, A) == A

        Complexity: O(N + E) where N = total nodes, E = total edges.

        Parameters
        ----------
        other : MemoryMeshCore
            The replica to merge with.

        Returns
        -------
        MemoryMeshCore
            A new instance representing the merged state.

        Raises
        ------
        NamespaceMismatchError
            If the two replicas have different namespaces.
        """
        if self.namespace != other.namespace:
            raise NamespaceMismatchError(self.namespace, other.namespace)

        merged = MemoryMeshCore(namespace=self.namespace)

        # --- Merge Lamport vectors (TMP-002) ---
        merged._lamport = merge_lamport_vectors(self._lamport, other._lamport)

        # --- Union CRDT sets (O(N + E)) ---
        merged._add_nodes = self._add_nodes | other._add_nodes
        merged._rem_nodes = self._rem_nodes | other._rem_nodes
        merged._add_edges = self._add_edges | other._add_edges
        merged._rem_edges = self._rem_edges | other._rem_edges

        # --- Merge node payloads ---
        all_ids = set(self._nodes) | set(other._nodes)
        for nid in all_ids:
            left = self._nodes.get(nid)
            right = other._nodes.get(nid)
            if left is None:
                merged._nodes[nid] = right  # type: ignore[assignment]
            elif right is None:
                merged._nodes[nid] = left
            else:
                merged._nodes[nid] = left.merge_with(right)

        # --- Rebuild NetworkX graph from observed state ---
        for nid in merged.observed_nodes():
            merged._graph.add_node(nid)

        for src, dst, label in merged.observed_edges():
            if (
                src in merged.observed_nodes()
                and dst in merged.observed_nodes()
            ):
                # Skip edges that would create cycles
                if merged._graph.has_node(dst) and merged._graph.has_node(src):
                    if nx.has_path(merged._graph, dst, src):
                        continue
                merged._graph.add_edge(src, dst, label=label)

        return merged

    # =================================================================
    # Snapshot (for testing / comparison)
    # =================================================================

    def snapshot(self) -> Dict[str, Any]:
        """Return a deterministic snapshot of the CRDT state.

        Used primarily for testing CRDT properties (commutativity,
        idempotency) by comparing snapshot dicts.
        """
        node_snap: Dict[str, Any] = {}
        for nid in sorted(self._nodes):
            node = self._nodes[nid]
            node_snap[nid] = {
                "id": node.id,
                "embedding_norm": float(np.linalg.norm(node.embedding)),
                "lamport_vector": dict(sorted(node.lamport_vector.items())),
                "causal_edges": sorted(node.causal_edges),
                "confidence": node.confidence,
                "content": node.content,
            }

        return {
            "namespace": self.namespace,
            "add_nodes": sorted(self._add_nodes),
            "rem_nodes": sorted(self._rem_nodes),
            "add_edges": sorted(self._add_edges),
            "rem_edges": sorted(self._rem_edges),
            "lamport": dict(sorted(self._lamport.items())),
            "nodes": node_snap,
        }

    def crdt_state(self) -> CRDTState:
        """Return the raw CRDT quadruple as frozen sets."""
        return CRDTState(
            add_nodes=frozenset(self._add_nodes),
            rem_nodes=frozenset(self._rem_nodes),
            add_edges=frozenset(self._add_edges),
            rem_edges=frozenset(self._rem_edges),
        )

    # =================================================================
    # WAL replay (TMP-004)
    # =================================================================

    def _replay_from_wal(self) -> None:
        """Replay all WAL entries to reconstruct in-memory state.

        Called during __init__ when a WAL file exists.
        Temporarily disables WAL recording to avoid re-logging
        entries that are already persisted.
        """
        if self._wal is None:
            return

        wal_ref = self._wal
        self._wal = None  # Disable WAL during replay

        try:
            for entry in wal_ref.replay():
                if entry.namespace != self.namespace:
                    continue
                self._apply_wal_entry(entry)
        finally:
            self._wal = wal_ref  # Re-enable WAL

    def _apply_wal_entry(self, entry) -> None:
        """Apply a single WAL entry to the in-memory state."""
        p = entry.payload

        if entry.op == WALOp.WRITE_NODE:
            self.write_memory(
                content=p["content"],
                agent_id=entry.agent_id,
                confidence=p["confidence"],
            )

        elif entry.op == WALOp.REMOVE_NODE:
            node_id = p["node_id"]
            if node_id in self._add_nodes and node_id not in self._rem_nodes:
                self.remove_node(node_id)

        elif entry.op == WALOp.ADD_EDGE:
            src, dst, label = p["source_id"], p["target_id"], p["label"]
            if (
                src in self.observed_nodes()
                and dst in self.observed_nodes()
            ):
                try:
                    self.add_causal_edge(src, dst, label)
                except (CycleDetectedError, ValueError, NodeNotFoundError):
                    pass  # Edge may conflict after replay ordering

        elif entry.op == WALOp.REMOVE_EDGE:
            edge = (p["source_id"], p["target_id"], p["label"])
            if edge in self._add_edges and edge not in self._rem_edges:
                self.remove_causal_edge(*edge)

        # SNAPSHOT entries are handled by query_at, not replay

    # =================================================================
    # Point-in-time queries (TMP-003)
    # =================================================================

    def query_at(
        self,
        timestamp_utc: str,
    ) -> "MemoryMeshCore":
        """Reconstruct the graph state at a specific point in time (TMP-003).

        Replays the WAL up to ``timestamp_utc``, building a fresh
        MemoryMeshCore representing the historical belief state.

        Parameters
        ----------
        timestamp_utc : str
            ISO 8601 UTC timestamp.  The returned graph reflects
            exactly the state at this moment.

        Returns
        -------
        MemoryMeshCore
            A new in-memory instance (no WAL) representing historical state.

        Raises
        ------
        ValueError
            If no WAL is configured (in-memory-only mode).
        """
        if self._wal is None:
            raise ValueError(
                "query_at() requires WAL persistence. "
                "Initialize with wal_path to enable time-travel queries."
            )

        # Build a fresh mesh by replaying WAL up to the target time
        historical = MemoryMeshCore(
            namespace=self.namespace,
            wal_path=None,  # No WAL for the historical view
            decay_config=self._decay_config,
            replay_wal=False,
        )

        for entry in self._wal.replay(up_to=timestamp_utc):
            if entry.namespace != self.namespace:
                continue
            historical._apply_wal_entry(entry)

        return historical

    # =================================================================
    # WAL snapshot + compaction (TMP-009)
    # =================================================================

    def create_snapshot(self) -> None:
        """Write a snapshot to the WAL (TMP-009).

        After a snapshot, older WAL entries can be compacted.
        """
        if self._wal is None:
            return
        self._wal.write_snapshot(
            namespace=self.namespace,
            snapshot_data=self.snapshot(),
        )

    def compact_wal(self) -> int:
        """Compact the WAL by removing entries before the latest snapshot.

        Returns
        -------
        int
            Number of entries removed.
        """
        if self._wal is None:
            return 0
        snap = self._wal.find_latest_snapshot()
        if snap is None:
            return 0
        return self._wal.compact(snap)

    @property
    def wal(self) -> Optional[WriteAheadLog]:
        """Access the WAL instance (for testing/inspection)."""
        return self._wal

    def __repr__(self) -> str:
        wal_info = f", wal={self._wal.path}" if self._wal else ""
        return (
            f"MemoryMeshCore(namespace='{self.namespace}', "
            f"nodes={self.node_count}, edges={self.edge_count}{wal_info})"
        )

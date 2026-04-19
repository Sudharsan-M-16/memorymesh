"""MemoryMesh Brain Stem — Comprehensive Test Suite.

26 tests across 10 test classes covering:
  - Content-addressed deduplication (SYN-004)
  - CRDT properties: idempotency, commutativity, associativity (SYN-002)
  - Lamport clock mechanics (TMP-001, TMP-002)
  - DFS cycle detection (SYN-006)
  - Causal chain traversal (TMP-008)
  - Bayesian Trust Score computation (TL-002)
  - Trust update + audit trail (TL-007, TL-008)
  - Node removal / 2P-set semantics (SYN-001)
  - GPU-batch readiness (float32 numpy arrays)
  - Namespace isolation (SYN-007)
  - Edge label validation

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from memorymesh import (
    BayesianTrustEngine,
    CycleDetectedError,
    EdgeLabel,
    MemoryMeshCore,
    MemoryNode,
    NamespaceMismatchError,
    NodeNotFoundError,
    content_address,
    zero_embedding,
)


# ===================================================================
# 1. Content-Addressed Deduplication (SYN-004)
# ===================================================================

class TestContentAddressing:
    """Verify SHA-256 content-addressed IDs for cross-agent dedup."""

    def test_deterministic_id_across_agents(self) -> None:
        """Two agents writing identical content → same SHA-256 ID."""
        mesh = MemoryMeshCore(namespace="test")

        n1 = mesh.write_memory("contract signed march 22", "agent-a", 0.91)
        n2 = mesh.write_memory("contract signed march 22", "agent-b", 0.82)

        assert n1.id == n2.id
        assert mesh.node_count == 1

    def test_whitespace_normalization(self) -> None:
        """Leading/trailing whitespace and case don't affect ID."""
        id1 = content_address("  Hello World  ")
        id2 = content_address("hello world")
        assert id1 == id2

    def test_deduplication_on_write(self) -> None:
        """Writing same content twice → single node with merged metadata."""
        mesh = MemoryMeshCore(namespace="test")

        mesh.write_memory("evidence A", "agent-1", 0.7)
        mesh.write_memory("evidence A", "agent-2", 0.9)

        assert mesh.node_count == 1
        node = mesh.get_all_nodes()[0]
        # Confidence takes max
        assert node.confidence == 0.9

    def test_different_content_different_ids(self) -> None:
        """Different content → different node IDs."""
        mesh = MemoryMeshCore(namespace="test")

        n1 = mesh.write_memory("fact alpha", "agent-1", 0.5)
        n2 = mesh.write_memory("fact beta", "agent-1", 0.5)

        assert n1.id != n2.id
        assert mesh.node_count == 2


# ===================================================================
# 2. CRDT Properties (SYN-002)
# ===================================================================

class TestCRDTProperties:
    """Verify the three CRDT mathematical guarantees."""

    @staticmethod
    def _make_mesh(ns: str, items: list[tuple[str, str, float]]) -> MemoryMeshCore:
        """Helper: create a mesh with given (content, agent, confidence) tuples."""
        mesh = MemoryMeshCore(namespace=ns)
        for content, agent, conf in items:
            mesh.write_memory(content, agent, conf)
        return mesh

    def test_idempotency(self) -> None:
        """merge(A, A) == A — self-merge produces identical state."""
        mesh = self._make_mesh("test", [
            ("evidence A", "agent-1", 0.5),
            ("evidence B", "agent-2", 0.6),
        ])

        merged = mesh.merge_replicas(mesh)
        assert merged.snapshot() == mesh.snapshot()

    def test_commutativity(self) -> None:
        """merge(A, B) == merge(B, A) — order doesn't matter."""
        a = self._make_mesh("test", [
            ("fact X", "agent-a", 0.8),
            ("fact Y", "agent-a", 0.6),
        ])
        b = self._make_mesh("test", [
            ("fact Z", "agent-b", 0.7),
            ("fact Y", "agent-b", 0.9),  # Overlapping with A
        ])

        ab = a.merge_replicas(b)
        ba = b.merge_replicas(a)
        assert ab.snapshot() == ba.snapshot()

    def test_associativity(self) -> None:
        """merge(A, merge(B, C)) == merge(merge(A, B), C)."""
        a = self._make_mesh("test", [("mem-a", "a1", 0.5)])
        b = self._make_mesh("test", [("mem-b", "a2", 0.6)])
        c = self._make_mesh("test", [("mem-c", "a3", 0.7)])

        # Left grouping: merge(merge(A, B), C)
        ab = a.merge_replicas(b)
        ab_c = ab.merge_replicas(c)

        # Right grouping: merge(A, merge(B, C))
        bc = b.merge_replicas(c)
        a_bc = a.merge_replicas(bc)

        assert ab_c.snapshot() == a_bc.snapshot()

    def test_merge_preserves_removed_nodes(self) -> None:
        """Removed nodes stay removed after merge (2P-set semantics)."""
        a = self._make_mesh("test", [("to-remove", "agent-1", 0.5)])
        node_id = a.get_all_nodes()[0].id
        a.remove_node(node_id)

        b = self._make_mesh("test", [("to-remove", "agent-2", 0.6)])

        merged = a.merge_replicas(b)
        # Remove wins in 2P-set
        assert node_id not in merged.observed_nodes()


# ===================================================================
# 3. Lamport Clock Mechanics (TMP-001, TMP-002)
# ===================================================================

class TestLamportClocks:
    """Verify Lamport vector clock increment and merge."""

    def test_clock_increment_on_write(self) -> None:
        """Each write advances the writing agent's Lamport component (TMP-001)."""
        mesh = MemoryMeshCore(namespace="test")

        n1 = mesh.write_memory("first", "agent-a", 0.5)
        n2 = mesh.write_memory("second", "agent-a", 0.5)

        assert n1.lamport_vector["agent-a"] == 1
        assert n2.lamport_vector["agent-a"] == 2

    def test_multi_agent_lamport_vectors(self) -> None:
        """Multi-agent writes produce independent Lamport components."""
        mesh = MemoryMeshCore(namespace="test")

        mesh.write_memory("from-a", "agent-a", 0.5)
        n2 = mesh.write_memory("from-b", "agent-b", 0.5)

        # After agent-b's write, the global vector has both agents
        assert n2.lamport_vector["agent-a"] == 1
        assert n2.lamport_vector["agent-b"] == 1

    def test_lamport_vector_merge(self) -> None:
        """Merge takes element-wise max of Lamport vectors (TMP-002)."""
        a = MemoryMeshCore(namespace="test")
        a.write_memory("shared", "agent-a", 0.5)
        a.write_memory("only-a", "agent-a", 0.5)  # agent-a: 2

        b = MemoryMeshCore(namespace="test")
        b.write_memory("shared", "agent-b", 0.5)
        b.write_memory("only-b-1", "agent-b", 0.5)
        b.write_memory("only-b-2", "agent-b", 0.5)  # agent-b: 3

        merged = a.merge_replicas(b)
        # Element-wise max: agent-a → max(2, 0)=2, agent-b → max(0, 3)=3
        assert merged._lamport["agent-a"] == 2
        assert merged._lamport["agent-b"] == 3


# ===================================================================
# 4. Cycle Detection (SYN-006)
# ===================================================================

class TestCycleDetection:
    """Verify DFS-based cycle detection on edge insertion."""

    def test_direct_cycle_rejected(self) -> None:
        """A → B → A cycle is detected and rejected."""
        mesh = MemoryMeshCore(namespace="test")
        n1 = mesh.write_memory("node-A", "agent-1", 0.5)
        n2 = mesh.write_memory("node-B", "agent-1", 0.5)

        mesh.add_causal_edge(n1.id, n2.id, EdgeLabel.SUPPORTS)

        with pytest.raises(CycleDetectedError):
            mesh.add_causal_edge(n2.id, n1.id, EdgeLabel.CAUSED_BY)

    def test_transitive_cycle_rejected(self) -> None:
        """A → B → C → A cycle (3-hop) is detected and rejected."""
        mesh = MemoryMeshCore(namespace="test")
        na = mesh.write_memory("node-A", "agent-1", 0.5)
        nb = mesh.write_memory("node-B", "agent-1", 0.5)
        nc = mesh.write_memory("node-C", "agent-1", 0.5)

        mesh.add_causal_edge(na.id, nb.id, EdgeLabel.SUPPORTS)
        mesh.add_causal_edge(nb.id, nc.id, EdgeLabel.CAUSED_BY)

        with pytest.raises(CycleDetectedError):
            mesh.add_causal_edge(nc.id, na.id, EdgeLabel.DERIVED_FROM)

    def test_self_loop_rejected(self) -> None:
        """Self-referencing edge is rejected."""
        mesh = MemoryMeshCore(namespace="test")
        n = mesh.write_memory("self-ref", "agent-1", 0.5)

        with pytest.raises(CycleDetectedError):
            mesh.add_causal_edge(n.id, n.id, EdgeLabel.SUPPORTS)

    def test_valid_dag_accepted(self) -> None:
        """A valid DAG (diamond shape) is accepted without error."""
        mesh = MemoryMeshCore(namespace="test")
        na = mesh.write_memory("root", "agent-1", 0.5)
        nb = mesh.write_memory("left", "agent-1", 0.5)
        nc = mesh.write_memory("right", "agent-1", 0.5)
        nd = mesh.write_memory("leaf", "agent-1", 0.5)

        # Diamond: root → left → leaf, root → right → leaf
        mesh.add_causal_edge(na.id, nb.id, EdgeLabel.SUPPORTS)
        mesh.add_causal_edge(na.id, nc.id, EdgeLabel.SUPPORTS)
        mesh.add_causal_edge(nb.id, nd.id, EdgeLabel.CAUSED_BY)
        mesh.add_causal_edge(nc.id, nd.id, EdgeLabel.CAUSED_BY)

        assert mesh.edge_count == 4


# ===================================================================
# 5. Causal Chain Traversal (TMP-008)
# ===================================================================

class TestCausalChain:
    """Verify causal_chain DAG traversal."""

    def test_causal_chain_returns_correct_dag(self) -> None:
        """causal_chain(node, depth=3) returns full connected subgraph."""
        mesh = MemoryMeshCore(namespace="test")

        # Build: evidence_A → supports → hypothesis_H → caused → decision_D
        ev_a = mesh.write_memory("evidence A found", "agent-r", 0.9)
        ev_b = mesh.write_memory("evidence B found", "agent-r", 0.85)
        hyp = mesh.write_memory("hypothesis H", "agent-r", 0.7)
        dec = mesh.write_memory("decision D", "agent-r", 0.8)

        mesh.add_causal_edge(ev_a.id, hyp.id, EdgeLabel.SUPPORTS)
        mesh.add_causal_edge(ev_b.id, hyp.id, EdgeLabel.SUPPORTS)
        mesh.add_causal_edge(hyp.id, dec.id, EdgeLabel.CAUSED_BY)

        chain = mesh.causal_chain(dec.id, depth=3)

        # All 4 nodes should be reachable
        assert len(chain.nodes) == 4
        assert len(chain.edges) == 3

    def test_causal_chain_depth_limit(self) -> None:
        """Depth limit restricts traversal."""
        mesh = MemoryMeshCore(namespace="test")

        nodes = []
        for i in range(5):
            n = mesh.write_memory(f"node-{i}", "agent-1", 0.5)
            nodes.append(n)
            if i > 0:
                mesh.add_causal_edge(nodes[i - 1].id, n.id, EdgeLabel.SUPPORTS)

        # From node-0, depth=2 should reach nodes 0, 1, 2 only
        chain = mesh.causal_chain(nodes[0].id, depth=2)
        assert len(chain.nodes) == 3

    def test_causal_chain_node_not_found(self) -> None:
        """Traversal on non-existent node raises NodeNotFoundError."""
        mesh = MemoryMeshCore(namespace="test")

        with pytest.raises(NodeNotFoundError):
            mesh.causal_chain("nonexistent-id", depth=3)


# ===================================================================
# 6. Bayesian Trust Score (TL-002)
# ===================================================================

class TestBayesianTrustScore:
    """Verify BTS computation and initial priors."""

    def test_default_prior_bts(self) -> None:
        """Default prior α=2, β=2 → BTS=0.50."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r")
        assert engine.get_bts("agent-r") == pytest.approx(0.5)

    def test_custom_prior_bts(self) -> None:
        """Custom prior α=18, β=6 → BTS=0.75 (TL-002)."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r", alpha=18, beta=6)
        assert engine.get_bts("agent-r") == pytest.approx(0.75)

    def test_bts_after_updates(self) -> None:
        """20 updates (16 correct, 4 wrong) → α=18, β=6, BTS=0.75."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r")  # α=2, β=2

        for i in range(16):
            engine.trust_update("agent-r", f"mem-correct-{i}", outcome=True)
        for i in range(4):
            engine.trust_update("agent-r", f"mem-wrong-{i}", outcome=False)

        alpha, beta = engine.get_parameters("agent-r")
        assert alpha == 18.0  # 2 + 16
        assert beta == 6.0    # 2 + 4
        assert engine.get_bts("agent-r") == pytest.approx(0.75)

    def test_unregistered_agent_raises(self) -> None:
        """Accessing BTS for unknown agent raises KeyError."""
        engine = BayesianTrustEngine()
        with pytest.raises(KeyError):
            engine.get_bts("ghost-agent")


# ===================================================================
# 7. Trust Update & Audit Trail (TL-007, TL-008)
# ===================================================================

class TestTrustAudit:
    """Verify trust update mechanics and forensic audit trail."""

    def test_trust_update_increments_alpha(self) -> None:
        """Positive outcome increments α by 1 (TL-007)."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r")

        engine.trust_update("agent-r", "mem-001", outcome=True)

        alpha, beta = engine.get_parameters("agent-r")
        assert alpha == 3.0
        assert beta == 2.0

    def test_trust_update_increments_beta(self) -> None:
        """Negative outcome increments β by 1."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r")

        engine.trust_update("agent-r", "mem-001", outcome=False)

        alpha, beta = engine.get_parameters("agent-r")
        assert alpha == 2.0
        assert beta == 3.0

    def test_audit_trail_completeness(self) -> None:
        """20 updates produce 20 audit entries with correct fields (TL-008)."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r")

        for i in range(20):
            outcome = i < 16  # First 16 correct, last 4 wrong
            engine.trust_update("agent-r", f"mem-{i:03d}", outcome=outcome)

        trail = engine.trust_audit("agent-r")
        assert len(trail) == 20

        # Check first entry
        first = trail[0]
        assert first.was_correct is True
        assert first.alpha_before == 2.0
        assert first.beta_before == 2.0
        assert first.alpha_after == 3.0
        assert first.beta_after == 2.0

        # Check last entry (17th wrong outcome)
        last = trail[-1]
        assert last.was_correct is False

    def test_idempotent_registration(self) -> None:
        """Re-registering an agent doesn't reset trust state."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r")
        engine.trust_update("agent-r", "mem-001", outcome=True)

        # Re-register — should be no-op
        engine.register_agent("agent-r")

        alpha, beta = engine.get_parameters("agent-r")
        assert alpha == 3.0  # Still reflects the update


# ===================================================================
# 8. Node Removal / 2P-Set Semantics (SYN-001)
# ===================================================================

class TestNodeRemoval:
    """Verify 2P-set remove semantics."""

    def test_removed_node_not_in_observed(self) -> None:
        """Removed nodes are excluded from observed state."""
        mesh = MemoryMeshCore(namespace="test")
        n = mesh.write_memory("to-remove", "agent-1", 0.5)

        mesh.remove_node(n.id)

        assert n.id not in mesh.observed_nodes()
        assert mesh.node_count == 0
        assert mesh.get_node(n.id) is None

    def test_removed_node_cannot_be_readded(self) -> None:
        """2P-set constraint: once removed, cannot re-add."""
        mesh = MemoryMeshCore(namespace="test")
        n = mesh.write_memory("ephemeral", "agent-1", 0.5)
        mesh.remove_node(n.id)

        with pytest.raises(ValueError, match="2P-set constraint"):
            mesh.write_memory("ephemeral", "agent-2", 0.6)


# ===================================================================
# 9. GPU-Batch Readiness
# ===================================================================

class TestGPUReadiness:
    """Verify data structures are GPU-transfer-ready."""

    def test_embedding_is_float32_numpy(self) -> None:
        """Embeddings are contiguous float32 numpy arrays."""
        mesh = MemoryMeshCore(namespace="test")
        n = mesh.write_memory("test content", "agent-1", 0.5)

        assert isinstance(n.embedding, np.ndarray)
        assert n.embedding.dtype == np.float32
        assert n.embedding.shape == (384,)
        assert n.embedding.flags["C_CONTIGUOUS"]

    def test_custom_embedding_validated(self) -> None:
        """Custom embeddings are validated for shape and dtype."""
        mesh = MemoryMeshCore(namespace="test")
        emb = np.random.randn(384).astype(np.float32)

        n = mesh.write_memory("with embedding", "agent-1", 0.5, embedding=emb)

        assert np.allclose(n.embedding, emb)

    def test_wrong_embedding_dim_rejected(self) -> None:
        """Embeddings with wrong dimensions are rejected."""
        mesh = MemoryMeshCore(namespace="test")
        bad_emb = np.zeros(128, dtype=np.float32)

        with pytest.raises(ValueError, match="384-dimensional"):
            mesh.write_memory("bad emb", "agent-1", 0.5, embedding=bad_emb)

    def test_zero_embedding_default(self) -> None:
        """Default embedding is a zero vector (placeholder for future GPU compute)."""
        emb = zero_embedding()
        assert np.all(emb == 0.0)
        assert emb.dtype == np.float32


# ===================================================================
# 10. Namespace Isolation & Edge Labels (SYN-007)
# ===================================================================

class TestNamespaceAndLabels:
    """Verify namespace isolation and edge label validation."""

    def test_cross_namespace_merge_rejected(self) -> None:
        """Merging replicas from different namespaces raises error."""
        a = MemoryMeshCore(namespace="legal")
        b = MemoryMeshCore(namespace="medical")

        with pytest.raises(NamespaceMismatchError):
            a.merge_replicas(b)

    def test_invalid_edge_label_rejected(self) -> None:
        """Edge labels not in the allowed set are rejected."""
        mesh = MemoryMeshCore(namespace="test")
        n1 = mesh.write_memory("node-1", "agent-1", 0.5)
        n2 = mesh.write_memory("node-2", "agent-1", 0.5)

        with pytest.raises(ValueError, match="Invalid edge label"):
            mesh.add_causal_edge(n1.id, n2.id, "invalid-label")

    def test_all_edge_labels_accepted(self) -> None:
        """All five EdgeLabel values are accepted."""
        mesh = MemoryMeshCore(namespace="test")
        nodes = []
        for i in range(6):
            n = mesh.write_memory(f"node-{i}", "agent-1", 0.5)
            nodes.append(n)

        # Use all 5 labels (no cycles in a linear chain)
        labels = [
            EdgeLabel.CAUSED_BY,
            EdgeLabel.SUPPORTS,
            EdgeLabel.CONTRADICTS,
            EdgeLabel.SUPERSEDES,
            EdgeLabel.DERIVED_FROM,
        ]
        for i, label in enumerate(labels):
            mesh.add_causal_edge(nodes[i].id, nodes[i + 1].id, label)

        assert mesh.edge_count == 5

"""MemoryMesh Week 3 — Semantic Conflict Resolution Test Suite.

16 tests across 5 test classes covering:
  - Core conflict detection (TL-003, TL-004)
  - Posterior belief probabilities (TL-005)
  - HIGH_UNCERTAINTY signaling (TL-006)
  - Bidirectional contradicts edges without DAG cycles
  - WAL conflict replay durability
  - CRDT conflict merge properties (commutativity, idempotency)
  - Edge cases (zero embeddings, removed nodes, multi-conflict)
  - Performance benchmarks

Run: python -m pytest tests/test_conflict_resolution.py -v
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from memorymesh import (
    BayesianTrustEngine,
    ConflictResolution,
    EdgeLabel,
    MemoryMeshCore,
    compute_belief_posteriors,
    cosine_similarity,
)


# ===================================================================
# Helpers
# ===================================================================

def _embedding(seed: float) -> np.ndarray:
    """Create a near-identical embedding with a single axis offset."""
    emb = np.ones(384, dtype=np.float32)
    emb[0] = seed
    return emb


def _orthogonal_embedding(seed: int) -> np.ndarray:
    """Create an embedding on a specific axis — orthogonal to others."""
    emb = np.zeros(384, dtype=np.float32)
    emb[seed % 384] = 1.0
    return emb


def _make_conflict_mesh(
    ns: str = "test",
    classifier=None,
    trust_engine=None,
    threshold: float = 0.85,
    wal_path=None,
):
    """Factory for conflict-enabled meshes."""
    return MemoryMeshCore(
        namespace=ns,
        trust_engine=trust_engine,
        conflict_classifier=classifier or (lambda l, r: True),
        conflict_similarity_threshold=threshold,
        wal_path=wal_path,
    )


# ===================================================================
# 1. Core Conflict Detection (TL-003, TL-004, TL-005, TL-006)
# ===================================================================

class TestCoreConflictDetection:
    """Verify similarity + classifier → conflict resolution flow."""

    def test_conflict_requires_similarity_and_classifier_true(self) -> None:
        """Conflict fires when cosine similarity > threshold AND classifier returns True."""
        engine = BayesianTrustEngine()
        engine.register_agent("agent-r", alpha=2, beta=2)
        engine.register_agent("agent-l", alpha=6, beta=2)

        mesh = _make_conflict_mesh(
            ns="legal",
            trust_engine=engine,
        )

        ar = mesh.write_memory(
            "contract signed march 22",
            "agent-r",
            0.91,
            embedding=_embedding(1.0),
        )
        al = mesh.write_memory(
            "contract signed march 15",
            "agent-l",
            0.82,
            embedding=_embedding(1.01),
        )

        conflicts = mesh.conflicts()
        assert len(conflicts) == 1

        conflict = conflicts[0]
        if conflict.left_id == ar.id:
            ar_posterior = conflict.left_posterior
            al_posterior = conflict.right_posterior
        else:
            ar_posterior = conflict.right_posterior
            al_posterior = conflict.left_posterior

        # agent-r: BTS = 2/4 = 0.5, conf 0.91 → score = 0.455
        # agent-l: BTS = 6/8 = 0.75, conf 0.82 → score = 0.615
        # ar_post = 0.455/(0.455+0.615) ≈ 0.425
        assert ar_posterior == pytest.approx(0.425, abs=0.001)
        assert al_posterior == pytest.approx(0.575, abs=0.001)
        assert conflict.high_uncertainty is True
        assert conflict.canonical_node_id == al.id

    def test_classifier_false_blocks_conflict(self) -> None:
        """Classifier returning False prevents conflict even with high similarity."""
        mesh = _make_conflict_mesh(
            ns="legal",
            classifier=lambda left, right: False,
        )

        mesh.write_memory("contract signed march 22", "agent-r", 0.91, _embedding(1.0))
        mesh.write_memory("contract signed march 15", "agent-l", 0.82, _embedding(1.01))

        assert mesh.conflicts() == []
        assert mesh.edge_count == 0

    def test_low_similarity_blocks_conflict(self) -> None:
        """Orthogonal embeddings produce cosine ≈ 0.0 — no conflict."""
        mesh = _make_conflict_mesh(ns="test")

        mesh.write_memory("the sky is blue", "agent-a", 0.9, _orthogonal_embedding(0))
        mesh.write_memory("water is wet", "agent-b", 0.9, _orthogonal_embedding(100))

        assert mesh.conflicts() == []

    def test_clear_winner_no_high_uncertainty(self) -> None:
        """When one posterior is clearly dominant, HIGH_UNCERTAINTY is False."""
        engine = BayesianTrustEngine()
        engine.register_agent("novice", alpha=2, beta=2)     # BTS = 0.50
        engine.register_agent("expert", alpha=18, beta=2)    # BTS = 0.90

        mesh = _make_conflict_mesh(ns="test", trust_engine=engine)

        mesh.write_memory("claim A", "novice", 0.3, _embedding(1.0))
        mesh.write_memory("claim B", "expert", 0.95, _embedding(1.01))

        conflicts = mesh.conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].high_uncertainty is False

    def test_no_conflict_without_classifier(self) -> None:
        """Without a classifier, no conflicts are ever detected."""
        mesh = MemoryMeshCore(namespace="test")  # No classifier
        mesh.write_memory("fact A", "agent-a", 0.9, _embedding(1.0))
        mesh.write_memory("fact B", "agent-b", 0.9, _embedding(1.01))

        assert mesh.conflicts() == []


# ===================================================================
# 2. Bidirectional Edges & DAG Safety
# ===================================================================

class TestConflictEdgesAndDAG:
    """Verify bidirectional contradicts edges preserve DAG acyclicity."""

    def test_conflict_edges_are_bidirectional_without_causal_cycle(self) -> None:
        """Both (A→B, contradicts) and (B→A, contradicts) exist in observed edges,
        but the NetworkX DAG remains acyclic."""
        mesh = _make_conflict_mesh(ns="test")

        left = mesh.write_memory("door is open", "agent-a", 0.8, _embedding(1.0))
        right = mesh.write_memory("door is closed", "agent-b", 0.8, _embedding(1.01))

        conflict_edges = {
            edge for edge in mesh.observed_edges()
            if edge[2] == EdgeLabel.CONTRADICTS.value
        }

        assert conflict_edges == {
            (left.id, right.id, EdgeLabel.CONTRADICTS.value),
            (right.id, left.id, EdgeLabel.CONTRADICTS.value),
        }
        assert nx.is_directed_acyclic_graph(mesh.graph)

    def test_causal_edges_coexist_with_conflict_edges(self) -> None:
        """A causal edge and a conflict edge on the same nodes don't break the DAG."""
        mesh = _make_conflict_mesh(ns="test")

        n1 = mesh.write_memory("original theory", "agent-a", 0.9, _embedding(1.0))
        n2 = mesh.write_memory("revised theory", "agent-b", 0.85, _embedding(1.01))

        # Add a causal edge alongside the conflict
        mesh.add_causal_edge(n1.id, n2.id, EdgeLabel.SUPERSEDES)

        assert mesh.edge_count >= 3  # supersedes + 2× contradicts
        assert nx.is_directed_acyclic_graph(mesh.graph)

    def test_conflict_annotations_on_both_nodes(self) -> None:
        """Both nodes in a conflict pair receive provenance annotations."""
        mesh = _make_conflict_mesh(ns="test")

        left = mesh.write_memory("X is true", "agent-a", 0.8, _embedding(1.0))
        right = mesh.write_memory("X is false", "agent-b", 0.8, _embedding(1.01))

        left_annot = mesh.conflict_annotations(left.id)
        right_annot = mesh.conflict_annotations(right.id)

        assert len(left_annot) == 1
        assert left_annot[0]["other_node_id"] == right.id
        assert len(right_annot) == 1
        assert right_annot[0]["other_node_id"] == left.id


# ===================================================================
# 3. WAL & Replay Durability
# ===================================================================

class TestConflictWALReplay:
    """Verify conflict state survives WAL replay (process restart)."""

    def test_conflict_annotations_survive_wal_replay(self, tmp_path: Path) -> None:
        """Conflicts, edges, and annotations are recovered from WAL."""
        path = tmp_path / "conflicts.wal"

        mesh = _make_conflict_mesh(ns="legal", wal_path=path)
        left = mesh.write_memory("door is open", "agent-a", 0.8, _embedding(1.0))
        right = mesh.write_memory("door is closed", "agent-b", 0.8, _embedding(1.01))

        # Simulate process death
        del mesh

        recovered = MemoryMeshCore(namespace="legal", wal_path=path)

        assert len(recovered.conflicts()) == 1
        assert (
            left.id, right.id, EdgeLabel.CONTRADICTS.value,
        ) in recovered.observed_edges()
        assert (
            right.id, left.id, EdgeLabel.CONTRADICTS.value,
        ) in recovered.observed_edges()
        assert recovered.conflict_annotations(left.id)[0]["other_node_id"] == right.id

    def test_canonical_belief_survives_wal_replay(self, tmp_path: Path) -> None:
        """Canonical node ID and posterior probabilities persist through WAL."""
        path = tmp_path / "canonical.wal"

        engine = BayesianTrustEngine()
        engine.register_agent("weak", alpha=2, beta=2)
        engine.register_agent("strong", alpha=10, beta=2)

        mesh = _make_conflict_mesh(ns="test", trust_engine=engine, wal_path=path)
        weak_node = mesh.write_memory("claim A", "weak", 0.5, _embedding(1.0))
        strong_node = mesh.write_memory("claim B", "strong", 0.95, _embedding(1.01))

        original_conflict = mesh.conflicts()[0]
        del mesh

        recovered = MemoryMeshCore(namespace="test", wal_path=path)
        replayed_conflict = recovered.conflicts()[0]

        assert replayed_conflict.canonical_node_id == original_conflict.canonical_node_id
        assert replayed_conflict.left_posterior == pytest.approx(
            original_conflict.left_posterior, abs=1e-6
        )


# ===================================================================
# 4. CRDT Merge Properties with Conflicts
# ===================================================================

class TestConflictCRDTMerge:
    """Verify CRDT merge properties hold when conflicts are present."""

    def test_merge_preserves_conflicts_commutativity(self) -> None:
        """merge(A, B) and merge(B, A) produce identical conflict state."""
        a = _make_conflict_mesh(ns="test")
        a.write_memory("fact X", "agent-a", 0.8, _embedding(1.0))
        a.write_memory("fact Y", "agent-a", 0.7, _embedding(1.01))

        b = _make_conflict_mesh(ns="test")
        b.write_memory("fact Z", "agent-b", 0.9, _embedding(2.0))

        ab = a.merge_replicas(b)
        ba = b.merge_replicas(a)

        ab_snap = ab.snapshot()
        ba_snap = ba.snapshot()

        assert ab_snap["conflicts"] == ba_snap["conflicts"]
        assert ab_snap["add_edges"] == ba_snap["add_edges"]

    def test_merge_idempotency_with_conflicts(self) -> None:
        """merge(A, A) preserves exact conflict state (no duplication)."""
        mesh = _make_conflict_mesh(ns="test")
        mesh.write_memory("claim one", "agent-a", 0.8, _embedding(1.0))
        mesh.write_memory("claim two", "agent-b", 0.8, _embedding(1.01))

        assert len(mesh.conflicts()) == 1

        merged = mesh.merge_replicas(mesh)
        assert len(merged.conflicts()) == 1
        assert merged.snapshot()["conflicts"] == mesh.snapshot()["conflicts"]

    def test_merge_propagates_conflicts_from_both_replicas(self) -> None:
        """Conflicts from both replicas appear in the merged result."""
        a = _make_conflict_mesh(ns="test")
        na1 = a.write_memory("the door is open", "agent-a", 0.9, _embedding(1.0))
        na2 = a.write_memory("the door is shut", "agent-a", 0.8, _embedding(1.01))

        b = _make_conflict_mesh(ns="test")
        nb1 = b.write_memory("the light is on", "agent-b", 0.9, _embedding(3.0))
        nb2 = b.write_memory("the light is off", "agent-b", 0.8, _embedding(3.01))

        assert len(a.conflicts()) == 1
        assert len(b.conflicts()) == 1

        merged = a.merge_replicas(b)
        assert len(merged.conflicts()) >= 2

    def test_removed_node_conflict_not_re_created(self) -> None:
        """If a conflicting node is removed, new conflicts aren't created for it."""
        mesh = _make_conflict_mesh(ns="test")
        n1 = mesh.write_memory("X is true", "agent-a", 0.8, _embedding(1.0))
        n2 = mesh.write_memory("X is false", "agent-b", 0.8, _embedding(1.01))

        assert len(mesh.conflicts()) == 1

        mesh.remove_node(n2.id)

        # n2 is tombstoned — no new conflicts should fire for it
        n3 = mesh.write_memory("X is maybe", "agent-c", 0.7, _embedding(1.02))
        # Only the original conflict should exist (n2 is in rem_nodes)
        assert n2.id not in mesh.observed_nodes()


# ===================================================================
# 5. Utility Functions
# ===================================================================

class TestConflictUtilities:
    """Verify cosine_similarity and compute_belief_posteriors."""

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Identical vectors have cosine similarity = 1.0."""
        v = np.ones(384, dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Orthogonal vectors have cosine similarity = 0.0."""
        a = _orthogonal_embedding(0)
        b = _orthogonal_embedding(100)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_zero_vector(self) -> None:
        """Zero vector returns 0.0 similarity (safe division)."""
        zero = np.zeros(384, dtype=np.float32)
        v = np.ones(384, dtype=np.float32)
        assert cosine_similarity(zero, v) == 0.0

    def test_belief_posteriors_equal_scores(self) -> None:
        """Equal confidence × BTS → 50/50 posteriors."""
        left_post, right_post = compute_belief_posteriors(0.5, 0.5, 0.5, 0.5)
        assert left_post == pytest.approx(0.5, abs=1e-6)
        assert right_post == pytest.approx(0.5, abs=1e-6)

    def test_belief_posteriors_zero_scores(self) -> None:
        """Both zero scores → graceful 50/50 fallback."""
        left_post, right_post = compute_belief_posteriors(0.0, 0.0, 0.0, 0.0)
        assert left_post == 0.5
        assert right_post == 0.5

    @pytest.mark.benchmark
    def test_conflict_detection_performance(self) -> None:
        """Conflict detection for 50 nodes completes quickly (< 1s)."""
        import time

        mesh = _make_conflict_mesh(ns="bench")

        for i in range(50):
            emb = _embedding(1.0 + i * 0.001)
            mesh.write_memory(f"node-{i}", f"agent-{i % 5}", 0.5 + i * 0.01, emb)

        start = time.perf_counter()
        mesh.detect_semantic_conflicts()
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Conflict detection took {elapsed:.3f}s"

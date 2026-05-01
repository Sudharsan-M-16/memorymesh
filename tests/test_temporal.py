"""MemoryMesh Week 2 — Temporal Persistence Test Suite.

22 tests across 6 test classes covering:
  - WAL basics: write, read, replay, fsync, sequential numbering
  - WAL crash recovery: truncated lines, partial writes
  - Point-in-time queries: historical state reconstruction (TMP-003)
  - Confidence decay: Ebbinghaus formula validation (TMP-005, TMP-006)
  - Spaced repetition: boost threshold and capping (TMP-007)
  - WAL compaction: snapshot + archive (TMP-009)

Run: python -m pytest tests/test_temporal.py -v
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memorymesh import (
    AccessTracker,
    DecayConfig,
    EdgeLabel,
    MemoryMeshCore,
    WALEntry,
    WALOp,
    WriteAheadLog,
    compute_decayed_confidence,
    compute_effective_confidence,
)


# ===================================================================
# Helpers
# ===================================================================

def _wal_path(tmp_path: Path, name: str = "test.wal") -> Path:
    """Create a WAL path inside pytest's tmp_path."""
    return tmp_path / name


def _utc_iso(dt: datetime) -> str:
    """Format a datetime as ISO 8601 UTC string."""
    return dt.isoformat()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ===================================================================
# 1. WAL Basics
# ===================================================================

class TestWALBasics:
    """Verify core WAL write/read/replay behavior."""

    def test_append_and_replay(self, tmp_path: Path) -> None:
        """Entries appended to WAL are recoverable via replay."""
        wal = WriteAheadLog(_wal_path(tmp_path))

        wal.append(WALOp.WRITE_NODE, "agent-1", "ns", {"content": "hello"})
        wal.append(WALOp.WRITE_NODE, "agent-2", "ns", {"content": "world"})

        entries = wal.replay_all()
        assert len(entries) == 2
        assert entries[0].payload["content"] == "hello"
        assert entries[1].payload["content"] == "world"

    def test_sequential_numbering(self, tmp_path: Path) -> None:
        """Entries get monotonically increasing sequence numbers."""
        wal = WriteAheadLog(_wal_path(tmp_path))

        for i in range(5):
            wal.append(WALOp.WRITE_NODE, "agent-1", "ns", {"i": i})

        entries = wal.replay_all()
        seqs = [e.seq for e in entries]
        assert seqs == [1, 2, 3, 4, 5]

    def test_entry_count(self, tmp_path: Path) -> None:
        """entry_count accurately reflects number of WAL entries."""
        wal = WriteAheadLog(_wal_path(tmp_path))

        for i in range(3):
            wal.append(WALOp.WRITE_NODE, "agent-1", "ns", {"i": i})

        assert wal.entry_count == 3

    def test_wal_file_created_on_disk(self, tmp_path: Path) -> None:
        """WAL file is physically created and contains valid JSON lines."""
        path = _wal_path(tmp_path)
        wal = WriteAheadLog(path)
        wal.append(WALOp.WRITE_NODE, "agent-1", "ns", {"content": "test"})

        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

    def test_replay_with_timestamp_cutoff(self, tmp_path: Path) -> None:
        """Replay respects up_to timestamp — only returns entries before cutoff."""
        wal = WriteAheadLog(_wal_path(tmp_path))

        t1 = "2026-04-03T14:30:00+00:00"
        t2 = "2026-04-03T14:32:00+00:00"
        t3 = "2026-04-03T14:35:00+00:00"

        wal.append(WALOp.WRITE_NODE, "a1", "ns", {"c": "early"}, timestamp_utc=t1)
        wal.append(WALOp.WRITE_NODE, "a1", "ns", {"c": "mid"}, timestamp_utc=t2)
        wal.append(WALOp.WRITE_NODE, "a1", "ns", {"c": "late"}, timestamp_utc=t3)

        # Replay up to t2 — should get 2 entries
        entries = list(wal.replay(up_to=t2))
        assert len(entries) == 2
        assert entries[-1].payload["c"] == "mid"


# ===================================================================
# 2. WAL Crash Recovery
# ===================================================================

class TestWALCrashRecovery:
    """Verify WAL resilience to corrupt/truncated data."""

    def test_truncated_line_skipped(self, tmp_path: Path) -> None:
        """Corrupt (non-JSON) lines are silently skipped during replay."""
        path = _wal_path(tmp_path)
        wal = WriteAheadLog(path)
        wal.append(WALOp.WRITE_NODE, "agent-1", "ns", {"content": "valid"})

        # Manually append a corrupt line
        with open(path, "a", encoding="utf-8") as f:
            f.write("THIS IS NOT JSON\n")

        # Re-open and replay — corrupt line should be skipped
        wal2 = WriteAheadLog(path)
        entries = wal2.replay_all()
        assert len(entries) == 1
        assert entries[0].payload["content"] == "valid"

    def test_empty_wal_replays_cleanly(self, tmp_path: Path) -> None:
        """An empty WAL file produces zero entries on replay."""
        path = _wal_path(tmp_path)
        path.touch()

        wal = WriteAheadLog(path)
        assert wal.replay_all() == []
        assert wal.current_seq == 0

    def test_seq_continues_after_reopen(self, tmp_path: Path) -> None:
        """After reopening a WAL, new entries continue from last seq."""
        path = _wal_path(tmp_path)

        wal1 = WriteAheadLog(path)
        wal1.append(WALOp.WRITE_NODE, "a1", "ns", {"i": 0})
        wal1.append(WALOp.WRITE_NODE, "a1", "ns", {"i": 1})

        # Reopen
        wal2 = WriteAheadLog(path)
        entry = wal2.append(WALOp.WRITE_NODE, "a1", "ns", {"i": 2})

        assert entry.seq == 3
        assert wal2.entry_count == 3


# ===================================================================
# 3. Point-in-Time Queries (TMP-003)
# ===================================================================

class TestPointInTimeQuery:
    """Verify query_at() time-travel functionality."""

    def test_query_at_reconstructs_historical_state(self, tmp_path: Path) -> None:
        """query_at(t) returns graph state as it was at time t."""
        path = _wal_path(tmp_path)
        mesh = MemoryMeshCore(namespace="test", wal_path=path)

        t1 = "2026-04-03T14:30:00+00:00"
        t2 = "2026-04-03T14:32:00+00:00"
        t3 = "2026-04-03T14:35:00+00:00"

        # Write entries with explicit timestamps in the WAL
        mesh._wal.append(
            WALOp.WRITE_NODE, "agent-r", "test",
            {"content": "fact A", "confidence": 0.9},
            timestamp_utc=t1,
        )
        mesh._wal.append(
            WALOp.WRITE_NODE, "agent-r", "test",
            {"content": "fact B", "confidence": 0.8},
            timestamp_utc=t2,
        )
        mesh._wal.append(
            WALOp.WRITE_NODE, "agent-r", "test",
            {"content": "fact C", "confidence": 0.7},
            timestamp_utc=t3,
        )

        # Rebuild mesh to pick up manual WAL entries
        mesh2 = MemoryMeshCore(namespace="test", wal_path=path)

        # Query at t2 — should only see facts A and B
        historical = mesh2.query_at(t2)
        assert historical.node_count == 2

        # Query at t1 — should only see fact A
        historical_t1 = mesh2.query_at(t1)
        assert historical_t1.node_count == 1

    def test_query_at_includes_edges(self, tmp_path: Path) -> None:
        """Historical state includes causal edges that existed at time t."""
        path = _wal_path(tmp_path)

        t1 = "2026-04-03T14:30:00+00:00"
        t2 = "2026-04-03T14:31:00+00:00"
        t3 = "2026-04-03T14:35:00+00:00"

        # Build WAL manually for precise timestamp control
        wal = WriteAheadLog(path)
        wal.append(WALOp.WRITE_NODE, "a1", "test",
                   {"content": "evidence X", "confidence": 0.9}, timestamp_utc=t1)
        wal.append(WALOp.WRITE_NODE, "a1", "test",
                   {"content": "conclusion Y", "confidence": 0.8}, timestamp_utc=t2)

        # Replay to get node IDs
        temp_mesh = MemoryMeshCore(namespace="test", wal_path=path)
        nodes = temp_mesh.get_all_nodes()
        if len(nodes) == 2:
            n_ids = [n.id for n in nodes]
            wal.append(WALOp.ADD_EDGE, "a1", "test",
                       {"source_id": n_ids[0], "target_id": n_ids[1],
                        "label": EdgeLabel.SUPPORTS},
                       timestamp_utc=t3)

        # Query before edge was added
        mesh = MemoryMeshCore(namespace="test", wal_path=path)
        historical = mesh.query_at(t2)
        assert historical.edge_count == 0

        # Query after edge was added
        full = mesh.query_at(t3)
        assert full.edge_count == 1

    def test_query_at_requires_wal(self) -> None:
        """query_at() raises ValueError if no WAL is configured."""
        mesh = MemoryMeshCore(namespace="test")  # No WAL

        with pytest.raises(ValueError, match="requires WAL"):
            mesh.query_at("2026-04-03T14:32:00+00:00")

    def test_full_state_recovery_after_restart(self, tmp_path: Path) -> None:
        """WAL replay on restart recovers full CRDT state (US-001)."""
        path = _wal_path(tmp_path)

        # Session 1: write nodes and edges
        mesh1 = MemoryMeshCore(namespace="test", wal_path=path)
        ev_a = mesh1.write_memory("evidence A found", "agent-r", 0.9)
        hyp = mesh1.write_memory("hypothesis H derived", "agent-r", 0.7)
        mesh1.add_causal_edge(ev_a.id, hyp.id, EdgeLabel.SUPPORTS)

        assert mesh1.node_count == 2
        assert mesh1.edge_count == 1

        # Simulate process death — destroy in-memory state
        del mesh1

        # Session 2: reconstruct from WAL
        mesh2 = MemoryMeshCore(namespace="test", wal_path=path)

        assert mesh2.node_count == 2
        assert mesh2.edge_count == 1

        # Verify causal chain survives restart
        chain = mesh2.causal_chain(hyp.id, depth=2)
        assert len(chain.nodes) == 2


# ===================================================================
# 4. Confidence Decay — Ebbinghaus (TMP-005, TMP-006)
# ===================================================================

class TestConfidenceDecay:
    """Verify Ebbinghaus-inspired confidence decay formula."""

    def test_no_decay_at_creation(self) -> None:
        """R(0) = R₀ — zero elapsed time means no decay."""
        now = _utc_iso(_now_utc())
        result = compute_decayed_confidence(
            initial_confidence=0.9,
            created_utc=now,
            now_utc=now,
        )
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_decay_after_24_hours(self) -> None:
        """After 24h with λ=0.01, R(24) = 0.9 × exp(−0.24) ≈ 0.708."""
        created = _now_utc() - timedelta(hours=24)
        now = _now_utc()

        result = compute_decayed_confidence(
            initial_confidence=0.9,
            created_utc=_utc_iso(created),
            now_utc=_utc_iso(now),
        )

        expected = 0.9 * math.exp(-0.01 * 24)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_custom_lambda_rate(self) -> None:
        """Custom λ=0.05 causes faster decay."""
        config = DecayConfig(lambda_rate=0.05)
        created = _now_utc() - timedelta(hours=10)
        now = _now_utc()

        result = compute_decayed_confidence(
            initial_confidence=1.0,
            created_utc=_utc_iso(created),
            now_utc=_utc_iso(now),
            config=config,
        )

        expected = 1.0 * math.exp(-0.05 * 10)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_decay_is_lazy_no_write_amplification(self, tmp_path: Path) -> None:
        """Decay does NOT modify stored confidence — it's computed on read (TMP-006)."""
        path = _wal_path(tmp_path)
        mesh = MemoryMeshCore(namespace="test", wal_path=path)

        node = mesh.write_memory("test fact", "agent-1", 0.9)
        original_confidence = node.confidence

        # Read the node — stored confidence should not change
        stored = mesh.get_node(node.id)
        assert stored is not None
        assert stored.confidence == original_confidence  # Unchanged in storage

        # Decayed confidence is computed separately via get_node_with_decay
        result = mesh.get_node_with_decay(node.id)
        assert result is not None
        _, effective_conf, _ = result
        # The effective confidence may differ from stored, but stored stays same
        assert mesh._nodes[node.id].confidence == original_confidence


# ===================================================================
# 5. Spaced Repetition Boost (TMP-007)
# ===================================================================

class TestSpacedRepetition:
    """Verify spaced repetition boost mechanics."""

    def test_boost_applied_above_threshold(self) -> None:
        """Access rate ≥ 3/hour triggers 1.15× boost."""
        config = DecayConfig(boost_threshold=3.0, boost_factor=1.15)
        conf, was_boosted = compute_effective_confidence(
            initial_confidence=0.8,
            created_utc=_utc_iso(_now_utc()),  # Just created, no decay
            access_rate=4.0,  # Above threshold
            now_utc=_utc_iso(_now_utc()),
            config=config,
        )
        assert was_boosted is True
        assert conf == pytest.approx(0.8 * 1.15, abs=1e-6)

    def test_no_boost_below_threshold(self) -> None:
        """Access rate < 3/hour does NOT trigger boost."""
        config = DecayConfig(boost_threshold=3.0, boost_factor=1.15)
        conf, was_boosted = compute_effective_confidence(
            initial_confidence=0.8,
            created_utc=_utc_iso(_now_utc()),
            access_rate=2.0,  # Below threshold
            now_utc=_utc_iso(_now_utc()),
            config=config,
        )
        assert was_boosted is False
        assert conf == pytest.approx(0.8, abs=1e-6)

    def test_boost_capped_at_one(self) -> None:
        """Boosted confidence cannot exceed 1.0."""
        config = DecayConfig(boost_threshold=1.0, boost_factor=1.5, boost_cap=1.0)
        conf, was_boosted = compute_effective_confidence(
            initial_confidence=0.95,
            created_utc=_utc_iso(_now_utc()),
            access_rate=5.0,
            now_utc=_utc_iso(_now_utc()),
            config=config,
        )
        assert was_boosted is True
        assert conf == 1.0  # Capped

    def test_access_tracker_records_reads(self) -> None:
        """AccessTracker correctly counts and reports access frequency."""
        tracker = AccessTracker()
        now = _now_utc()

        # Record 5 accesses in the last 30 minutes
        for i in range(5):
            ts = now - timedelta(minutes=30 - i)
            tracker.record_access("node-1", _utc_iso(ts))

        rate = tracker.access_rate_per_hour("node-1", _utc_iso(now))
        assert rate == 5.0  # 5 accesses in 1 hour window
        assert tracker.access_count("node-1") == 5


# ===================================================================
# 6. WAL Compaction (TMP-009)
# ===================================================================

class TestWALCompaction:
    """Verify WAL snapshot + compaction behavior."""

    def test_snapshot_written_to_wal(self, tmp_path: Path) -> None:
        """create_snapshot() writes a SNAPSHOT entry to the WAL."""
        path = _wal_path(tmp_path)
        mesh = MemoryMeshCore(namespace="test", wal_path=path)
        mesh.write_memory("fact A", "agent-1", 0.9)

        mesh.create_snapshot()

        entries = mesh.wal.replay_all()
        snapshot_entries = [e for e in entries if e.op == WALOp.SNAPSHOT]
        assert len(snapshot_entries) == 1
        assert "snapshot_data" in snapshot_entries[0].payload

    def test_compact_removes_old_entries(self, tmp_path: Path) -> None:
        """compact_wal() removes entries before the snapshot."""
        path = _wal_path(tmp_path)
        mesh = MemoryMeshCore(namespace="test", wal_path=path)

        # Write 3 nodes
        mesh.write_memory("fact A", "agent-1", 0.9)
        mesh.write_memory("fact B", "agent-1", 0.8)
        mesh.write_memory("fact C", "agent-1", 0.7)

        # Snapshot
        mesh.create_snapshot()

        # Write more
        mesh.write_memory("fact D", "agent-1", 0.6)

        # Compact — should remove the 3 write entries before snapshot
        removed = mesh.compact_wal()
        assert removed == 3  # 3 writes removed, snapshot + 1 write kept

        # WAL should still have snapshot + post-snapshot write
        remaining = mesh.wal.replay_all()
        assert len(remaining) == 2

    def test_no_compact_without_snapshot(self, tmp_path: Path) -> None:
        """compact_wal() does nothing if no snapshot exists."""
        path = _wal_path(tmp_path)
        mesh = MemoryMeshCore(namespace="test", wal_path=path)
        mesh.write_memory("fact A", "agent-1", 0.9)

        removed = mesh.compact_wal()
        assert removed == 0

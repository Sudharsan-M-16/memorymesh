"""Write-Ahead Log (WAL) — append-only persistence for MemoryMesh.

Every mutation to the CRDT graph is captured as a structured log entry
before it is applied.  This gives us:

    1. Crash recovery    — replay the WAL to reconstruct full state
    2. Time-travel       — replay up to timestamp T to see historical state
    3. Audit compliance  — every operation is permanently recorded

Format: JSON-lines (one JSON object per line, newline-delimited).
Each entry is fsync'd to disk before the operation is committed.

PRD Requirements covered:
    TMP-004  Append-only WAL, no destructive updates
    TMP-009  WAL compaction via snapshot + archive (stub)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# WAL operation types
# ---------------------------------------------------------------------------

class WALOp(str, Enum):
    """Operation types recorded in the WAL.

    Each enum value maps to a specific CRDT mutation.
    """
    WRITE_NODE = "write_node"
    REMOVE_NODE = "remove_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    SNAPSHOT = "snapshot"


# ---------------------------------------------------------------------------
# WAL Entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WALEntry:
    """A single immutable WAL record.

    Attributes
    ----------
    seq : int
        Monotonically increasing sequence number (1-based).
    op : WALOp
        The operation type.
    timestamp_utc : str
        ISO 8601 UTC timestamp when the operation occurred.
    agent_id : str
        Agent that initiated the operation (empty for system ops).
    namespace : str
        Namespace scope for the operation.
    payload : dict
        Operation-specific data.  Structure depends on ``op``:

        WRITE_NODE:
            {content, confidence, lamport_vector, embedding_norm}
        REMOVE_NODE:
            {node_id}
        ADD_EDGE:
            {source_id, target_id, label}
        REMOVE_EDGE:
            {source_id, target_id, label}
        SNAPSHOT:
            {snapshot_data}  (full serialized CRDT state)
    """
    seq: int
    op: WALOp
    timestamp_utc: str
    agent_id: str
    namespace: str
    payload: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize to a single-line JSON string."""
        return json.dumps({
            "seq": self.seq,
            "op": self.op.value,
            "timestamp_utc": self.timestamp_utc,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "payload": self.payload,
        }, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def from_json(line: str) -> "WALEntry":
        """Deserialize from a JSON string.

        Raises
        ------
        ValueError
            If the line is not valid JSON or missing required fields.
        """
        data = json.loads(line)
        return WALEntry(
            seq=data["seq"],
            op=WALOp(data["op"]),
            timestamp_utc=data["timestamp_utc"],
            agent_id=data["agent_id"],
            namespace=data["namespace"],
            payload=data["payload"],
        )


# ---------------------------------------------------------------------------
# WAL Engine
# ---------------------------------------------------------------------------

class WriteAheadLog:
    """Append-only, fsync'd write-ahead log.

    Each entry is written as a single JSON line followed by a newline.
    After writing, the file is flushed and fsync'd to guarantee
    durability even on power failure.

    Parameters
    ----------
    wal_path : Path or str
        Filesystem path for the WAL file.  Created if it doesn't exist.
        Parent directories are created automatically.

    Usage
    -----
    >>> wal = WriteAheadLog("/tmp/mesh.wal")
    >>> wal.append(WALOp.WRITE_NODE, "agent-1", "default", {"content": "hello"})
    >>> entries = list(wal.replay())
    >>> len(entries)
    1
    """

    def __init__(self, wal_path: str | Path) -> None:
        self._path = Path(wal_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seq: int = 0
        self._fd: Optional[Any] = None

        # If WAL already exists, count existing entries to set seq
        if self._path.exists():
            for entry in self._iter_raw():
                self._seq = max(self._seq, entry.seq)

    # -----------------------------------------------------------------
    # Core append (TMP-004: append-only, fsync'd)
    # -----------------------------------------------------------------

    def append(
        self,
        op: WALOp,
        agent_id: str,
        namespace: str,
        payload: Dict[str, Any],
        timestamp_utc: Optional[str] = None,
    ) -> WALEntry:
        """Append a new entry to the WAL.

        The entry is:
          1. Assigned a monotonic sequence number
          2. Serialized to JSON
          3. Written + flushed + fsync'd

        Parameters
        ----------
        op : WALOp
            Operation type.
        agent_id : str
            Agent performing the operation.
        namespace : str
            Target namespace.
        payload : dict
            Operation-specific data.
        timestamp_utc : str, optional
            Override timestamp (for testing).  Defaults to now.

        Returns
        -------
        WALEntry
            The committed entry with assigned seq number.
        """
        self._seq += 1
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

        entry = WALEntry(
            seq=self._seq,
            op=op,
            timestamp_utc=ts,
            agent_id=agent_id,
            namespace=namespace,
            payload=payload,
        )

        line = entry.to_json() + "\n"

        # Append with fsync for durability
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        return entry

    # -----------------------------------------------------------------
    # Replay (full log or up to a timestamp)
    # -----------------------------------------------------------------

    def replay(
        self,
        up_to: Optional[str] = None,
    ) -> Iterator[WALEntry]:
        """Replay WAL entries, optionally stopping at a timestamp.

        Parameters
        ----------
        up_to : str, optional
            ISO 8601 UTC timestamp.  If provided, only entries with
            ``timestamp_utc <= up_to`` are yielded.  This is the core
            primitive for ``query_at()`` (TMP-003).

        Yields
        ------
        WALEntry
            Entries in sequential order.

        Notes
        -----
        Corrupt or truncated lines are silently skipped — this is
        intentional for crash recovery.  If the process was killed
        mid-write, the partial last line is discarded.
        """
        for entry in self._iter_raw():
            if up_to is not None and entry.timestamp_utc > up_to:
                return
            yield entry

    def replay_all(self) -> List[WALEntry]:
        """Return all WAL entries as a list (convenience method)."""
        return list(self.replay())

    # -----------------------------------------------------------------
    # Snapshot support (TMP-009)
    # -----------------------------------------------------------------

    def write_snapshot(
        self,
        namespace: str,
        snapshot_data: Dict[str, Any],
    ) -> WALEntry:
        """Write a snapshot entry to the WAL.

        After a snapshot, WAL entries older than the snapshot may be
        archived to cold storage.  The snapshot serves as the
        authoritative base state for future replays.
        """
        return self.append(
            op=WALOp.SNAPSHOT,
            agent_id="__system__",
            namespace=namespace,
            payload={"snapshot_data": snapshot_data},
        )

    def find_latest_snapshot(self) -> Optional[WALEntry]:
        """Find the most recent SNAPSHOT entry in the WAL.

        Returns None if no snapshot exists.
        """
        latest: Optional[WALEntry] = None
        for entry in self._iter_raw():
            if entry.op == WALOp.SNAPSHOT:
                latest = entry
        return latest

    # -----------------------------------------------------------------
    # Compaction (TMP-009)
    # -----------------------------------------------------------------

    def compact(self, snapshot_entry: WALEntry) -> int:
        """Compact the WAL by removing entries before a snapshot.

        All entries with seq < snapshot_entry.seq are discarded.
        The snapshot entry and all subsequent entries are preserved.

        Parameters
        ----------
        snapshot_entry : WALEntry
            The snapshot to use as the new base state.

        Returns
        -------
        int
            Number of entries removed.
        """
        entries = list(self._iter_raw())
        kept = [e for e in entries if e.seq >= snapshot_entry.seq]
        removed_count = len(entries) - len(kept)

        if removed_count > 0:
            # Rewrite the WAL with only kept entries
            with open(self._path, "w", encoding="utf-8") as f:
                for entry in kept:
                    f.write(entry.to_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

        return removed_count

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Path to the WAL file."""
        return self._path

    @property
    def entry_count(self) -> int:
        """Total number of entries in the WAL."""
        return sum(1 for _ in self._iter_raw())

    @property
    def current_seq(self) -> int:
        """Current sequence number (last assigned)."""
        return self._seq

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _iter_raw(self) -> Iterator[WALEntry]:
        """Iterate over all parseable entries in the WAL file.

        Silently skips corrupt/truncated lines for crash safety.
        """
        if not self._path.exists():
            return

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield WALEntry.from_json(line)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Corrupt line — skip silently (crash recovery)
                    continue

    def __repr__(self) -> str:
        return (
            f"WriteAheadLog(path='{self._path}', "
            f"entries={self.entry_count}, seq={self._seq})"
        )

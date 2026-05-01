# MemoryMesh

MemoryMesh is a Python package for durable, causal, trust-aware memory shared by AI agents. It implements a content-addressed memory graph using CRDT semantics, Lamport clocks, Bayesian trust scoring, write-ahead logging, and lazy temporal confidence decay.

The current codebase exposes the Week 2 public API in `memorymesh.__version__ == "0.2.0"`.

## What It Provides

- Content-addressed memory nodes using SHA-256 over canonical content.
- Immutable `MemoryNode` values with Lamport vectors, provenance, confidence, embeddings, and causal edges.
- A 2P2P-Graph CRDT core with grow-only add/remove sets for nodes and edges.
- DAG-safe causal edge insertion with cycle detection.
- Namespace isolation for replica merges.
- Bayesian per-agent trust scores with an audit trail.
- Append-only JSON-lines WAL persistence with replay and point-in-time queries.
- Lazy Ebbinghaus-style confidence decay.
- Spaced repetition boost based on recent read frequency.
- GPU-ready placeholder embeddings as contiguous `float32` vectors of shape `(384,)`.

## Installation

```bash
pip install -e ".[dev]"
```

Runtime dependencies are declared in `pyproject.toml`:

- `networkx`
- `numpy`

Development dependencies include:

- `pytest`
- `pytest-benchmark`

## Quick Start

```python
from memorymesh import EdgeLabel, MemoryMeshCore

mesh = MemoryMeshCore(namespace="demo")

evidence = mesh.write_memory(
    content="Contract signed on March 22",
    agent_id="agent-researcher",
    confidence=0.91,
)

conclusion = mesh.write_memory(
    content="The contract is active",
    agent_id="agent-analyst",
    confidence=0.74,
)

mesh.add_causal_edge(evidence.id, conclusion.id, EdgeLabel.SUPPORTS)

chain = mesh.causal_chain(conclusion.id, depth=2)
print(mesh.node_count, mesh.edge_count)
print(list(chain.nodes))
```

## Persistent Memory

Pass a WAL path to make writes durable. A new `MemoryMeshCore` with the same namespace and WAL path replays the log on startup.

```python
from memorymesh import MemoryMeshCore

mesh = MemoryMeshCore(namespace="demo", wal_path="data/demo.wal")
node = mesh.write_memory("Persistent fact", "agent-1", confidence=0.8)

recovered = MemoryMeshCore(namespace="demo", wal_path="data/demo.wal")
assert recovered.get_node(node.id) is not None
```

## Point-In-Time Queries

`query_at()` reconstructs state by replaying WAL entries up to an ISO 8601 UTC timestamp.

```python
from datetime import datetime, timezone
from memorymesh import MemoryMeshCore

mesh = MemoryMeshCore(namespace="audit", wal_path="data/audit.wal")
mesh.write_memory("Initial claim", "agent-1", confidence=0.6)

checkpoint = datetime.now(timezone.utc).isoformat()

mesh.write_memory("Later claim", "agent-1", confidence=0.7)

historical = mesh.query_at(checkpoint)
print(historical.node_count)
```

## Temporal Confidence

Stored confidence is not rewritten by decay. The effective confidence is computed lazily on read.

```python
from memorymesh import MemoryMeshCore

mesh = MemoryMeshCore(namespace="temporal")
node = mesh.write_memory("Frequently referenced memory", "agent-1", confidence=0.9)

result = mesh.get_node_with_decay(node.id)
if result is not None:
    raw_node, effective_confidence, was_boosted = result
    print(raw_node.confidence, effective_confidence, was_boosted)
```

The default decay model is:

```text
R(t) = R0 * exp(-lambda_rate * delta_hours)
```

The default `lambda_rate` is `0.01` per hour. Frequent reads can apply a multiplicative boost capped at `1.0`.

## Bayesian Trust

```python
from memorymesh import BayesianTrustEngine

trust = BayesianTrustEngine()
trust.register_agent("agent-researcher")

score_before = trust.get_bts("agent-researcher")
score_after = trust.trust_update(
    agent_id="agent-researcher",
    memory_id="memory-001",
    outcome=True,
)

audit_log = trust.trust_audit("agent-researcher")
print(score_before, score_after, audit_log[-1])
```

Trust scores use the Beta distribution mean:

```text
BTS = alpha / (alpha + beta)
```

Default priors are `alpha=2.0` and `beta=2.0`.

## Architecture

```text
memorymesh/
  types.py         Edge labels and custom exceptions
  memory_node.py   Immutable content-addressed memory node
  mesh_core.py     CRDT graph engine, WAL integration, temporal reads
  trust_engine.py  Bayesian trust scoring and audit trail
  wal.py           Append-only JSON-lines write-ahead log
  temporal.py      Confidence decay and access tracking
  __init__.py      Public exports
tests/
  test_brain_stem.py  Week 1 core correctness
  test_temporal.py    Week 2 temporal persistence
```

The CRDT state is represented as four grow-only sets:

```text
add_nodes
rem_nodes
add_edges
rem_edges
```

Observed state is derived as:

```text
observed_nodes = add_nodes - rem_nodes
observed_edges = add_edges - rem_edges
```

## Core Invariants

- Tombstones are append-only.
- CRDT merges use set union.
- Lamport vectors merge by elementwise maximum.
- Causal edges must keep the graph acyclic.
- Node IDs come from `content_address(content)`.
- Removed nodes cannot be re-added.
- Replica merges are only valid inside the same namespace.
- WAL-backed mutations are logged before commit.
- Lazy decay does not mutate stored node confidence.

## Testing

Run the full suite:

```bash
python -m pytest tests/ -v
```

Run without benchmarks:

```bash
python -m pytest tests/ -v -m "not benchmark"
```

Run a single suite:

```bash
python -m pytest tests/test_temporal.py -v
```

If local Windows temp/cache directories are permission-blocked, keep pytest artifacts in the repository workspace:

```bash
python -m pytest tests/ -v -o cache_dir=.tmp/pytest-cache --basetemp=.tmp/pytest-tmp
```

## Current Limits

- Semantic conflict detection and resolution are not implemented yet.
- LangChain and LlamaIndex integrations are not implemented yet.
- GPU acceleration packages are listed as optional future-facing dependencies, but the current implementation uses CPU data structures.
- The trust engine is single-threaded; wrap it externally if used concurrently.

## Public API Snapshot

Common imports:

```python
from memorymesh import (
    AccessTracker,
    BayesianTrustEngine,
    DecayConfig,
    EdgeLabel,
    MemoryMeshCore,
    MemoryNode,
    WALEntry,
    WALOp,
    WriteAheadLog,
    compute_decayed_confidence,
    compute_effective_confidence,
    content_address,
    zero_embedding,
)
```

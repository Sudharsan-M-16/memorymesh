# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

MemoryMesh is **"The Persistent Cognitive Substrate for the Agentic Internet"** — a distributed memory system for AI agents built on CRDT (Conflict-Free Replicated Data Type) theory. It solves **Agent Amnesia**: the inability of AI agents to persist causal, trust-weighted, temporally-consistent memory across sessions.

| Property | Value |
|----------|-------|
| **Current Phase** | Temporal Persistence — Week 2 (v0.2.0) |
| **Stack** | Pure Python 3.11+, NetworkX, NumPy |
| **Not yet active** | FastAPI (Week 6), GPU acceleration |

---

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Quick smoke test (check imports)
python -c "from memorymesh import MemoryNode, MemoryMeshCore, BayesianTrustEngine; print('OK')"

# Run full test suite (must pass before any commit)
python -m pytest tests/ -v

# Run without benchmarks (faster feedback)
python -m pytest tests/ -v -m "not benchmark"

# Run a single test class
python -m pytest tests/test_brain_stem.py::TestCRDTMerge -v

# Run a single test
python -m pytest tests/test_brain_stem.py::TestContentAddressing::test_deterministic_id_across_agents -v

# Run with benchmarks only
python -m pytest tests/ -v --benchmark-only
```

---

## Architecture

### Module Dependency Map

```
types.py              → (no internal deps, vocabulary only)
memory_node.py        → types.py
trust_engine.py       → (standalone, no internal deps)
wal.py                → (standalone, JSON-lines persistence)
temporal.py           → (standalone, decay/boost math)
mesh_core.py          → memory_node.py, types.py, wal.py, temporal.py
__init__.py           → aggregates all above (public API)
```

### Core Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `types.py` | `EdgeLabel`, exceptions | Shared vocabulary — 5 edge labels + error types |
| `memory_node.py` | `MemoryNode` | Immutable 7-tuple cognitive unit `(id, e, τ, C, R, T, κ)` |
| `trust_engine.py` | `BayesianTrustEngine` | Per-agent Beta-Binomial trust tracking `BTS = α/(α+β)` |
| `wal.py` | `WriteAheadLog`, `WALEntry` | Append-only WAL: fsync'd JSON-lines, crash recovery, replay |
| `temporal.py` | `DecayConfig`, `AccessTracker` | Ebbinghaus decay + spaced repetition boost (lazy on read) |
| `mesh_core.py` | `MemoryMeshCore`, `CRDTState` | 2P2P-Graph CRDT engine + WAL integration + time-travel |

---

## The 7-Tuple

Every memory is `m = (id, e, τ, C, R, T, κ)`:

| Symbol | Field | Type | Description |
|--------|-------|------|-------------|
| `id` | `id` | `str` | SHA-256 of canonical content (deterministic across agents) |
| `e` | `embedding` | `np.ndarray (384,) float32` | Semantic vector (zero placeholder until GPU week) |
| `τ` | `lamport_vector` | `dict[str, int]` | Logical clock per agent (causal ordering) |
| `C` | `causal_edges` | `frozenset[tuple[str, str, str]]` | `(src, dst, label)` triples — immutable |
| `R` | `confidence` | `float [0, 1]` | Belief strength |
| `T` | `trust_provenance` | `dict` | Agent ID, timestamp, namespace |
| `κ` | `content` | `str` | Raw canonical payload |

---

## CRDT State (`mesh_core.py`)

Four **grow-only sets** — never shrink, only union:

- `add_nodes`: `set[str]`
- `rem_nodes`: `set[str]` *(tombstones)*
- `add_edges`: `set[tuple]`
- `rem_edges`: `set[tuple]` *(tombstones)*

**Derived state:**
```
observed_nodes = add_nodes - rem_nodes
observed_edges = add_edges - rem_edges
```

---

## Coding Standards

### 1. Immutability First
`MemoryNode` is a frozen dataclass. To "update," use `merge_with()` to return a new instance.

### 2. Content Addressing
Node IDs must always be the SHA-256 hash of the content via `content_address(content)`. No UUIDs.

### 3. Lamport Merge
Always use **elementwise maximum** for logical clocks. Never sum or average.

### 4. Tombstones Are Forever
Never `discard()` from `add` sets. Deletion is strictly performed by adding the ID to the `rem` sets.

### 5. Cycle Detection
All causal edges must be added via `add_causal_edge()` to trigger the DFS cycle check.

### 6. Float32 Embeddings
NumPy arrays for embeddings must use `dtype=np.float32` for future CUDA compatibility.

### 7. Module Discipline
- `types.py`: Vocabulary only (Enums/Exceptions). **No logic.**
- `trust_engine.py`: **Single-threaded only.**

---

## Architectural Rules

| Rule | Description |
|------|-------------|
| **Rule 1 — No Cycles** | The graph is a DAG. Bypassing cycle detection is a **critical failure**. |
| **Rule 2 — Commutativity** | `merge(A, B)` must equal `merge(B, A)`. |
| **Rule 3 — Deduplication** | Identical content **MUST** result in identical IDs. |
| **Rule 4 — Mathematical Trust** | Trust updates follow Beta distribution mean: `BTS = α/(α+β)` |

---

## PRD Requirements Mapping

| Req ID | Description | Status | File |
|--------|-------------|--------|------|
| **TL-001** | Beta-Binomial trust model | ✅ | `trust_engine.py` |
| **TL-002** | BTS computation < 1ms | ✅ | `trust_engine.py` |
| **TL-007** | `trust_update(agent_id, memory_id, outcome)` | ✅ | `trust_engine.py` |
| **TL-008** | `trust_audit(agent_id)` | ✅ | `trust_engine.py` |
| **TMP-001** | Wall-clock + Lamport timestamps | ✅ | `memory_node.py` |
| **TMP-002** | Lamport element-wise max merge | ✅ | `memory_node.py` |
| **TMP-003** | Point-in-time `query_at()` | ✅ | `mesh_core.py` |
| **TMP-004** | Append-only WAL replay | ✅ | `wal.py`, `mesh_core.py` |
| **TMP-005** | Ebbinghaus confidence decay | ✅ | `temporal.py` |
| **TMP-006** | Lazy decay on read | ✅ | `temporal.py`, `mesh_core.py` |
| **TMP-007** | Spaced repetition boost | ✅ | `temporal.py`, `mesh_core.py` |
| **TMP-008** | `causal_chain(node_id, depth)` | ✅ | `mesh_core.py` |
| **TMP-009** | WAL compaction via snapshot | ✅ | `wal.py`, `mesh_core.py` |
| **SYN-001** | 2P2P-Graph CRDT (4 grow-only sets) | ✅ | `mesh_core.py` |
| **SYN-002** | CRDT merge properties | ✅ | `mesh_core.py` |
| **SYN-004** | Content-addressed dedup (SHA-256) | ✅ | `memory_node.py` |
| **SYN-006** | DFS cycle detection | ✅ | `mesh_core.py` |
| **SYN-007** | Namespace isolation | ✅ | `mesh_core.py` |
| **TL-003/004** | Semantic conflict detection | ❌ | (Week 3) |
| **TL-005/006** | Conflict resolution + belief probs | ❌ | (Week 3) |
| **LangChain** | Drop-in integration | ❌ | (Week 4+) |

---

## What's Not Implemented (Yet)

| Feature | PRD Ref | Planned |
|---------|---------|---------|
| Semantic conflict detection | TL-003, TL-004, TL-005, TL-006 | Week 3 |
| GPU acceleration (FAISS, cuGraph) | Performance targets | Week 4+ |
| LangChain integration | Interoperability | Week 5+ |
| LlamaIndex integration | Interoperability | Week 5+ |
| Rust hot-path rewrite | Performance | Week 9-10 |

---

## Plan Mode Template

Before editing core logic, fill this out:

```
Plan: [Feature Name]
Requirement ID:  [e.g., SYN-001]
Files affected:  [list]
CRDT Safety:     Does this preserve Commutativity / Associativity / Idempotency?
DAG Safety:      Does this touch edges? If so, is cycle detection active?
Test Plan:       Which mathematical property does the new test prove?
```

---

## What MemoryMesh Is Not

> MemoryMesh is a **cognitive substrate**. Everything else is built *on* it, not *in* it.

- ❌ Not a vector database
- ❌ Not a RAG wrapper
- ❌ Not a chatbot plugin
- ❌ Not a message queue

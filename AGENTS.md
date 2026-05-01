# Repository Guidelines

## Project Structure & Module Organization

MemoryMesh is a Python 3.11+ package. Source code lives in `memorymesh/`; tests live in `tests/`.

- `memorymesh/types.py`: shared edge labels and custom exceptions. Keep this vocabulary-only.
- `memorymesh/memory_node.py`: immutable, content-addressed 7-tuple memory node model.
- `memorymesh/mesh_core.py`: 2P2P-Graph CRDT engine, Lamport clocks, causal graph operations, WAL integration, point-in-time queries, and lazy temporal reads.
- `memorymesh/trust_engine.py`: Bayesian trust scoring and append-only audit trail.
- `memorymesh/wal.py`: fsync'd JSON-lines write-ahead log, replay, snapshots, and compaction.
- `memorymesh/temporal.py`: Ebbinghaus confidence decay and spaced repetition access tracking.
- `memorymesh/__init__.py`: public package exports and package version.
- `tests/test_brain_stem.py`: Week 1 core correctness suite.
- `tests/test_temporal.py`: Week 2 temporal persistence suite.

Generated caches, virtual environments, egg metadata, temporary WAL files, and benchmark artifacts should not be committed.

## Build, Test, and Development Commands

Install in editable development mode:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run fast tests without benchmarks:

```bash
python -m pytest tests/ -v -m "not benchmark"
```

Run one focused suite:

```bash
python -m pytest tests/test_temporal.py -v
```

Smoke-test public imports:

```bash
python -c "from memorymesh import MemoryMeshCore, MemoryNode, BayesianTrustEngine, WriteAheadLog; print('OK')"
```

If Windows temp or cache directories are permission-blocked, keep pytest artifacts inside the workspace:

```bash
python -m pytest tests/ -v -o cache_dir=.tmp/pytest-cache --basetemp=.tmp/pytest-tmp
```

## Coding Style & Naming Conventions

Use 4-space indentation, type annotations, and small modules with clear ownership. Use descriptive snake_case names for functions, variables, and modules. Class names use PascalCase.

Keep module responsibilities strict:

- CRDT graph logic belongs in `mesh_core.py`.
- Trust and audit logic belongs in `trust_engine.py`.
- WAL serialization and replay belong in `wal.py`.
- Decay and access-frequency math belong in `temporal.py`.
- Shared labels and exceptions belong in `types.py`.

Memory nodes are immutable. Update behavior should return new values or merge via `MemoryNode.merge_with()`. Always use `content_address()` for node IDs. Do not introduce UUID-based node IDs.

NumPy embeddings must remain contiguous `float32` arrays with shape `(384,)` unless the embedding model contract is deliberately changed and tested.

## Testing Guidelines

Tests use `pytest`. Add focused tests for every behavioral change, especially CRDT properties, DAG cycle safety, Lamport vector semantics, WAL replay, time-travel queries, decay calculations, and trust updates.

Name test classes by feature, for example `TestCycleDetection`, and test functions by expected behavior, for example `test_transitive_cycle_rejected`.

Any edge insertion must be tested through `add_causal_edge()` so label validation and cycle detection are exercised.

Temporal tests that write WAL data should use `tmp_path`; do not write durable test artifacts into the repository root.

## Core Invariants

Preserve these invariants unless a change explicitly updates the data model and tests:

- Tombstones are append-only. Removing a node or edge adds to the remove set; it never deletes from the add set.
- CRDT merges use set union for add/remove sets.
- Lamport vectors merge by elementwise maximum.
- The causal graph must remain acyclic.
- Identical canonical content must produce identical SHA-256 node IDs.
- Removed nodes cannot be re-added because the graph uses 2P-set semantics.
- Cross-namespace replica merges must be rejected.
- WAL-backed mutations must be logged before in-memory commit.
- Lazy confidence decay must not mutate stored node confidence.
- Access tracking is ephemeral metadata, not durable CRDT state.

## Commit & Pull Request Guidelines

The current history uses short, direct commit subjects such as `Week 1` and cleanup commits. Prefer concise imperative messages, for example `Add WAL replay tests` or `Fix Lamport merge snapshot`.

Pull requests should include a summary, the requirement or behavior changed, test evidence, and any performance or CRDT-safety implications. Link related issues when available.

## Agent-Specific Instructions

Before editing core logic, identify which invariant the change touches. If a change affects CRDT state, WAL replay, or graph edges, add or update tests that prove the expected mathematical property.

Do not bypass public methods by mutating private sets, `_nodes`, or `_graph` outside tests. If tests need direct setup, keep it isolated and explain why the public API cannot express the case.

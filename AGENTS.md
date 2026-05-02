# Repository Guidelines

## Project Structure & Module Organization

MemoryMesh is a Python 3.11+ package with a small visualization stack.
Source code for the core library lives in `memorymesh/`; tests live in
`tests/`.

- `memorymesh/types.py`: closed edge-label vocabulary and custom exceptions.
  Keep this vocabulary-only.
- `memorymesh/memory_node.py`: immutable, content-addressed 7-tuple memory node
  model, embedding validation, and Lamport-vector helpers.
- `memorymesh/mesh_core.py`: 2P2P-Graph CRDT engine, Lamport clocks, causal
  graph operations, WAL integration, point-in-time queries, snapshots, and lazy
  temporal reads.
- `memorymesh/trust_engine.py`: Bayesian trust scoring and append-only in-memory
  audit trail.
- `memorymesh/wal.py`: fsync'd JSON-lines write-ahead log, replay, snapshots,
  and compaction.
- `memorymesh/temporal.py`: Ebbinghaus confidence decay and spaced repetition
  access tracking.
- `memorymesh/__init__.py`: public package exports and package version.
- `tests/test_brain_stem.py`: Week 1 core correctness suite.
- `tests/test_temporal.py`: Week 2 temporal persistence suite.
- `api.py`: read-only FastAPI bridge that serves `GET /api/graph` from
  `sandbox.wal`.
- `sandbox.py` and `swarm_simulation.py`: local demo scripts that regenerate
  `sandbox.wal` and populate the visualization graph.
- `mesh-dashboard/`: Vite + React dashboard that polls the FastAPI graph API and
  renders the DAG with `react-force-graph-2d`.

Generated caches, virtual environments, egg metadata, `.vite/`, `dist/`,
temporary WAL files, `sandbox.wal`, and benchmark artifacts should not be
committed.

## Build, Test, and Development Commands

Install the Python package in editable development mode:

```bash
pip install -e ".[dev]"
```

Run the full Python test suite:

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

If Windows temp or cache directories are permission-blocked, keep pytest
artifacts inside the workspace:

```bash
python -m pytest tests/ -v -o cache_dir=.tmp/pytest-cache --basetemp=.tmp/pytest-tmp
```

Run the demo data generator:

```bash
python swarm_simulation.py
```

Run the graph API for the dashboard. `fastapi` and `uvicorn` are used by
`api.py` but are not part of the core package dependencies:

```bash
uvicorn api:app --reload --port 8000
```

Run the dashboard from `mesh-dashboard/`:

```bash
npm install
npm run dev
npm run build
npm run lint
```

## Coding Style & Naming Conventions

Use 4-space indentation, type annotations, and small modules with clear
ownership. Use descriptive snake_case names for Python functions, variables, and
modules. Class names use PascalCase. JavaScript and React code in
`mesh-dashboard/` follows the existing Vite/React style: functional components,
hooks, double-quoted strings in JSX files, and utility-first Tailwind classes.

Keep module responsibilities strict:

- CRDT graph logic belongs in `mesh_core.py`.
- Trust and audit logic belongs in `trust_engine.py`.
- WAL serialization and replay belong in `wal.py`.
- Decay and access-frequency math belong in `temporal.py`.
- Shared labels and exceptions belong in `types.py`.
- API serialization belongs in `api.py`; do not move visualization-only JSON
  shaping into the core package.
- Dashboard rendering, polling, and inspector UI belong in `mesh-dashboard/src/`.

Memory nodes are immutable. Update behavior should return new values or merge
via `MemoryNode.merge_with()`. Always use `content_address()` for node IDs. Do
not introduce UUID-based node IDs.

NumPy embeddings must remain contiguous `float32` arrays with shape `(384,)`
unless the embedding model contract is deliberately changed and tested.

## Testing Guidelines

Tests use `pytest`. Add focused tests for every behavioral change, especially
CRDT properties, DAG cycle safety, Lamport vector semantics, WAL replay,
time-travel queries, decay calculations, and trust updates.

Name test classes by feature, for example `TestCycleDetection`, and test
functions by expected behavior, for example `test_transitive_cycle_rejected`.

Any edge insertion must be tested through `add_causal_edge()` so label
validation and cycle detection are exercised.

Temporal tests that write WAL data should use `tmp_path`; do not write durable
test artifacts into the repository root.

If a change touches `api.py`, add or update tests when feasible for response
shape and namespace behavior. If a change touches `mesh-dashboard/`, run
`npm run lint` and `npm run build` from `mesh-dashboard/`.

## Core Invariants

Preserve these invariants unless a change explicitly updates the data model and
tests:

- Tombstones are append-only. Removing a node or edge adds to the remove set; it
  never deletes from the add set.
- CRDT merges use set union for add/remove sets.
- Lamport vectors merge by elementwise maximum.
- The causal graph must remain acyclic.
- Identical canonical content must produce identical SHA-256 node IDs.
- Removed nodes cannot be re-added because the graph uses 2P-set semantics.
- Cross-namespace replica merges must be rejected.
- WAL-backed mutations must be logged before in-memory commit.
- WAL replay must not re-log replayed entries.
- Lazy confidence decay must not mutate stored node confidence.
- Access tracking is ephemeral metadata, not durable CRDT state.
- API and dashboard code must treat the graph API as read-only unless a feature
  explicitly introduces write endpoints and tests their CRDT/WAL safety.

## Frontend and API Notes

The dashboard expects the API at `http://localhost:8000/api/graph` and polls it
once per second. Keep `api.py` response objects compatible with the current
shape:

- Nodes include `id`, `label`, `confidence`, `lamport_vector`, and
  `full_content`.
- Edges include `source`, `target`, and `label`.

The API rebuilds its in-memory mesh from `sandbox.wal` at startup. Demo scripts
may remove and regenerate that file; tests should not depend on it.

Keep dashboard UI changes operational and dense rather than marketing-oriented.
The first screen is the graph workspace, not a landing page.

## Commit & Pull Request Guidelines

The current history uses short, direct commit subjects such as `Week 1` and
cleanup commits. Prefer concise imperative messages, for example `Add WAL
replay tests` or `Fix Lamport merge snapshot`.

Pull requests should include a summary, the requirement or behavior changed,
test evidence, and any performance or CRDT-safety implications. Link related
issues when available.

## Agent-Specific Instructions

Before editing core logic, identify which invariant the change touches. If a
change affects CRDT state, WAL replay, or graph edges, add or update tests that
prove the expected mathematical property.

Do not bypass public methods by mutating private sets, `_nodes`, or `_graph`
outside tests. If tests need direct setup, keep it isolated and explain why the
public API cannot express the case.

Do not delete or rewrite user-generated WAL/demo data unless explicitly asked.
Generated files may be ignored by git, but they can still be useful for local
dashboard state.

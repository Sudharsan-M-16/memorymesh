# Repository Guidelines

## Project Structure & Module Organization

MemoryMesh is a Python 3.11+ package. Source code lives in `memorymesh/`; tests live in `tests/`.

- `memorymesh/types.py`: shared edge labels and custom exceptions.
- `memorymesh/memory_node.py`: immutable content-addressed memory node model.
- `memorymesh/mesh_core.py`: 2P2P-Graph CRDT engine, Lamport clocks, causal graph operations.
- `memorymesh/trust_engine.py`: Bayesian trust scoring and audit trail.
- `memorymesh/__init__.py`: public package exports.
- `tests/test_brain_stem.py`: current correctness suite for the Week 1 core.

Generated caches, virtual environments, and benchmark artifacts should not be committed.

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

Smoke-test package imports:

```bash
python -c "from memorymesh import MemoryMeshCore, MemoryNode, BayesianTrustEngine; print('OK')"
```

## Coding Style & Naming Conventions

Use 4-space indentation, type annotations, and small modules with clear ownership. Keep CRDT logic in `mesh_core.py`, trust logic in `trust_engine.py`, and vocabulary-only definitions in `types.py`.

Memory nodes are immutable; update behavior should return new values or merge via `MemoryNode.merge_with()`. Always use `content_address()` for node IDs. Do not introduce UUID-based node IDs.

Use descriptive snake_case names for functions, variables, and modules. Class names use PascalCase.

## Testing Guidelines

Tests use `pytest`. Add focused tests for every behavioral change, especially CRDT properties, DAG cycle safety, Lamport vector semantics, and trust updates.

Name test classes by feature, for example `TestCycleDetection`, and test functions by expected behavior, for example `test_transitive_cycle_rejected`.

Any edge insertion must be tested through `add_causal_edge()` so cycle detection is exercised.

## Commit & Pull Request Guidelines

The current history uses short, direct commit subjects such as `Week 1` and cleanup commits. Prefer concise imperative messages, for example `Add WAL replay tests` or `Fix Lamport merge snapshot`.

Pull requests should include a summary, the requirement or behavior changed, test evidence, and any performance or CRDT-safety implications. Link related issues when available.

## Agent-Specific Instructions

Preserve the core invariants: tombstones are append-only, CRDT merges use set union, Lamport vectors merge by elementwise maximum, and the causal graph must remain acyclic.

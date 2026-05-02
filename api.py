"""Read-only FastAPI bridge for MemoryMesh DAG visualization.

Exposes endpoints that format the in-memory CRDT graph into JSON
structures compatible with standard graph visualizers.

Endpoints:
    GET /api/graph           — nodes + edges (basic, backward-compatible)
    GET /api/graph/conflicts — nodes + edges + conflict overlays

Run:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from memorymesh import MemoryMeshCore

# ---------------------------------------------------------------------------
# Global mesh instance — replays sandbox.wal on startup to reconstruct state
# ---------------------------------------------------------------------------

mesh = MemoryMeshCore(namespace="sandbox", wal_path="sandbox.wal")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MemoryMesh DAG API",
    description="Read-only bridge for visualizing the MemoryMesh causal graph.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/graph")
def get_graph() -> dict:
    """Return the full DAG as nodes + edges for visualization."""
    nodes = [
        {
            "id": node.id,
            "label": node.content[:20],
            "confidence": node.confidence,
            "lamport_vector": node.lamport_vector,
            "full_content": node.content,
        }
        for node in mesh.get_all_nodes()
    ]

    edges = [
        {
            "source": src,
            "target": dst,
            "label": label,
        }
        for src, dst, label in mesh.observed_edges()
    ]

    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph/conflicts")
def get_graph_with_conflicts() -> dict:
    """Return the full DAG with conflict overlays.

    Each node includes ``is_canonical``, ``posterior``,
    ``high_uncertainty``, and ``conflict_count`` fields.
    The response also includes a top-level ``conflicts`` array
    with full conflict resolution records.
    """
    return mesh.get_graph_with_conflicts()

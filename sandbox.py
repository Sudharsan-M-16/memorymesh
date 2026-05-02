import json
import os
from memorymesh import MemoryMeshCore, EdgeLabel, CycleDetectedError

# Clean up old WAL file from previous runs so we start fresh
if os.path.exists("sandbox.wal"):
    os.remove("sandbox.wal")

print("=== 🧠 MemoryMesh v0.2.0 Sandbox ===")

# 1. Initialize the Mesh (This triggers the WAL persistence)
mesh = MemoryMeshCore(namespace="sandbox", wal_path="sandbox.wal")
print("\n[System] Mesh initialized.")

# 2. Agent 1 writes a memory
print("\n[Agent-Monitor] Observing the environment...")
obs_node = mesh.write_memory(
    content="The API is returning 500 Internal Server Error",
    agent_id="Agent-Monitor",
    confidence=1.0
)
print(f"  -> Node ID generated: {obs_node.id}")
print(f"  -> Lamport Clock: {obs_node.lamport_vector}")

# 3. Agent 2 writes a memory based on Agent 1's observation
print("\n[Agent-DevOps] Deriving a conclusion...")
action_node = mesh.write_memory(
    content="Restart the main database container",
    agent_id="Agent-DevOps",
    confidence=0.85
)
print(f"  -> Lamport Clock: {action_node.lamport_vector}")

# 4. Connect the thoughts (The Directed Acyclic Graph)
print("\n[System] Linking observation to conclusion...")
mesh.add_causal_edge(
    source_id=obs_node.id,
    target_id=action_node.id,
    label=EdgeLabel.CAUSED_BY
)
print(f"  -> Total Nodes in Mesh: {mesh.node_count}")
print(f"  -> Total Edges in Mesh: {mesh.edge_count}")

# 5. Inspect the "Flight Recorder" (The WAL)
# 5. The Paradox Test: Intentionally cause a cycle
print("\n[System] Attempting to create a paradox...")
try:
    mesh.add_causal_edge(
        source_id=action_node.id,  # The conclusion
        target_id=obs_node.id,     # The observation
        label=EdgeLabel.CAUSED_BY
    )
except CycleDetectedError as e:
    print(f"🛑 FATAL: {e}")
print("\n=== 📜 Inspecting the Write-Ahead Log ===")
with open("sandbox.wal", "r") as f:
    for line in f:
        entry = json.loads(line)
        print(f" Seq {entry['seq']} | Op: {entry['op']} | Agent: {entry['agent_id']}")

print("\nSandbox execution complete!")
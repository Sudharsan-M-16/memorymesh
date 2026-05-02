import time
import os
from memorymesh import MemoryMeshCore, EdgeLabel

# 1. Clear the old graph so we start with a blank canvas
wal_file = "sandbox.wal"
if os.path.exists(wal_file):
    os.remove(wal_file)
    print("🧹 Cleared old sandbox.wal")

# 2. Boot up the mesh
mesh = MemoryMeshCore(namespace="sandbox", wal_path=wal_file)

print("🚀 Starting Swarm Simulation... Watch your React Dashboard!")
time.sleep(2) # Give you time to switch windows

# 3. Agent Alpha starts the sequence
node_1 = mesh.write_memory(content="System boot sequence initiated.", agent_id="Agent-Alpha", confidence=0.95)
print(f"Alpha added: {node_1.id[:8]}")
time.sleep(1.5) # Watch the UI!

# 4. Agent Beta reacts to Alpha
node_2 = mesh.write_memory(content="Boot sequence verified. Starting API gateway.", agent_id="Agent-Beta", confidence=0.88)
mesh.add_causal_edge(source_id=node_1.id, target_id=node_2.id, label=EdgeLabel.CAUSED_BY)
print(f"Beta added: {node_2.id[:8]}")
time.sleep(1.5)

# 5. Agent Charlie also reacts to Alpha (Branching the graph!)
node_3 = mesh.write_memory(content="Boot sequence logged in audit trail.", agent_id="Agent-Charlie", confidence=0.9)
mesh.add_causal_edge(source_id=node_1.id, target_id=node_3.id, label=EdgeLabel.CAUSED_BY)
print(f"Charlie added: {node_3.id[:8]}")
time.sleep(1.5)

# 6. Agent Alpha reacts to Beta
node_4 = mesh.write_memory(content="Traffic spike detected on API gateway.", agent_id="Agent-Alpha", confidence=0.6)
mesh.add_causal_edge(source_id=node_2.id, target_id=node_4.id, label=EdgeLabel.CAUSED_BY)
print(f"Alpha added: {node_4.id[:8]}")
time.sleep(1.5)

# 7. Agent Beta reacts to Alpha's warning
node_5 = mesh.write_memory(content="Auto-scaling web servers to handle traffic.", agent_id="Agent-Beta", confidence=0.85)
mesh.add_causal_edge(source_id=node_4.id, target_id=node_5.id, label=EdgeLabel.CAUSED_BY)
print(f"Beta added: {node_5.id[:8]}")
time.sleep(1.5)

# 8. Agent Charlie makes an independent observation (Floating node)
node_6 = mesh.write_memory(content="Background ML training job started on GPU-1.", agent_id="Agent-Charlie", confidence=0.99)
print(f"Charlie added: {node_6.id[:8]}")
time.sleep(1.5)

# 9. Agent Alpha reacts to BOTH the web servers (node_5) and the ML job (node_6) - Merging the graph!
node_7 = mesh.write_memory(content="CRITICAL: Server load at 99%. CPU and GPU maxed out.", agent_id="Agent-Alpha", confidence=0.35) # Red confidence!
mesh.add_causal_edge(source_id=node_5.id, target_id=node_7.id, label=EdgeLabel.CAUSED_BY)
mesh.add_causal_edge(source_id=node_6.id, target_id=node_7.id, label=EdgeLabel.CAUSED_BY)
print(f"Alpha added: {node_7.id[:8]}")

print("✅ Swarm simulation complete!")
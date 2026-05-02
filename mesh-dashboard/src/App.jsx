import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_URL = "http://localhost:8000/api/graph";
const POLL_INTERVAL_MS = 1000;

const confidenceTone = (confidence) => {
  const value = Number(confidence) || 0;

  if (value >= 0.8) {
    return {
      fill: "#22c55e",
      glow: "rgba(34, 197, 94, 0.32)",
      text: "text-emerald-300",
      bar: "bg-emerald-400",
    };
  }

  if (value >= 0.4) {
    return {
      fill: "#eab308",
      glow: "rgba(234, 179, 8, 0.3)",
      text: "text-yellow-300",
      bar: "bg-yellow-400",
    };
  }

  return {
    fill: "#ef4444",
    glow: "rgba(239, 68, 68, 0.32)",
    text: "text-red-300",
    bar: "bg-red-400",
  };
};

const getNodeColor = (confidence) => confidenceTone(confidence).fill;

const shortId = (id, head = 12, tail = 8) => {
  const value = String(id ?? "");
  if (value.length <= head + tail + 3) return value;
  return `${value.slice(0, head)}...${value.slice(-tail)}`;
};

const normalizeEndpointGraph = (payload, previousNodeState) => {
  const rawNodes = Array.isArray(payload?.nodes) ? payload.nodes : [];
  const nodes = rawNodes
    .filter((node) => node && node.id !== undefined && node.id !== null)
    .map((node) => {
      const id = String(node.id);
      const previous = previousNodeState.get(id);

      return {
        ...node,
        id,
        confidence: Number.isFinite(Number(node.confidence))
          ? Number(node.confidence)
          : 0,
        x: previous?.x,
        y: previous?.y,
        vx: previous?.vx,
        vy: previous?.vy,
      };
    });

  const nodeIds = new Set(nodes.map((node) => node.id));
  const rawEdges = Array.isArray(payload?.edges) ? payload.edges : [];
  const links = rawEdges
    .map((edge, index) => {
      const source =
        edge?.source ?? edge?.source_id ?? edge?.from ?? edge?.parent;
      const target = edge?.target ?? edge?.target_id ?? edge?.to ?? edge?.child;

      return {
        ...edge,
        id: edge?.id ?? `${source ?? "unknown"}-${target ?? "unknown"}-${index}`,
        source: source === undefined || source === null ? "" : String(source),
        target: target === undefined || target === null ? "" : String(target),
        label: edge?.label === undefined || edge?.label === null ? "" : String(edge.label),
      };
    })
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));

  return { nodes, links };
};

const graphFingerprint = (graph) =>
  JSON.stringify({
    nodes: graph.nodes
      .map((node) => ({
        id: node.id,
        confidence: node.confidence,
        lamport_vector: node.lamport_vector,
        full_content: node.full_content,
      }))
      .sort((a, b) => a.id.localeCompare(b.id)),
    links: graph.links
      .map((link) => ({
        source:
          typeof link.source === "object" && link.source !== null
            ? link.source.id
            : link.source,
        target:
          typeof link.target === "object" && link.target !== null
            ? link.target.id
            : link.target,
        label: link.label,
      }))
      .sort((a, b) =>
        `${a.source}:${a.target}:${a.label}`.localeCompare(
          `${b.source}:${b.target}:${b.label}`,
        ),
      ),
  });

const useViewportSize = () => {
  const [size, setSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const onResize = () => {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    };

    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return size;
};

const vectorEntries = (vector) => {
  if (!vector || typeof vector !== "object") return [];
  if (Array.isArray(vector)) {
    return vector.map((value, index) => [`${index}`, value]);
  }
  return Object.entries(vector).sort(([left], [right]) => left.localeCompare(right));
};

function InspectorPanel({ node, onClose }) {
  const confidence = Math.max(0, Math.min(1, Number(node?.confidence) || 0));
  const tone = confidenceTone(confidence);
  const vector = vectorEntries(node?.lamport_vector);

  return (
    <aside
      className={`absolute right-0 top-0 z-30 h-full w-full max-w-[26rem] transform border-l border-cyan-300/15 bg-slate-950/95 shadow-2xl shadow-cyan-950/40 backdrop-blur-xl transition-transform duration-300 ease-out ${node ? "translate-x-0" : "translate-x-full"
        }`}
    >
      {node ? (
        <div className="flex h-full flex-col">
          <div className="border-b border-white/10 px-5 py-4">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[0.68rem] font-semibold uppercase tracking-[0.24em] text-cyan-300">
                  Node Inspector
                </p>
                <p
                  className="mt-2 truncate font-mono text-sm text-slate-200"
                  title={String(node.id)}
                >
                  {shortId(node.id, 18, 12)}
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-white/10 bg-white/[0.04] text-lg leading-none text-slate-300 transition hover:border-cyan-300/50 hover:bg-cyan-300/10 hover:text-white"
                aria-label="Close node inspector"
              >
                x
              </button>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
            <section className="mb-6">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-[0.68rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Confidence
                </h2>
                <span className={`font-mono text-sm ${tone.text}`}>
                  {confidence.toFixed(3)}
                </span>
              </div>
              <div className="h-3 overflow-hidden rounded-full border border-white/10 bg-slate-900">
                <div
                  className={`h-full rounded-full ${tone.bar} transition-all duration-300`}
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
            </section>

            <section className="mb-6">
              <h2 className="mb-3 text-[0.68rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
                Lamport Vector
              </h2>
              <div className="flex flex-wrap gap-2">
                {vector.length > 0 ? (
                  vector.map(([replica, clock]) => (
                    <span
                      key={replica}
                      className="max-w-full rounded-md border border-cyan-300/20 bg-cyan-300/10 px-2.5 py-1 font-mono text-xs text-cyan-100 shadow-sm shadow-cyan-950/30"
                      title={`${replica}: ${String(clock)}`}
                    >
                      <span className="text-cyan-300">{replica}</span>
                      <span className="mx-1 text-slate-500">/</span>
                      <span className="text-slate-100">{String(clock)}</span>
                    </span>
                  ))
                ) : (
                  <span className="rounded-md border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono text-xs text-slate-500">
                    empty
                  </span>
                )}
              </div>
            </section>

            <section>
              <h2 className="mb-3 text-[0.68rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
                Full Content
              </h2>
              <div className="max-h-[42vh] overflow-y-auto rounded-md border border-white/10 bg-black/25 p-4 font-mono text-sm leading-6 text-slate-200 shadow-inner shadow-black/30">
                <p className="whitespace-pre-wrap break-words">
                  {String(node.full_content ?? "")}
                </p>
              </div>
            </section>
          </div>
        </div>
      ) : null}
    </aside>
  );
}

export default function App() {
  const viewport = useViewportSize();
  const fgRef = useRef(null);
  const lastFingerprintRef = useRef("");
  const nodePhysicsRef = useRef(new Map());
  const requestInFlightRef = useRef(false);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [connectionState, setConnectionState] = useState("connecting");
  const [lastUpdated, setLastUpdated] = useState(null);

  const selectedNode = useMemo(
    () => graphData.nodes.find((node) => node.id === selectedNodeId) ?? null,
    [graphData.nodes, selectedNodeId],
  );

  const rememberPhysics = useCallback((nodes) => {
    const next = new Map();
    nodes.forEach((node) => {
      next.set(node.id, {
        x: node.x,
        y: node.y,
        vx: node.vx,
        vy: node.vy,
      });
    });
    nodePhysicsRef.current = next;
  }, []);

  const fetchGraph = useCallback(
    async (signal) => {
      if (requestInFlightRef.current) return;
      requestInFlightRef.current = true;

      try {
        rememberPhysics(graphData.nodes);

        const response = await fetch(API_URL, {
          cache: "no-store",
          signal,
        });

        if (!response.ok) {
          throw new Error(`Graph endpoint returned ${response.status}`);
        }

        const payload = await response.json();
        const normalized = normalizeEndpointGraph(payload, nodePhysicsRef.current);
        const fingerprint = graphFingerprint(normalized);

        if (fingerprint !== lastFingerprintRef.current) {
          lastFingerprintRef.current = fingerprint;
          setGraphData(normalized);
        }

        setConnectionState("live");
        setLastUpdated(new Date());
      } catch (error) {
        if (error.name !== "AbortError") {
          setConnectionState("offline");
        }
      } finally {
        requestInFlightRef.current = false;
      }
    },
    [graphData.nodes, rememberPhysics],
  );

  useEffect(() => {
    const controller = new AbortController();

    fetchGraph(controller.signal);
    const intervalId = window.setInterval(() => {
      fetchGraph(controller.signal);
    }, POLL_INTERVAL_MS);

    return () => {
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [fetchGraph]);

  useEffect(() => {
    if (!fgRef.current) return;

    fgRef.current.d3Force("charge")?.strength(-400);
    fgRef.current.d3Force("link")?.distance(150);
    fgRef.current.d3ReheatSimulation?.();
  }, []);

  const drawNode = useCallback(
    (node, ctx, globalScale) => {
      const confidence = Math.max(0, Math.min(1, Number(node.confidence) || 0));
      const tone = confidenceTone(confidence);
      const selected = node.id === selectedNodeId;
      const radius = selected ? 8.5 : 7;
      const scaledRadius = radius / Math.sqrt(Math.max(globalScale, 0.6));

      ctx.save();
      ctx.beginPath();
      ctx.arc(node.x, node.y, scaledRadius + 8 / globalScale, 0, Math.PI * 2);
      ctx.fillStyle = tone.glow;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(node.x, node.y, scaledRadius, 0, Math.PI * 2);
      ctx.fillStyle = tone.fill;
      ctx.fill();

      ctx.lineWidth = selected ? 2.5 / globalScale : 1.2 / globalScale;
      ctx.strokeStyle = selected ? "#67e8f9" : "rgba(226, 232, 240, 0.72)";
      ctx.stroke();

      if (globalScale >= 0.9) {
        ctx.font = `${Math.max(8, 10 / globalScale)}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = selected
          ? "rgba(224, 242, 254, 0.96)"
          : "rgba(203, 213, 225, 0.78)";
        ctx.fillText(shortId(node.id, 6, 4), node.x, node.y + scaledRadius + 4);
      }

      ctx.restore();
    },
    [selectedNodeId],
  );

  const drawLinkLabel = useCallback((link, ctx, globalScale) => {
    if (!link.label) return;

    const source = link.source;
    const target = link.target;
    if (
      !source ||
      !target ||
      typeof source !== "object" ||
      typeof target !== "object"
    ) {
      return;
    }

    const text = String(link.label);
    const middleX = source.x + (target.x - source.x) * 0.54;
    const middleY = source.y + (target.y - source.y) * 0.54;
    const angle = Math.atan2(target.y - source.y, target.x - source.x);
    const readableAngle =
      angle > Math.PI / 2 || angle < -Math.PI / 2 ? angle + Math.PI : angle;
    const fontSize = Math.max(8, 11 / globalScale);
    const paddingX = 6 / globalScale;
    const paddingY = 3 / globalScale;

    ctx.save();
    ctx.translate(middleX, middleY);
    ctx.rotate(readableAngle);
    ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const width = ctx.measureText(text).width + paddingX * 2;
    const height = fontSize + paddingY * 2;
    const radius = 4 / globalScale;
    const x = -width / 2;
    const y = -height / 2;

    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    ctx.fillStyle = "rgba(2, 6, 23, 0.86)";
    ctx.strokeStyle = "rgba(103, 232, 249, 0.28)";
    ctx.lineWidth = 1 / globalScale;
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "rgba(224, 242, 254, 0.95)";
    ctx.fillText(text, 0, 0.5 / globalScale);
    ctx.restore();
  }, []);

  const statusClass =
    connectionState === "live"
      ? "bg-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.9)]"
      : connectionState === "offline"
        ? "bg-red-400 shadow-[0_0_16px_rgba(248,113,113,0.9)]"
        : "bg-yellow-400 shadow-[0_0_16px_rgba(250,204,21,0.9)]";

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(14,165,233,0.18),transparent_30%),radial-gradient(circle_at_80%_20%,rgba(34,197,94,0.08),transparent_26%),linear-gradient(135deg,#020617_0%,#0f172a_48%,#020617_100%)]" />
      <div className="absolute inset-0 opacity-[0.08] [background-image:linear-gradient(rgba(255,255,255,0.7)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.7)_1px,transparent_1px)] [background-size:32px_32px]" />

      <header className="pointer-events-none absolute left-0 right-0 top-0 z-20 flex h-16 items-center justify-between border-b border-white/10 bg-slate-950/70 px-5 backdrop-blur-xl">
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold uppercase tracking-[0.28em] text-slate-100">
            MemoryMesh DAG
          </h1>
          <p className="mt-1 font-mono text-xs text-slate-500">
            {graphData.nodes.length} nodes / {graphData.links.length} causal edges
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className={`h-2.5 w-2.5 rounded-full ${statusClass}`} />
          <div className="text-right font-mono text-xs">
            <p className="uppercase tracking-[0.2em] text-slate-300">
              {connectionState}
            </p>
            <p className="text-slate-600">
              {lastUpdated ? lastUpdated.toLocaleTimeString() : "--:--:--"}
            </p>
          </div>
        </div>
      </header>

      <section className="absolute inset-0">
        <ForceGraph2D
          ref={fgRef}
          width={viewport.width}
          height={viewport.height}
          graphData={graphData}
          nodeColor={(node) => getNodeColor(node.confidence)}
          backgroundColor="rgba(0,0,0,0)"
          cooldownTicks={90}
          d3AlphaDecay={0.035}
          d3VelocityDecay={0.24}
          nodeRelSize={6}
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(node.x, node.y, 14, 0, Math.PI * 2);
            ctx.fill();
          }}
          nodeLabel={(node) =>
            `${node.id}\nconfidence: ${(Number(node.confidence) || 0).toFixed(3)}`
          }
          linkColor={() => "rgba(148, 163, 184, 0.42)"}
          linkWidth={1.25}
          linkDirectionalArrowLength={8}
          linkDirectionalArrowRelPos={0.93}
          linkDirectionalArrowColor={() => "rgba(103, 232, 249, 0.82)"}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleSpeed={0.0035}
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={drawLinkLabel}
          onNodeClick={(node) => {
            setSelectedNodeId(node.id);
            fgRef.current?.centerAt(node.x, node.y, 450);
            fgRef.current?.zoom(2.1, 450);
          }}
          onBackgroundClick={() => setSelectedNodeId(null)}
        />
      </section>

      <div className="pointer-events-none absolute bottom-4 left-4 z-20 rounded-md border border-white/10 bg-slate-950/75 px-3 py-2 font-mono text-xs text-slate-400 shadow-lg shadow-black/30 backdrop-blur-xl">
        <span className="mr-2 text-slate-500">confidence</span>
        <span className="mr-2 text-emerald-300">&gt;= 0.8</span>
        <span className="mr-2 text-yellow-300">0.4-0.79</span>
        <span className="text-red-300">&lt; 0.4</span>
      </div>

      <InspectorPanel node={selectedNode} onClose={() => setSelectedNodeId(null)} />
    </main>
  );
}

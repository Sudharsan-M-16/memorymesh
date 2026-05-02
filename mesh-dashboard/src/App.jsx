import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_URL = "http://localhost:8000/api/graph/conflicts";
const POLL_INTERVAL_MS = 1000;
const CONTRADICTS_LABEL = "contradicts";

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

const confidenceTone = (value) => {
  const v = Number(value) || 0;
  if (v >= 0.8) return { fill: "#22c55e", glow: "rgba(34,197,94,0.32)", text: "text-emerald-300", bar: "bg-emerald-400" };
  if (v >= 0.4) return { fill: "#eab308", glow: "rgba(234,179,8,0.3)", text: "text-yellow-300", bar: "bg-yellow-400" };
  return { fill: "#ef4444", glow: "rgba(239,68,68,0.32)", text: "text-red-300", bar: "bg-red-400" };
};

const shortId = (id, head = 12, tail = 8) => {
  const s = String(id ?? "");
  return s.length <= head + tail + 3 ? s : `${s.slice(0, head)}...${s.slice(-tail)}`;
};

const isConflictEdge = (link) => {
  const label = typeof link.label === "string" ? link.label : "";
  return label.toLowerCase() === CONTRADICTS_LABEL;
};

/* ------------------------------------------------------------------ */
/*  Graph normaliser                                                  */
/* ------------------------------------------------------------------ */

const normalizeEndpointGraph = (payload, prevPhysics) => {
  const rawNodes = Array.isArray(payload?.nodes) ? payload.nodes : [];
  const nodes = rawNodes
    .filter((n) => n && n.id != null)
    .map((n) => {
      const id = String(n.id);
      const prev = prevPhysics.get(id);
      return {
        ...n,
        id,
        confidence: Number.isFinite(Number(n.confidence)) ? Number(n.confidence) : 0,
        posterior: Number.isFinite(Number(n.posterior)) ? Number(n.posterior) : 1,
        is_canonical: n.is_canonical !== false,
        high_uncertainty: !!n.high_uncertainty,
        x: prev?.x, y: prev?.y, vx: prev?.vx, vy: prev?.vy,
      };
    });

  const nodeIds = new Set(nodes.map((n) => n.id));
  const rawEdges = Array.isArray(payload?.edges) ? payload.edges : [];
  const links = rawEdges
    .map((e, i) => {
      const src = e?.source ?? e?.source_id ?? e?.from ?? e?.parent;
      const tgt = e?.target ?? e?.target_id ?? e?.to ?? e?.child;
      return {
        ...e,
        id: e?.id ?? `${src ?? "x"}-${tgt ?? "x"}-${i}`,
        source: src == null ? "" : String(src),
        target: tgt == null ? "" : String(tgt),
        label: e?.label == null ? "" : String(e.label),
      };
    })
    .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

  return { nodes, links };
};

const graphFingerprint = (g) =>
  JSON.stringify({
    nodes: g.nodes
      .map((n) => ({ id: n.id, confidence: n.confidence, posterior: n.posterior, is_canonical: n.is_canonical, high_uncertainty: n.high_uncertainty, full_content: n.full_content }))
      .sort((a, b) => a.id.localeCompare(b.id)),
    links: g.links
      .map((l) => ({ source: typeof l.source === "object" ? l.source?.id : l.source, target: typeof l.target === "object" ? l.target?.id : l.target, label: l.label }))
      .sort((a, b) => `${a.source}:${a.target}:${a.label}`.localeCompare(`${b.source}:${b.target}:${b.label}`)),
  });

/* ------------------------------------------------------------------ */
/*  Hooks                                                             */
/* ------------------------------------------------------------------ */

const useViewportSize = () => {
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight });
  useEffect(() => {
    const h = () => setSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, []);
  return size;
};

const vectorEntries = (v) => {
  if (!v || typeof v !== "object") return [];
  if (Array.isArray(v)) return v.map((val, i) => [`${i}`, val]);
  return Object.entries(v).sort(([a], [b]) => a.localeCompare(b));
};

/* ------------------------------------------------------------------ */
/*  Inspector Panel                                                   */
/* ------------------------------------------------------------------ */

function InspectorPanel({ node, onClose }) {
  if (!node) {
    return (
      <aside className="absolute right-0 top-0 z-30 h-full w-full max-w-[26rem] translate-x-full transform border-l border-cyan-300/15 bg-slate-950/95 shadow-2xl shadow-cyan-950/40 backdrop-blur-xl transition-transform duration-300 ease-out" />
    );
  }

  const posterior = Math.max(0, Math.min(1, Number(node.posterior) || 0));
  const confidence = Math.max(0, Math.min(1, Number(node.confidence) || 0));
  const posteriorTone = confidenceTone(posterior);
  const isCanonical = node.is_canonical !== false;
  const highUncertainty = !!node.high_uncertainty;
  const vector = vectorEntries(node.lamport_vector);

  return (
    <aside className="absolute right-0 top-0 z-30 h-full w-full max-w-[26rem] translate-x-0 transform border-l border-cyan-300/15 bg-slate-950/95 shadow-2xl shadow-cyan-950/40 backdrop-blur-xl transition-transform duration-300 ease-out">
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="border-b border-white/10 px-5 py-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <p className="text-[0.68rem] font-semibold uppercase tracking-[0.24em] text-cyan-300">
                Node Inspector
              </p>
              <p className="mt-2 truncate font-mono text-sm text-slate-200" title={String(node.id)}>
                {shortId(node.id, 18, 12)}
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-white/10 bg-white/[0.04] text-lg leading-none text-slate-300 transition hover:border-cyan-300/50 hover:bg-cyan-300/10 hover:text-white"
              aria-label="Close node inspector"
            >
              ✕
            </button>
          </div>

          {/* Status pills */}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {isCanonical ? (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2.5 py-0.5 text-[0.68rem] font-semibold uppercase tracking-wide text-emerald-300">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                Canonical Truth
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-500/30 bg-slate-500/10 px-2.5 py-0.5 text-[0.68rem] font-semibold uppercase tracking-wide text-slate-400">
                <span className="h-1.5 w-1.5 rounded-full bg-slate-500" />
                Shadow Belief
              </span>
            )}
            {highUncertainty && (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 px-2.5 py-0.5 text-[0.68rem] font-semibold uppercase tracking-wide text-amber-300 animate-pulse">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                High Uncertainty
              </span>
            )}
          </div>
        </div>

        {/* Body */}
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          {/* Posterior Probability */}
          <section className="mb-6">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-[0.68rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
                Posterior Probability
              </h2>
              <span className={`font-mono text-sm ${posteriorTone.text}`}>
                {posterior.toFixed(3)}
              </span>
            </div>
            <div className="h-3 overflow-hidden rounded-full border border-white/10 bg-slate-900">
              <div
                className={`h-full rounded-full ${posteriorTone.bar} transition-all duration-300`}
                style={{ width: `${posterior * 100}%` }}
              />
            </div>
          </section>

          {/* Raw Confidence */}
          <section className="mb-6">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-[0.68rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
                Raw Confidence
              </h2>
              <span className="font-mono text-xs text-slate-400">
                {confidence.toFixed(3)}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full border border-white/10 bg-slate-900">
              <div
                className="h-full rounded-full bg-slate-500/60 transition-all duration-300"
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
          </section>

          {/* Lamport Vector */}
          <section className="mb-6">
            <h2 className="mb-3 text-[0.68rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
              Lamport Vector
            </h2>
            <div className="flex flex-wrap gap-2">
              {vector.length > 0 ? (
                vector.map(([replica, clock]) => (
                  <span key={replica} className="max-w-full rounded-md border border-cyan-300/20 bg-cyan-300/10 px-2.5 py-1 font-mono text-xs text-cyan-100 shadow-sm shadow-cyan-950/30" title={`${replica}: ${String(clock)}`}>
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

          {/* Full Content */}
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
    </aside>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                               */
/* ------------------------------------------------------------------ */

export default function App() {
  const viewport = useViewportSize();
  const fgRef = useRef(null);
  const lastFingerprintRef = useRef("");
  const nodePhysicsRef = useRef(new Map());
  const requestInFlightRef = useRef(false);
  const pulseRef = useRef(0);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [connectionState, setConnectionState] = useState("connecting");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [conflictCount, setConflictCount] = useState(0);

  const selectedNode = useMemo(
    () => graphData.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [graphData.nodes, selectedNodeId],
  );

  // Pulse animation timer for high-uncertainty rings
  useEffect(() => {
    let raf;
    const tick = () => {
      pulseRef.current = (Date.now() % 2000) / 2000;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const rememberPhysics = useCallback((nodes) => {
    const m = new Map();
    nodes.forEach((n) => m.set(n.id, { x: n.x, y: n.y, vx: n.vx, vy: n.vy }));
    nodePhysicsRef.current = m;
  }, []);

  const fetchGraph = useCallback(
    async (signal) => {
      if (requestInFlightRef.current) return;
      requestInFlightRef.current = true;
      try {
        rememberPhysics(graphData.nodes);
        const res = await fetch(API_URL, { cache: "no-store", signal });
        if (!res.ok) throw new Error(`${res.status}`);
        const payload = await res.json();
        const normalized = normalizeEndpointGraph(payload, nodePhysicsRef.current);
        const fp = graphFingerprint(normalized);
        if (fp !== lastFingerprintRef.current) {
          lastFingerprintRef.current = fp;
          setGraphData(normalized);
        }
        setConflictCount(Array.isArray(payload.conflicts) ? payload.conflicts.length : 0);
        setConnectionState("live");
        setLastUpdated(new Date());
      } catch (e) {
        if (e.name !== "AbortError") setConnectionState("offline");
      } finally {
        requestInFlightRef.current = false;
      }
    },
    [graphData.nodes, rememberPhysics],
  );

  useEffect(() => {
    const ac = new AbortController();
    fetchGraph(ac.signal);
    const id = setInterval(() => fetchGraph(ac.signal), POLL_INTERVAL_MS);
    return () => { ac.abort(); clearInterval(id); };
  }, [fetchGraph]);

  // D3 physics overrides
  useEffect(() => {
    if (!fgRef.current) return;
    fgRef.current.d3Force("charge")?.strength(-400);
    fgRef.current.d3Force("link")?.distance(150);
    fgRef.current.d3ReheatSimulation?.();
  }, []);

  /* ---- Canvas: draw nodes ---- */
  const drawNode = useCallback(
    (node, ctx, globalScale) => {
      const confidence = Math.max(0, Math.min(1, Number(node.confidence) || 0));
      const tone = confidenceTone(confidence);
      const selected = node.id === selectedNodeId;
      const isShadow = node.is_canonical === false;
      const isUncertain = !!node.high_uncertainty;
      const radius = selected ? 8.5 : 7;
      const sr = radius / Math.sqrt(Math.max(globalScale, 0.6));
      const nodeAlpha = isShadow ? 0.38 : 1.0;

      ctx.save();
      ctx.globalAlpha = nodeAlpha;

      // Glow
      ctx.beginPath();
      ctx.arc(node.x, node.y, sr + 8 / globalScale, 0, Math.PI * 2);
      ctx.fillStyle = isShadow ? "rgba(100,116,139,0.15)" : tone.glow;
      ctx.fill();

      // Body
      ctx.beginPath();
      ctx.arc(node.x, node.y, sr, 0, Math.PI * 2);
      ctx.fillStyle = isShadow ? "#64748b" : tone.fill;
      ctx.fill();

      // Stroke
      ctx.lineWidth = selected ? 2.5 / globalScale : 1.2 / globalScale;
      ctx.strokeStyle = selected ? "#67e8f9" : isShadow ? "rgba(148,163,184,0.4)" : "rgba(226,232,240,0.72)";
      ctx.stroke();

      ctx.globalAlpha = 1.0;

      // Uncertainty warning ring (pulsing dashed)
      if (isUncertain) {
        const pulse = pulseRef.current;
        const ringRadius = sr + 5 / globalScale;
        const dashLen = 4 / globalScale;
        ctx.beginPath();
        ctx.arc(node.x, node.y, ringRadius, 0, Math.PI * 2);
        ctx.setLineDash([dashLen, dashLen]);
        ctx.lineDashOffset = pulse * dashLen * 8;
        ctx.lineWidth = 1.8 / globalScale;
        ctx.strokeStyle = `rgba(251,191,36,${0.5 + 0.4 * Math.sin(pulse * Math.PI * 2)})`;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Label
      if (globalScale >= 0.9) {
        ctx.font = `${Math.max(8, 10 / globalScale)}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = selected ? "rgba(224,242,254,0.96)" : isShadow ? "rgba(148,163,184,0.55)" : "rgba(203,213,225,0.78)";
        ctx.fillText(shortId(node.id, 6, 4), node.x, node.y + sr + 4);
      }

      ctx.restore();
    },
    [selectedNodeId],
  );

  /* ---- Canvas: draw link labels + conflict dashes ---- */
  const drawLinkLabel = useCallback((link, ctx, globalScale) => {
    const source = link.source;
    const target = link.target;
    if (!source || !target || typeof source !== "object" || typeof target !== "object") return;

    const conflict = isConflictEdge(link);
    const text = String(link.label || "");

    // For conflict edges: draw dashed red line (no arrows drawn by the library for these)
    if (conflict) {
      const dashLen = 6 / globalScale;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.setLineDash([dashLen, dashLen * 0.8]);
      ctx.strokeStyle = "rgba(239,68,68,0.55)";
      ctx.lineWidth = 1.6 / globalScale;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    if (!text) return;

    const mx = source.x + (target.x - source.x) * 0.54;
    const my = source.y + (target.y - source.y) * 0.54;
    const angle = Math.atan2(target.y - source.y, target.x - source.x);
    const ra = angle > Math.PI / 2 || angle < -Math.PI / 2 ? angle + Math.PI : angle;
    const fs = Math.max(8, 11 / globalScale);
    const px = 6 / globalScale;
    const py = 3 / globalScale;

    ctx.save();
    ctx.translate(mx, my);
    ctx.rotate(ra);
    ctx.font = `${fs}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const tw = ctx.measureText(text).width + px * 2;
    const th = fs + py * 2;
    const r = 4 / globalScale;
    const bx = -tw / 2;
    const by = -th / 2;

    ctx.beginPath();
    ctx.moveTo(bx + r, by);
    ctx.lineTo(bx + tw - r, by);
    ctx.quadraticCurveTo(bx + tw, by, bx + tw, by + r);
    ctx.lineTo(bx + tw, by + th - r);
    ctx.quadraticCurveTo(bx + tw, by + th, bx + tw - r, by + th);
    ctx.lineTo(bx + r, by + th);
    ctx.quadraticCurveTo(bx, by + th, bx, by + th - r);
    ctx.lineTo(bx, by + r);
    ctx.quadraticCurveTo(bx, by, bx + r, by);
    ctx.closePath();
    ctx.fillStyle = conflict ? "rgba(127,29,29,0.80)" : "rgba(2,6,23,0.86)";
    ctx.strokeStyle = conflict ? "rgba(239,68,68,0.40)" : "rgba(103,232,249,0.28)";
    ctx.lineWidth = 1 / globalScale;
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = conflict ? "rgba(252,165,165,0.95)" : "rgba(224,242,254,0.95)";
    ctx.fillText(text, 0, 0.5 / globalScale);
    ctx.restore();
  }, []);

  /* ---- Status indicator ---- */
  const statusClass =
    connectionState === "live"
      ? "bg-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.9)]"
      : connectionState === "offline"
        ? "bg-red-400 shadow-[0_0_16px_rgba(248,113,113,0.9)]"
        : "bg-yellow-400 shadow-[0_0_16px_rgba(250,204,21,0.9)]";

  const causalEdgeCount = graphData.links.filter((l) => !isConflictEdge(l)).length;
  const conflictEdgeCount = graphData.links.filter((l) => isConflictEdge(l)).length;

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
            {graphData.nodes.length} nodes / {causalEdgeCount} causal{conflictCount > 0 ? ` / ${conflictCount} conflicts` : ""}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`h-2.5 w-2.5 rounded-full ${statusClass}`} />
          <div className="text-right font-mono text-xs">
            <p className="uppercase tracking-[0.2em] text-slate-300">{connectionState}</p>
            <p className="text-slate-600">{lastUpdated ? lastUpdated.toLocaleTimeString() : "--:--:--"}</p>
          </div>
        </div>
      </header>

      <section className="absolute inset-0">
        <ForceGraph2D
          ref={fgRef}
          width={viewport.width}
          height={viewport.height}
          graphData={graphData}
          nodeColor={(n) => (n.is_canonical === false ? "#64748b" : confidenceTone(n.confidence).fill)}
          backgroundColor="rgba(0,0,0,0)"
          cooldownTicks={90}
          d3AlphaDecay={0.035}
          d3VelocityDecay={0.24}
          nodeRelSize={6}
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={(n, c, ctx) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(n.x, n.y, 14, 0, Math.PI * 2); ctx.fill(); }}
          nodeLabel={(n) => `${n.id}\nposterior: ${(Number(n.posterior) || 0).toFixed(3)}`}
          linkColor={(l) => isConflictEdge(l) ? "rgba(239,68,68,0.0)" : "rgba(148,163,184,0.42)"}
          linkWidth={(l) => isConflictEdge(l) ? 0 : 1.25}
          linkDirectionalArrowLength={(l) => isConflictEdge(l) ? 0 : 8}
          linkDirectionalArrowRelPos={0.93}
          linkDirectionalArrowColor={() => "rgba(103,232,249,0.82)"}
          linkDirectionalParticles={(l) => isConflictEdge(l) ? 0 : 1}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleSpeed={0.0035}
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={drawLinkLabel}
          onNodeClick={(n) => { setSelectedNodeId(n.id); fgRef.current?.centerAt(n.x, n.y, 450); fgRef.current?.zoom(2.1, 450); }}
          onBackgroundClick={() => setSelectedNodeId(null)}
        />
      </section>

      {/* Legend */}
      <div className="pointer-events-none absolute bottom-4 left-4 z-20 flex flex-col gap-1.5 rounded-md border border-white/10 bg-slate-950/75 px-3 py-2 font-mono text-xs text-slate-400 shadow-lg shadow-black/30 backdrop-blur-xl">
        <div>
          <span className="mr-2 text-slate-500">confidence</span>
          <span className="mr-2 text-emerald-300">&gt;= 0.8</span>
          <span className="mr-2 text-yellow-300">0.4–0.79</span>
          <span className="text-red-300">&lt; 0.4</span>
        </div>
        <div>
          <span className="mr-2 text-slate-500">edges</span>
          <span className="mr-2 text-cyan-300">causal →</span>
          <span className="text-red-400">--- conflict</span>
        </div>
        {conflictEdgeCount > 0 && (
          <div>
            <span className="mr-2 text-slate-500">nodes</span>
            <span className="mr-2 text-emerald-300">canonical</span>
            <span className="text-slate-500">shadow</span>
          </div>
        )}
      </div>

      <InspectorPanel node={selectedNode} onClose={() => setSelectedNodeId(null)} />
    </main>
  );
}

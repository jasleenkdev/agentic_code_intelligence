import '@xyflow/react/dist/style.css'
import {
  Background, Controls, MarkerType, Position, ReactFlow, type Edge, type Node,
} from '@xyflow/react'
import { useEffect, useMemo, useState } from 'react'
import { api, type GraphData } from '../lib/api'
import { useCurrentTheme } from '../lib/theme'
import type { FileTarget } from './FileViewer'
import { EmptyState, ErrorBanner, Spinner } from './ui'

const COL = 280
const ROW = 64

/** Layered left-to-right layout: column = longest distance from a root (no incoming edge). */
function layout(g: GraphData): Record<string, { x: number; y: number }> {
  const incoming = new Map<string, number>()
  const out = new Map<string, string[]>()
  for (const n of g.nodes) { incoming.set(n.id, 0); out.set(n.id, []) }
  for (const e of g.edges) {
    incoming.set(e.target, (incoming.get(e.target) ?? 0) + 1)
    out.get(e.source)?.push(e.target)
  }
  const layer = new Map<string, number>()
  const queue = g.nodes.filter((n) => !incoming.get(n.id)).map((n) => n.id)
  if (!queue.length && g.nodes.length) queue.push(g.nodes[0].id)
  queue.forEach((id) => layer.set(id, 0))
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i]
    for (const t of out.get(id) ?? []) {
      const next = (layer.get(id) ?? 0) + 1
      if (!layer.has(t) || (layer.get(t)! < next && next < 8)) {
        layer.set(t, next)
        queue.push(t)
      }
    }
  }
  const rows = new Map<number, number>()
  const pos: Record<string, { x: number; y: number }> = {}
  for (const n of g.nodes) {
    const l = layer.get(n.id) ?? 0
    const r = rows.get(l) ?? 0
    rows.set(l, r + 1)
    pos[n.id] = { x: l * COL, y: r * ROW }
  }
  // Centre each column vertically against the tallest one.
  const tallest = Math.max(1, ...rows.values())
  for (const n of g.nodes) {
    const l = layer.get(n.id) ?? 0
    pos[n.id].y += ((tallest - (rows.get(l) ?? 1)) * ROW) / 2
  }
  return pos
}

function parseLocation(loc: string | null): FileTarget | null {
  const m = loc?.match(/^(.+):(\d+)(?:-(\d+))?$/)
  if (!m) return null
  const start = Number(m[2])
  return { path: m[1], start, end: m[3] ? Number(m[3]) : start }
}

export function CallGraph({ resultId, onOpen }: {
  resultId: string; onOpen: (t: FileTarget) => void
}) {
  const theme = useCurrentTheme()
  const [data, setData] = useState<GraphData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    setData(null); setError(null)
    api.callpaths(resultId).then((d) => live && setData(d)).catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [resultId])

  const { nodes, edges } = useMemo(() => {
    if (!data) return { nodes: [] as Node[], edges: [] as Edge[] }
    const pos = layout(data)
    const nodes: Node[] = data.nodes.map((n) => ({
      id: n.id,
      position: pos[n.id],
      data: { label: n.label, location: n.location },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      className: n.role === 'root' ? 'cg-node cg-root' : 'cg-node',
      style: { cursor: parseLocation(n.location) ? 'pointer' : 'default' },
    }))
    const edges: Edge[] = data.edges.map((e) => ({
      id: `${e.source}->${e.target}`,
      source: e.source,
      target: e.target,
      markerEnd: { type: MarkerType.ArrowClosed },
      style: { strokeWidth: 1.5 },
    }))
    return { nodes, edges }
  }, [data])

  if (error) return <ErrorBanner message={error} />
  if (!data) return <div className="flex justify-center py-10"><Spinner /></div>
  if (!data.nodes.length) {
    return <EmptyState title="No call paths for this answer" />
  }
  return (
    <div className="h-[340px] overflow-hidden rounded-lg border border-app" data-testid="call-graph">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        colorMode={theme}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.2}
        nodesConnectable={false}
        onNodeClick={(_, n) => {
          const t = parseLocation((n.data as { location: string | null }).location)
          if (t) onOpen(t)
        }}
      >
        <Background gap={20} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}

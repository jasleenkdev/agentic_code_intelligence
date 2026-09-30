// Typed client for the FastAPI backend (src/api/app.py). All paths are relative (/api/...):
// same-origin in `./run.sh app`, proxied to :8000 by Vite in dev.

export type StepKind =
  | 'plan' | 'search' | 'read' | 'assess' | 'refine' | 'verify' | 'stop' | 'llm'

export interface Step {
  index: number
  kind: StepKind
  name: string
  inputs: Record<string, unknown>
  outputs: Record<string, unknown>
  summary: string
  result_count: number
  elapsed_ms: number
  ok: boolean
  phase: 'first' | 'refine'
}

export interface Snippet {
  file_path: string
  start_line: number
  end_line: number
  text: string
  symbol: string | null
  reason: string
  highlight_lines: number[]
  verified: boolean
  truncated: boolean
  location: string
}

export interface RouteInfo {
  type: string
  category: string
  method?: string
  params?: Record<string, unknown>
  rewritten_query?: string
}

export interface SubPlan { text: string; route: RouteInfo; steps: string[] }

export interface AskResult {
  id: string
  phase: 'first' | 'final'
  changed: boolean
  query: string
  route: RouteInfo
  answer: string
  snippets: Snippet[]
  call_paths: string[][]
  confidence: string
  plan: { query?: string; method?: string; decomposed?: boolean; subplans?: SubPlan[] }
  latency_ms: number
  final_latency_ms: number
  refinement_rounds: number
  results_changed_by_llm: boolean
  optimizations: Finding[]
  error: string | null
}

export interface DoneInfo {
  id: string
  llm_available: boolean
  latency_ms: number
  final_latency_ms: number
}

export interface Finding {
  check: string
  location: string
  file_path: string
  line: number
  end_line: number
  symbol: string | null
  message: string
  evidence: string
  explanation?: string
}

export interface Stats {
  repo: { name: string; url: string; commit: string }
  files: number
  functions: number
  chunks: number
  call_edges: number
  catalog_rows: number
  embedding: { model: string; text: string }
  built_at: string
  llm: { model: string; available: boolean; cpu_only: boolean }
}

export interface FileWindow {
  path: string
  total_lines: number
  start_line: number
  end_line: number
  highlight: [number, number]
  lines: string[]
}

export interface GraphData {
  nodes: { id: string; label: string; location: string | null; role: string }[]
  edges: { source: string; target: string }[]
  route: string
}

export interface CatalogKind { kind: string; entries: number; usages: number }
export interface CatalogItem {
  kind: string
  name: string
  value: string
  detail: string | null
  file_path: string | null
  line: number | null
  usages: number
}
export interface CatalogRow {
  role: string
  value: string
  detail: string | null
  line: number
  file_path: string
  symbol: string | null
}
export interface CatalogEntry {
  kind: string
  name: string
  definitions: CatalogRow[]
  usages: CatalogRow[]
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function getJson<T>(url: string): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(url)
  } catch {
    throw new ApiError('Cannot reach the local API. Is `./run.sh app` running?', 0)
  }
  if (!resp.ok) {
    let msg = `Request failed (${resp.status})`
    try {
      const body = await resp.json()
      msg = body.detail || body.error || msg
    } catch { /* keep default */ }
    throw new ApiError(typeof msg === 'string' ? msg : JSON.stringify(msg), resp.status)
  }
  return resp.json() as Promise<T>
}

export const api = {
  stats: () => getJson<Stats>('/api/stats'),
  file: (path: string, start: number, end: number, context = 30) =>
    getJson<FileWindow>(
      `/api/file?path=${encodeURIComponent(path)}&start=${start}&end=${end}&context=${context}`,
    ),
  callpaths: (id: string) => getJson<GraphData>(`/api/results/${id}/callpaths`),
  optimizations: (id: string, explain = true) =>
    getJson<{ findings: Finding[]; explained: number; llm_available: boolean }>(
      `/api/results/${id}/optimizations?explain=${explain}`,
    ),
  catalogKinds: () => getJson<CatalogKind[]>('/api/catalog/kinds'),
  catalog: (kind: string | null, q: string, limit = 200) => {
    const p = new URLSearchParams({ limit: String(limit) })
    if (kind) p.set('kind', kind)
    if (q.trim()) p.set('q', q.trim())
    return getJson<{ items: CatalogItem[]; count: number; ranked_by: string }>(`/api/catalog?${p}`)
  },
  catalogEntry: (kind: string, name: string) =>
    getJson<CatalogEntry>(
      `/api/catalog/entry?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}`,
    ),
  transcribe: async (blob: Blob): Promise<{ text: string; elapsed_ms: number }> => {
    const form = new FormData()
    form.append('audio', blob, 'recording.webm')
    let resp: Response
    try {
      resp = await fetch('/api/transcribe', { method: 'POST', body: form })
    } catch {
      throw new ApiError('Cannot reach the local API for transcription.', 0)
    }
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new ApiError(body.detail || 'Transcription failed.', resp.status)
    }
    return resp.json()
  },
}

export interface AskOptions {
  query: string
  use_llm?: boolean
  explain?: boolean
  optimize?: boolean
  refine?: boolean
}

export interface AskHandlers {
  onStep?: (s: Step) => void
  onResults?: (r: AskResult) => void
  onExplanation?: (text: string) => void
  onExplanationDone?: (info: { first_token_ms: number; total_ms: number }) => void
  onDone?: (d: DoneInfo) => void
  onError?: (message: string) => void
}

/** POST /api/ask and dispatch its Server-Sent Events as they arrive. */
export async function askStream(opts: AskOptions, h: AskHandlers, signal?: AbortSignal) {
  let resp: Response
  try {
    resp = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
      signal,
    })
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    h.onError?.('Cannot reach the local API. Is `./run.sh app` running?')
    return
  }
  if (!resp.ok || !resp.body) {
    const body = await resp.json().catch(() => ({}))
    const detail = Array.isArray(body.detail) ? 'Please enter a longer question.' : body.detail
    h.onError?.(detail || body.error || `Request failed (${resp.status})`)
    return
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let cut: number
      while ((cut = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, cut)
        buffer = buffer.slice(cut + 2)
        dispatch(block, h)
      }
    }
  } catch (e) {
    if ((e as Error).name !== 'AbortError') h.onError?.('The connection to the API was interrupted.')
  }
}

function dispatch(block: string, h: AskHandlers) {
  let event = 'message'
  const data: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue
    if (line.startsWith('event: ')) event = line.slice(7)
    else if (line.startsWith('data: ')) data.push(line.slice(6))
  }
  if (!data.length) return
  const payload = JSON.parse(data.join('\n'))
  switch (event) {
    case 'step': h.onStep?.(payload); break
    case 'results': h.onResults?.(payload); break
    case 'explanation': h.onExplanation?.(payload.text); break
    case 'explanation_done': h.onExplanationDone?.(payload); break
    case 'done': h.onDone?.(payload); break
    case 'error': h.onError?.(payload.detail ? `${payload.error} (${payload.detail})` : payload.error); break
  }
}

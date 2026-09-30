// Client for the snippet-library search engine (src/api/codesearch_routes.py, /api/cs/...).
import { ApiError } from './api'

export interface LibraryVersion {
  version: string
  snippets: number
  built_at: string
  encoded: Record<string, number>
  total_s: number
  commit: string
}

export interface Library {
  name: string
  kind: 'snippets' | 'codebase'
  description: string
  models: string[]
  versions: LibraryVersion[]
  has_lineages: boolean
}

export interface PassTrace { name: string; detail: string; ms: number }

export interface LineageMember {
  version: string
  id: string
  score: number | null
  same_text: boolean
}

export interface SearchHit {
  rank: number
  id: string
  version: string
  score: number
  text: string
  path: string
  symbol: string
  start_line: number
  end_line: number
  style: string
  matched_literals?: string[]
  lineage?: { id: number; versions: LineageMember[] }
}

export interface SearchResponse {
  query: string
  library: string
  version: string
  versions: string[]
  library_size: number
  category: {
    style: string
    multi_test: boolean
    interactive: boolean
    topics: string[]
    sections: Record<string, number>
    function_names: string[]
  }
  preprocessing: {
    views: string[]
    cleaned: string
    raw_chars: number
    hint: { kind: string; label: string | null; phrase: string; cleaned_query: string } | null
  }
  pipeline: Record<string, unknown>
  passes: PassTrace[]
  results: SearchHit[]
  latency_ms: number
}

export interface Example { label: string; query: string }

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(url, init)
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

export const cs = {
  libraries: () => request<Library[]>('/api/cs/libraries'),
  examples: () => request<Record<string, Example[]>>('/api/cs/examples'),
  search: (query: string, library: string, version: string | null, k = 10) =>
    request<SearchResponse>('/api/cs/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, library, version, k }),
    }),
  evaluation: () => request<any>('/api/cs/eval'),
}

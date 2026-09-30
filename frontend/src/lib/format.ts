export function ms(v: number | undefined | null): string {
  if (v === undefined || v === null || Number.isNaN(v)) return '—'
  if (v < 1000) return `${Math.round(v)} ms`
  return `${(v / 1000).toFixed(v < 10000 ? 2 : 1)} s`
}

export function pct(v: number | undefined | null, digits = 0): string {
  if (v === undefined || v === null) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export function num(v: number | undefined | null, digits = 3): string {
  if (v === undefined || v === null) return '—'
  return v.toFixed(digits)
}

export const KIND_LABELS: Record<string, string> = {
  mcp_tool: 'MCP tools',
  front_route: 'Deeplinks & routes',
  api_route: 'API routes',
  event: 'Events',
  intent: 'Intents',
  service: 'Services',
  ws_message: 'WebSocket messages',
  action: 'Scene actions',
  job: 'Jobs',
  uri: 'URIs',
  constant: 'Constants',
}

export const ROUTE_LABELS: Record<string, string> = {
  sequence: 'call order',
  callers: 'callers',
  callees: 'callees',
  definition: 'definition',
  usage: 'usage',
  semantic: 'plain English',
  multi: 'multi-part',
}

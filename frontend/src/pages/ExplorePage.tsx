import clsx from 'clsx'
import { ArrowRight, FileCode2, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import type { FileTarget } from '../components/FileViewer'
import { Badge, EmptyState, ErrorBanner, Spinner } from '../components/ui'
import {
  api, type CatalogEntry, type CatalogItem, type CatalogKind, type CatalogRow,
} from '../lib/api'
import { KIND_LABELS } from '../lib/format'

function useDebounced<T>(value: T, delay = 250): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return v
}

export function ExplorePage({ onOpen }: { onOpen: (t: FileTarget) => void }) {
  const [kinds, setKinds] = useState<CatalogKind[]>([])
  const [kind, setKind] = useState<string | null>('front_route')
  const [q, setQ] = useState('')
  const dq = useDebounced(q)
  const [items, setItems] = useState<CatalogItem[] | null>(null)
  const [rankedBy, setRankedBy] = useState('substring')
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<CatalogItem | null>(null)

  useEffect(() => { api.catalogKinds().then(setKinds).catch((e) => setError(e.message)) }, [])

  useEffect(() => {
    let live = true
    setItems(null)
    api.catalog(kind, dq, 300)
      .then((d) => {
        if (!live) return
        setItems(d.items); setRankedBy(d.ranked_by); setError(null)
        setSelected((s) => d.items.find((i) => s && i.kind === s.kind && i.name === s.name)
          ?? d.items[0] ?? null)
      })
      .catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [kind, dq])

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <button
          onClick={() => setKind(null)}
          className={chip(kind === null)}
        >
          All
        </button>
        {kinds.map((k) => (
          <button key={k.kind} onClick={() => setKind(k.kind)} className={chip(kind === k.kind)}>
            {KIND_LABELS[k.kind] ?? k.kind}
            <span className="ml-1.5 tabular-nums text-muted">{k.entries.toLocaleString()}</span>
          </button>
        ))}
      </div>
      {error && <div className="mb-4"><ErrorBanner message={error} /></div>}

      <div className="grid gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
        <div className="panel flex h-[calc(100vh-190px)] min-h-[420px] flex-col">
          <div className="relative border-b border-app p-3">
            <Search className="pointer-events-none absolute left-6 top-5.5 h-4 w-4 text-muted" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search names, values, descriptions…"
              className="w-full rounded-md border border-app bg-transparent py-2 pl-9 pr-3 text-sm outline-none focus:border-teal-500"
              data-testid="catalog-search"
            />
            {items && (
              <p className="mt-2 text-xs text-muted">
                {items.length} {items.length === 300 ? '+ ' : ''}entries
                {rankedBy === 'phrase' && ' · ranked by phrase match'}
              </p>
            )}
          </div>
          <ul className="flex-1 overflow-y-auto p-1.5" data-testid="catalog-list">
            {!items && <li className="flex justify-center py-10"><Spinner /></li>}
            {items && !items.length && (
              <li><EmptyState title="No catalog entries match">Try a shorter search.</EmptyState></li>
            )}
            {items?.map((it) => (
              <li key={`${it.kind}:${it.name}`}>
                <button
                  onClick={() => setSelected(it)}
                  className={clsx(
                    'w-full rounded-md px-3 py-2 text-left transition',
                    selected?.kind === it.kind && selected.name === it.name
                      ? 'bg-teal-500/10 ring-1 ring-teal-500/30' : 'hover:bg-zinc-500/10',
                  )}
                >
                  <div className="flex items-center gap-2">
                    <span className="truncate font-mono text-[13px]">{it.name}</span>
                    <span className="ml-auto shrink-0 text-xs tabular-nums text-muted">
                      {it.usages} use{it.usages === 1 ? '' : 's'}
                    </span>
                  </div>
                  {(it.detail || it.value !== it.name) && (
                    <p className="truncate text-xs text-muted">{it.detail || it.value}</p>
                  )}
                  {kind === null && (
                    <span className="text-[10px] uppercase tracking-wide text-muted">
                      {KIND_LABELS[it.kind] ?? it.kind}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="min-w-0">
          {selected
            ? <EntryDetail item={selected} onOpen={onOpen} />
            : <div className="panel"><EmptyState title="Select an entry to see its definition and usages" /></div>}
        </div>
      </div>
    </div>
  )
}

const chip = (active: boolean) => clsx(
  'rounded-full border px-3 py-1 text-xs transition',
  active ? 'border-teal-500/60 bg-teal-500/10 font-medium' : 'border-app hover:bg-zinc-500/10',
)

function EntryDetail({ item, onOpen }: { item: CatalogItem; onOpen: (t: FileTarget) => void }) {
  const [entry, setEntry] = useState<CatalogEntry | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    setEntry(null); setError(null)
    api.catalogEntry(item.kind, item.name)
      .then((e) => live && setEntry(e)).catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [item.kind, item.name])

  const byFile = useMemo(() => {
    const m = new Map<string, CatalogRow[]>()
    for (const u of entry?.usages ?? []) m.set(u.file_path, [...(m.get(u.file_path) ?? []), u])
    return [...m.entries()]
  }, [entry])

  const question = item.kind === 'front_route'
    ? `Where is the ${item.name} route used?` : `Where is ${item.name} used?`

  return (
    <div className="panel p-5" data-testid="catalog-detail">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0">
          <Badge tone="violet">{KIND_LABELS[item.kind] ?? item.kind}</Badge>
          <h2 className="mt-2 break-all font-mono text-lg font-semibold">{item.name}</h2>
          {item.detail && <p className="mt-1 text-sm text-muted">{item.detail}</p>}
        </div>
        <Link
          to={`/agent?q=${encodeURIComponent(question)}`}
          className="ml-auto flex items-center gap-1 rounded-md border border-app px-3 py-1.5 text-xs hover:bg-zinc-500/10"
        >
          Ask the agent <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>
      {error && <div className="mt-4"><ErrorBanner message={error} /></div>}
      {!entry && !error && <div className="flex justify-center py-10"><Spinner /></div>}
      {entry && (
        <>
          <h3 className="mb-2 mt-5 text-sm font-semibold">
            Definition{entry.definitions.length === 1 ? '' : 's'}
          </h3>
          <RowList rows={entry.definitions} onOpen={onOpen} />
          <h3 className="mb-2 mt-5 text-sm font-semibold">
            Usages <span className="font-normal text-muted">· {entry.usages.length} in {byFile.length} files</span>
          </h3>
          {!entry.usages.length && <p className="text-sm text-muted">No usages found in the index.</p>}
          <div className="space-y-3">
            {byFile.map(([file, rows]) => (
              <div key={file}>
                <p className="mb-1 flex items-center gap-1.5 font-mono text-xs text-muted">
                  <FileCode2 className="h-3.5 w-3.5" /> {file}
                </p>
                <RowList rows={rows} onOpen={onOpen} hideFile />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function RowList({ rows, onOpen, hideFile }: {
  rows: CatalogRow[]; onOpen: (t: FileTarget) => void; hideFile?: boolean
}) {
  return (
    <ul className="divide-y divide-[var(--border)] overflow-hidden rounded-lg border border-app">
      {rows.map((r, i) => (
        <li key={i}>
          <button
            onClick={() => onOpen({ path: r.file_path, start: r.line, end: r.line })}
            className="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-zinc-500/10"
          >
            <span className="w-14 shrink-0 font-mono text-xs text-muted">:{r.line}</span>
            {!hideFile && <span className="truncate font-mono text-xs">{r.file_path}</span>}
            {r.symbol && <span className="shrink-0 truncate font-mono text-xs text-teal-600 dark:text-teal-300">{r.symbol}</span>}
            {r.value && <span className="min-w-0 truncate font-mono text-xs text-muted" title={r.value}>{r.value}</span>}
            <span className="ml-auto shrink-0 text-[11px] text-muted">{r.role}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}

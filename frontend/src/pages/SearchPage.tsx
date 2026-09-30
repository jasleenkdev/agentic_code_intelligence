import clsx from 'clsx'
import { GitBranch, Layers, Search, Timer } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { Badge, Card, EmptyState, ErrorBanner, SectionTitle, Spinner } from '../components/ui'
import {
  cs, type Example, type Library, type PassTrace, type SearchHit, type SearchResponse,
} from '../lib/codesearch'
import { ms } from '../lib/format'

const ALL = 'all'
const LIB_LABELS: Record<string, string> = { apps: 'APPS corpus', gladys: 'Gladys codebase' }

export function SearchPage() {
  const [params, setParams] = useSearchParams()
  const [libs, setLibs] = useState<Library[] | null>(null)
  const [examples, setExamples] = useState<Record<string, Example[]>>({})
  const [wanted, setLibrary] = useState(params.get('library') ?? 'apps')
  const [version, setVersion] = useState<string>(params.get('version') ?? '')
  const [query, setQuery] = useState(params.get('q') ?? '')
  const [res, setRes] = useState<SearchResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    cs.libraries().then(setLibs).catch((e) => setError(e.message))
    cs.examples().then(setExamples).catch(() => {})
  }, [])

  // Fall back to the first indexed library when the wanted one is not indexed.
  const lib = useMemo(
    () => libs?.find((l) => l.name === wanted) ?? libs?.[0] ?? null, [libs, wanted])
  const library = lib?.name ?? wanted

  const run = (text = query, v = version) => {
    if (text.trim().length < 2 || !lib) return
    setBusy(true)
    setError(null)
    setParams({ library, ...(v ? { version: v } : {}) }, { replace: true })
    cs.search(text, library, v || null, 10)
      .then(setRes)
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false))
  }

  const language = lib?.kind === 'codebase' ? 'jsx' : 'python'

  return (
    <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 lg:grid-cols-[minmax(0,1fr)_340px]">
      <section className="min-w-0 space-y-4">
        <Card className="space-y-3 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex rounded-lg bg-zinc-500/10 p-0.5" role="tablist" data-testid="library-switch">
              {(libs ?? []).map((l) => (
                <button
                  key={l.name}
                  role="tab"
                  aria-selected={l.name === library}
                  onClick={() => { setLibrary(l.name); setVersion(''); setRes(null) }}
                  className={clsx('rounded-md px-3 py-1.5 text-sm transition',
                    l.name === library ? 'bg-app font-medium shadow-sm' : 'text-muted hover:text-inherit')}
                >
                  {LIB_LABELS[l.name] ?? l.name}
                </button>
              ))}
            </div>
            {lib && (
              <label className="flex items-center gap-2 text-sm">
                <GitBranch className="h-4 w-4 text-muted" />
                <select
                  className="rounded-md border border-app bg-app px-2 py-1 text-sm"
                  value={version}
                  onChange={(e) => setVersion(e.target.value)}
                  data-testid="version-select"
                >
                  <option value="">latest ({lib.versions[lib.versions.length - 1]?.version})</option>
                  {[...lib.versions].reverse().map((v) => (
                    <option key={v.version} value={v.version}>
                      {v.version} · {v.snippets.toLocaleString()} snippets
                    </option>
                  ))}
                  {lib.versions.length > 1 && <option value={ALL}>All versions (lineages)</option>}
                </select>
              </label>
            )}
            {lib && (
              <span className="ml-auto text-xs text-muted">
                {lib.description} · model {lib.models.join(' + ')}
              </span>
            )}
          </div>
          <textarea
            className="min-h-[120px] w-full resize-y rounded-lg border border-app bg-app p-3 font-mono text-sm outline-none focus:ring-2 focus:ring-teal-500/40"
            placeholder={lib?.kind === 'codebase'
              ? 'Describe the code you are looking for…'
              : 'Paste a programming problem (statement, input/output format, examples)…'}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run()
            }}
            data-testid="query"
          />
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => run()}
              disabled={busy || query.trim().length < 2}
              className="inline-flex items-center gap-2 rounded-lg bg-teal-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
              data-testid="search-button"
            >
              {busy ? <Spinner /> : <Search className="h-4 w-4" />} Search
            </button>
            <span className="text-xs text-muted">⌘/Ctrl + Enter</span>
            <div className="ml-2 flex flex-wrap gap-1.5">
              {(examples[library] ?? []).map((ex) => (
                <button
                  key={ex.label}
                  onClick={() => {
                    setQuery(ex.query)
                    const v = /version/i.test(ex.query) && lib && lib.versions.length > 1 ? ALL : version
                    setVersion(v)
                    run(ex.query, v)
                  }}
                  className="rounded-full border border-app px-2.5 py-1 text-xs hover:bg-zinc-500/10"
                  data-testid="example-chip"
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </div>
        </Card>

        {error && <ErrorBanner message={error} />}
        {libs && libs.length === 0 && (
          <EmptyState title="No snippet library is indexed yet">
            Run <code>./run.sh index-apps</code> (APPS corpus) or{' '}
            <code>./run.sh index-version gladys --ref v5.1.1</code>.
          </EmptyState>
        )}

        {res && <LatencyBanner res={res} />}
        {res && (
          <div className="space-y-3" data-testid="results">
            {res.results.map((h) => (
              <HitCard key={`${h.id}@${h.version}`} hit={h} language={language}
                best={res.results[0]?.score ?? 1} allVersions={res.version === ALL} />
            ))}
          </div>
        )}
        {!res && !busy && libs && libs.length > 0 && (
          <EmptyState title="Ranked code snippets appear here" icon={<Search className="h-8 w-8" />}>
            Pick an example or paste a problem statement. Every search shows how the query was
            categorised and pre-processed, each retrieval pass with its time, and the total latency.
          </EmptyState>
        )}
      </section>

      <aside className="space-y-4">
        {res ? <QueryPanel res={res} /> : (
          <Card className="p-4 text-sm text-muted">
            <SectionTitle>How it works</SectionTitle>
            <ol className="list-decimal space-y-1 pl-4">
              <li>Categorise and clean the query (statement, I/O format, examples, LaTeX).</li>
              <li>Dense recall over every snippet with a CPU code-embedding model.</li>
              <li>Rerank the top candidates with lexical and program-style signals.</li>
              <li>All-versions mode: collapse near-identical versions into lineages.</li>
            </ol>
          </Card>
        )}
      </aside>
    </div>
  )
}

function LatencyBanner({ res }: { res: SearchResponse }) {
  return (
    <div className="flex flex-wrap items-center gap-4 rounded-xl border border-teal-500/30 bg-teal-500/5 px-4 py-3"
      data-testid="latency">
      <Timer className="h-5 w-5 text-teal-600 dark:text-teal-300" />
      <span className="text-2xl font-semibold tabular-nums">{ms(res.latency_ms)}</span>
      <span className="text-sm text-muted">
        to rank {res.library_size.toLocaleString()} snippets · {res.passes.length} passes · CPU only
      </span>
      <Badge tone="accent" className="ml-auto">{res.library}@{res.version}</Badge>
    </div>
  )
}

function HitCard({ hit, language, best, allVersions }: {
  hit: SearchHit; language: string; best: number; allVersions: boolean
}) {
  const loc = hit.path ? `${hit.path}:${hit.start_line}-${hit.end_line}` : hit.id
  const width = Math.max(4, Math.min(100, (hit.score / (best || 1)) * 100))
  return (
    <Card className="p-3" >
      <div className="mb-2 flex flex-wrap items-center gap-2 text-sm" data-testid="hit">
        <span className="w-7 font-semibold tabular-nums text-muted">#{hit.rank}</span>
        <span className="font-mono font-medium" data-testid="hit-id">{loc}</span>
        {hit.symbol && hit.symbol !== '<module>' && (
          <span className="font-mono text-xs text-muted">{hit.symbol}</span>
        )}
        <Badge tone="violet">@{hit.version}</Badge>
        <Badge>{hit.style}</Badge>
        {(hit.matched_literals ?? []).slice(0, 4).map((l) => (
          <Badge key={l} tone="good" title="Literal shared by the problem text and the code">
            “{l}”
          </Badge>
        ))}
        <span className="ml-auto flex items-center gap-2 text-xs text-muted">
          <span className="h-1.5 w-20 overflow-hidden rounded bg-zinc-500/15">
            <span className="block h-full bg-teal-500" style={{ width: `${width}%` }} />
          </span>
          <span className="tabular-nums">{hit.score.toFixed(3)}</span>
        </span>
      </div>
      {allVersions && hit.lineage && (
        <div className="mb-2 flex flex-wrap items-center gap-1.5 text-xs" data-testid="lineage">
          <Layers className="h-3.5 w-3.5 text-muted" />
          <span className="text-muted">lineage:</span>
          {hit.lineage.versions.map((m) => (
            <span
              key={`${m.id}@${m.version}`}
              title={m.same_text ? 'identical to the shown version' : 'differs from the shown version'}
              className={clsx('rounded px-1.5 py-0.5 font-mono ring-1 ring-inset',
                m.version === hit.version ? 'bg-teal-500/15 ring-teal-500/40' : 'ring-zinc-500/25')}
            >
              {m.version}{m.same_text ? ' =' : ' ≠'}
              {m.score !== null && <span className="text-muted"> {m.score.toFixed(2)}</span>}
            </span>
          ))}
        </div>
      )}
      <CodeBlock code={hit.text} startLine={hit.start_line || 1} maxHeight="320px" language={language} />
    </Card>
  )
}

function QueryPanel({ res }: { res: SearchResponse }) {
  const c = res.category
  const maxMs = Math.max(...res.passes.map((p) => p.ms), 1)
  return (
    <>
      <Card className="space-y-3 p-4 text-sm" >
        <SectionTitle>Query category</SectionTitle>
        <div className="flex flex-wrap gap-1.5" data-testid="category">
          <Badge tone="accent">{c.style === 'code description' ? c.style : `${c.style} program`}</Badge>
          {c.multi_test && <Badge>multiple test cases</Badge>}
          {c.interactive && <Badge tone="warn">interactive</Badge>}
          {c.topics.map((t) => <Badge key={t} tone="violet">{t}</Badge>)}
        </div>
        <div className="text-xs text-muted">
          sections: {Object.entries(c.sections).map(([k, v]) => `${k} ${v} chars`).join(' · ') || '—'}
        </div>
        {res.preprocessing.hint && (
          <div className="text-xs">
            version hint <Badge tone="warn">{res.preprocessing.hint.phrase}</Badge> →{' '}
            {res.preprocessing.hint.label ?? res.preprocessing.hint.kind}
          </div>
        )}
        <details className="text-xs">
          <summary className="cursor-pointer text-muted">
            pre-processed text ({res.preprocessing.views.join(', ')}; {res.preprocessing.raw_chars} raw chars)
          </summary>
          <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded bg-zinc-500/10 p-2 font-mono">
            {res.preprocessing.cleaned}
          </pre>
        </details>
      </Card>
      <Card className="p-4 text-sm">
        <SectionTitle>Retrieval passes</SectionTitle>
        <ol className="space-y-2" data-testid="passes">
          {res.passes.map((p: PassTrace, i) => (
            <li key={i}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-medium">{i + 1}. {p.name}</span>
                <span className="tabular-nums text-xs text-muted">{ms(p.ms)}</span>
              </div>
              <div className="my-1 h-1 overflow-hidden rounded bg-zinc-500/10">
                <div className="h-full bg-teal-500/70" style={{ width: `${Math.max(2, (p.ms / maxMs) * 100)}%` }} />
              </div>
              <div className="text-xs text-muted">{p.detail}</div>
            </li>
          ))}
        </ol>
      </Card>
    </>
  )
}

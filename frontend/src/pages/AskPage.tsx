import clsx from 'clsx'
import {
  ArrowRight, Clock, GitBranch, Lightbulb, ListChecks, Search, Sparkles, X,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { CallGraph } from '../components/CallGraph'
import type { FileTarget } from '../components/FileViewer'
import { MicButton } from '../components/MicButton'
import { OptimizationPanel } from '../components/OptimizationPanel'
import { ResultCard } from '../components/ResultCard'
import { StepTimeline } from '../components/StepTimeline'
import { Badge, EmptyState, ErrorBanner, Spinner } from '../components/ui'
import { askStream, type AskResult, type DoneInfo, type Step } from '../lib/api'
import { ms, ROUTE_LABELS } from '../lib/format'

const EXAMPLES: { kind: string; q: string }[] = [
  { kind: 'usage', q: 'Where is the Bluetooth-settings deeplink used?' },
  { kind: 'structural', q: 'Which files call tool setValue before tool emit?' },
  { kind: 'plain English', q: 'How does Gladys purge old device states?' },
  { kind: 'structural', q: 'Who calls purgeStates and what does it call?' },
  { kind: 'usage', q: 'Where is EVENTS.DEVICE.NEW_STATE used?' },
  { kind: 'plain English', q: 'How are Zigbee2mqtt devices discovered?' },
]

const STRUCTURAL = new Set(['sequence', 'callers', 'callees', 'multi'])

interface RunState {
  query: string
  steps: Step[]
  result: AskResult | null
  explanation: string
  explanationDone: boolean
  explanationMs: number | null
  done: DoneInfo | null
  error: string | null
  startedAt: number
  totalMs: number | null
  explain: boolean // whether the optional LLM explanation was requested
}

export function AskPage({ llmAvailable, onOpen }: {
  llmAvailable: boolean | undefined; onOpen: (t: FileTarget) => void
}) {
  const [params, setParams] = useSearchParams()
  const [query, setQuery] = useState(params.get('q') ?? '')
  const [optimize, setOptimize] = useState(false)
  const [explain, setExplain] = useState(false) // optional, off by default (D-033)
  const [run, setRun] = useState<RunState | null>(null)
  const [micError, setMicError] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)
  const input = useRef<HTMLTextAreaElement>(null)

  const ask = useCallback((q: string) => {
    const text = q.trim()
    if (text.length < 2) return
    abort.current?.abort()
    const ctl = new AbortController()
    abort.current = ctl
    setMicError(null)
    setParams({ q: text }, { replace: true })
    setRun({
      query: text, steps: [], result: null, explanation: '', explanationDone: false,
      explanationMs: null, done: null, error: null, startedAt: performance.now(), totalMs: null,
      explain,
    })
    const up = (f: (r: RunState) => RunState) => {
      if (!ctl.signal.aborted) setRun((r) => (r ? f(r) : r))
    }
    askStream({ query: text, use_llm: true, explain, optimize }, {
      onStep: (s) => up((r) => ({ ...r, steps: [...r.steps, s] })),
      onResults: (res) => up((r) => ({ ...r, result: res })),
      onExplanation: (t) => up((r) => ({ ...r, explanation: r.explanation + t })),
      onExplanationDone: (i) => up((r) => ({ ...r, explanationDone: true, explanationMs: i.total_ms })),
      onDone: (d) => up((r) => ({
        ...r, done: d, explanationDone: true, totalMs: performance.now() - r.startedAt,
      })),
      onError: (m) => up((r) => ({ ...r, error: m, done: r.done ?? {
        id: '', llm_available: false, latency_ms: 0, final_latency_ms: 0,
      } })),
    }, ctl.signal)
  }, [optimize, explain, setParams])

  // Run a query passed in the URL once (shareable links / Explore → Ask).
  const initial = useRef(params.get('q'))
  useEffect(() => {
    if (initial.current) ask(initial.current)
    input.current?.focus()
    return () => abort.current?.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const submit = (e: FormEvent) => { e.preventDefault(); ask(query) }
  const pick = (q: string) => { setQuery(q); ask(q) }

  const res = run?.result
  const firstResultsIn = run && res && run.steps.length ? res.latency_ms : null
  const working = !!run && !run.done
  const phaseA = working && !res
  const phaseB = working && !!res

  return (
    <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6">
      <form onSubmit={submit} className="panel p-4" aria-label="Ask a question">
        <label htmlFor="q" className="mb-2 block text-sm font-medium">
          Ask about the Gladys Assistant codebase
        </label>
        <div className="flex items-start gap-2">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-3.5 h-4 w-4 text-muted" />
            <textarea
              id="q"
              ref={input}
              data-testid="query-input"
              value={query}
              rows={2}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(query) }
              }}
              placeholder="e.g. Which files call tool setValue before tool emit?"
              className="w-full resize-none rounded-lg border border-app bg-transparent py-3 pl-9 pr-9 text-base outline-none placeholder:text-zinc-400 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
            />
            {query && (
              <button
                type="button"
                onClick={() => { setQuery(''); input.current?.focus() }}
                className="absolute right-2 top-3 rounded p-1 text-muted hover:bg-zinc-500/10"
                aria-label="Clear question"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          <MicButton
            onText={(t) => { setQuery(t); ask(t) }}
            onError={setMicError}
            disabled={phaseA}
          />
          <button
            type="submit"
            data-testid="ask-button"
            disabled={query.trim().length < 2 || phaseA}
            className="flex h-11 items-center gap-2 rounded-lg bg-teal-600 px-4 text-sm font-medium text-white transition hover:bg-teal-500 disabled:opacity-50"
          >
            {phaseA ? <Spinner /> : <ArrowRight className="h-4 w-4" />} Ask
          </button>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.q}
              type="button"
              onClick={() => pick(ex.q)}
              className="group flex items-center gap-1.5 rounded-full border border-app px-3 py-1 text-xs transition hover:border-teal-500/60 hover:bg-teal-500/5"
            >
              <span className="text-[10px] font-semibold uppercase tracking-wide text-muted group-hover:text-teal-600 dark:group-hover:text-teal-300">
                {ex.kind}
              </span>
              {ex.q}
            </button>
          ))}
          <label className="ml-auto flex cursor-pointer items-center gap-2 text-xs text-muted">
            <input
              type="checkbox"
              checked={explain}
              onChange={(e) => setExplain(e.target.checked)}
              className="accent-teal-600"
              data-testid="explain-toggle"
            />
            LLM explanation (optional)
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-xs text-muted">
            <input
              type="checkbox"
              checked={optimize}
              onChange={(e) => setOptimize(e.target.checked)}
              className="accent-teal-600"
            />
            also suggest optimisations
          </label>
        </div>
        {micError && <div className="mt-3"><ErrorBanner message={micError} /></div>}
      </form>

      {!run && (
        <EmptyState title="Ask a question to see the agent work" icon={<Sparkles className="h-8 w-8" />}>
          Answers come from the call graph, the domain catalog and hybrid search over{' '}
          7&nbsp;000+ functions, with exact file paths and line ranges. Try one of the examples above.
        </EmptyState>
      )}

      {run && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
          <div className="min-w-0 space-y-4">
            <div className="flex flex-wrap items-center gap-2" data-testid="answer-meta">
              {res && (
                <Badge tone="violet" title={`Planner method: ${res.route.method ?? 'rules'}`}>
                  <GitBranch className="h-3 w-3" />
                  <span data-testid="route-badge">{ROUTE_LABELS[res.route.type] ?? res.route.type}</span>
                </Badge>
              )}
              {res && (
                <Badge tone={res.confidence === 'high' ? 'good' : 'warn'}>
                  confidence {res.confidence}
                </Badge>
              )}
              {firstResultsIn !== null && (
                <Badge tone="accent" title="Time to first results (deterministic phase)">
                  <Clock className="h-3 w-3" /> results in {ms(firstResultsIn)}
                </Badge>
              )}
              {run.totalMs !== null && !run.error && (
                <Badge title="Wall-clock time including the LLM phase and the explanation">
                  total {ms(run.totalMs)}
                </Badge>
              )}
              {res?.results_changed_by_llm && res.phase === 'final' && (
                <Badge tone="violet"><Sparkles className="h-3 w-3" /> refined by LLM</Badge>
              )}
              {res?.route.rewritten_query && (
                <span className="truncate text-xs text-muted">
                  rewritten: <span className="font-mono">{res.route.rewritten_query}</span>
                </span>
              )}
            </div>

            {run.error && <ErrorBanner message={run.error} />}

            {res && (
              <Explanation run={run} llmAvailable={run.done ? run.done.llm_available : llmAvailable} />
            )}

            {phaseA && !run.error && (
              <div className="panel flex items-center gap-3 p-6 text-sm text-muted">
                <Spinner className="text-teal-500" /> Planning and searching…
              </div>
            )}

            {res && res.snippets.length > 0 && STRUCTURAL.has(res.route.type) && res.call_paths.length > 0 && (
              <section>
                <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
                  <GitBranch className="h-4 w-4 text-teal-500" /> Call paths
                  <span className="font-normal text-muted">· click a node to open its code</span>
                </h3>
                <CallGraph resultId={res.id} onOpen={onOpen} />
              </section>
            )}

            {res && (
              <section className="space-y-3" data-testid="results">
                <h3 className="flex items-center gap-2 text-sm font-semibold">
                  <ListChecks className="h-4 w-4 shrink-0 text-teal-500" />
                  <span className="shrink-0 whitespace-nowrap">
                    {res.snippets.length} result{res.snippets.length === 1 ? '' : 's'}
                  </span>
                  {res.answer && <span className="min-w-0 truncate font-normal text-muted" title={res.answer}>· {res.answer}</span>}
                </h3>
                {res.snippets.length === 0 && (
                  <EmptyState title="Nothing matched">
                    Try naming a function, event or deeplink, or rephrase the question.
                  </EmptyState>
                )}
                {res.snippets.map((s, i) => (
                  <ResultCard key={`${s.location}-${i}`} rank={i + 1} snippet={s} onOpen={onOpen} />
                ))}
              </section>
            )}

            {res && (optimize || res.optimizations.length > 0) && run.done && res.id && (
              <OptimizationPanel resultId={res.id} initial={res.optimizations} onOpen={onOpen} />
            )}
            {res && !optimize && !res.optimizations.length && run.done && (
              <button
                onClick={() => { setOptimize(true) }}
                className="flex items-center gap-2 text-xs text-muted hover:text-amber-600"
              >
                <Lightbulb className="h-3.5 w-3.5" /> Check these snippets for optimisation opportunities
              </button>
            )}
          </div>

          <aside className="panel h-fit p-3 lg:sticky lg:top-20">
            <div className="mb-2 flex items-center justify-between px-2">
              <h3 className="text-sm font-semibold">Agent trace</h3>
              <span className="text-xs text-muted">{run.steps.length} steps</span>
            </div>
            <StepTimeline steps={run.steps} running={phaseA} phaseBRunning={phaseB} />
          </aside>
        </div>
      )}
    </div>
  )
}

function Explanation({ run, llmAvailable }: { run: RunState; llmAvailable: boolean | undefined }) {
  const streaming = run.explain && !run.explanationDone && !run.done
  if (!run.explain) {
    return run.done ? <span className="sr-only" data-testid="explanation-off">explanation off</span> : null
  }
  if (!run.explanation && run.done && !llmAvailable) {
    return (
      <div className="rounded-lg border border-app px-4 py-3 text-sm text-muted" data-testid="llm-offline">
        The local LLM is offline, so there is no written explanation. The results below are complete
        and verified.
      </div>
    )
  }
  if (!run.explanation && !streaming) return null
  return (
    <section
      className="rounded-lg border border-fuchsia-500/25 bg-fuchsia-500/5 px-4 py-3"
      data-testid="explanation"
    >
      <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-fuchsia-600 dark:text-fuchsia-300">
        <Sparkles className="h-3.5 w-3.5" /> Explanation
        {run.explanationMs !== null && (
          <span className="font-normal text-muted">· {ms(run.explanationMs)}</span>
        )}
      </div>
      <p className={clsx('whitespace-pre-wrap text-sm leading-relaxed', streaming && 'caret')}>
        {run.explanation || <span className="text-muted">Writing an explanation from the verified snippets…</span>}
      </p>
    </section>
  )
}

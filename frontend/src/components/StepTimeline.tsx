import clsx from 'clsx'
import {
  BookOpen, Flag, Gauge, ListTree, RefreshCcw, Search, ShieldCheck, Sparkles,
} from 'lucide-react'
import type { Step, StepKind } from '../lib/api'
import { ms } from '../lib/format'
import { Badge, Spinner } from './ui'

const KIND: Record<StepKind, { icon: typeof Search; color: string; label: string }> = {
  plan: { icon: ListTree, color: 'text-violet-500', label: 'Plan' },
  search: { icon: Search, color: 'text-sky-500', label: 'Search' },
  read: { icon: BookOpen, color: 'text-indigo-400', label: 'Read' },
  assess: { icon: Gauge, color: 'text-amber-500', label: 'Assess' },
  refine: { icon: RefreshCcw, color: 'text-teal-500', label: 'Refine' },
  verify: { icon: ShieldCheck, color: 'text-emerald-500', label: 'Verify' },
  stop: { icon: Flag, color: 'text-zinc-400', label: 'Stop' },
  llm: { icon: Sparkles, color: 'text-fuchsia-500', label: 'LLM' },
}

interface SubPlanOut { text: string; route: { type: string }; steps: string[] }

function PlanDetail({ step }: { step: Step }) {
  const subs = (step.outputs.subplans as SubPlanOut[] | undefined) ?? []
  return (
    <div className="mt-2 space-y-2">
      {subs.map((sp, i) => (
        <div key={i} className="rounded-md border border-app px-3 py-2">
          <div className="mb-1 flex items-center gap-2 text-xs">
            <Badge tone="violet">{sp.route.type}</Badge>
            <span className="truncate text-muted">{sp.text}</span>
          </div>
          <ol className="space-y-0.5 text-xs">
            {sp.steps.map((s, j) => (
              <li key={j} className="flex items-baseline gap-2">
                <span className="w-3 shrink-0 text-right tabular-nums text-muted">{j + 1}</span>
                <span className="font-mono">{s}</span>
              </li>
            ))}
          </ol>
        </div>
      ))}
    </div>
  )
}

function AssessDetail({ step }: { step: Step }) {
  const o = step.outputs as Record<string, unknown>
  const signals: [string, string, boolean][] = [
    ['margin', String(o.margin ?? '—'), Number(o.margin) >= 0.2],
    ['BM25 ↔ vector agree', o.agreement ? 'yes' : 'no', Boolean(o.agreement)],
    ['catalog hit in top 3', o.catalog_in_top3 ? 'yes' : 'no', Boolean(o.catalog_in_top3)],
    ['verified', String(o.verified_ratio ?? '—'), Number(o.verified_ratio) >= 0.5],
  ]
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      <Badge tone={o.label === 'high' ? 'good' : 'warn'}>confidence {String(o.label)}</Badge>
      {signals.map(([k, v, good]) => (
        <span key={k} className={clsx('rounded px-1.5 py-0.5 text-xs',
          good ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-300' : 'bg-zinc-500/10 text-muted')}>
          {k}: {v}
        </span>
      ))}
    </div>
  )
}

export function StepTimeline({ steps, running, phaseBRunning }: {
  steps: Step[]; running: boolean; phaseBRunning: boolean
}) {
  const firstB = steps.findIndex((s) => s.phase === 'refine')
  return (
    <ol className="relative space-y-1" aria-label="Agent steps" data-testid="timeline">
      {steps.map((s, i) => {
        const k = KIND[s.kind] ?? KIND.stop
        const Icon = k.icon
        return (
          <li key={s.index}>
            {i === firstB && (
              <div className="my-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-fuchsia-500">
                <Sparkles className="h-3 w-3" /> after first results · LLM phase
                <span className="h-px flex-1 bg-fuchsia-500/30" />
              </div>
            )}
            <div className={clsx('flex gap-3 rounded-lg px-2 py-1.5',
              s.kind === 'stop' && 'bg-zinc-500/5')}>
              <div className="flex flex-col items-center">
                <Icon className={clsx('mt-0.5 h-4 w-4 shrink-0', k.color, !s.ok && 'opacity-60')} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className={clsx('text-xs font-semibold', k.color)}>{k.label}</span>
                  <span className="truncate text-sm">{s.name}</span>
                  <span className="ml-auto shrink-0 font-mono text-[11px] text-muted">
                    {s.elapsed_ms ? ms(s.elapsed_ms) : ''}
                  </span>
                </div>
                {s.summary && s.kind !== 'plan' && (
                  <p className="truncate text-xs text-muted" title={s.summary}>{s.summary}</p>
                )}
                {s.kind === 'plan' && s.outputs.subplans ? <PlanDetail step={s} /> : null}
                {s.kind === 'assess' && <AssessDetail step={s} />}
              </div>
            </div>
          </li>
        )
      })}
      {(running || phaseBRunning) && (
        <li className="flex items-center gap-3 px-2 py-1.5 text-sm text-muted">
          <Spinner className="text-teal-500" />
          {running ? 'Working…' : 'Refining with the local LLM (results are already shown)…'}
        </li>
      )}
    </ol>
  )
}

import clsx from 'clsx'
import { useEffect, useState, type ReactNode } from 'react'
import { cs } from '../lib/codesearch'
import { num } from '../lib/format'
import { Card, ErrorBanner, SectionTitle, Spinner } from './ui'

// APPS (P0), P1 and evolutionary-retrieval results, read from results/ and release/ via
// GET /api/cs/eval. Nothing here is hard-coded except labels.

type M = Record<string, number>
interface CsEval {
  official: M | null
  run_info: { wall_clock_s: number; index_s: number; search_s: number; machine: string } | null
  baselines: { system: string; ndcg_at_10: number; mrr_at_10?: number | null; source: string }[] | null
  benchmark: {
    model: string; model_id: string; params_m: number; max_seq: number; doc_encode_s: number
    sets: Record<string, M>
  }[] | null
  ablation: { name: string; note: string; sets: Record<string, M> }[] | null
  p1: { model: string; rows: Record<string, string | number | null>[] } | null
  evolution: {
    model: string; entries: number; lineages_inferred: number; lineages_true: number
    grouping: M; sets: Record<string, Record<string, M>>
  } | null
}

function T({ head, rows, hl }: { head: string[]; rows: ReactNode[][]; hl?: (i: number) => boolean }) {
  return (
    <div className="overflow-x-auto p-2">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-muted">
            {head.map((h, i) => (
              <th key={h} className={clsx('px-3 py-2 font-medium', i > 0 && 'text-right')}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {rows.map((row, i) => (
            <tr key={i} className={clsx('border-t border-app', hl?.(i) && 'bg-teal-500/10 font-medium')}>
              {row.map((c, j) => (
                <td key={j} className={clsx('px-3 py-1.5', j > 0 && 'whitespace-nowrap text-right')}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Big({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card className="p-4">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="mt-1 text-xs text-muted">{sub}</div>}
    </Card>
  )
}

export function AppsMetrics() {
  const [d, setD] = useState<CsEval | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { cs.evaluation().then(setD).catch((e) => setErr(e.message)) }, [])
  if (err) return <ErrorBanner message={err} />
  if (!d) return <div className="flex justify-center py-10"><Spinner /></div>
  const o = d.official
  return (
    <div className="space-y-6" data-testid="apps-metrics">
      <div>
        <h1 className="text-xl font-semibold">CoIR APPS retrieval (MTEB AppsRetrieval)</h1>
        <p className="text-sm text-muted">
          Official score = test split, one run with the submitted pipeline. Every choice was made on
          stdin-style APPS <b>train</b> problems (dev-tune), confirmed on dev-holdout; the test split was
          never used for tuning.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Big label="Test NDCG@10 (official)" value={o ? num(o.ndcg_at_10, 4) : 'not run yet'} />
        <Big label="Test MRR@10 (official)" value={o ? num(o.mrr_at_10, 4) : '—'} />
        <Big label="Full test run (CPU)"
          value={d.run_info ? `${Math.round(d.run_info.wall_clock_s / 60)} min` : '—'}
          sub={d.run_info ? `index ${Math.round(d.run_info.index_s)} s · search ${Math.round(d.run_info.search_s)} s` : undefined} />
        <Big label="Recall@100 (test)" value={o ? num(o.recall_at_100, 3) : '—'} />
      </div>

      {d.baselines && (
        <Card className="overflow-hidden">
          <div className="p-4 pb-0"><SectionTitle>Test split vs baselines</SectionTitle></div>
          <T head={['System', 'NDCG@10', 'MRR@10', 'Source']}
            rows={d.baselines.map((b) => [b.system, num(b.ndcg_at_10, 4), b.mrr_at_10 == null ? '—' : num(b.mrr_at_10, 4), b.source])}
            hl={(i) => d.baselines![i].source.startsWith('ours')} />
        </Card>
      )}

      {d.ablation && (
        <Card className="overflow-hidden">
          <div className="p-4 pb-0"><SectionTitle>Ablations (tuning splits)</SectionTitle></div>
          <T head={['Variant', 'What changes', 'dev-tune NDCG@10', 'MRR@10', 'dev-holdout NDCG@10', 'MRR@10']}
            rows={d.ablation.map((r) => [
              <span key="n" className="font-mono text-xs">{r.name}</span>,
              <span key="d" className="text-xs text-muted">{r.note}</span>,
              num(r.sets['dev-tune']?.['ndcg@10'], 4), num(r.sets['dev-tune']?.['mrr@10'], 4),
              num(r.sets['dev-holdout']?.['ndcg@10'], 4), num(r.sets['dev-holdout']?.['mrr@10'], 4),
            ])}
            hl={(i) => d.ablation![i].name === 'final'} />
        </Card>
      )}

      {d.benchmark && (
        <Card className="overflow-hidden">
          <div className="p-4 pb-0"><SectionTitle>Embedding models (tuning splits, CPU)</SectionTitle></div>
          <T head={['Model', 'Params', 'dev-tune NDCG@10', 'MRR@10', 'R@100', 'dev-holdout NDCG@10', 'Encode 5k snippets']}
            rows={[...d.benchmark].sort((a, b) => (b.sets['dev-tune']?.['ndcg@10'] ?? 0) - (a.sets['dev-tune']?.['ndcg@10'] ?? 0))
              .map((r) => [r.model, `${r.params_m}M`, num(r.sets['dev-tune']?.['ndcg@10'], 4),
                num(r.sets['dev-tune']?.['mrr@10'], 4), num(r.sets['dev-tune']?.['recall@100'], 3),
                num(r.sets['dev-holdout']?.['ndcg@10'], 4), `${Math.round(r.doc_encode_s)} s`])} />
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-2">
        {d.p1 && (
          <Card className="overflow-hidden">
            <div className="p-4 pb-0"><SectionTitle>P1 · full vs incremental rebuild ({d.p1.model})</SectionTitle></div>
            <T head={['Library', 'Version', 'Build', 'Snippets', 'Re-encoded', 'Total s']}
              rows={d.p1.rows.map((r) => [String(r.library), String(r.version), String(r.build),
                String(r.snippets), `${r.encoded} / ${r.snippets}`, String(r.total_s)])}
              hl={(i) => d.p1!.rows[i].build === 'incremental'} />
          </Card>
        )}
        {d.evolution && (
          <Card className="overflow-hidden">
            <div className="p-4 pb-0">
              <SectionTitle>Bonus · all-versions retrieval ({d.evolution.entries.toLocaleString()} entries)</SectionTitle>
              <p className="text-xs text-muted">
                lineages inferred {d.evolution.lineages_inferred} (true {d.evolution.lineages_true}) · pair
                precision {num(d.evolution.grouping.pair_precision, 3)} · recall {num(d.evolution.grouping.pair_recall, 3)}
              </p>
            </div>
            <T head={['Set', 'System', 'NDCG@10 dedup', 'Exact version @1', 'Lineage MRR@10', 'Distinct @10']}
              rows={Object.entries(d.evolution.sets).flatMap(([s, systems]) =>
                Object.entries(systems).map(([name, m]) => [s, name, num(m['ndcg@10_dedup'], 4),
                  num(m['exact_version@1'], 4), num(m['lineage_mrr@10'], 4),
                  num(m['distinct_lineages@10'], 1)]))}
              hl={(i) => i % 4 === 2} />
          </Card>
        )}
      </div>
    </div>
  )
}

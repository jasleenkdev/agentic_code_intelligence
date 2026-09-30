import { Lightbulb, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, type Finding } from '../lib/api'
import type { FileTarget } from './FileViewer'
import { Badge, ErrorBanner, Spinner } from './ui'

/** Static optimisation findings for the answer's snippets, with LLM explanations when available. */
export function OptimizationPanel({ resultId, initial, onOpen }: {
  resultId: string; initial: Finding[]; onOpen: (t: FileTarget) => void
}) {
  const [findings, setFindings] = useState<Finding[]>(initial)
  const [loading, setLoading] = useState(true)
  const [llm, setLlm] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    setLoading(true); setError(null)
    api.optimizations(resultId, true)
      .then((d) => { if (live) { setFindings(d.findings); setLlm(d.llm_available) } })
      .catch((e) => live && setError(e.message))
      .finally(() => live && setLoading(false))
    return () => { live = false }
  }, [resultId])

  return (
    <section className="panel p-4" data-testid="optimizations">
      <div className="mb-3 flex items-center gap-2">
        <Lightbulb className="h-4 w-4 text-amber-500" />
        <h3 className="text-sm font-semibold">Optimisation suggestions</h3>
        {loading && <Spinner className="text-muted" />}
        {llm === false && <Badge tone="warn">LLM offline · static findings only</Badge>}
      </div>
      {error && <ErrorBanner message={error} />}
      {!loading && !error && !findings.length && (
        <p className="text-sm text-muted">No obvious inefficiencies in these snippets.</p>
      )}
      <ul className="space-y-3">
        {findings.map((f, i) => (
          <li key={i} className="rounded-lg border border-app p-3">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge tone="warn">{f.check.replaceAll('_', ' ')}</Badge>
              <button
                className="font-mono text-xs text-muted hover:text-teal-600 dark:hover:text-teal-300"
                onClick={() => onOpen({ path: f.file_path, start: f.line, end: f.end_line })}
              >
                {f.location}
              </button>
            </div>
            <p className="mt-2 text-sm">{f.message}</p>
            {f.evidence && (
              <pre className="code-surface mt-2 overflow-x-auto rounded-md px-3 py-2 font-mono text-xs">
                {f.evidence}
              </pre>
            )}
            {f.explanation && (
              <p className="mt-2 flex gap-2 text-sm text-fuchsia-700 dark:text-fuchsia-300">
                <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                {f.explanation}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

import { CheckCircle2, ExternalLink, FileCode2, TriangleAlert } from 'lucide-react'
import type { Snippet } from '../lib/api'
import { CodeBlock } from './CodeBlock'
import type { FileTarget } from './FileViewer'
import { Badge } from './ui'

export function ResultCard({ rank, snippet, onOpen }: {
  rank: number; snippet: Snippet; onOpen: (t: FileTarget) => void
}) {
  const s = snippet
  const hl = s.highlight_lines.length ? s.highlight_lines : []
  const open = () => onOpen({
    path: s.file_path,
    start: hl.length ? Math.min(...hl) : s.start_line,
    end: hl.length ? Math.max(...hl) : s.end_line,
  })
  return (
    <article className="panel overflow-hidden" data-testid="result-card">
      <header className="flex flex-wrap items-center gap-2 border-b border-app px-4 py-2.5">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-teal-500/15 font-mono text-xs font-semibold text-teal-700 dark:text-teal-300">
          {rank}
        </span>
        <button
          onClick={open}
          className="flex min-w-0 items-center gap-1.5 font-mono text-sm hover:text-teal-600 dark:hover:text-teal-300"
          title="Open the full file at these lines"
        >
          <FileCode2 className="h-4 w-4 shrink-0 text-muted" />
          <span className="truncate" data-testid="result-path">{s.file_path}</span>
          <span className="shrink-0 text-muted">:{s.start_line}–{s.end_line}</span>
        </button>
        {s.symbol && <Badge className="font-mono">{s.symbol}</Badge>}
        <div className="ml-auto flex items-center gap-2">
          {s.verified ? (
            <Badge tone="good" title="Lines re-read from disk and matched to the index">
              <CheckCircle2 className="h-3 w-3" /> verified
            </Badge>
          ) : (
            <Badge tone="warn" title="Could not be re-verified against the file on disk">
              <TriangleAlert className="h-3 w-3" /> unverified
            </Badge>
          )}
          <button
            onClick={open}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted hover:bg-zinc-500/10"
          >
            <ExternalLink className="h-3.5 w-3.5" /> open file
          </button>
        </div>
      </header>
      {s.reason && <p className="px-4 pt-2 text-xs text-muted">{s.reason}</p>}
      <div className="p-2">
        <CodeBlock code={s.text} startLine={s.start_line} highlight={hl} maxHeight="320px" />
        {s.truncated && (
          <p className="px-2 pt-1 text-[11px] text-muted">
            Snippet truncated — open the file to see the rest.
          </p>
        )}
      </div>
    </article>
  )
}

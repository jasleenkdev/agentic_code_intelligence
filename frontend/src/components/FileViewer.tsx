import { ExternalLink, FileCode2, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, type FileWindow } from '../lib/api'
import { CodeBlock } from './CodeBlock'
import { ErrorBanner, Spinner } from './ui'

export interface FileTarget { path: string; start: number; end: number }

/** Modal showing the full file around a range (click-through from result cards). */
export function FileViewer({ target, onClose }: { target: FileTarget | null; onClose: () => void }) {
  const [data, setData] = useState<FileWindow | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!target) return
    setData(null)
    setError(null)
    api.file(target.path, target.start, target.end, 60).then(setData, (e) => setError(e.message))
  }, [target])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!target) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`File ${target.path}`}
    >
      <div
        className="panel flex max-h-[88vh] w-full max-w-5xl flex-col overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-app px-4 py-3">
          <FileCode2 className="h-4 w-4 text-teal-500" />
          <span className="truncate font-mono text-sm">{target.path}</span>
          <span className="text-xs text-muted">
            lines {target.start}–{target.end}
            {data && ` · showing ${data.start_line}–${data.end_line} of ${data.total_lines}`}
          </span>
          <a
            className="ml-auto inline-flex items-center gap-1 text-xs text-muted hover:text-teal-500"
            href={`https://github.com/GladysAssistant/Gladys/blob/8574722f2ffb1fafc10868396cfefd38a1880586/${target.path}#L${target.start}-L${target.end}`}
            target="_blank"
            rel="noreferrer"
          >
            GitHub <ExternalLink className="h-3 w-3" />
          </a>
          <button onClick={onClose} className="rounded p-1 text-muted hover:bg-zinc-500/15" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="overflow-auto p-3">
          {error && <ErrorBanner message={error} />}
          {!data && !error && (
            <div className="flex items-center gap-2 p-6 text-sm text-muted"><Spinner /> Loading file…</div>
          )}
          {data && (
            <CodeBlock
              code={data.lines.join('\n')}
              startLine={data.start_line}
              highlight={data.highlight}
              highlightRange
              scrollToHighlight
            />
          )}
        </div>
      </div>
    </div>
  )
}

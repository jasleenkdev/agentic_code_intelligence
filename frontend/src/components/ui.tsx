import clsx from 'clsx'
import { AlertTriangle, Inbox, Loader2 } from 'lucide-react'
import type { ReactNode } from 'react'

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx('panel', className)}>{children}</div>
}

const TONES = {
  neutral: 'bg-zinc-500/10 text-zinc-600 dark:text-zinc-300 ring-zinc-500/20',
  accent: 'bg-teal-500/10 text-teal-700 dark:text-teal-300 ring-teal-500/25',
  good: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 ring-emerald-500/25',
  warn: 'bg-amber-500/10 text-amber-700 dark:text-amber-300 ring-amber-500/25',
  bad: 'bg-rose-500/10 text-rose-700 dark:text-rose-300 ring-rose-500/25',
  violet: 'bg-violet-500/10 text-violet-700 dark:text-violet-300 ring-violet-500/25',
} as const
export type Tone = keyof typeof TONES

export function Badge({ tone = 'neutral', children, className, title }: {
  tone?: Tone; children: ReactNode; className?: string; title?: string
}) {
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset',
        TONES[tone], className,
      )}
    >
      {children}
    </span>
  )
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={clsx('h-4 w-4 animate-spin', className)} aria-hidden />
}

export function EmptyState({ title, children, icon }: {
  title: string; children?: ReactNode; icon?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      <div className="text-muted">{icon ?? <Inbox className="h-8 w-8" />}</div>
      <p className="font-medium">{title}</p>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
    </div>
  )
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300"
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  )
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">{children}</h2>
      {right}
    </div>
  )
}

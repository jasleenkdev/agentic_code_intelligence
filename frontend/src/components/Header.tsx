import clsx from 'clsx'
import { Code2, Moon, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { api, type Stats } from '../lib/api'
import type { Theme } from '../lib/theme'

export function useStats(pollMs = 15000) {
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    const load = () => api.stats()
      .then((s) => { if (live) { setStats(s); setError(null) } })
      .catch((e) => live && setError(e.message))
    load()
    const t = setInterval(load, pollMs)
    return () => { live = false; clearInterval(t) }
  }, [pollMs])
  return { stats, error }
}

function Pill({ label, value }: { label: string; value: string }) {
  return (
    <span className="hidden items-baseline gap-1 rounded-md bg-zinc-500/10 px-2 py-1 text-xs lg:flex">
      <span className="font-semibold tabular-nums">{value}</span>
      <span className="text-muted">{label}</span>
    </span>
  )
}

const NAV = [
  { to: '/', label: 'Search' },
  { to: '/agent', label: 'Gladys agent' },
  { to: '/explore', label: 'Explore' },
  { to: '/metrics', label: 'Metrics' },
]

export function Header({ stats, statsError, theme, toggleTheme }: {
  stats: Stats | null; statsError: string | null; theme: Theme; toggleTheme: () => void
}) {
  const llm = stats?.llm
  return (
    <header className="sticky top-0 z-30 border-b border-app header-glass">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4">
        <NavLink to="/" className="flex items-center gap-2 font-semibold">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-500/15 text-teal-600 dark:text-teal-300">
            <Code2 className="h-4 w-4" />
          </span>
          <span className="hidden sm:inline">Code Search</span>
        </NavLink>
        <nav className="flex items-center gap-1">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === '/'}
              className={({ isActive }) => clsx(
                'rounded-md px-3 py-1.5 text-sm transition',
                isActive ? 'bg-zinc-500/15 font-medium' : 'text-muted hover:bg-zinc-500/10',
              )}
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2" data-testid="status">
          {statsError && !stats && (
            <span className="rounded-md bg-rose-500/10 px-2 py-1 text-xs text-rose-600 dark:text-rose-300">
              API unreachable
            </span>
          )}
          {stats && (
            <>
              <a
                href={`${stats.repo.url}/tree/${stats.repo.commit}`}
                target="_blank"
                rel="noreferrer"
                className="hidden items-center gap-1 rounded-md px-2 py-1 text-xs hover:bg-zinc-500/10 md:flex"
                title={`Indexed at commit ${stats.repo.commit}`}
              >
                <span className="font-medium">{stats.repo.name}</span>
                <span className="font-mono text-muted">@{stats.repo.commit.slice(0, 7)}</span>
              </a>
              <Pill value={stats.files.toLocaleString()} label="files" />
              <Pill value={stats.functions.toLocaleString()} label="functions" />
              <span
                className="flex items-center gap-1.5 rounded-md bg-zinc-500/10 px-2 py-1 text-xs"
                title={llm?.available
                  ? `Local LLM ${llm.model} is running (CPU)`
                  : 'Local LLM is offline: answers still work, without explanations'}
                data-testid="llm-status"
              >
                <span className={clsx('h-2 w-2 rounded-full',
                  llm?.available ? 'bg-emerald-500' : 'bg-zinc-400')} />
                <span className="font-mono">{llm?.model ?? 'LLM'}</span>
                <span className="text-muted">{llm?.available ? 'online' : 'offline'}</span>
              </span>
            </>
          )}
          <button
            onClick={toggleTheme}
            className="rounded-md p-2 text-muted hover:bg-zinc-500/10"
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </header>
  )
}

import { createContext, useContext, useEffect, useState } from 'react'

export type Theme = 'dark' | 'light'

function read(): Theme {
  try {
    return localStorage.getItem('theme') === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

/** Dark by default; the choice is remembered per browser. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(read)
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    try {
      localStorage.setItem('theme', theme)
    } catch { /* private mode: ignore */ }
  }, [theme])
  return [theme, () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))]
}

export function isDark(): boolean {
  return document.documentElement.classList.contains('dark')
}

export const ThemeContext = createContext<Theme>('dark')
export const useCurrentTheme = () => useContext(ThemeContext)

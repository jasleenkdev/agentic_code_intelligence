import { useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { FileViewer, type FileTarget } from './components/FileViewer'
import { Header, useStats } from './components/Header'
import { EmptyState } from './components/ui'
import { ThemeContext, useTheme } from './lib/theme'
import { AskPage } from './pages/AskPage'
import { ExplorePage } from './pages/ExplorePage'
import { MetricsPage } from './pages/MetricsPage'
import { SearchPage } from './pages/SearchPage'

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const { stats, error } = useStats()
  const [file, setFile] = useState<FileTarget | null>(null)

  return (
    <ThemeContext.Provider value={theme}>
      <BrowserRouter>
        <div className="min-h-screen bg-app">
          <Header stats={stats} statsError={error} theme={theme} toggleTheme={toggleTheme} />
          <main>
            <Routes>
              <Route path="/" element={<SearchPage />} />
              <Route path="/agent" element={<AskPage llmAvailable={stats?.llm.available} onOpen={setFile} />} />
              <Route path="/explore" element={<ExplorePage onOpen={setFile} />} />
              <Route path="/metrics" element={<MetricsPage />} />
              <Route path="*" element={<EmptyState title="Page not found" />} />
            </Routes>
          </main>
          <FileViewer target={file} onClose={() => setFile(null)} />
        </div>
      </BrowserRouter>
    </ThemeContext.Provider>
  )
}

import { AppsMetrics } from '../components/AppsMetrics'

// APPS, P1 and bonus results, read from results/ and release/ via GET /api/cs/eval.

export function MetricsPage() {
  return (
    <div className="mx-auto max-w-7xl space-y-10 px-4 py-6">
      <AppsMetrics />
    </div>
  )
}

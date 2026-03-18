import { useEffect, useState } from 'react'

interface Stats {
  providers: number
  keys: number
  recentRequests: number
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats>({ providers: 0, keys: 0, recentRequests: 0 })

  useEffect(() => {
    Promise.all([
      fetch('/api/admin/providers').then(r => r.json()),
      fetch('/api/admin/keys').then(r => r.json()),
      fetch('/api/admin/usage').then(r => r.json()),
    ]).then(([providers, keys, usage]) => {
      setStats({
        providers: providers.length,
        keys: keys.length,
        recentRequests: usage.length,
      })
    }).catch(() => {})
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard label="Providers" value={stats.providers} />
        <StatCard label="API Keys" value={stats.keys} />
        <StatCard label="Recent Requests" value={stats.recentRequests} />
      </div>
      <div className="mt-8 p-4 rounded-lg bg-gray-900 border border-gray-800">
        <h2 className="text-lg font-semibold mb-2">Quick Start</h2>
        <p className="text-gray-400 text-sm mb-3">
          Point your AI applications to Louter:
        </p>
        <pre className="bg-gray-950 p-3 rounded text-sm text-green-400 overflow-x-auto">
{`OPENAI_API_BASE=http://localhost:6188/v1
OPENAI_API_KEY=lot_your_key_here`}
        </pre>
      </div>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-3xl font-bold text-blue-400">{value}</div>
      <div className="text-sm text-gray-400 mt-1">{label}</div>
    </div>
  )
}

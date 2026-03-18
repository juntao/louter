import { useEffect, useState } from 'react'

interface UsageLog {
  id: string
  key_id: string | null
  provider_id: string | null
  model_id: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  latency_ms: number
  status_code: number
  created_at: string
}

interface DailyUsage {
  day: string
  requests: number
  total_tokens: number
  prompt_tokens: number
  completion_tokens: number
  avg_latency_ms: number
}

interface ModelUsage {
  model_id: string
  requests: number
  total_tokens: number
}

interface Stats {
  days: number
  total_requests: number
  total_tokens: number
  total_prompt_tokens: number
  total_completion_tokens: number
  daily: DailyUsage[]
  by_model: ModelUsage[]
}

type TimeRange = '7' | '30' | '90'

function formatTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return n.toString()
}

export default function Usage() {
  const [logs, setLogs] = useState<UsageLog[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [range, setRange] = useState<TimeRange>('30')

  const loadLogs = () => {
    fetch('/api/admin/usage').then(r => r.json()).then(setLogs).catch(() => {})
  }
  const loadStats = (days: string) => {
    fetch(`/api/admin/usage/stats?days=${days}`).then(r => r.json()).then(setStats).catch(() => {})
  }

  useEffect(() => { loadLogs(); loadStats(range) }, [])
  useEffect(() => { loadStats(range) }, [range])

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Usage</h1>
        <div className="flex gap-1 bg-gray-900 rounded p-0.5">
          {(['7', '30', '90'] as TimeRange[]).map(d => (
            <button
              key={d}
              onClick={() => setRange(d)}
              className={`px-3 py-1 text-xs rounded ${range === d ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-200'}`}
            >
              {d === '7' ? '7 Days' : d === '30' ? '30 Days' : '90 Days'}
            </button>
          ))}
        </div>
      </div>

      {/* Summary Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <StatCard label="Total Requests" value={stats.total_requests.toLocaleString()} />
          <StatCard label="Total Tokens" value={formatTokens(stats.total_tokens)} />
          <StatCard label="Prompt Tokens" value={formatTokens(stats.total_prompt_tokens)} sub={`${stats.total_tokens > 0 ? Math.round(stats.total_prompt_tokens / stats.total_tokens * 100) : 0}%`} />
          <StatCard label="Completion Tokens" value={formatTokens(stats.total_completion_tokens)} sub={`${stats.total_tokens > 0 ? Math.round(stats.total_completion_tokens / stats.total_tokens * 100) : 0}%`} />
        </div>
      )}

      {/* Chart */}
      {stats && stats.daily.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
          <h2 className="text-sm font-medium text-gray-400 mb-3">Daily Token Usage</h2>
          <TokenChart data={stats.daily} />
        </div>
      )}

      {/* Per-model breakdown */}
      {stats && stats.by_model.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
          <h2 className="text-sm font-medium text-gray-400 mb-3">Usage by Model</h2>
          <div className="space-y-2">
            {stats.by_model.map(m => {
              const pct = stats.total_tokens > 0 ? (m.total_tokens / stats.total_tokens * 100) : 0
              return (
                <div key={m.model_id} className="flex items-center gap-3 text-sm">
                  <code className="text-blue-400 text-xs w-48 truncate">{m.model_id}</code>
                  <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
                    <div className="bg-blue-500 h-full rounded-full" style={{ width: `${Math.max(pct, 1)}%` }} />
                  </div>
                  <span className="text-gray-400 text-xs w-20 text-right">{formatTokens(m.total_tokens)}</span>
                  <span className="text-gray-500 text-xs w-16 text-right">{m.requests} req</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Log table */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Recent Requests</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-left text-gray-500 text-xs">
                <th className="py-2 px-3">Time</th>
                <th className="py-2 px-3">Model</th>
                <th className="py-2 px-3">Prompt</th>
                <th className="py-2 px-3">Completion</th>
                <th className="py-2 px-3">Total</th>
                <th className="py-2 px-3">Latency</th>
                <th className="py-2 px-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {logs.map(log => (
                <tr key={log.id} className="border-b border-gray-800/30 hover:bg-gray-800/30">
                  <td className="py-1.5 px-3 text-gray-500 text-xs">{new Date(log.created_at).toLocaleString()}</td>
                  <td className="py-1.5 px-3"><code className="text-blue-400 text-xs">{log.model_id}</code></td>
                  <td className="py-1.5 px-3 text-gray-400 text-xs">{log.prompt_tokens > 0 ? log.prompt_tokens.toLocaleString() : '-'}</td>
                  <td className="py-1.5 px-3 text-gray-400 text-xs">{log.completion_tokens > 0 ? log.completion_tokens.toLocaleString() : '-'}</td>
                  <td className="py-1.5 px-3 text-gray-300 text-xs font-medium">{log.total_tokens > 0 ? log.total_tokens.toLocaleString() : '-'}</td>
                  <td className="py-1.5 px-3 text-gray-400 text-xs">{log.latency_ms}ms</td>
                  <td className="py-1.5 px-3">
                    <span className={`text-xs ${log.status_code === 200 ? 'text-green-400' : 'text-red-400'}`}>{log.status_code}</span>
                  </td>
                </tr>
              ))}
              {logs.length === 0 && (
                <tr><td colSpan={7} className="text-center text-gray-500 py-8">No usage logs yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <div className="text-2xl font-bold text-blue-400">{value}</div>
      <div className="text-xs text-gray-400 mt-0.5">
        {label}
        {sub && <span className="text-gray-600 ml-1">({sub})</span>}
      </div>
    </div>
  )
}

/** Pure SVG line chart — no dependencies */
function TokenChart({ data }: { data: DailyUsage[] }) {
  if (data.length === 0) return null

  const W = 800, H = 200, PX = 50, PY = 20, PB = 30
  const chartW = W - PX - 10
  const chartH = H - PY - PB

  const maxTokens = Math.max(...data.map(d => d.total_tokens), 1)
  const maxRequests = Math.max(...data.map(d => d.requests), 1)

  const xStep = data.length > 1 ? chartW / (data.length - 1) : chartW

  const tokenPoints = data.map((d, i) => {
    const x = PX + i * xStep
    const y = PY + chartH - (d.total_tokens / maxTokens) * chartH
    return `${x},${y}`
  })

  const requestPoints = data.map((d, i) => {
    const x = PX + i * xStep
    const y = PY + chartH - (d.requests / maxRequests) * chartH
    return `${x},${y}`
  })

  // Fill area under token line
  const tokenArea = `${PX},${PY + chartH} ${tokenPoints.join(' ')} ${PX + (data.length - 1) * xStep},${PY + chartH}`

  // Y-axis labels (tokens)
  const yLabels = [0, 0.25, 0.5, 0.75, 1].map(pct => ({
    y: PY + chartH - pct * chartH,
    label: formatTokens(Math.round(maxTokens * pct)),
  }))

  // X-axis labels (dates)
  const labelEvery = Math.max(1, Math.floor(data.length / 7))
  const xLabels = data
    .map((d, i) => ({ i, label: d.day.slice(5) })) // MM-DD
    .filter((_, i) => i % labelEvery === 0 || i === data.length - 1)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 240 }}>
      {/* Grid lines */}
      {yLabels.map((yl, i) => (
        <g key={i}>
          <line x1={PX} y1={yl.y} x2={W - 10} y2={yl.y} stroke="#1f2937" strokeWidth="1" />
          <text x={PX - 6} y={yl.y + 4} textAnchor="end" fill="#6b7280" fontSize="10">{yl.label}</text>
        </g>
      ))}

      {/* Token area fill */}
      <polygon points={tokenArea} fill="rgba(59,130,246,0.1)" />

      {/* Token line */}
      <polyline points={tokenPoints.join(' ')} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinejoin="round" />

      {/* Request line (dashed) */}
      <polyline points={requestPoints.join(' ')} fill="none" stroke="#a78bfa" strokeWidth="1.5" strokeDasharray="4,3" strokeLinejoin="round" />

      {/* Data points */}
      {data.map((d, i) => (
        <g key={i}>
          <circle cx={PX + i * xStep} cy={PY + chartH - (d.total_tokens / maxTokens) * chartH} r="3" fill="#3b82f6" />
          <title>{`${d.day}\nTokens: ${d.total_tokens.toLocaleString()}\nRequests: ${d.requests}`}</title>
          <rect x={PX + i * xStep - xStep / 2} y={PY} width={xStep} height={chartH} fill="transparent" />
        </g>
      ))}

      {/* X-axis labels */}
      {xLabels.map(xl => (
        <text key={xl.i} x={PX + xl.i * xStep} y={H - 5} textAnchor="middle" fill="#6b7280" fontSize="10">{xl.label}</text>
      ))}

      {/* Legend */}
      <line x1={W - 180} y1={10} x2={W - 160} y2={10} stroke="#3b82f6" strokeWidth="2" />
      <text x={W - 155} y={14} fill="#9ca3af" fontSize="10">Tokens</text>
      <line x1={W - 100} y1={10} x2={W - 80} y2={10} stroke="#a78bfa" strokeWidth="1.5" strokeDasharray="4,3" />
      <text x={W - 75} y={14} fill="#9ca3af" fontSize="10">Requests</text>
    </svg>
  )
}

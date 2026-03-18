import { useEffect, useState } from 'react'

interface Provider {
  id: string
  name: string
  provider_type: string
  base_url: string
  api_key: string
  is_enabled: boolean
  config_json: string
  created_at: string
}

const PROVIDER_PRESETS: Record<string, { name: string; base_url: string; group: string }> = {
  openai:    { name: 'openai',    base_url: 'https://api.openai.com/v1',         group: 'Major Providers' },
  anthropic: { name: 'anthropic', base_url: 'https://api.anthropic.com',         group: 'Major Providers' },
  deepseek:  { name: 'deepseek',  base_url: 'https://api.deepseek.com/v1',      group: 'Major Providers' },
  azure:     { name: 'azure',     base_url: 'https://your-resource.openai.azure.com', group: 'Major Providers' },
  qwen:      { name: 'qwen',      base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', group: 'Chinese Providers' },
  groq:      { name: 'groq',      base_url: 'https://api.groq.com/openai/v1',   group: 'Inference Platforms' },
  together:  { name: 'together',  base_url: 'https://api.together.xyz/v1',      group: 'Inference Platforms' },
  fireworks: { name: 'fireworks', base_url: 'https://api.fireworks.ai/inference/v1', group: 'Inference Platforms' },
  ollama:    { name: 'ollama',    base_url: 'http://localhost:11434',            group: 'Local / Self-hosted' },
  custom:    { name: '',          base_url: '',                                   group: 'Custom' },
}

function resolveProviderType(preset: string): string {
  if (['openai', 'anthropic', 'azure', 'ollama', 'deepseek'].includes(preset)) return preset
  return 'custom'
}

type FormMode = 'manual' | 'auto'

export default function Providers() {
  const [providers, setProviders] = useState<Provider[]>([])
  const [showForm, setShowForm] = useState(false)
  const [formMode, setFormMode] = useState<FormMode>('manual')
  const [preset, setPreset] = useState('openai')
  const [form, setForm] = useState({ name: 'openai', provider_type: 'openai', base_url: 'https://api.openai.com/v1', api_key: '' })
  const [testing, setTesting] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<Record<string, { success: boolean; message: string } | null>>({})

  // Auto-configure state
  const [autoDocUrl, setAutoDocUrl] = useState('')
  const [autoApiKey, setAutoApiKey] = useState('')
  const [autoAnalyzing, setAutoAnalyzing] = useState(false)
  const [autoResult, setAutoResult] = useState<{ success: boolean; error?: string; config?: any; raw_analysis?: string } | null>(null)

  const loadProviders = () => {
    fetch('/api/admin/providers')
      .then(r => r.json())
      .then(setProviders)
      .catch(() => {})
  }

  useEffect(() => { loadProviders() }, [])

  const handlePresetChange = (key: string) => {
    setPreset(key)
    const p = PROVIDER_PRESETS[key] || { name: '', base_url: '' }
    setForm({ ...form, provider_type: resolveProviderType(key), name: p.name, base_url: p.base_url })
  }

  const handleCreate = async () => {
    await fetch('/api/admin/providers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    })
    setShowForm(false)
    setPreset('openai')
    setForm({ name: 'openai', provider_type: 'openai', base_url: 'https://api.openai.com/v1', api_key: '' })
    setAutoResult(null)
    loadProviders()
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this provider?')) return
    await fetch(`/api/admin/providers/${id}`, { method: 'DELETE' })
    loadProviders()
  }

  const handleToggle = async (p: Provider) => {
    await fetch(`/api/admin/providers/${p.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_enabled: !p.is_enabled }),
    })
    loadProviders()
  }

  const handleTest = async (id: string) => {
    setTesting(id)
    setTestResult(prev => ({ ...prev, [id]: null }))
    try {
      const r = await fetch(`/api/admin/providers/${id}/test`, { method: 'POST' })
      const data = await r.json()
      setTestResult(prev => ({ ...prev, [id]: data }))
    } catch {
      setTestResult(prev => ({ ...prev, [id]: { success: false, message: 'Request failed' } }))
    }
    setTesting(null)
  }

  const handleAutoAnalyze = async () => {
    setAutoAnalyzing(true)
    setAutoResult(null)
    try {
      const r = await fetch('/api/admin/providers/auto-configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_url: autoDocUrl, api_key: autoApiKey }),
      })
      const data = await r.json()
      setAutoResult(data)

      // If successful, auto-fill the manual form
      if (data.success && data.config) {
        const cfg = data.config
        setPreset('custom')
        setForm({
          name: cfg.name || '',
          provider_type: cfg.provider_type || 'custom',
          base_url: cfg.base_url || '',
          api_key: autoApiKey,
        })
        setFormMode('manual') // Switch to manual to show the filled form
      }
    } catch {
      setAutoResult({ success: false, error: 'Request failed' })
    }
    setAutoAnalyzing(false)
  }

  const hasExistingProviders = providers.length > 0

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Providers</h1>
        <button
          onClick={() => { setShowForm(!showForm); setAutoResult(null) }}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
        >
          Add Provider
        </button>
      </div>

      {showForm && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
          {/* Mode tabs */}
          <div className="flex gap-1 mb-4">
            <button
              onClick={() => setFormMode('manual')}
              className={`px-3 py-1.5 text-sm rounded ${formMode === 'manual' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-200'}`}
            >
              Manual Setup
            </button>
            <button
              onClick={() => setFormMode('auto')}
              className={`px-3 py-1.5 text-sm rounded ${formMode === 'auto' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-200'} ${!hasExistingProviders ? 'opacity-50' : ''}`}
              disabled={!hasExistingProviders}
              title={!hasExistingProviders ? 'Add at least one provider manually first' : 'Auto-configure from API documentation'}
            >
              Auto Configure from Docs
            </button>
          </div>

          {formMode === 'auto' ? (
            /* ── Auto Configure Form ── */
            <div>
              <p className="text-sm text-gray-400 mb-3">
                Provide the API documentation URL. An existing LLM will analyze it and extract the configuration automatically.
              </p>
              <div className="grid grid-cols-1 gap-3">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">API Documentation URL</label>
                  <input
                    placeholder="https://platform.openai.com/docs/api-reference or any API doc page"
                    value={autoDocUrl}
                    onChange={e => setAutoDocUrl(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">API Key for the new provider</label>
                  <input
                    type="password"
                    placeholder="sk-... (will be pre-filled in the config)"
                    value={autoApiKey}
                    onChange={e => setAutoApiKey(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
              </div>
              <div className="mt-4 flex gap-2">
                <button
                  onClick={handleAutoAnalyze}
                  disabled={autoAnalyzing || !autoDocUrl}
                  className="px-4 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
                >
                  {autoAnalyzing ? 'Analyzing documentation...' : 'Analyze & Configure'}
                </button>
                <button onClick={() => setShowForm(false)} className="px-4 py-2 bg-gray-700 text-white rounded text-sm hover:bg-gray-600">Cancel</button>
              </div>

              {/* Auto result */}
              {autoResult && (
                <div className={`mt-4 rounded p-3 text-sm ${autoResult.success ? 'bg-green-950 border border-green-800' : 'bg-red-950 border border-red-800'}`}>
                  {autoResult.success && autoResult.config ? (
                    <div>
                      <div className="text-green-400 font-medium mb-2">Configuration extracted successfully!</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div><span className="text-gray-500">Name:</span> <span className="text-green-300">{autoResult.config.name}</span></div>
                        <div><span className="text-gray-500">Base URL:</span> <span className="text-green-300">{autoResult.config.base_url}</span></div>
                        <div><span className="text-gray-500">OpenAI Compatible:</span> <span className="text-green-300">{autoResult.config.is_openai_compatible ? 'Yes' : 'No'}</span></div>
                        <div><span className="text-gray-500">Auth:</span> <span className="text-green-300">{autoResult.config.auth_header}</span></div>
                      </div>
                      {autoResult.config.models?.length > 0 && (
                        <div className="mt-2 text-xs">
                          <span className="text-gray-500">Models: </span>
                          <span className="text-green-300">{autoResult.config.models.join(', ')}</span>
                        </div>
                      )}
                      <p className="text-gray-400 text-xs mt-3">
                        The form below has been filled with the extracted configuration. Review and click "Create" to save.
                      </p>
                    </div>
                  ) : (
                    <div className="text-red-400">
                      {autoResult.error || 'Failed to extract configuration'}
                      {autoResult.raw_analysis && (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-xs text-gray-500">Show raw LLM response</summary>
                          <pre className="mt-1 text-xs text-gray-400 whitespace-pre-wrap max-h-40 overflow-y-auto">{autoResult.raw_analysis}</pre>
                        </details>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            /* ── Manual Setup Form ── */
            <div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Provider</label>
                  <select
                    value={preset}
                    onChange={e => handlePresetChange(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
                  >
                    {(() => {
                      const groups: Record<string, string[]> = {}
                      for (const [key, val] of Object.entries(PROVIDER_PRESETS)) {
                        ;(groups[val.group] ??= []).push(key)
                      }
                      return Object.entries(groups).map(([group, keys]) => (
                        <optgroup key={group} label={group}>
                          {keys.map(k => (
                            <option key={k} value={k}>
                              {k === 'custom' ? 'Custom (OpenAI Compatible)' : PROVIDER_PRESETS[k].name}
                            </option>
                          ))}
                        </optgroup>
                      ))
                    })()}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Name</label>
                  <input
                    value={form.name}
                    onChange={e => setForm({ ...form, name: e.target.value })}
                    className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Base URL</label>
                  <input
                    value={form.base_url}
                    onChange={e => setForm({ ...form, base_url: e.target.value })}
                    className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">API Key</label>
                  <input
                    type="password"
                    placeholder={preset === 'ollama' ? '(not required)' : 'sk-... or your API key'}
                    value={form.api_key}
                    onChange={e => setForm({ ...form, api_key: e.target.value })}
                    className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
              </div>
              <div className="mt-4 flex gap-2">
                <button onClick={handleCreate} className="px-4 py-2 bg-green-600 text-white rounded text-sm hover:bg-green-700">Create</button>
                <button onClick={() => setShowForm(false)} className="px-4 py-2 bg-gray-700 text-white rounded text-sm hover:bg-gray-600">Cancel</button>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="space-y-3">
        {providers.map(p => (
          <div key={p.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium">{p.name}</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-400">{p.provider_type}</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${p.is_enabled ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}`}>
                    {p.is_enabled ? 'enabled' : 'disabled'}
                  </span>
                </div>
                <div className="text-sm text-gray-500 mt-1">{p.base_url}</div>
                <div className="text-xs text-gray-600 mt-0.5">
                  API Key: {p.api_key ? '****' + p.api_key.slice(-4) : '(none)'}
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleTest(p.id)}
                  disabled={testing === p.id}
                  className="px-3 py-1 text-sm rounded bg-blue-900 hover:bg-blue-800 text-blue-300 disabled:opacity-50"
                >
                  {testing === p.id ? 'Testing...' : 'Test'}
                </button>
                <button
                  onClick={() => handleToggle(p)}
                  className="px-3 py-1 text-sm rounded bg-gray-800 hover:bg-gray-700"
                >
                  {p.is_enabled ? 'Disable' : 'Enable'}
                </button>
                <button
                  onClick={() => handleDelete(p.id)}
                  className="px-3 py-1 text-sm rounded bg-red-900 hover:bg-red-800 text-red-300"
                >
                  Delete
                </button>
              </div>
            </div>
            {testResult[p.id] && (
              <div className={`mt-3 text-sm px-3 py-2 rounded ${testResult[p.id]!.success ? 'bg-green-950 text-green-400 border border-green-800' : 'bg-red-950 text-red-400 border border-red-800'}`}>
                {testResult[p.id]!.message}
              </div>
            )}
          </div>
        ))}
        {providers.length === 0 && (
          <p className="text-gray-500 text-center py-8">No providers configured. Click "Add Provider" to get started.</p>
        )}
      </div>
    </div>
  )
}

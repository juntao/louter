export default function Setup() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Setup Guide</h1>

      <div className="space-y-6">
        <Section title="1. Configure Providers">
          <p className="text-gray-400 text-sm mb-3">
            Add your LLM providers via the <a href="/providers" className="text-blue-400 hover:underline">Providers</a> page
            or by editing <code className="text-xs bg-gray-800 px-1 py-0.5 rounded">louter.toml</code>.
          </p>
        </Section>

        <Section title="2. Create an API Key">
          <p className="text-gray-400 text-sm mb-3">
            Go to <a href="/keys" className="text-blue-400 hover:underline">Keys</a> and create a key.
            Keys follow the format <code className="text-xs bg-gray-800 px-1 py-0.5 rounded text-yellow-400">lot_xxxx</code>.
          </p>
        </Section>

        <Section title="3. Point Your App to Louter">
          <CodeBlock title="Environment Variables">
{`OPENAI_API_BASE=http://localhost:6188/v1
OPENAI_API_KEY=lot_your_key_here`}
          </CodeBlock>

          <CodeBlock title="OpenAI Python SDK">
{`from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6188/v1",
    api_key="lot_your_key_here",
)

# Models auto-route to the right provider
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}],
)`}
          </CodeBlock>

          <CodeBlock title="OpenAI Node.js SDK">
{`import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:6188/v1",
  apiKey: "lot_your_key_here",
});

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello!" }],
});`}
          </CodeBlock>

          <CodeBlock title="curl">
{`curl http://localhost:6188/v1/chat/completions \\
  -H "Authorization: Bearer lot_your_key_here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'`}
          </CodeBlock>
        </Section>

        <Section title="4. Model Routing">
          <p className="text-gray-400 text-sm mb-3">
            Models are automatically routed by prefix. You can also configure custom rules per key.
          </p>
          <div className="bg-gray-950 rounded-lg p-4 text-sm space-y-1">
            <RouteRow pattern="claude-*" provider="Anthropic" />
            <RouteRow pattern="gpt-*, o1-*, dall-e-*" provider="OpenAI" />
            <RouteRow pattern="deepseek-*" provider="DeepSeek" />
            <RouteRow pattern="qwen-*" provider="Qwen (custom)" />
            <RouteRow pattern="llama-*, mistral-*, etc." provider="Ollama" />
            <RouteRow pattern="{provider-name}-*" provider="Custom (by name match)" />
          </div>
        </Section>

        <Section title="5. Multimodal Endpoints">
          <p className="text-gray-400 text-sm mb-3">
            Beyond chat completions, Louter proxies these OpenAI-compatible endpoints:
          </p>
          <div className="bg-gray-950 rounded-lg p-4 text-sm space-y-1">
            <RouteRow pattern="POST /v1/images/generations" provider="Text-to-Image (DALL-E, etc.)" />
            <RouteRow pattern="POST /v1/images/edits" provider="Image Editing" />
            <RouteRow pattern="POST /v1/embeddings" provider="Text Embeddings" />
            <RouteRow pattern="POST /v1/audio/speech" provider="Text-to-Speech" />
            <RouteRow pattern="POST /v1/audio/transcriptions" provider="Speech-to-Text" />
          </div>
          <p className="text-gray-500 text-xs mt-2">
            These endpoints auto-route to the correct provider based on the model field in the request.
          </p>
        </Section>

        <Section title="6. Custom Providers">
          <p className="text-gray-400 text-sm mb-3">
            Any OpenAI-compatible API can be added as a "Custom" provider. Just provide the base URL and API key.
            This works with Qwen, Groq, Together AI, Fireworks, vLLM, LM Studio, and any other OpenAI-compatible service.
          </p>
          <p className="text-gray-400 text-sm">
            Model names starting with the provider name are auto-routed.
            For example, if you add a provider named "qwen", all <code className="text-xs bg-gray-800 px-1 py-0.5 rounded text-blue-400">qwen-*</code> models route to it automatically.
          </p>
        </Section>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
      <h2 className="text-lg font-semibold mb-3">{title}</h2>
      {children}
    </div>
  )
}

function CodeBlock({ title, children }: { title: string; children: string }) {
  return (
    <div className="mb-4">
      <div className="text-xs text-gray-500 mb-1">{title}</div>
      <pre className="bg-gray-950 p-3 rounded text-xs text-green-400 overflow-x-auto whitespace-pre">
        {children}
      </pre>
    </div>
  )
}

function RouteRow({ pattern, provider }: { pattern: string; provider: string }) {
  return (
    <div className="flex items-center gap-2">
      <code className="text-blue-400">{pattern}</code>
      <span className="text-gray-600">→</span>
      <span className="text-green-400">{provider}</span>
    </div>
  )
}

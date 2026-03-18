use std::sync::Arc;

use axum::extract::State;
use axum::response::Html;

use crate::AppState;

/// GET /api/docs - Agent auto-configuration documentation
pub async fn docs(State(state): State<Arc<AppState>>) -> Html<String> {
    let addr = format!("{}:{}", state.config.server.host, state.config.server.port);

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Louter - Agent Configuration Guide</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; }}
        h2 {{ color: #79c0ff; margin-top: 2em; }}
        code {{ background: #161b22; padding: 2px 6px; border-radius: 4px; color: #f0883e; }}
        pre {{ background: #161b22; padding: 16px; border-radius: 8px; overflow-x: auto; border: 1px solid #30363d; }}
        .endpoint {{ color: #7ee787; }}
    </style>
</head>
<body>
    <h1>Louter - Local AI API Router</h1>
    <p>Base URL: <code>http://{addr}/v1</code></p>

    <h2>Quick Setup</h2>
    <p>Point your AI application to Louter instead of directly to the provider:</p>
    <pre>
OPENAI_API_BASE=http://{addr}/v1
OPENAI_API_KEY=lot_your_key_here</pre>

    <h2>OpenAI Python SDK</h2>
    <pre>
from openai import OpenAI

client = OpenAI(
    base_url="http://{addr}/v1",
    api_key="lot_your_key_here",
)

# Use any model from any configured provider
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",  # Routes to Anthropic
    messages=[{{"role": "user", "content": "Hello!"}}],
)

response = client.chat.completions.create(
    model="gpt-4o",  # Routes to OpenAI
    messages=[{{"role": "user", "content": "Hello!"}}],
)</pre>

    <h2>curl Example</h2>
    <pre>
curl http://{addr}/v1/chat/completions \
  -H "Authorization: Bearer lot_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{{
    "model": "claude-sonnet-4-20250514",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "stream": true
  }}'</pre>

    <h2>API Endpoints</h2>
    <ul>
        <li><span class="endpoint">POST /v1/chat/completions</span> - Chat completions (streaming supported)</li>
        <li><span class="endpoint">GET /v1/models</span> - List available models</li>
        <li><span class="endpoint">GET /api/admin/providers</span> - List providers</li>
        <li><span class="endpoint">GET /api/admin/keys</span> - List API keys</li>
        <li><span class="endpoint">GET /api/admin/usage</span> - Usage logs</li>
    </ul>

    <h2>Model Routing</h2>
    <p>Models are automatically routed to the correct provider by prefix:</p>
    <ul>
        <li><code>claude-*</code> → Anthropic</li>
        <li><code>gpt-*</code>, <code>o1-*</code>, <code>o3-*</code>, <code>o4-*</code> → OpenAI</li>
        <li><code>deepseek-*</code> → DeepSeek</li>
        <li><code>llama-*</code>, <code>mistral-*</code>, etc. → Ollama</li>
    </ul>
    <p>Custom routing rules can be configured per API key via the admin API.</p>
</body>
</html>"#,
    );

    Html(html)
}

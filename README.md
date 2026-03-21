<p align="center">
  <h1 align="center">Louter</h1>
  <p align="center">Local LLM gateway for your AI agents.</p>
</p>

<p align="center">
  <a href="#install">Install</a> &bull;
  <a href="#agent-setup">Agent Setup</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#routing">Routing</a> &bull;
  <a href="#%E4%B8%AD%E6%96%87%E8%AF%B4%E6%98%8E">中文说明</a>
</p>

---

Run one local gateway, connect every agent to every LLM.

```
  Claude Code ──┐
    OpenClaw ───┤                      ┌── OpenAI
       Cline ───┼──→  Louter (:6188)  ─┼── Anthropic
       aider ───┤     localhost        ├── DeepSeek
    your app ───┘                      ├── Ollama (local)
                                       └── Qwen / Groq / Azure / ...
```

Louter is a single-binary LLM API gateway that runs on your machine. It gives all your local agents — Claude Code, OpenClaw, Cline, aider, or any tool that speaks the OpenAI protocol — a single `localhost` endpoint that routes to any LLM provider.

**No Docker. No Redis. No Postgres.** Just one binary with embedded SQLite and a Web UI.

## Install

**One-line install** (requires Rust and Node.js):

```bash
curl -fsSL https://raw.githubusercontent.com/Drlucaslu/louter/main/install.sh | bash
```

**Or build manually:**

```bash
git clone https://github.com/Drlucaslu/louter.git && cd louter
cd web && npm install && npm run build && cd ..
cargo build --release
./target/release/louter
```

Then open **http://localhost:6188** — add your providers and create an API key (`lot_xxxx`).

## Agent Setup

Once Louter is running and you've added providers via the Web UI, configure your agents:

### Claude Code

```bash
# Set environment variables before launching Claude Code
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here

# Now Claude Code can use any model through Louter
claude --model gpt-4o          # → routes to OpenAI
claude --model deepseek-chat   # → routes to DeepSeek
```

Or add to your shell profile (`~/.zshrc` / `~/.bashrc`) to make it permanent:

```bash
echo 'export OPENAI_API_BASE=http://localhost:6188/v1' >> ~/.zshrc
echo 'export OPENAI_API_KEY=lot_your_key_here' >> ~/.zshrc
```

### OpenClaw

In your OpenClaw configuration, set the API base URL:

```yaml
# OpenClaw config
api_base: http://localhost:6188/v1
api_key: lot_your_key_here
```

OpenClaw will route all model requests through Louter automatically.

### Cline (VS Code)

1. Open Cline settings in VS Code
2. Set **API Provider** to "OpenAI Compatible"
3. Set **Base URL** to `http://localhost:6188/v1`
4. Set **API Key** to your `lot_xxxx` key
5. Use any model name — Louter routes it to the right provider

### aider

```bash
aider --openai-api-base http://localhost:6188/v1 \
      --openai-api-key lot_your_key_here \
      --model gpt-4o
```

Or via environment variables:

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
aider --model deepseek-chat
```

### Any OpenAI-compatible agent

Any tool that accepts `OPENAI_API_BASE` and `OPENAI_API_KEY` works out of the box:

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
# Now run your agent — it will use Louter as the gateway
```

**Python:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6188/v1",
    api_key="lot_your_key_here",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",  # routes to Anthropic
    messages=[{"role": "user", "content": "Hello!"}],
)
```

**Node.js:**

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:6188/v1",
  apiKey: "lot_your_key_here",
});

const response = await client.chat.completions.create({
  model: "gpt-4o",  // routes to OpenAI
  messages: [{ role: "user", content: "Hello!" }],
});
```

## Why Louter?

| Problem | Solution |
|---|---|
| Each agent needs its own API keys and provider config | One gateway, one key — all agents share it |
| Switching LLM providers means reconfiguring every agent | Swap providers in the Web UI; agents don't change |
| Anthropic, Azure, etc. have different APIs | Louter converts everything to/from OpenAI format |
| Want to use local models (Ollama) alongside cloud APIs | Same endpoint for both — route by model name |
| Gateway tools need Docker + Redis + Postgres | Louter is a single binary, zero dependencies |
| Adding a new provider means editing config and restarting | Add providers at runtime via Web UI, or auto-configure from docs |

## Features

- **One endpoint for all providers** — OpenAI, Anthropic, Azure, DeepSeek, Ollama, and any OpenAI-compatible API (Qwen, Groq, Together, Fireworks, vLLM, LM Studio)
- **Smart routing** — `claude-*` → Anthropic, `gpt-*` → OpenAI, `deepseek-*` → DeepSeek, etc. Custom glob rules with priorities per key. Use `model: "auto"` for content-based routing
- **Full API coverage** — Chat completions, streaming, models, images, embeddings, audio — all OpenAI-compatible endpoints
- **Native format conversion** — Anthropic tool use, Azure deployment URLs, provider-specific auth — handled transparently
- **Built-in Web UI** — Manage providers, keys, routing rules, and view usage analytics
- **Auto-configure** — Paste an API doc URL, an existing LLM analyzes it and fills in the config
- **Single binary** — Rust + embedded SQLite + embedded React UI. No external services needed
- **Runtime reconfiguration** — Add, update, or disable providers without restarting

## Routing

Three-tier routing — no configuration needed for common providers:

| Model name | Routes to | How |
|---|---|---|
| `claude-*` | Anthropic | Built-in prefix match |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI | Built-in prefix match |
| `deepseek-*` | DeepSeek | Built-in prefix match |
| `llama*`, `mistral*`, `gemma*` | Ollama | Built-in prefix match |
| `qwen-*` | Qwen (custom) | Provider name match |

Override with per-key rules in the Web UI:

| Pattern | Provider | Priority |
|---|---|---|
| `gpt-4o*` | Azure OpenAI | 10 |
| `gpt-*` | OpenAI | 1 |
| `*` | DeepSeek | 0 |

### Smart Routing (`model: "auto"`)

Send `model: "auto"` and Louter automatically classifies your message and routes to the best provider. Pure keyword-based detection, zero latency, no LLM calls.

**Categories** (checked in priority order):

| Category | Detection | Examples |
|---|---|---|
| `code` | Code blocks (` ``` `), keywords (`function`, `def`, `class`, `import`, `bug`, `fix`, `debug`...) | "Help me write a Python function" |
| `math` | Math symbols (`∑`, `∫`, `√`), LaTeX (`\frac`, `$...$`), keywords (`equation`, `theorem`, `计算`...) | "Solve this equation" |
| `translation` | `translate`, `translation`, `翻译` | "Translate this to Chinese" |
| `general` | Default fallback | "Tell me a joke" |

**Configuration** — add to `louter.toml`:

```toml
[smart_routing]
code = "anthropic/claude-sonnet-4-20250514"
math = "deepseek/deepseek-chat"
translation = "qwen/qwen-plus"
general = "qwen/qwen-plus"
```

Format: `category = "provider_name/model_name"`. The provider name must match a provider you've added in the Web UI.

**Usage:**

```bash
curl http://localhost:6188/v1/chat/completions \
  -H "Authorization: Bearer lot_xxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Help me write a Python function"}]}'
# → routes to anthropic/claude-sonnet-4-20250514 (code category)
```

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat (streaming + non-streaming) |
| `GET /v1/models` | List all models from all providers |
| `POST /v1/images/generations` | Text-to-image |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/speech` | Text-to-speech |
| `POST /v1/audio/transcriptions` | Speech-to-text |

## Architecture

```
┌─────────────────────────────────────────────┐
│               Louter (single binary)        │
│                                             │
│  ┌──────────┐  ┌────────────────────────┐   │
│  │  Web UI  │  │  /v1/* API Gateway     │   │
│  │  :6188   │  │  (OpenAI-compatible)   │   │
│  └──────────┘  └───────────┬────────────┘   │
│                            │                │
│                 ┌──────────▼──────────┐      │
│                 │   Router Engine     │      │
│                 │ rules → prefix →    │      │
│                 │ default provider    │      │
│                 └──────────┬──────────┘      │
│                            │                │
│  ┌──────────┐   ┌─────────▼────────────┐    │
│  │  SQLite  │   │  Provider Registry   │    │
│  │  (keys,  │   │                      │    │
│  │  rules,  │   │ OpenAI   Anthropic   │    │
│  │  usage)  │   │ Azure    DeepSeek    │    │
│  └──────────┘   │ Ollama   Custom...   │    │
│                 └──────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Tech stack:** Rust (Axum + Tokio) / SQLite / React + Tailwind / single binary via `rust-embed`

## License

MIT

---

# 中文说明

## Louter — 本地 Agent 的 LLM 统一网关

在本地运行一个网关，让你所有的 AI Agent 连接所有大模型。

```
  Claude Code ──┐
    OpenClaw ───┤                      ┌── OpenAI
       Cline ───┼──→  Louter (:6188)  ─┼── Anthropic
       aider ───┤     localhost        ├── DeepSeek
      你的应用 ──┘                      ├── Ollama（本地）
                                       └── 通义千问 / Groq / Azure / ...
```

Louter 是一个单一二进制的大模型 API 网关，运行在你的本地机器上。它为你所有的本地 Agent — Claude Code、OpenClaw、Cline、aider，或任何支持 OpenAI 协议的工具 — 提供一个统一的 `localhost` 端点，自动路由到任意大模型供应商。

**无需 Docker。无需 Redis。无需 Postgres。** 只需一个二进制文件，内嵌 SQLite 和 Web 管理界面。

## 安装

**一键安装**（需要 Rust 和 Node.js）：

```bash
curl -fsSL https://raw.githubusercontent.com/Drlucaslu/louter/main/install.sh | bash
```

**或手动构建：**

```bash
git clone https://github.com/Drlucaslu/louter.git && cd louter
cd web && npm install && npm run build && cd ..
cargo build --release
./target/release/louter
```

打开 **http://localhost:6188**，在 Web UI 中添加供应商并创建 API Key（`lot_xxxx`）。

## Agent 配置

Louter 运行后，在 Web UI 中添加供应商并创建 Key，然后配置你的 Agent：

### Claude Code

```bash
# 启动 Claude Code 前设置环境变量
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here

# 现在 Claude Code 可以通过 Louter 使用任意模型
claude --model gpt-4o          # → 路由到 OpenAI
claude --model deepseek-chat   # → 路由到 DeepSeek
```

写入 shell 配置文件使其永久生效：

```bash
echo 'export OPENAI_API_BASE=http://localhost:6188/v1' >> ~/.zshrc
echo 'export OPENAI_API_KEY=lot_your_key_here' >> ~/.zshrc
```

### OpenClaw

在 OpenClaw 配置中设置 API 地址：

```yaml
# OpenClaw 配置
api_base: http://localhost:6188/v1
api_key: lot_your_key_here
```

### Cline (VS Code)

1. 打开 VS Code 中的 Cline 设置
2. **API Provider** 选择 "OpenAI Compatible"
3. **Base URL** 填入 `http://localhost:6188/v1`
4. **API Key** 填入你的 `lot_xxxx` Key
5. 使用任意模型名称 — Louter 自动路由到对应供应商

### aider

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
aider --model deepseek-chat
```

### 通用方式

任何支持 `OPENAI_API_BASE` 环境变量的工具都可以直接使用：

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
# 启动你的 Agent — 它将通过 Louter 访问所有大模型
```

## 为什么选择 Louter？

| 痛点 | 解决方案 |
|---|---|
| 每个 Agent 都需要单独配置 API Key 和供应商 | 一个网关、一个 Key，所有 Agent 共享 |
| 切换大模型供应商需要重新配置每个 Agent | 在 Web UI 中切换，Agent 无需任何改动 |
| Anthropic、Azure 等各有不同的 API 格式 | Louter 统一转换为 OpenAI 兼容格式 |
| 想同时使用本地模型（Ollama）和云端 API | 同一个端点，按模型名自动路由 |
| 网关工具依赖 Docker + Redis + Postgres | 单一二进制文件，零外部依赖 |
| 添加新供应商需要改配置、重启服务 | Web UI 实时添加，或通过文档自动配置 |

## 核心特性

- **一个端点接入所有供应商** — OpenAI、Anthropic、Azure、DeepSeek、Ollama，以及任意 OpenAI 兼容 API（通义千问、Groq、Together、Fireworks、vLLM、LM Studio）
- **智能路由** — `claude-*` → Anthropic、`gpt-*` → OpenAI、`deepseek-*` → DeepSeek，支持自定义规则和优先级。使用 `model: "auto"` 按消息内容自动路由
- **完整的 API 覆盖** — 聊天补全、流式响应、模型列表、图像、嵌入、音频 — 全部 OpenAI 兼容
- **原生格式转换** — Anthropic 工具调用、Azure 部署 URL、供应商特定认证 — 透明处理
- **内置 Web UI** — 管理供应商、Key、路由规则，查看用量分析
- **自动配置** — 粘贴 API 文档地址，已有的 LLM 自动分析并填入配置
- **单一二进制** — Rust + 内嵌 SQLite + 内嵌 React UI，无需任何外部服务
- **运行时动态配置** — 添加、更新、禁用供应商无需重启

## 路由机制

三级路由，常用供应商无需配置：

| 模型名称 | 路由到 | 方式 |
|---|---|---|
| `claude-*` | Anthropic | 内置前缀匹配 |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI | 内置前缀匹配 |
| `deepseek-*` | DeepSeek | 内置前缀匹配 |
| `llama*`, `mistral*`, `gemma*` | Ollama | 内置前缀匹配 |
| `qwen-*` | 通义千问（自定义） | 供应商名称匹配 |

可在 Web UI 中为每个 Key 设置自定义规则覆盖默认行为。

### 智能路由（`model: "auto"`）

发送 `model: "auto"`，Louter 自动分析消息内容并路由到最合适的供应商。纯关键词检测，零延迟，不调用任何 LLM。

**分类**（按优先级检测）：

| 分类 | 检测方式 | 示例 |
|---|---|---|
| `code` | 代码块（` ``` `）、编程关键词（`function`、`def`、`class`、`import`、`bug`、`fix`、`debug`...） | "帮我写一个排序函数" |
| `math` | 数学符号（`∑`、`∫`、`√`）、LaTeX（`\frac`、`$...$`）、关键词（`equation`、`计算`、`方程`...） | "解这个方程" |
| `translation` | `translate`、`translation`、`翻译` | "翻译这段话" |
| `general` | 默认分类 | "今天天气怎么样" |

**配置** — 在 `louter.toml` 中添加：

```toml
[smart_routing]
code = "anthropic/claude-sonnet-4-20250514"
math = "deepseek/deepseek-chat"
translation = "qwen/qwen-plus"
general = "qwen/qwen-plus"
```

格式：`分类 = "供应商名称/模型名称"`。供应商名称需与 Web UI 中添加的供应商一致。

## 许可证

MIT

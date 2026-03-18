<p align="center">
  <h1 align="center">Louter</h1>
  <p align="center">One endpoint. Every LLM.</p>
</p>

<p align="center">
  <a href="#features">Features</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#routing">Routing</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#%E4%B8%AD%E6%96%87%E8%AF%B4%E6%98%8E">中文说明</a>
</p>

---

Louter is a lightweight, self-hosted LLM API gateway that unifies OpenAI, Anthropic, Azure, DeepSeek, Ollama, and any OpenAI-compatible provider behind a single OpenAI-compatible endpoint. One binary, zero external dependencies, built-in Web UI.

```
Your App (OpenAI SDK)  ──→  Louter (:6188)  ──→  OpenAI
                                              ──→  Anthropic
                                              ──→  Azure OpenAI
                                              ──→  DeepSeek
                                              ──→  Ollama
                                              ──→  Qwen / Groq / Together / ...
```

## Why Louter?

| Pain point | How Louter solves it |
|---|---|
| Switching providers means changing SDK code | Point all apps at Louter; swap providers in the Web UI |
| Each provider has its own auth, format, and quirks | Louter speaks OpenAI format — it converts to Anthropic, Azure, etc. under the hood |
| Managing keys across teams and services | Issue `lot_*` keys with per-key routing rules and usage tracking |
| Adding a new provider means config files and restarts | Add providers in the Web UI at runtime — or let an LLM auto-configure it from docs |
| Observability requires external tooling | Built-in usage analytics: tokens, latency, per-model breakdown |
| Gateway tools require Docker, Redis, Postgres, etc. | Louter is a single binary with embedded SQLite and Web UI |

## Features

- **Unified OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`, images, embeddings, audio — all work with standard OpenAI SDKs
- **Native provider support** — OpenAI, Anthropic (full format conversion including tool use), Azure OpenAI, DeepSeek, Ollama
- **Any OpenAI-compatible API** — Qwen, Groq, Together AI, Fireworks, vLLM, LM Studio — just add a base URL
- **Smart routing** — Auto-routes `claude-*` → Anthropic, `gpt-*` → OpenAI, etc. Custom glob rules with priorities per key
- **Streaming** — Full SSE streaming support with proper token counting across all providers
- **Auto-configure** — Paste an API doc URL, Louter uses an existing LLM to extract the config automatically
- **Built-in Web UI** — Dark-themed admin dashboard for providers, keys, routing rules, and usage analytics
- **Single binary** — Rust + embedded SQLite + embedded React frontend. No Docker, no external services
- **Runtime reconfiguration** — Add, update, disable providers without restart

## Quick Start

### Build from source

```bash
# Build frontend
cd web && npm install && npm run build && cd ..

# Build backend (embeds frontend into binary)
cargo build --release

# Run
./target/release/louter
```

Open **http://localhost:6188** — the Web UI guides you through setup.

### Configure

Louter works with zero config (defaults to `localhost:6188` + `louter.db`). To customize:

```bash
cp louter.example.toml louter.toml
# Edit host, port, database path
./target/release/louter louter.toml
```

## Usage

### 1. Add providers in the Web UI

Go to **Providers** → **Add Provider**, pick a preset (OpenAI, Anthropic, etc.) or add any OpenAI-compatible API.

### 2. Create an API key

Go to **Keys** → **Create Key**. Keys use the format `lot_xxxx`.

### 3. Point your app at Louter

**Python (OpenAI SDK)**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6188/v1",
    api_key="lot_your_key_here",
)

# Auto-routes to the right provider
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

**Node.js (OpenAI SDK)**

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:6188/v1",
  apiKey: "lot_your_key_here",
});

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello!" }],
});
```

**curl**

```bash
curl http://localhost:6188/v1/chat/completions \
  -H "Authorization: Bearer lot_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Environment variables** — works with any app that uses the OpenAI SDK:

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
```

## Routing

Louter resolves which provider handles a request using a three-tier system:

### 1. Per-key routing rules (highest priority)

Custom glob rules configured in the Web UI per API key:

| Pattern | Provider | Priority |
|---|---|---|
| `gpt-4o*` | Azure OpenAI | 10 |
| `gpt-*` | OpenAI | 1 |
| `claude-*` | Anthropic | 1 |

### 2. Auto-routing by model prefix

Built-in mappings — no configuration needed:

| Model prefix | Routes to |
|---|---|
| `claude-*` | Anthropic |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `dall-e-*` | OpenAI |
| `deepseek-*` | DeepSeek |
| `llama*`, `mistral*`, `gemma*`, `phi*` | Ollama |
| `{provider-name}-*` | Custom provider by name match |

### 3. Default provider (fallback)

Each API key can have a default provider for unmatched models.

## Supported Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat (streaming + non-streaming) |
| `GET /v1/models` | List models from all providers |
| `POST /v1/images/generations` | Text-to-image |
| `POST /v1/images/edits` | Image editing |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/speech` | Text-to-speech |
| `POST /v1/audio/transcriptions` | Speech-to-text |
| `POST /v1/audio/translations` | Speech translation |

## Architecture

```
┌─────────────────────────────────────────┐
│              Louter Binary              │
│                                         │
│  ┌──────────┐  ┌─────────────────────┐  │
│  │ Web UI   │  │  OpenAI-Compatible  │  │
│  │ (React)  │  │    API Gateway      │  │
│  └──────────┘  └────────┬────────────┘  │
│                         │               │
│              ┌──────────▼──────────┐    │
│              │   Router Engine     │    │
│              │  (rules → prefix    │    │
│              │   → default)        │    │
│              └──────────┬──────────┘    │
│                         │               │
│  ┌──────────┐  ┌───────▼───────────┐   │
│  │ SQLite   │  │ Provider Registry │   │
│  │ (keys,   │  │                   │   │
│  │  rules,  │  │ OpenAI  Anthropic │   │
│  │  usage)  │  │ Azure   DeepSeek  │   │
│  └──────────┘  │ Ollama  Custom... │   │
│                └───────────────────┘   │
└─────────────────────────────────────────┘
```

**Tech stack**: Rust (Axum + Tokio) / SQLite (sqlx) / React 19 + Tailwind CSS (Vite) / Single binary via `rust-embed`

## License

MIT

---

# 中文说明

## Louter — 一个端点，所有大模型

Louter 是一个轻量级、可自部署的大模型 API 网关。它将 OpenAI、Anthropic、Azure、DeepSeek、Ollama 以及任何 OpenAI 兼容的服务统一到一个 OpenAI 兼容的端点之后。单一二进制文件，零外部依赖，内置 Web 管理界面。

## 为什么选择 Louter？

| 痛点 | Louter 的解决方案 |
|---|---|
| 切换供应商需要修改代码 | 所有应用指向 Louter，在 Web UI 中切换供应商即可 |
| 每个供应商有不同的认证方式和格式 | Louter 统一使用 OpenAI 格式，自动转换为 Anthropic、Azure 等原生格式 |
| 跨团队管理多个 API Key | 发放 `lot_*` 格式的 Key，支持按 Key 设置路由规则和用量追踪 |
| 添加新供应商需要改配置、重启服务 | 在 Web UI 中实时添加，或让 LLM 通过文档自动配置 |
| 可观测性依赖外部工具 | 内置用量分析：Token 数、延迟、按模型维度的统计 |
| 网关工具依赖 Docker、Redis、Postgres 等 | Louter 是单一二进制文件，内嵌 SQLite 和 Web UI |

## 核心特性

- **统一的 OpenAI 兼容 API** — `/v1/chat/completions`、`/v1/models`、图像、嵌入、音频端点，全部兼容 OpenAI SDK
- **原生供应商支持** — OpenAI、Anthropic（完整格式转换，包括工具调用）、Azure OpenAI、DeepSeek、Ollama
- **支持任意 OpenAI 兼容 API** — 通义千问、Groq、Together AI、Fireworks、vLLM、LM Studio 等，只需填入 Base URL
- **智能路由** — 自动将 `claude-*` 路由到 Anthropic、`gpt-*` 路由到 OpenAI 等；支持按 Key 配置自定义 Glob 规则和优先级
- **流式响应** — 全面支持 SSE 流式传输，跨所有供应商正确统计 Token
- **自动配置** — 粘贴 API 文档地址，Louter 利用已有的 LLM 自动提取配置
- **内置 Web UI** — 暗色主题的管理后台，管理供应商、Key、路由规则和用量分析
- **单一二进制** — Rust + 内嵌 SQLite + 内嵌 React 前端，无需 Docker，无需外部服务
- **运行时动态配置** — 添加、更新、禁用供应商无需重启

## 快速开始

```bash
# 构建前端
cd web && npm install && npm run build && cd ..

# 构建后端（前端会嵌入到二进制中）
cargo build --release

# 运行
./target/release/louter
```

打开 **http://localhost:6188**，Web UI 会引导你完成配置。

## 使用方法

### 1. 在 Web UI 中添加供应商

进入 **Providers** → **Add Provider**，选择预设（OpenAI、Anthropic 等）或添加任意 OpenAI 兼容的 API。

### 2. 创建 API Key

进入 **Keys** → **Create Key**。Key 格式为 `lot_xxxx`。

### 3. 将应用指向 Louter

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6188/v1",
    api_key="lot_your_key_here",
)

# 自动路由到正确的供应商
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

或者设置环境变量，对所有使用 OpenAI SDK 的应用生效：

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
```

## 路由机制

Louter 通过三级路由系统决定请求由哪个供应商处理：

1. **按 Key 的自定义规则**（最高优先级）— 在 Web UI 中为每个 Key 配置 Glob 模式匹配规则
2. **按模型前缀自动路由** — 内置映射：`claude-*` → Anthropic，`gpt-*` → OpenAI 等
3. **默认供应商**（兜底）— 每个 Key 可设置一个默认供应商

## 许可证

MIT

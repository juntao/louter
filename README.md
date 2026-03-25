<p align="center">
  <h1 align="center">Louter</h1>
  <p align="center">Local LLM gateway with hybrid inference and distillation.</p>
</p>

<p align="center">
  <a href="#install">Install</a> &bull;
  <a href="#agent-setup">Agent Setup</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#hybrid-inference">Hybrid Inference</a> &bull;
  <a href="#distillation">Distillation</a> &bull;
  <a href="#routing">Routing</a> &bull;
  <a href="#%E4%B8%AD%E6%96%87%E8%AF%B4%E6%98%8E">中文说明</a>
</p>

---

Run one local gateway, route every agent to the right model — cloud or local.

```
  Claude Code ──┐                         ┌── OpenAI
    OpenClaw ───┤                         ├── Anthropic
       Cline ───┼──→  Louter (:6188)  ───┼── DeepSeek
       aider ───┤     localhost           ├── Ollama (local, distilled)
    your app ───┘                         └── Qwen / Groq / Azure / ...
                        │
                  ┌─────▼─────┐
                  │  Hybrid   │  Session-aware routing
                  │  Router   │  Local-first + cloud fallback
                  │           │  Auto-escalation on failure
                  └─────┬─────┘
                        │
                  ┌─────▼─────┐
                  │ Distill   │  Collect cloud responses
                  │ Pipeline  │  Compress → Train → Deploy
                  └───────────┘  Data flywheel
```

Louter is a single-binary LLM API gateway that sits between your AI agents and LLM providers. Beyond routing, it implements **hybrid inference** — automatically routing simple requests to a fast local model and complex ones to the cloud — and a **distillation pipeline** that continuously improves the local model from cloud responses.

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
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here

claude --model gpt-4o          # → routes to OpenAI
claude --model deepseek-chat   # → routes to DeepSeek
```

### OpenClaw

```yaml
api_base: http://localhost:6188/v1
api_key: lot_your_key_here
```

### Cline (VS Code)

1. Open Cline settings → **API Provider** → "OpenAI Compatible"
2. **Base URL**: `http://localhost:6188/v1`
3. **API Key**: your `lot_xxxx` key

### aider

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
aider --model deepseek-chat
```

### Any OpenAI-compatible agent

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

## Features

- **One endpoint for all providers** — OpenAI, Anthropic, Azure, DeepSeek, Ollama, and any OpenAI-compatible API
- **Hybrid inference** — Route simple requests to a fast local model, complex ones to the cloud. Session-aware: same model throughout a conversation, with auto-escalation on failure
- **Distillation pipeline** — Automatically collect cloud responses as training data, compress, fine-tune a local model via LoRA, and deploy to Ollama
- **Smart routing** — `claude-*` → Anthropic, `gpt-*` → OpenAI, `deepseek-*` → DeepSeek. Custom glob rules with priorities. `model: "auto"` for content-based routing
- **Tool call normalizer** — Parse Hermes/Qwen/ReAct/JSON tool call formats from local models and convert to standard OpenAI format
- **Implicit feedback** — Detect agent retries to automatically label training samples as success/failure
- **Runtime tuning** — Adjust routing thresholds via API without restarting. Relax limits as the distilled model improves
- **Full API coverage** — Chat, streaming, models, images, embeddings, audio — all OpenAI-compatible
- **Native format conversion** — Anthropic tool use, Azure deployment URLs, provider-specific auth — handled transparently
- **Built-in Web UI** — Manage providers, keys, routing rules, distillation stats, and usage analytics
- **Single binary** — Rust (Axum + Tokio) / SQLite / React + Tailwind / embedded via `rust-embed`

## Hybrid Inference

Louter can route requests between a local model (via Ollama) and a cloud model, based on task type, context size, and historical success rates.

```toml
# louter.toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "qwen2.5:1.5b"           # Fast local model (~90 tok/s)
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514" # Powerful cloud fallback
min_local_success_rate = 0.7             # Route to cloud if local success < 70%
min_samples = 20                         # Min data before trusting success rate
fallback_enabled = true                  # Try local first, fall back to cloud
local_task_types = ["general"]           # Only route simple conversations locally
max_local_context_tokens = 2000          # Large contexts go to cloud
max_local_latency_ms = 30000             # Slow responses go to cloud
```

### How it works

1. **New conversation** — Hybrid router decides local vs cloud based on task type, context size, and historical success rate
2. **Continuation** — Same model is used for all turns in the conversation (session-aware routing)
3. **Local failure** — If the local model returns a bad response, the session is escalated to cloud for all remaining turns
4. **Data collection** — Cloud responses are automatically saved as training samples for distillation

### Runtime adjustment

As your distilled model improves, relax the limits without restarting:

```bash
# After first distillation round — try tool_call locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -H "Content-Type: application/json" \
  -d '{"local_task_types": ["general", "tool_call"], "max_local_context_tokens": 4000}'

# After more training — route everything locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -d '{"local_task_types": ["general", "tool_call", "code"], "max_local_context_tokens": 8000}'
```

## Distillation

Louter automatically collects cloud model responses as training data, then provides tools to compress, fine-tune, and deploy a local model — creating a **data flywheel** where the local model gets better over time.

```
Use Louter normally
    ↓
Cloud responses saved to training_samples table
    ↓
Export & compress: ./distill/run_distill.sh --export-only
    ↓
Fine-tune local model (LoRA on Qwen2.5-1.5B)
    ↓
Deploy to Ollama or LlamaEdge → local model handles more requests
    ↓
Fewer cloud calls → lower cost → repeat
```

### Prerequisites

**Python 3.10+** and a virtual environment:

```bash
cd distill
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This installs PyTorch, Transformers, PEFT, TRL, and other dependencies. The base model (**Qwen2.5-1.5B-Instruct**) is downloaded automatically from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) the first time you run training — no manual download needed. It caches to `~/.cache/huggingface/hub/` (~3 GB).

> To use a different base model (e.g. `Qwen/Qwen2.5-3B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`), set the `BASE_MODEL` environment variable or pass `--base-model` to `train.py`.

**Hardware requirements:**

| Setup | What to expect |
|-------|----------------|
| NVIDIA GPU (8 GB+ VRAM) | Fastest. Uses bfloat16 automatically |
| Apple Silicon (M1/M2/M3/M4) | Good. Uses MPS with float16. 1.5B model trains fine on 16 GB RAM |
| CPU only | Slow but works. Uses float32. Expect hours for 1000+ samples |

### Step 1: Collect training data

Enable data collection in `louter.toml` (enabled by default):

```toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "qwen2.5:1.5b"
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"
fallback_enabled = true

[distillation]
collect_training_data = true        # default: true
max_samples = 100000                # prune oldest when exceeded
only_successful = true              # only collect status-200 responses
```

Use Louter normally. Every cloud response is automatically saved to the `training_samples` table in SQLite. Check progress:

```bash
# Via API
curl http://localhost:6188/api/admin/distill/stats

# Or via the Web UI at http://localhost:6188 → Distillation dashboard
```

Aim for **1000+ samples** before your first training run (more is better).

### Step 2: Export and compress

```bash
cd distill
source venv/bin/activate

# Export from SQLite + compress in one step
./run_distill.sh --export-only

# Or manually:
python export.py --db ../louter.db --output output/training_data.jsonl --mark-exported --format openai
python compress.py output/training_data.jsonl -o output/training_data_compressed.jsonl --stats
```

Output files:
- `distill/output/training_data.jsonl` — raw exported samples
- `distill/output/training_data_compressed.jsonl` — deduplicated and truncated (30-50% smaller)

### Step 3: Fine-tune with LoRA

```bash
# Full pipeline (export + compress + train + merge + deploy)
./run_distill.sh

# Or train only (data must already be exported)
./run_distill.sh --train-only

# Or run train.py directly for more control
python train.py \
    --data output/training_data_compressed.jsonl \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir ./output \
    --epochs 5 --batch-size 8 --lr 5e-4

# Resume from a checkpoint
python train.py --data output/training_data_compressed.jsonl \
    --resume-from ./output/checkpoint-500
```

**Choosing a different base model:** You can fine-tune any HuggingFace causal LM — pass `--base-model` to `train.py` or set `BASE_MODEL` for `run_distill.sh`:

```bash
# Larger Qwen for better quality (needs more VRAM/RAM)
python train.py --data output/training_data_compressed.jsonl --base-model Qwen/Qwen2.5-3B-Instruct

# Llama-based model
BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct ./run_distill.sh
```

Any model with standard attention layers works (Qwen, Llama, Mistral, Gemma, Phi, etc.). The model is downloaded automatically from Hugging Face on first use.

**Output files after training:**
- `distill/output/` — LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
- Checkpoints at `distill/output/checkpoint-*/`

### Step 4: Merge and convert to GGUF

Ollama requires **GGUF format** for Qwen models (direct safetensors import only works for Llama/Mistral/Gemma/Phi3). The pipeline handles this automatically if [llama.cpp](https://github.com/ggerganov/llama.cpp) is available:

```bash
# Set up llama.cpp for GGUF conversion (one-time)
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp
pip install -r ../llama.cpp/requirements.txt

# Merge + convert to GGUF (auto-detected by train.py)
python train.py --merge \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path ./output \
    --output-dir ./merged_model

# Or as part of the full pipeline:
./run_distill.sh --deploy-only
```

You can choose a quantization type with `--gguf-type` (default: `q4_k_m`):
- `f16` — full precision, largest file, best quality
- `q8_0` — 8-bit, good balance
- `q4_k_m` — 4-bit (recommended for 1.5B), small and fast
- `q4_0` — 4-bit, smallest file

**Output files after merge:**
- `distill/merged_model/` — full merged model (safetensors + tokenizer + config)
- `distill/merged_model/model-q4_k_m.gguf` — quantized GGUF model for Ollama
- `distill/merged_model/Modelfile` — Ollama import file (auto-generated, points to GGUF)

> If llama.cpp is not found, `train.py` will still save the merged safetensors model and print manual conversion instructions.

### Step 5: Deploy the distilled model

#### Option A: Deploy to Ollama (recommended)

1. **Install Ollama** if you haven't already:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

2. **Start Ollama** (runs as a background service on port 11434):

```bash
ollama serve
# On macOS, the Ollama app starts the server automatically
```

3. **Import the distilled model** (requires GGUF — see Step 4):

```bash
cd distill
ollama create louter-distilled -f merged_model/Modelfile
```

4. **Verify it works:**

```bash
ollama run louter-distilled "Hello, what can you do?"
```

The model is now available at `http://localhost:11434` as `louter-distilled`.

#### Option B: Deploy with LlamaEdge (WasmEdge)

LlamaEdge also uses the GGUF file produced in Step 4.

1. **Install LlamaEdge** — follow instructions at [LlamaEdge docs](https://llamaedge.com/docs/user-guide/get-started-with-llamaedge):

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s -- -v 0.14.1
```

2. **Run the model** with an OpenAI-compatible API:

```bash
wasmedge --dir .:. \
    --nn-preload default:GGML:AUTO:distill/merged_model/model-q4_k_m.gguf \
    llama-api-server.wasm \
    --prompt-template chatml \
    --ctx-size 4096 \
    --model-name louter-distilled \
    --socket-addr 0.0.0.0:8080
```

The model is now available at `http://localhost:8080` as an OpenAI-compatible endpoint.

### Step 6: Configure Louter to use the distilled model

#### For Ollama deployment

Add Ollama as a provider in Louter's Web UI (`http://localhost:6188`) or via API, then update `louter.toml`:

```toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "louter-distilled"         # ← your distilled model name
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"
min_local_success_rate = 0.7
min_samples = 20
fallback_enabled = true                  # try local first, fall back to cloud on failure
local_task_types = ["general"]           # start conservative, expand later
max_local_context_tokens = 2000
max_local_latency_ms = 30000
```

#### For LlamaEdge deployment

Register LlamaEdge as a custom OpenAI-compatible provider in the Web UI:
- **Provider type**: OpenAI Compatible
- **Base URL**: `http://localhost:8080/v1`
- **Model name**: `louter-distilled`

Then in `louter.toml`:

```toml
[hybrid]
enabled = true
local_provider = "llamaedge"             # ← name you gave it in Web UI
local_model = "louter-distilled"
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"
fallback_enabled = true
local_task_types = ["general"]
```

#### Gradually expand local routing

As the distilled model proves itself, relax limits at runtime without restarting:

```bash
# After first distillation — also route tool_call tasks locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -H "Content-Type: application/json" \
  -d '{"local_task_types": ["general", "tool_call"], "max_local_context_tokens": 4000}'

# After more training rounds — route everything locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -d '{"local_task_types": ["general", "tool_call", "code"], "max_local_context_tokens": 8000}'
```

### Pipeline components

| Tool | Purpose |
|------|---------|
| `distill/export.py` | Export training samples from SQLite as JSONL (OpenAI / ShareGPT format) |
| `distill/compress.py` | Compress training data: dedup system prompts, truncate tool results, strip verbose sections |
| `distill/train.py` | LoRA fine-tuning on Qwen2.5-1.5B (supports CUDA / MPS / CPU) |
| `distill/run_distill.sh` | End-to-end: export → compress → train → merge → deploy to Ollama |

### Training parameters (optimized for 1.5B model)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Base model | Qwen2.5-1.5B-Instruct | Fast inference (~90 tok/s on Apple Silicon) |
| LoRA rank | 8 | Small model, low rank sufficient |
| LoRA alpha | 16 | 2x rank |
| Batch size | 8 | Small model fits larger batches |
| Learning rate | 5e-4 | Small models learn faster |
| Epochs | 5 | More passes for limited data |

### Monitoring

The Web UI dashboard (`http://localhost:6188/distill`) shows:

- Training sample counts by task type
- Local vs cloud routing ratio
- Local success rate per task type
- Session statistics (active, escalated)
- Current hybrid config + dynamic overrides

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/admin/distill/stats` | Training sample statistics |
| `GET /api/admin/distill/routing` | Routing history and success rates |
| `POST /api/admin/distill/export` | Export training samples as JSON |
| `GET /api/admin/distill/config` | Current hybrid + distillation config |
| `PUT /api/admin/distill/config` | Update routing thresholds at runtime |

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

Send `model: "auto"` and Louter classifies your message and routes to the best provider. Zero latency overhead.

```toml
[smart_routing]
code = "anthropic/claude-sonnet-4-20250514"
math = "deepseek/deepseek-chat"
translation = "openai/gpt-4o"
general = "openai/gpt-4o"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Louter (single binary)              │
│                                                     │
│  ┌──────────┐  ┌──────────────────────────────┐     │
│  │  Web UI  │  │  /v1/* API Gateway           │     │
│  │  React   │  │  (OpenAI-compatible)         │     │
│  └──────────┘  └──────────────┬───────────────┘     │
│                               │                     │
│                    ┌──────────▼──────────┐           │
│                    │   Session Router    │           │
│                    │   (per-conversation │           │
│                    │    model affinity)  │           │
│                    └──────────┬──────────┘           │
│                               │                     │
│                    ┌──────────▼──────────┐           │
│                    │   Hybrid Router     │           │
│                    │   task type / ctx   │           │
│                    │   size / success    │           │
│                    │   rate filtering    │           │
│                    └────┬──────────┬─────┘           │
│                         │          │                │
│              ┌──────────▼┐   ┌────▼──────────┐      │
│              │   Local   │   │    Cloud      │      │
│              │  (Ollama) │   │  (Anthropic,  │      │
│              │  + Tool   │   │   OpenAI...)  │      │
│              │  Call     │   │              │      │
│              │  Normal.  │   │  → training  │      │
│              └───────────┘   │    samples   │      │
│                              └──────────────┘      │
│                                                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐     │
│  │  SQLite  │  │ Feedback  │  │  Dynamic     │     │
│  │  (keys,  │  │ Tracker   │  │  Config      │     │
│  │  rules,  │  │ (retry    │  │  (runtime    │     │
│  │  usage,  │  │  detect)  │  │   tunable)   │     │
│  │  samples)│  └───────────┘  └──────────────┘     │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
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

## License

MIT

---

# 中文说明

## Louter — 本地 Agent 的 LLM 统一网关 + 混合推理 + 蒸馏

在本地运行一个网关，让你所有的 AI Agent 智能路由到最合适的模型 — 本地或云端。

```
  Claude Code ──┐                         ┌── OpenAI
    OpenClaw ───┤                         ├── Anthropic
       Cline ───┼──→  Louter (:6188)  ───┼── DeepSeek
       aider ───┤     localhost           ├── Ollama（本地蒸馏模型）
      你的应用 ──┘                         └── 通义千问 / Groq / Azure / ...
```

Louter 不仅是一个 API 网关，更是一个**混合推理 + 蒸馏**系统：

- **混合推理 (Hybrid Inference)** — 简单请求走快速本地模型，复杂请求走云端大模型，同一会话内保持同一模型不切换
- **蒸馏飞轮 (Distillation)** — 自动收集云端响应作为训练数据，压缩后微调本地模型，本地模型越来越强，云端调用越来越少

**无需 Docker。无需 Redis。无需 Postgres。** 单一二进制文件，内嵌 SQLite 和 Web 管理界面。

## 安装

```bash
git clone https://github.com/Drlucaslu/louter.git && cd louter
cd web && npm install && npm run build && cd ..
cargo build --release
./target/release/louter
```

打开 **http://localhost:6188**，添加供应商并创建 API Key。

## 混合推理

```toml
# louter.toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "qwen2.5:1.5b"            # 快速本地模型（~90 tok/s）
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"  # 强大的云端后备
min_local_success_rate = 0.7              # 本地成功率 < 70% 就走云端
fallback_enabled = true                   # 先试本地，失败转云端
local_task_types = ["general"]            # 只让简单对话走本地
max_local_context_tokens = 2000           # 大上下文走云端
```

### 工作原理

1. **新会话** — 根据任务类型、上下文大小、历史成功率决定走本地还是云端
2. **会话延续** — 同一会话始终使用同一模型（Session 级路由）
3. **本地失败** — 自动升级到云端，该会话后续全部走云端
4. **数据收集** — 云端响应自动保存为蒸馏训练数据

### 运行时调整

随着蒸馏模型改善，逐步放宽限制，无需重启：

```bash
# 蒸馏后 → 放开 tool_call
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -d '{"local_task_types": ["general", "tool_call"], "max_local_context_tokens": 4000}'
```

## 蒸馏流水线

### 环境准备

```bash
cd distill
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

基础模型（**Qwen2.5-1.5B-Instruct**）在首次训练时自动从 [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 下载，缓存在 `~/.cache/huggingface/hub/`（约 3 GB），无需手动下载。

### 完整流程

```bash
# 1. 正常使用 Louter，云端响应自动收集到 SQLite

# 2. 积累 1000+ 样本后，一键蒸馏
cd distill && source venv/bin/activate
./run_distill.sh   # 导出 → 压缩 → 训练 → 合并 → 部署到 Ollama

# 3. 或分步执行：
./run_distill.sh --export-only   # 导出 + 压缩训练数据
./run_distill.sh --train-only    # 训练 LoRA 适配器
./run_distill.sh --deploy-only   # 合并 + 部署到 Ollama
```

### 输出文件

| 路径 | 内容 |
|------|------|
| `distill/output/` | LoRA 适配器权重（`adapter_model.safetensors`） |
| `distill/merged_model/` | 合并后的完整模型（safetensors + tokenizer） |
| `distill/merged_model/Modelfile` | Ollama 导入文件（自动生成） |

### 转换为 GGUF 并部署到 Ollama

Ollama 对 Qwen 模型需要 **GGUF 格式**（仅 Llama/Mistral/Gemma/Phi3 支持直接导入 safetensors）。如果本地有 [llama.cpp](https://github.com/ggerganov/llama.cpp)，流水线会自动转换：

```bash
# 一次性准备 llama.cpp（用于 GGUF 转换）
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp
pip install -r ../llama.cpp/requirements.txt

# 安装 Ollama（如未安装）
curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama 服务
ollama serve

# 运行完整流水线（含自动 GGUF 转换 + 部署）
cd distill && source venv/bin/activate
./run_distill.sh

# 验证
ollama run louter-distilled "你好"
```

### 部署到 LlamaEdge（可选）

LlamaEdge 同样使用上述步骤生成的 GGUF 文件（详见 [LlamaEdge 文档](https://llamaedge.com/docs/user-guide/get-started-with-llamaedge)）：

```bash
wasmedge --dir .:. \
    --nn-preload default:GGML:AUTO:distill/merged_model/model-q4_k_m.gguf \
    llama-api-server.wasm \
    --prompt-template chatml --ctx-size 4096 \
    --model-name louter-distilled --socket-addr 0.0.0.0:8080
```

### 配置 Louter 使用蒸馏模型

```toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "louter-distilled"         # ← 蒸馏模型名称
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"
min_local_success_rate = 0.7
fallback_enabled = true
local_task_types = ["general"]           # 先保守，后续逐步放开
```

| 工具 | 用途 |
|------|------|
| `distill/export.py` | 从 SQLite 导出训练数据（OpenAI / ShareGPT 格式） |
| `distill/compress.py` | 压缩训练数据：去重 system prompt、截断工具结果 |
| `distill/train.py` | LoRA 微调（基于 Qwen2.5-1.5B，支持 CUDA / MPS / CPU） |
| `distill/run_distill.sh` | 端到端：导出 → 压缩 → 训练 → 合并 → 部署 Ollama |

### 数据飞轮

```
正常使用 Louter → 云端响应自动收集 → 压缩 + 微调 → 部署到 Ollama / LlamaEdge
    ↑                                                    │
    └──── 本地模型更强 → 更多请求走本地 → 成本更低 ←───────┘
```

## 核心特性

- **一个端点接入所有供应商** — OpenAI、Anthropic、Azure、DeepSeek、Ollama 及任意 OpenAI 兼容 API
- **混合推理** — 本地优先 + 云端回退，Session 级路由，失败自动升级
- **蒸馏飞轮** — 自动收集、压缩、训练、部署，本地模型持续进化
- **工具调用标准化** — 解析 Hermes/Qwen/ReAct/JSON 格式，统一转换为 OpenAI 格式
- **隐式反馈** — 检测 Agent 重试行为，自动标注训练样本的成功/失败
- **运行时可调** — 通过 API 动态调整路由阈值，无需重启
- **智能路由** — 按模型名前缀自动匹配，或用 `model: "auto"` 按内容分类路由
- **内置 Web UI** — 管理供应商、Key、路由规则、蒸馏状态和用量统计

## 许可证

MIT

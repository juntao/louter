# End-to-End Testing Plan: SFT + RL Pipeline

A step-by-step plan for testing Louter's distillation and RL pipelines using external conversation logs. Designed to be followed by any coding agent or human operator.

## Prerequisites

- Python 3.10+
- ~10 GB disk space (model weights + training artifacts)
- GPU recommended (CUDA or Apple Silicon MPS), CPU works but is slow
- Internet access (to download the base model on first run)

## Overview

```
Input:  folder of JSONL conversation logs
                ↓
        ingest_conversations.py  (new script)
                ↓
        louter.db  (SQLite with training_samples table)
               ╱            ╲
          SFT path          RL path
         export.py       export_episodes.py
         compress.py     generate_rollouts.py
         train.py        reward_rollouts.py
              ↓           train_grpo.py
        merged model          ↓
              ↓          merged model
        serve locally    serve locally
```

---

## Step 0: Setup Environment

```bash
cd /path/to/louter

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (SFT + RL)
pip install -r distill/requirements.txt
pip install -r distill/rl/requirements.txt
```

## Step 1: Prepare Conversation Logs

Place JSONL conversation log files in a single folder. Each `.jsonl` file contains one JSON object per line.

**Supported input formats** (the ingestion script auto-detects):

### Format A: OpenAI chat format (most common)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

### Format B: OpenClaw session format

```json
{
  "session_id": "abc123",
  "turn": 1,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "response_text": "The assistant's response...",
  "tool_calls": null,
  "next_state": {"score": 1}
}
```

### Format C: ShareGPT format

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What is 2+2?"},
    {"from": "gpt", "value": "2+2 equals 4."}
  ]
}
```

### Format D: Simple prompt/response

```json
{
  "prompt": "What is 2+2?",
  "response": "2+2 equals 4.",
  "system": "You are a helpful assistant."
}
```

**Example folder structure:**

```
test_data/
├── conversations_01.jsonl
├── conversations_02.jsonl
└── conversations_03.jsonl
```

## Step 2: Create the Ingestion Script

Create `distill/ingest_conversations.py` — converts external JSONL logs into Louter's `training_samples` SQLite table.

```python
#!/usr/bin/env python3
"""
Ingest external conversation JSONL files into Louter's training_samples table.

Supports: OpenAI chat format, OpenClaw session format, ShareGPT format,
          and simple prompt/response format.

Usage:
    python ingest_conversations.py --input-dir ./test_data --db ./louter.db
    python ingest_conversations.py --input-dir ./test_data --db ./louter.db --source cloud
    python ingest_conversations.py --input-dir ./test_data --db ./louter.db --dry-run
"""

import argparse
import glob
import json
import sqlite3
import sys
import uuid
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS training_samples (
    id TEXT PRIMARY KEY,
    request_messages TEXT NOT NULL,
    request_tools TEXT,
    response_content TEXT NOT NULL,
    request_model TEXT NOT NULL DEFAULT '',
    actual_model TEXT NOT NULL DEFAULT '',
    provider_type TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT 'general',
    has_tool_calls INTEGER NOT NULL DEFAULT 0,
    is_successful INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'cloud',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    is_exported INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ts_task_type ON training_samples(task_type);
CREATE INDEX IF NOT EXISTS idx_ts_created_at ON training_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_ts_is_successful ON training_samples(is_successful);
CREATE INDEX IF NOT EXISTS idx_ts_source ON training_samples(source);
CREATE INDEX IF NOT EXISTS idx_ts_is_exported ON training_samples(is_exported);
"""


def detect_and_convert(record: dict) -> dict | None:
    """Auto-detect format and convert to (messages_without_response, response, tools).

    Returns {"messages": [...], "response": {...}, "tools": [...] | None}
    or None if unparseable.
    """
    # Format A: OpenAI chat {"messages": [...]}
    if "messages" in record and isinstance(record["messages"], list):
        messages = record["messages"]
        # If last message is assistant, split it off as the response
        if messages and messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]
            response = messages[-1]
        # OpenClaw variant: messages + response_text
        elif "response_text" in record:
            prompt_messages = messages
            response = {"role": "assistant", "content": record["response_text"]}
        else:
            # All messages but no clear response — skip
            return None

        tools = record.get("tools")
        has_tool_calls = bool(response.get("tool_calls"))

        # OpenClaw: use next_state.score for success signal
        is_successful = 1
        if "next_state" in record and isinstance(record.get("next_state"), dict):
            score = record["next_state"].get("score", 1)
            is_successful = 0 if score < 0 else 1

        return {
            "messages": prompt_messages,
            "response": response,
            "tools": tools,
            "has_tool_calls": int(has_tool_calls),
            "is_successful": is_successful,
        }

    # Format C: ShareGPT {"conversations": [...]}
    if "conversations" in record and isinstance(record["conversations"], list):
        role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
        messages = []
        for turn in record["conversations"]:
            role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
            messages.append({"role": role, "content": turn.get("value", "")})

        if messages and messages[-1]["role"] == "assistant":
            return {
                "messages": messages[:-1],
                "response": messages[-1],
                "tools": None,
                "has_tool_calls": 0,
                "is_successful": 1,
            }
        return None

    # Format D: Simple {"prompt": "...", "response": "..."}
    if "prompt" in record and "response" in record:
        messages = []
        if record.get("system"):
            messages.append({"role": "system", "content": record["system"]})
        messages.append({"role": "user", "content": record["prompt"]})
        return {
            "messages": messages,
            "response": {"role": "assistant", "content": record["response"]},
            "tools": None,
            "has_tool_calls": 0,
            "is_successful": 1,
        }

    return None


def classify_task_type(messages: list, response: dict) -> str:
    """Simple heuristic task classification."""
    all_text = " ".join(
        m.get("content", "") or "" for m in messages + [response]
        if isinstance(m.get("content"), str)
    ).lower()

    if response.get("tool_calls"):
        return "tool_call"
    for kw in ["translate", "翻译", "translation"]:
        if kw in all_text[:500]:
            return "translation"
    for kw in ["def ", "function ", "class ", "```", "import ", "console.log"]:
        if kw in (response.get("content") or ""):
            return "code"
    return "general"


def ingest(input_dir: str, db_path: str, source: str = "cloud",
           dry_run: bool = False) -> int:
    """Ingest all JSONL files from input_dir into the database."""
    files = sorted(glob.glob(str(Path(input_dir) / "*.jsonl")))
    if not files:
        print(f"No .jsonl files found in {input_dir}", file=sys.stderr)
        return 0

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    total = 0
    skipped = 0

    for filepath in files:
        filename = Path(filepath).name
        file_count = 0
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  SKIP {filename}:{line_num} — invalid JSON", file=sys.stderr)
                    skipped += 1
                    continue

                converted = detect_and_convert(record)
                if converted is None:
                    skipped += 1
                    continue

                task_type = classify_task_type(
                    converted["messages"], converted["response"]
                )

                if not dry_run:
                    conn.execute(
                        "INSERT OR IGNORE INTO training_samples "
                        "(id, request_messages, request_tools, response_content, "
                        " request_model, actual_model, provider_type, task_type, "
                        " has_tool_calls, is_successful, source) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            json.dumps(converted["messages"], ensure_ascii=False),
                            json.dumps(converted["tools"]) if converted["tools"] else None,
                            json.dumps(converted["response"], ensure_ascii=False),
                            "",  # request_model
                            "",  # actual_model
                            "",  # provider_type
                            task_type,
                            converted["has_tool_calls"],
                            converted["is_successful"],
                            source,
                        ),
                    )

                file_count += 1
                total += 1

        print(f"  {filename}: {file_count} samples ingested", file=sys.stderr)

    if not dry_run:
        conn.commit()
    conn.close()

    print(f"\nTotal: {total} samples ingested, {skipped} skipped", file=sys.stderr)
    if dry_run:
        print("(dry run — nothing written to database)", file=sys.stderr)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Ingest external conversation logs into Louter's training_samples table"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing .jsonl conversation log files",
    )
    parser.add_argument(
        "--db", default="louter.db",
        help="Path to louter.db (will be created if missing)",
    )
    parser.add_argument(
        "--source", default="cloud", choices=["cloud", "local"],
        help="Label all ingested samples as this source (default: cloud)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate without writing to database",
    )

    args = parser.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting from: {args.input_dir}", file=sys.stderr)
    print(f"Database: {args.db}", file=sys.stderr)
    print(f"Source label: {args.source}", file=sys.stderr)
    print(file=sys.stderr)

    ingest(args.input_dir, args.db, args.source, args.dry_run)


if __name__ == "__main__":
    main()
```

## Step 3: Download Qwen3 1.5B (if not cached)

The model downloads automatically on first use via HuggingFace. To pre-download:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen3-1.5B'
print(f'Downloading {model_name}...')
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print('Done. Model cached at ~/.cache/huggingface/')
"
```

Verify download:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.5B', trust_remote_code=True)
print(f'Vocab size: {t.vocab_size}')
print(f'Model loaded successfully')
"
```

> **Note**: The existing pipeline defaults to `Qwen/Qwen2.5-1.5B-Instruct`. To use Qwen3 instead, set `BASE_MODEL=Qwen/Qwen3-1.5B` in all commands below. The scripts accept any HuggingFace model name.

## Step 4: Ingest Conversation Logs

```bash
# Ingest external JSONL files into the SQLite database
python3 distill/ingest_conversations.py \
    --input-dir ./test_data \
    --db ./louter.db

# Verify
python3 distill/export.py --db ./louter.db --stats
```

Expected output:

```
Ingesting from: ./test_data
Database: ./louter.db

  conversations_01.jsonl: 150 samples ingested
  conversations_02.jsonl: 200 samples ingested

Total: 350 samples ingested, 5 skipped

=== Training Sample Statistics ===

Task Type        Total  Success    Tools Unexport       Tokens
-----------------------------------------------------------------
general            280      280        0      280            0
code                50       50        0       50            0
tool_call           20       20       20       20            0
-----------------------------------------------------------------
TOTAL              350
```

## Step 5: Choose a Pipeline Mode

### Option A: SFT Only

Supervised fine-tuning on successful conversations. Best for initial training when you have good reference data.

```bash
BASE_MODEL="Qwen/Qwen3-1.5B" \
LOUTER_DB="./louter.db" \
./distill/run_distill.sh
```

This runs: export → compress → train (LoRA) → merge → deploy.

Output: `distill/merged_model/` (full merged model with Modelfile)

**Verify SFT output:**

```bash
# Check that the merged model exists
ls -la distill/merged_model/
# Should contain: config.json, model*.safetensors, tokenizer*, Modelfile

# Quick sanity test
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('./distill/merged_model', trust_remote_code=True)
t = AutoTokenizer.from_pretrained('./distill/merged_model', trust_remote_code=True)
inputs = t.apply_chat_template(
    [{'role': 'user', 'content': 'Hello!'}],
    tokenize=True, add_generation_prompt=True, return_tensors='pt'
)
out = m.generate(inputs, max_new_tokens=50, do_sample=False)
print(t.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
"
```

### Option B: RL Only

Reinforcement learning using GRPO. Requires data already in the database. Works best when you have a mix of successful and failed samples.

```bash
BASE_MODEL="Qwen/Qwen3-1.5B" \
LOUTER_DB="./louter.db" \
JUDGE_PROVIDER="ollama" \
JUDGE_MODEL="qwen2.5:7b" \
MIN_EPISODES=10 \
./distill/rl/run_rl.sh
```

> Set `MIN_EPISODES=10` (or lower) for testing with small datasets. Default is 100.
>
> Set `JUDGE_PROVIDER=ollama` to avoid needing Anthropic/OpenAI API keys. Requires `ollama pull qwen2.5:7b` first. Alternatively, use `JUDGE_PROVIDER=anthropic` with a valid `ANTHROPIC_API_KEY`.

Output: `distill/rl/rl_merged/` (full merged model with Modelfile)

**Verify RL output:**

```bash
ls -la distill/rl/rl_merged/
# Should contain: config.json, model*.safetensors, tokenizer*, Modelfile

# Check training artifacts
cat distill/rl/output/rl_adapter/training_meta.json | python3 -m json.tool
# Should show hyperparameters, steps completed, final KL divergence
```

### Option C: SFT + RL (Recommended)

Run SFT first to get a baseline, then RL to refine it. The RL pipeline automatically uses the SFT adapter as its starting point.

```bash
export BASE_MODEL="Qwen/Qwen3-1.5B"
export LOUTER_DB="./louter.db"

# Step C.1: SFT
./distill/run_distill.sh

# Step C.2: RL (starts from SFT adapter at distill/output)
ADAPTER_PATH="./distill/output" \
JUDGE_PROVIDER="ollama" \
JUDGE_MODEL="qwen2.5:7b" \
MIN_EPISODES=10 \
./distill/rl/run_rl.sh
```

Output: `distill/rl/rl_merged/` (RL-refined model, built on top of SFT)

## Step 6: Serve the Merged Model

Pick one serving method. All expose an OpenAI-compatible API at `http://localhost:8000/v1`.

### Option 1: HuggingFace Transformers (works everywhere)

```bash
# Serve the SFT model
python3 distill/rl/serve_hf.py --model ./distill/merged_model --port 8000

# Or serve the RL model
python3 distill/rl/serve_hf.py --model ./distill/rl/rl_merged --port 8000

# Or serve base model + LoRA adapter (no merge step needed)
python3 distill/rl/serve_hf.py \
    --model Qwen/Qwen3-1.5B \
    --adapter ./distill/output \
    --port 8000
```

### Option 2: Ollama (if installed)

```bash
# Import merged model
ollama create louter-test -f ./distill/rl/rl_merged/Modelfile

# Ollama serves on port 11434 by default
ollama run louter-test "Hello, how are you?"
```

### Option 3: vLLM (GPU only, highest throughput)

```bash
MODEL_PATH=./distill/rl/rl_merged PORT=8000 ./distill/rl/serve_vllm.sh
```

## Step 7: Verify the Served Model

Test that the served model responds correctly:

```bash
# Health check (serve_hf.py only)
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "louter-rl",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "louter-rl",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding."}
    ],
    "stream": true
  }'
```

Expected: a valid JSON response with `choices[0].message.content` containing the model's answer.

---

## Quick Reference: Complete Commands

### Minimal test (SFT only, ~30 min on GPU)

```bash
source .venv/bin/activate
python3 distill/ingest_conversations.py --input-dir ./test_data --db ./louter.db
BASE_MODEL="Qwen/Qwen3-1.5B" LOUTER_DB="./louter.db" ./distill/run_distill.sh
python3 distill/rl/serve_hf.py --model ./distill/merged_model --port 8000
```

### Full test (SFT + RL, ~1-2 hours on GPU)

```bash
source .venv/bin/activate
python3 distill/ingest_conversations.py --input-dir ./test_data --db ./louter.db
export BASE_MODEL="Qwen/Qwen3-1.5B" LOUTER_DB="./louter.db"
./distill/run_distill.sh
ADAPTER_PATH="./distill/output" MIN_EPISODES=10 \
  JUDGE_PROVIDER="ollama" JUDGE_MODEL="qwen2.5:7b" \
  ./distill/rl/run_rl.sh
python3 distill/rl/serve_hf.py --model ./distill/rl/rl_merged --port 8000
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No .jsonl files found` | Check `--input-dir` points to a directory with `.jsonl` files |
| `Not enough episodes (N < 100)` | Set `MIN_EPISODES=10` for testing, or add more data |
| `CUDA out of memory` | Reduce batch size: edit `run_distill.sh` to use `--batch-size 2 --gradient-accumulation 8` |
| `MPS not available` | Update to macOS 12.3+ and PyTorch 2.0+ |
| `Model not found at ./rl_merged` | Run training first — the merged model is created by the train+merge steps |
| `score_with_judge fails` | Set `JUDGE_PROVIDER=ollama` and `ollama pull qwen2.5:7b`, or set `ANTHROPIC_API_KEY` |
| `vLLM import error` | vLLM requires CUDA GPU — use `serve_hf.py` on Mac/CPU instead |

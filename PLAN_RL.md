# Reinforcement Learning Plan for Louter

Inspired by [OpenClaw RL](https://github.com/Gen-Verse/OpenClaw-RL), this plan extends Louter's existing LoRA distillation pipeline with reinforcement learning to continuously improve the local model from natural conversation feedback.

## Background

### What we have today

Louter already collects training data from the hybrid routing pipeline:

1. **Data collection**: Every chat completion is saved to `training_samples` in SQLite (messages, tools, response, task type, success status)
2. **Implicit feedback**: The `FeedbackTracker` detects agent retries within 60 seconds and marks previous samples as failed — free success/failure labels with zero manual annotation
3. **LoRA distillation**: `distill/run_distill.sh` runs Export → Compress → Train (SFT) → Merge → Deploy to Ollama
4. **Routing history**: Per-task-type success rates tracked in `routing_history` table

This gives us supervised fine-tuning (SFT) on successful cloud responses. But SFT only imitates — it cannot learn *from mistakes* or optimize for specific reward signals.

### What OpenClaw RL adds

OpenClaw RL introduces three key ideas we can adapt:

| OpenClaw Concept | Louter Adaptation |
|---|---|
| **GRPO (Group Relative Policy Optimization)** | Train the local model with RL using grouped completions and relative advantages — no separate critic network needed |
| **Next-state reward** | Use Louter's existing retry detection + cloud model as judge to score local responses without manual labels |
| **On-Policy Distillation (OPD)** | When local fails and cloud succeeds, use the cloud response as a teacher signal at the token level |
| **Async training loop** | Model keeps serving while RL training runs in background; hot-swap weights on completion |

### Why RL on top of SFT

SFT teaches the model to imitate good responses. RL teaches it to *avoid bad responses* and *maximize reward*. Concretely:

- SFT cannot use failed samples (it only trains on good outputs)
- RL uses both positive and negative signal — a retry-marked sample becomes a negative reward
- GRPO compares multiple completions for the same prompt, learning which response *style* works better
- OPD provides token-level gradient signal from cloud→local, stronger than sequence-level SFT

## Architecture Overview

```
                         Louter Proxy (:6188)
                              │
                  ┌───────────┼───────────┐
                  │           │           │
              Serve local   Collect    Implicit
              model via     samples    feedback
              OpenAI API               (retry detection)
                  │           │           │
                  │           ▼           ▼
                  │     ┌─────────────────────┐
                  │     │   SQLite Database    │
                  │     │  training_samples    │
                  │     │  routing_history     │
                  │     │  rl_episodes (new)   │
                  │     └─────────────────────┘
                  │               │
                  │               ▼
                  │     ┌─────────────────────┐
                  │     │  RL Pipeline (new)   │
                  │     │                     │
                  │     │  1. Export episodes  │
                  │     │  2. Score (reward)   │
                  │     │  3. Generate rollouts│
                  │     │  4. GRPO training    │
                  │     │  5. Merge + deploy   │
                  │     └─────────────────────┘
                  │               │
                  ▼               ▼
              ┌───────────────────────┐
              │  Updated local model  │
              │  (vLLM / Ollama /     │
              │   HF Transformers)    │
              └───────────────────────┘
```

## Implementation Plan

### Phase 1: Reward Infrastructure

**Goal**: Enrich the existing data collection with reward signals usable for RL.

#### 1.1 New database table: `rl_episodes`

```sql
CREATE TABLE IF NOT EXISTS rl_episodes (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,               -- FK to training_samples.id
    prompt_messages TEXT NOT NULL,          -- JSON: input messages (without response)
    completion TEXT NOT NULL,               -- JSON: model response
    source TEXT NOT NULL,                   -- 'local' or 'cloud'
    reward REAL,                            -- scalar reward [-1.0, 1.0]
    reward_source TEXT,                     -- 'implicit', 'judge', 'environment', 'manual'
    reward_details TEXT,                    -- JSON: breakdown of reward components
    is_used_for_training INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Script**: `distill/rl/export_episodes.py`
- Read `training_samples` and convert to RL episodes
- Assign rewards from implicit feedback:
  - `is_successful=1` and no retry → reward = +1.0
  - `is_successful=0` or retry detected → reward = -1.0
  - Cloud fallback after local failure → local gets -0.5, cloud gets +1.0

#### 1.2 Judge-based reward scoring

**Script**: `distill/rl/score_with_judge.py`

Use a cloud model (or a local judge model) to score local model responses on a rubric:

```python
JUDGE_PROMPT = """Rate this assistant response on a scale of 1-5:
- Correctness: Is the response factually and logically correct?
- Helpfulness: Does it address the user's request?
- Completeness: Is anything important missing?

User request: {prompt}
Assistant response: {completion}

Return JSON: {"score": <1-5>, "reason": "<brief explanation>"}"""
```

- Batch process unscored local-model episodes
- Normalize 1-5 scores to [-1.0, 1.0] range
- Store as `reward_source='judge'`
- Support multiple judge backends: Anthropic API, OpenAI API, local model via Ollama
- Rate-limit and cache to control cost

#### 1.3 Environment-based rewards (for tool-calling tasks)

For `task_type='tool_call'`, we can extract richer signals:

- **Tool call validity**: Does the function name exist in the provided tools? Are arguments valid JSON matching the schema? → binary reward
- **Execution feedback**: If tool results are in subsequent messages, check for error patterns → negative reward for errors
- **Completion**: Did the conversation reach a final answer or keep looping? → reward for task completion

**Script**: `distill/rl/score_tool_calls.py`

### Phase 2: GRPO Training Pipeline

**Goal**: Implement Group Relative Policy Optimization adapted for small models on consumer hardware.

#### 2.1 Rollout generation

**Script**: `distill/rl/generate_rollouts.py`

For each prompt in the episode buffer, generate G completions (default G=4) from the current local model:

```python
def generate_rollouts(prompts, model, tokenizer, G=4, temperature=0.7, max_tokens=2048):
    """Generate G completions per prompt for GRPO."""
    rollouts = []
    for prompt in prompts:
        completions = []
        for _ in range(G):
            output = model.generate(prompt, temperature=temperature, max_new_tokens=max_tokens)
            completions.append(output)
        rollouts.append({"prompt": prompt, "completions": completions})
    return rollouts
```

Support multiple inference backends:
- **vLLM** (preferred for GPU): High-throughput batched generation
- **Ollama**: REST API generation (simpler setup)
- **HuggingFace Transformers**: Direct model loading (fallback)

#### 2.2 Reward assignment for rollouts

**Script**: `distill/rl/reward_rollouts.py`

Score each generated completion:

1. **Judge scoring**: Send each completion to the judge model
2. **Reference comparison**: If we have the cloud model's response for this prompt, compute similarity (ROUGE-L, BERTScore) as a soft reward
3. **Format/tool rewards**: Structural checks (valid JSON, correct tool call format)

Compute group-relative advantages (the GRPO core):

```python
def compute_grpo_advantages(rewards_per_group):
    """GRPO: advantages are relative within each group."""
    advantages = []
    for group_rewards in rewards_per_group:
        mean = sum(group_rewards) / len(group_rewards)
        std = max(statistics.stdev(group_rewards), 1e-8)
        group_advantages = [(r - mean) / std for r in group_rewards]
        advantages.append(group_advantages)
    return advantages
```

#### 2.3 GRPO training step

**Script**: `distill/rl/train_grpo.py`

Core training loop implementing GRPO with LoRA:

```python
# GRPO loss: clipped surrogate objective with KL penalty
# L = -E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ] + β * KL(π_θ || π_ref)
#
# Where:
#   r(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio)
#   A = group-relative advantage
#   ε = 0.2 (clip range)
#   β = 0.02 (KL penalty weight)
```

Key parameters (adapted for small models / consumer hardware):

| Parameter | Value | Rationale |
|---|---|---|
| Group size (G) | 4 | Balance between signal quality and compute |
| Clip epsilon | 0.2 | Standard PPO/GRPO clip range |
| KL penalty (beta) | 0.02 | Prevent divergence from base model |
| Learning rate | 1e-5 | Lower than SFT to avoid catastrophic forgetting |
| LoRA rank | 8 | Match existing SFT configuration |
| Batch size | 2 | Fit in consumer GPU memory (each sample has G completions) |
| Gradient accumulation | 8 | Effective batch = 16 |
| Max steps per round | 200 | Short RL rounds, frequent evaluation |

**LoRA reuse**: Start from the existing SFT adapter weights (if available) rather than from scratch. The RL phase refines what SFT learned.

#### 2.4 On-Policy Distillation (OPD) — optional hybrid mode

**Script**: `distill/rl/train_opd.py`

When we have both a failed local response and a successful cloud response for the same prompt, use OPD for stronger signal:

1. Compute token-level log-probabilities of the cloud response under the local model
2. Use the log-prob gap as a per-token advantage (tokens where local assigns low probability to the correct cloud token get stronger gradient)
3. Combine with GRPO loss: `L_total = W_RL * L_grpo + W_OPD * L_opd`

This is the most powerful training signal but requires prompt pairs where both local and cloud responded.

### Phase 3: Training Orchestration

**Goal**: End-to-end pipeline script, similar to the existing `run_distill.sh`.

#### 3.1 Pipeline orchestrator

**Script**: `distill/rl/run_rl.sh`

```
Usage:
  ./run_rl.sh                          # Full RL pipeline
  ./run_rl.sh --score-only             # Just score episodes with judge
  ./run_rl.sh --train-only             # Just run GRPO training (episodes must exist)
  ./run_rl.sh --rollout-only           # Just generate rollouts
  ./run_rl.sh --deploy-only            # Just merge and deploy

Environment:
  LOUTER_DB          Path to louter.db (default: ../louter.db)
  BASE_MODEL         Base model for RL (default: Qwen/Qwen2.5-1.5B-Instruct)
  ADAPTER_PATH       Starting adapter (default: ../distill/output — reuse SFT adapter)
  JUDGE_PROVIDER     Judge model provider: anthropic|openai|ollama (default: anthropic)
  JUDGE_MODEL        Judge model name (default: claude-sonnet-4-20250514)
  INFERENCE_BACKEND  For rollout generation: vllm|ollama|transformers (default: transformers)
  OLLAMA_MODEL       Name for deployed model (default: louter-rl)
```

Pipeline steps:

```
1. Export episodes    →  SQLite → rl_episodes.jsonl
2. Score episodes     →  Judge model assigns rewards to unscored episodes
3. Generate rollouts  →  G completions per prompt from current model
4. Score rollouts     →  Judge + structural rewards for each completion
5. GRPO training      →  LoRA update with clipped surrogate + KL penalty
6. Evaluate           →  Run on held-out prompts, compare to SFT baseline
7. Merge + deploy     →  Merge LoRA into base model, deploy via chosen backend
```

#### 3.2 Evaluation checkpoint

**Script**: `distill/rl/evaluate.py`

Before deploying, validate the RL model is actually better:

- Hold out 10% of episodes as eval set
- Generate completions from both SFT model and RL model
- Score both with the judge
- Only deploy if RL model wins on average reward
- Log comparison to `distill/rl/output/eval_report.json`

### Phase 4: Flexible Model Serving

**Goal**: Support serving the RL-trained model through multiple backends, not just Ollama.

#### 4.1 vLLM serving (recommended for GPU)

**Script**: `distill/rl/serve_vllm.sh`

```bash
# Serve merged model with vLLM (OpenAI-compatible API)
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --dtype auto \
    --gpu-memory-utilization 0.9
```

vLLM provides:
- 3-5x higher throughput than HuggingFace for batched inference
- OpenAI-compatible API (drop-in replacement for Ollama in Louter config)
- Continuous batching, PagedAttention
- LoRA adapter hot-loading (serve base model + swap adapters without restart)

#### 4.2 Ollama serving (existing)

Already supported. After `ollama create louter-rl -f merged_model/Modelfile`, update `louter.toml`:

```toml
[hybrid]
local_model = "louter-rl"
```

#### 4.3 HuggingFace Transformers serving

**Script**: `distill/rl/serve_hf.py`

Lightweight OpenAI-compatible server using FastAPI + Transformers for environments without vLLM or Ollama:

```python
# Minimal OpenAI-compatible server
# Supports: /v1/chat/completions (streaming + non-streaming)
# Usage: python serve_hf.py --model ./merged_model --port 8000
```

Useful for:
- Apple Silicon (MPS) where vLLM is not available
- CPU-only environments
- Quick testing before deploying to Ollama

#### 4.4 Louter provider configuration

Add vLLM and generic OpenAI-compatible endpoint support to `louter.toml`:

```toml
[hybrid]
local_provider = "vllm"        # or "ollama" or "openai-compatible"
local_endpoint = "http://localhost:8000/v1"
local_model = "louter-rl"
```

This requires a small Rust change to the provider layer to support configurable OpenAI-compatible endpoints for local inference (the existing `openai.rs` provider can be reused by pointing it at a local URL).

### Phase 5: Continuous RL Loop

**Goal**: Automate the RL improvement cycle so the model keeps getting better over time.

#### 5.1 Scheduled RL rounds

Add a cron-like mechanism or a manual trigger:

```bash
# Run RL round when enough new episodes have accumulated
# Recommended: after every 500+ new episodes
./run_rl.sh --min-episodes 500
```

The script checks `rl_episodes WHERE is_used_for_training = 0` and skips if below threshold.

#### 5.2 Progressive task expansion

As RL improves the model, gradually route more task types locally:

```
Round 1: general only          (lowest complexity)
Round 2: general + translation (pattern-heavy tasks)
Round 3: + code                (structured output)
Round 4: + tool_call           (most complex)
```

Track success rates per task type after each RL round. Only expand when `min_local_success_rate` is met for current types.

#### 5.3 Safety rails

- **KL divergence cap**: If KL between RL model and reference exceeds threshold (0.1), stop training early to prevent mode collapse
- **Reward hacking detection**: Monitor if judge scores go up but actual retry rates don't improve — indicates reward model exploitation
- **Rollback mechanism**: Keep the previous merged model as `louter-rl-prev` so Louter can fall back if the new model regresses

## File Structure

```
distill/
├── export.py                    # (existing) Export training samples
├── compress.py                  # (existing) Compress training data
├── train.py                     # (existing) SFT LoRA training
├── run_distill.sh               # (existing) SFT pipeline
├── requirements.txt             # (existing) SFT dependencies
└── rl/
    ├── requirements.txt         # RL-specific dependencies
    ├── run_rl.sh                # Main RL pipeline orchestrator
    ├── export_episodes.py       # Convert training_samples → rl_episodes
    ├── score_with_judge.py      # Judge-based reward scoring
    ├── score_tool_calls.py      # Structural rewards for tool-call tasks
    ├── generate_rollouts.py     # Generate G completions per prompt
    ├── reward_rollouts.py       # Score rollouts + compute GRPO advantages
    ├── train_grpo.py            # GRPO training with LoRA
    ├── train_opd.py             # On-Policy Distillation (optional)
    ├── evaluate.py              # Compare RL model vs SFT baseline
    ├── serve_vllm.sh            # vLLM serving script
    ├── serve_hf.py              # HuggingFace Transformers server
    └── output/                  # Training outputs, checkpoints, eval reports
```

## Dependencies

**`distill/rl/requirements.txt`**:

```
# Core RL training
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
datasets>=2.18.0
accelerate>=0.28.0
trl>=0.12.0                     # GRPOTrainer (trl 0.12+)
bitsandbytes>=0.43.0

# Rollout generation (choose one)
vllm>=0.6.0                     # GPU inference (optional)
# ollama via REST API — no pip dependency

# Reward scoring
rouge-score>=0.1.2              # ROUGE-L for reference comparison
anthropic>=0.40.0               # Judge model API (optional)
openai>=1.50.0                  # Alternative judge (optional)

# Serving
fastapi>=0.115.0                # HF Transformers server
uvicorn>=0.32.0

# Evaluation
numpy>=1.26.0
```

## Hardware Requirements

| Setup | Hardware | Notes |
|---|---|---|
| **Minimal (Apple Silicon)** | M1/M2/M3 with 16GB+ | MPS for training, HF Transformers for serving. G=2, batch=1. Slow but works. |
| **Recommended (single GPU)** | 1x RTX 3090/4090 (24GB) | vLLM for rollouts, GRPO training with LoRA. G=4, batch=2. ~1 hour per RL round (500 episodes). |
| **Fast (multi-GPU)** | 2-4x GPUs | vLLM on GPU 0 for rollouts, training on remaining GPUs. G=8, batch=4. |
| **Cloud** | Any cloud GPU (A100, H100) | Same scripts, just faster. |

For the 1.5B base model, a single 24GB GPU comfortably handles the full pipeline. The 3B model fits with gradient checkpointing.

## Implementation Order

1. **Phase 1.1**: `rl_episodes` schema + `export_episodes.py` — get reward data flowing
2. **Phase 1.2**: `score_with_judge.py` — enrich rewards beyond implicit feedback
3. **Phase 2.1-2.2**: `generate_rollouts.py` + `reward_rollouts.py` — GRPO data pipeline
4. **Phase 2.3**: `train_grpo.py` — core RL training loop
5. **Phase 3.1**: `run_rl.sh` — end-to-end orchestration
6. **Phase 3.2**: `evaluate.py` — safety gate before deployment
7. **Phase 4**: Serving scripts (vLLM, HF) + Louter provider config
8. **Phase 2.4**: `train_opd.py` — optional advanced mode
9. **Phase 5**: Continuous loop automation + progressive expansion

Phases 1-3 are the core value. Phase 4 broadens deployment options. Phase 5 makes it self-sustaining.

## Key Differences from OpenClaw RL

| Aspect | OpenClaw RL | Louter RL |
|---|---|---|
| **Scale** | 8+ GPUs, Megatron+SGLang | Single GPU / Apple Silicon, HF Transformers / vLLM |
| **Async loop** | 4 fully async components | Sequential pipeline with optional background rollouts |
| **Reward model** | Dedicated PRM with majority voting | Cloud model as judge + implicit retry feedback |
| **Data source** | Live conversation interception | Existing Louter proxy data collection |
| **Training framework** | Slime (Megatron+SGLang) | trl GRPOTrainer + LoRA (PEFT) |
| **Model size** | 4B-32B | 1.5B-8B (consumer hardware focus) |
| **Serving** | SGLang only | Ollama, vLLM, HF Transformers |

The core RL algorithm (GRPO) is the same. We adapt the infrastructure for single-machine, consumer-hardware deployment while reusing Louter's existing data pipeline.

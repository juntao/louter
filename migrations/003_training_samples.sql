-- Training samples for distillation pipeline
CREATE TABLE IF NOT EXISTS training_samples (
    id TEXT PRIMARY KEY,
    request_messages TEXT NOT NULL,           -- JSON: full messages array
    request_tools TEXT,                       -- JSON: tools array if present
    response_content TEXT NOT NULL,           -- JSON: full model response (choices)
    request_model TEXT NOT NULL,              -- original requested model name
    actual_model TEXT NOT NULL,               -- model that actually handled it
    provider_type TEXT NOT NULL DEFAULT '',   -- provider type (openai, anthropic, ollama, etc.)
    task_type TEXT NOT NULL DEFAULT 'general',-- classified task type
    has_tool_calls INTEGER NOT NULL DEFAULT 0,-- response contains tool_calls
    is_successful INTEGER NOT NULL DEFAULT 1, -- whether the request succeeded
    source TEXT NOT NULL DEFAULT 'cloud',     -- 'cloud' or 'local'
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    is_exported INTEGER NOT NULL DEFAULT 0,   -- whether already exported for training
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ts_task_type ON training_samples(task_type);
CREATE INDEX IF NOT EXISTS idx_ts_created_at ON training_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_ts_is_successful ON training_samples(is_successful);
CREATE INDEX IF NOT EXISTS idx_ts_source ON training_samples(source);
CREATE INDEX IF NOT EXISTS idx_ts_is_exported ON training_samples(is_exported);

-- Routing history for tracking local vs cloud routing decisions and outcomes
CREATE TABLE IF NOT EXISTS routing_history (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    routed_to TEXT NOT NULL,                  -- 'local' or 'cloud'
    was_successful INTEGER NOT NULL DEFAULT 1,
    was_fallback INTEGER NOT NULL DEFAULT 0,  -- was this a fallback after local failure
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_rh_task_type ON routing_history(task_type);
CREATE INDEX IF NOT EXISTS idx_rh_created_at ON routing_history(created_at);

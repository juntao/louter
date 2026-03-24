//! Runtime-adjustable routing parameters.
//!
//! These values override the static config from louter.toml and can be
//! changed via the admin API without restarting the server.
//! As the distilled model improves, these thresholds should be relaxed.

use tokio::sync::RwLock;

/// Dynamic overrides for hybrid routing parameters.
pub struct DynamicHybridConfig {
    inner: RwLock<Overrides>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Overrides {
    /// Max context tokens for local routing (0 = use static config)
    pub max_local_context_tokens: Option<u32>,
    /// Max latency for local routing (0 = use static config)
    pub max_local_latency_ms: Option<u32>,
    /// Override min success rate (None = use static config)
    pub min_local_success_rate: Option<f64>,
    /// Override local task types (None = use static config)
    pub local_task_types: Option<Vec<String>>,
}

impl Default for Overrides {
    fn default() -> Self {
        Self {
            max_local_context_tokens: None,
            max_local_latency_ms: None,
            min_local_success_rate: None,
            local_task_types: None,
        }
    }
}

impl DynamicHybridConfig {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Overrides::default()),
        }
    }

    pub async fn get(&self) -> Overrides {
        self.inner.read().await.clone()
    }

    pub async fn update(&self, patch: Overrides) {
        let mut current = self.inner.write().await;
        if patch.max_local_context_tokens.is_some() {
            current.max_local_context_tokens = patch.max_local_context_tokens;
        }
        if patch.max_local_latency_ms.is_some() {
            current.max_local_latency_ms = patch.max_local_latency_ms;
        }
        if patch.min_local_success_rate.is_some() {
            current.min_local_success_rate = patch.min_local_success_rate;
        }
        if patch.local_task_types.is_some() {
            current.local_task_types = patch.local_task_types;
        }
    }

    /// Get the effective max context tokens (dynamic override > static config).
    pub async fn effective_max_context_tokens(&self, static_val: u32) -> u32 {
        let overrides = self.inner.read().await;
        overrides.max_local_context_tokens.unwrap_or(static_val)
    }

    /// Get the effective max latency (dynamic override > static config).
    pub async fn effective_max_latency(&self, static_val: u32) -> u32 {
        let overrides = self.inner.read().await;
        overrides.max_local_latency_ms.unwrap_or(static_val)
    }

    /// Get effective min success rate.
    pub async fn effective_min_success_rate(&self, static_val: f64) -> f64 {
        let overrides = self.inner.read().await;
        overrides.min_local_success_rate.unwrap_or(static_val)
    }

    /// Get effective local task types.
    pub async fn effective_local_task_types(&self, static_val: &[String]) -> Vec<String> {
        let overrides = self.inner.read().await;
        overrides
            .local_task_types
            .clone()
            .unwrap_or_else(|| static_val.to_vec())
    }
}

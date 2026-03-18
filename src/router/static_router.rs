use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::db;
use crate::error::{AppError, AppResult};
use crate::providers::Provider;
use sqlx::SqlitePool;

/// ProviderRegistry holds all initialized Provider instances.
/// Wrapped in RwLock to allow runtime updates from admin API.
pub struct ProviderRegistry {
    providers: RwLock<HashMap<String, Arc<dyn Provider>>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register(&self, id: String, provider: Arc<dyn Provider>) {
        self.providers.write().await.insert(id, provider);
    }

    pub async fn remove(&self, id: &str) {
        self.providers.write().await.remove(id);
    }

    pub async fn get(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.read().await.get(id).cloned()
    }

    pub async fn all(&self) -> Vec<(String, Arc<dyn Provider>)> {
        self.providers
            .read()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// Resolve a model name to a Provider.
///
/// Priority:
/// 1. Key's routing_rules (by priority DESC, pattern match)
/// 2. Model name prefix auto-routing
/// 3. Key's default_provider
/// 4. NoRoute error
pub async fn resolve_provider(
    registry: &ProviderRegistry,
    pool: &SqlitePool,
    key_id: &str,
    default_provider_id: Option<&str>,
    model: &str,
) -> AppResult<Arc<dyn Provider>> {
    // 1. Check routing rules for this key
    let rules = db::get_routing_rules_for_key(pool, key_id).await?;
    for rule in &rules {
        if match_pattern(&rule.model_pattern, model) {
            if let Some(p) = registry.get(&rule.target_provider_id).await {
                return Ok(p);
            }
        }
    }

    // 2. Model name prefix auto-routing
    if let Some(provider_id) = auto_route_by_prefix(model, registry).await {
        if let Some(p) = registry.get(&provider_id).await {
            return Ok(p);
        }
    }

    // 3. Key's default provider
    if let Some(dp_id) = default_provider_id {
        if let Some(p) = registry.get(dp_id).await {
            return Ok(p);
        }
    }

    Err(AppError::NoRoute(format!(
        "No provider found for model '{model}'"
    )))
}

/// Match model name against a glob-like pattern.
/// Supports `*` as wildcard.
fn match_pattern(pattern: &str, model: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(prefix) = pattern.strip_suffix('*') {
        return model.starts_with(prefix);
    }

    if let Some(suffix) = pattern.strip_prefix('*') {
        return model.ends_with(suffix);
    }

    pattern == model
}

/// Auto-route based on well-known model name prefixes.
/// Also checks provider names for custom providers (e.g. a provider named "qwen" matches qwen-* models).
async fn auto_route_by_prefix(model: &str, registry: &ProviderRegistry) -> Option<String> {
    let model_lower = model.to_lowercase();

    // 1. Check well-known provider types by model prefix
    let target_type = if model_lower.starts_with("claude") {
        Some("anthropic")
    } else if model_lower.starts_with("gpt-")
        || model_lower.starts_with("o1")
        || model_lower.starts_with("o3")
        || model_lower.starts_with("o4")
        || model_lower.starts_with("chatgpt")
        || model_lower.starts_with("dall-e")
    {
        Some("openai")
    } else if model_lower.starts_with("deepseek") {
        Some("deepseek")
    } else {
        None
    };

    if let Some(tt) = target_type {
        for (id, provider) in registry.all().await {
            if provider.provider_type().to_string() == tt {
                return Some(id);
            }
        }
    }

    // 2. For custom/generic providers, match by provider name prefix.
    //    e.g. a provider named "qwen" matches model "qwen-turbo",
    //    a provider named "groq" matches model "groq-llama3", etc.
    let all = registry.all().await;
    for (id, provider) in &all {
        let name_lower = provider.name().to_lowercase();
        if model_lower.starts_with(&name_lower) {
            return Some(id.clone());
        }
    }

    // 3. Common open-source model names -> Ollama (local inference)
    if model_lower.contains("llama")
        || model_lower.contains("mistral")
        || model_lower.contains("gemma")
        || model_lower.contains("phi")
    {
        for (id, provider) in &all {
            if provider.provider_type().to_string() == "ollama" {
                return Some(id.clone());
            }
        }
    }

    None
}

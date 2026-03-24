use std::sync::Arc;

use crate::config::HybridConfig;
use crate::db;
use crate::dynamic_config::DynamicHybridConfig;
use crate::providers::Provider;
use crate::router::smart_router::{classify_request, TaskClassification};
use crate::router::static_router::ProviderRegistry;
use crate::types::chat::ChatCompletionRequest;
use sqlx::SqlitePool;

/// Estimate token count from a chat request.
/// Uses a rough heuristic: ~4 chars per token for English, ~2 for CJK.
pub fn estimate_tokens(req: &ChatCompletionRequest) -> u32 {
    let mut total_chars: usize = 0;
    for msg in &req.messages {
        if let Some(ref content) = msg.content {
            total_chars += content.as_text().len();
        }
    }
    // Add tool definitions if present
    if let Some(ref tools) = req.tools {
        for tool in tools {
            total_chars += tool.function.name.len();
            total_chars += tool.function.description.as_ref().map(|d| d.len()).unwrap_or(0);
            total_chars += tool.function.parameters.as_ref()
                .map(|p| p.to_string().len())
                .unwrap_or(0);
        }
    }
    // Rough estimate: ~3.5 chars per token on average (mix of EN/CJK)
    (total_chars as f64 / 3.5) as u32
}

/// Decision made by the hybrid router
#[derive(Clone)]
pub enum RoutingDecision {
    /// Route to local model
    Local {
        provider: Arc<dyn Provider>,
        model: String,
        task_type: String,
    },
    /// Route to cloud model
    Cloud {
        provider: Arc<dyn Provider>,
        model: String,
        task_type: String,
        reason: String,
    },
    /// Try local first, fallback to cloud if it fails (non-streaming only)
    LocalWithFallback {
        local_provider: Arc<dyn Provider>,
        local_model: String,
        cloud_provider: Arc<dyn Provider>,
        cloud_model: String,
        task_type: String,
    },
}

impl RoutingDecision {
    #[allow(dead_code)]
    pub fn task_type(&self) -> &str {
        match self {
            RoutingDecision::Local { task_type, .. } => task_type,
            RoutingDecision::Cloud { task_type, .. } => task_type,
            RoutingDecision::LocalWithFallback { task_type, .. } => task_type,
        }
    }

    #[allow(dead_code)]
    pub fn is_local(&self) -> bool {
        matches!(self, RoutingDecision::Local { .. })
    }
}

/// Resolve a provider by name from the registry.
async fn find_provider_by_name(
    registry: &ProviderRegistry,
    name: &str,
) -> Option<Arc<dyn Provider>> {
    let all = registry.all().await;
    all.into_iter()
        .find(|(_, p)| p.name().eq_ignore_ascii_case(name))
        .map(|(_, p)| p)
}

/// Make a hybrid routing decision based on config, classification, and historical data.
pub async fn hybrid_route(
    config: &HybridConfig,
    registry: &ProviderRegistry,
    pool: &SqlitePool,
    req: &ChatCompletionRequest,
    dynamic: Option<&DynamicHybridConfig>,
) -> Option<RoutingDecision> {
    if !config.enabled {
        return None;
    }

    // Resolve providers
    let local_provider = find_provider_by_name(registry, &config.local_provider).await?;
    let cloud_provider = find_provider_by_name(registry, &config.cloud_provider).await?;

    // Classify the request
    let classification: TaskClassification = classify_request(req);
    let task_type = classification.category.to_string();

    // Get effective config values (dynamic overrides > static config)
    let effective_task_types = if let Some(dc) = dynamic {
        dc.effective_local_task_types(&config.local_task_types).await
    } else {
        config.local_task_types.clone()
    };

    let effective_max_tokens = if let Some(dc) = dynamic {
        dc.effective_max_context_tokens(config.max_local_context_tokens).await
    } else {
        config.max_local_context_tokens
    };

    let effective_min_success_rate = if let Some(dc) = dynamic {
        dc.effective_min_success_rate(config.min_local_success_rate).await
    } else {
        config.min_local_success_rate
    };

    // Check if this task type is eligible for local routing
    if !effective_task_types.is_empty()
        && !effective_task_types.iter().any(|t| t == &task_type)
    {
        return Some(RoutingDecision::Cloud {
            provider: cloud_provider,
            model: config.cloud_model.clone(),
            task_type,
            reason: format!("task type '{}' not in local_task_types", classification.category),
        });
    }

    // Check context size — large contexts go to cloud
    if effective_max_tokens > 0 {
        let estimated_tokens = estimate_tokens(req);
        if estimated_tokens > effective_max_tokens {
            return Some(RoutingDecision::Cloud {
                provider: cloud_provider,
                model: config.cloud_model.clone(),
                task_type,
                reason: format!(
                    "context too large ({} tokens > {} limit)",
                    estimated_tokens, effective_max_tokens
                ),
            });
        }
    }

    // Check historical success rate for this task type
    let success_rates = db::get_local_success_rates(pool, 7).await.ok()?;
    let rate_for_type = success_rates.iter().find(|r| r.task_type == task_type);

    match rate_for_type {
        Some(rate) if rate.total >= config.min_samples => {
            if rate.success_rate >= effective_min_success_rate {
                // Good success rate → route locally
                if config.fallback_enabled && !req.stream {
                    Some(RoutingDecision::LocalWithFallback {
                        local_provider,
                        local_model: config.local_model.clone(),
                        cloud_provider,
                        cloud_model: config.cloud_model.clone(),
                        task_type,
                    })
                } else {
                    Some(RoutingDecision::Local {
                        provider: local_provider,
                        model: config.local_model.clone(),
                        task_type,
                    })
                }
            } else {
                // Low success rate → cloud
                Some(RoutingDecision::Cloud {
                    provider: cloud_provider,
                    model: config.cloud_model.clone(),
                    task_type,
                    reason: format!(
                        "local success rate {:.1}% < {:.1}% threshold",
                        rate.success_rate * 100.0,
                        effective_min_success_rate * 100.0
                    ),
                })
            }
        }
        _ => {
            // Not enough data yet → cold start strategy:
            // Alternate between local (with fallback) and cloud to collect
            // both routing history AND training samples from cloud.
            // Use TOTAL routing history count (all destinations) for alternation,
            // not just local count — otherwise we get stuck in a cloud-only loop.
            let all_stats = db::get_routing_stats(pool, 7).await.ok().unwrap_or_default();
            let total_for_type: i64 = all_stats
                .iter()
                .filter(|_| true)
                .map(|s| s.total)
                .sum();

            // Use modulo to alternate: even counts → cloud, odd → local
            // This ensures ~50% of requests go to cloud during cold start,
            // collecting training data for distillation.
            let try_local = total_for_type % 2 == 1;

            if try_local && config.fallback_enabled && !req.stream {
                Some(RoutingDecision::LocalWithFallback {
                    local_provider,
                    local_model: config.local_model.clone(),
                    cloud_provider,
                    cloud_model: config.cloud_model.clone(),
                    task_type,
                })
            } else {
                Some(RoutingDecision::Cloud {
                    provider: cloud_provider,
                    model: config.cloud_model.clone(),
                    task_type,
                    reason: format!(
                        "cold start: collecting data ({}/{} samples)",
                        total_for_type,
                        config.min_samples
                    ),
                })
            }
        }
    }
}

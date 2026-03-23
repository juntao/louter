use std::sync::Arc;

use crate::config::HybridConfig;
use crate::db;
use crate::providers::Provider;
use crate::router::smart_router::{classify_request, TaskClassification};
use crate::router::static_router::ProviderRegistry;
use crate::types::chat::ChatCompletionRequest;
use sqlx::SqlitePool;

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

    // Check if this task type is eligible for local routing
    if !config.local_task_types.is_empty()
        && !config.local_task_types.iter().any(|t| t == &task_type)
    {
        return Some(RoutingDecision::Cloud {
            provider: cloud_provider,
            model: config.cloud_model.clone(),
            task_type,
            reason: format!("task type '{}' not in local_task_types", classification.category),
        });
    }

    // Check historical success rate for this task type
    let success_rates = db::get_local_success_rates(pool, 7).await.ok()?;
    let rate_for_type = success_rates.iter().find(|r| r.task_type == task_type);

    match rate_for_type {
        Some(rate) if rate.total >= config.min_samples => {
            if rate.success_rate >= config.min_local_success_rate {
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
                        config.min_local_success_rate * 100.0
                    ),
                })
            }
        }
        _ => {
            // Not enough data yet → cold start strategy:
            // Alternate between local (with fallback) and cloud to collect
            // both routing history AND training samples from cloud.
            let current_count = rate_for_type.map(|r| r.total).unwrap_or(0);

            // Use modulo to alternate: even counts → cloud, odd → local
            // This ensures ~50% of requests go to cloud during cold start,
            // collecting training data for distillation.
            let try_local = current_count % 2 == 1;

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
                        current_count,
                        config.min_samples
                    ),
                })
            }
        }
    }
}

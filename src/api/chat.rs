use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::db;
use crate::db::schema::{RoutingHistoryRow, TrainingSampleRow, UsageLogRow};
use crate::error::AppError;
use crate::feedback;
use crate::router::hybrid_router::{self, RoutingDecision};
use crate::router::smart_router;
use crate::router::static_router::resolve_provider;
use crate::types::chat::ChatCompletionRequest;
use crate::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    // Auth: extract Bearer token
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let key_value = auth_header
        .strip_prefix("Bearer ")
        .unwrap_or(auth_header);

    // Look up the key
    let key = db::get_key_by_value(&state.db, key_value)
        .await?
        .ok_or_else(|| AppError::Unauthorized("Invalid API key".to_string()))?;

    // Capture the original requested model before routing
    let request_model = req.model.clone();

    // Classify the request for training data collection
    let classification = smart_router::classify_request(&req);
    let task_type = classification.category.to_string();

    // Implicit feedback: detect if this is a retry of a recent request.
    // If so, mark the previous training sample as failed.
    let content_hash = feedback::hash_last_user_message(&req.messages);
    let failed_sample = state
        .feedback
        .record_and_check_retry(content_hash, &task_type, None, "pending")
        .await;
    if let Some(sample_id) = failed_sample {
        let db = state.db.clone();
        tokio::spawn(async move {
            let _ = sqlx::query(
                "UPDATE training_samples SET is_successful = 0 WHERE id = ?",
            )
            .bind(&sample_id)
            .execute(&db)
            .await;
            tracing::info!("Marked sample {} as failed (retry detected)", sample_id);
        });
    }

    // Session-aware hybrid routing:
    // 1. Continuation → reuse same model as previous turns
    // 2. New conversation → make fresh routing decision via hybrid router
    // 3. Mid-session failure → escalate to cloud (handled in fallback path)
    let hybrid_decision = if state.config.hybrid.enabled {
        let cached_target = state.session_router.get_session_target(&req.messages).await;

        // Get the base routing decision (we always need it for provider references)
        let base_decision = hybrid_router::hybrid_route(
            &state.config.hybrid,
            &state.registry,
            &state.db,
            &req,
            Some(&state.dynamic_config),
        )
        .await;

        match (&cached_target, base_decision) {
            (_, None) => None, // No providers configured

            (None, Some(decision)) => {
                // New session: use hybrid router's decision and cache it
                let target = match &decision {
                    RoutingDecision::Local { .. } | RoutingDecision::LocalWithFallback { .. } => "local",
                    RoutingDecision::Cloud { .. } => "cloud",
                };
                state.session_router.set_session_target(&req.messages, target).await;
                tracing::info!("New session: target={}", target);
                Some(decision)
            }

            (Some(target), Some(decision)) => {
                // Existing session: override the decision to match cached target
                Some(override_decision_target(decision, target, &state.config.hybrid))
            }
        }
    } else {
        None
    };

    // Route the request
    if let Some(decision) = hybrid_decision {
        handle_hybrid_request(state, req, key.id, request_model, task_type, decision).await
    } else {
        // Standard routing (no hybrid)
        handle_standard_request(state, req, key, request_model, task_type).await
    }
}

/// Override a routing decision to match the cached session target.
///
/// Keeps provider references from the original decision but changes direction:
/// - target="cloud" → forces Cloud (session was escalated or started on cloud)
/// - target="local" → forces Local/LocalWithFallback (session started on local)
fn override_decision_target(
    decision: RoutingDecision,
    target: &str,
    _config: &crate::config::HybridConfig,
) -> RoutingDecision {
    match (target, &decision) {
        // Already matches target → pass through
        ("cloud", RoutingDecision::Cloud { .. }) => decision,
        ("local", RoutingDecision::Local { .. }) => decision,
        ("local", RoutingDecision::LocalWithFallback { .. }) => decision,

        // Session wants cloud, we have LocalWithFallback → extract cloud provider
        ("cloud", RoutingDecision::LocalWithFallback { .. }) => {
            tracing::info!("Session: overriding local → cloud (cached/escalated)");
            match decision {
                RoutingDecision::LocalWithFallback { cloud_provider, cloud_model, task_type, .. } => {
                    RoutingDecision::Cloud {
                        provider: cloud_provider,
                        model: cloud_model,
                        task_type,
                        reason: "session: staying on cloud".to_string(),
                    }
                }
                _ => unreachable!(),
            }
        }

        // Session wants cloud, we only have Local (no cloud ref) → pass through
        // (rare: only happens if fallback is disabled)
        ("cloud", RoutingDecision::Local { .. }) => {
            tracing::info!("Session: want cloud but only have Local decision, passing through");
            decision
        }

        // Session wants local, router wants cloud → keep cloud but log it
        // (don't force local if the router specifically chose cloud — respect the data)
        ("local", RoutingDecision::Cloud { .. }) => {
            tracing::info!("Session: want local but router chose cloud, respecting router");
            decision
        }

        _ => decision,
    }
}

/// Handle a request using standard (non-hybrid) routing.
async fn handle_standard_request(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    key: crate::db::schema::KeyRow,
    request_model: String,
    task_type: String,
) -> Result<Response, AppError> {
    let (provider, model) = if req.model == "auto" {
        if let Some(ref sr_config) = state.config.smart_routing {
            match smart_router::smart_route(sr_config, &state.registry, &req.messages).await {
                Some((p, m)) => (p, m),
                None => {
                    return Err(AppError::NoRoute(
                        "Smart routing: no matching provider found".into(),
                    ));
                }
            }
        } else {
            return Err(AppError::BadRequest(
                "Smart routing not configured. Add [smart_routing] to louter.toml".into(),
            ));
        }
    } else {
        let p = resolve_provider(
            &state.registry,
            &state.db,
            &key.id,
            key.default_provider_id.as_deref(),
            &req.model,
        )
        .await?;
        (p, req.model.clone())
    };

    let mut req = req;
    req.model = model.clone();
    let key_id = key.id.clone();

    execute_and_track(state, req, key_id, request_model, model, task_type, provider, "cloud").await
}

/// Handle a request using hybrid routing (local/cloud/fallback).
async fn handle_hybrid_request(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    key_id: String,
    request_model: String,
    _task_type: String,
    decision: RoutingDecision,
) -> Result<Response, AppError> {
    match decision {
        RoutingDecision::Local {
            provider,
            model,
            task_type,
        } => {
            tracing::info!("Hybrid: routing to local model '{}' (task: {})", model, task_type);
            let mut req = req;
            req.model = model.clone();
            execute_and_track(state, req, key_id, request_model, model, task_type, provider, "local").await
        }

        RoutingDecision::Cloud {
            provider,
            model,
            task_type,
            reason,
        } => {
            tracing::info!("Hybrid: routing to cloud '{}' (reason: {})", model, reason);
            let mut req = req;
            req.model = model.clone();
            execute_and_track(state, req, key_id, request_model, model, task_type, provider, "cloud").await
        }

        RoutingDecision::LocalWithFallback {
            local_provider,
            local_model,
            cloud_provider,
            cloud_model,
            task_type,
        } => {
            // Non-streaming only: try local, fallback to cloud
            if req.stream {
                // For streaming, just route to local (no fallback possible)
                tracing::info!("Hybrid: streaming → local model '{}'", local_model);
                let mut req = req;
                req.model = local_model.clone();
                return execute_and_track(
                    state, req, key_id, request_model, local_model, task_type, local_provider, "local",
                )
                .await;
            }

            tracing::info!("Hybrid: trying local '{}' with cloud fallback", local_model);
            let mut local_req = req.clone();
            local_req.model = local_model.clone();

            let start = Instant::now();
            let db_pool = state.db.clone();

            match local_provider.complete(&local_req).await {
                Ok(response) => {
                    let latency = start.elapsed().as_millis() as i32;

                    // Check if the response looks valid
                    let has_tools_in_req = local_req.tools.as_ref().is_some_and(|t| !t.is_empty());
                    let is_valid = validate_response(&response, has_tools_in_req);

                    if is_valid {
                        tracing::info!("Hybrid: local model succeeded");
                        log_usage(
                            &db_pool, &key_id, &request_model, &local_model,
                            response.usage.as_ref().map(|u| u.prompt_tokens as i32).unwrap_or(0),
                            response.usage.as_ref().map(|u| u.completion_tokens as i32).unwrap_or(0),
                            latency,
                        ).await;
                        log_routing(&db_pool, &task_type, "local", true, false, latency).await;

                        // Collect training sample from local success too
                        if state.config.distillation.collect_training_data {
                            collect_sample(
                                &db_pool, &local_req, &response, &request_model, &local_model,
                                "ollama", &task_type, "local", latency,
                            ).await;
                        }

                        return Ok(Json(response).into_response());
                    }

                    // Local response was bad → fallback to cloud + escalate session
                    tracing::info!("Hybrid: local response invalid, falling back to cloud");
                    log_routing(&db_pool, &task_type, "local", false, false, latency).await;
                    state.session_router.escalate_to_cloud(&local_req.messages).await;
                }
                Err(e) => {
                    let latency = start.elapsed().as_millis() as i32;
                    tracing::warn!("Hybrid: local model error: {e}, falling back to cloud");
                    log_routing(&db_pool, &task_type, "local", false, false, latency).await;
                    state.session_router.escalate_to_cloud(&local_req.messages).await;
                }
            }

            // Fallback to cloud
            let mut cloud_req = req;
            cloud_req.model = cloud_model.clone();
            let start = Instant::now();

            let response = cloud_provider.complete(&cloud_req).await?;
            let latency = start.elapsed().as_millis() as i32;

            let usage = response.usage.as_ref();
            log_usage(
                &db_pool, &key_id, &request_model, &cloud_model,
                usage.map(|u| u.prompt_tokens as i32).unwrap_or(0),
                usage.map(|u| u.completion_tokens as i32).unwrap_or(0),
                latency,
            ).await;
            log_routing(&db_pool, &task_type, "cloud", true, true, latency).await;

            // Collect training sample from cloud fallback
            if state.config.distillation.collect_training_data {
                collect_sample(
                    &db_pool, &cloud_req, &response, &request_model, &cloud_model,
                    "cloud", &task_type, "cloud", latency,
                ).await;
            }

            Ok(Json(response).into_response())
        }
    }
}

/// Validate a non-streaming response looks reasonable.
///
/// Beyond just checking for content presence, validates that the response
/// is substantive enough to be useful.
fn validate_response(
    response: &crate::types::chat::ChatCompletionResponse,
    has_tools_in_request: bool,
) -> bool {
    // Must have at least one choice
    if response.choices.is_empty() {
        return false;
    }

    let choice = &response.choices[0];

    // Check finish_reason
    if let Some(ref reason) = choice.finish_reason {
        if reason == "error" {
            return false;
        }
    }

    let has_tool_calls = choice
        .message
        .tool_calls
        .as_ref()
        .is_some_and(|tc| !tc.is_empty());

    // If request had tools but response has no tool_calls, check content quality
    if has_tools_in_request && !has_tool_calls {
        let content = choice
            .message
            .content
            .as_ref()
            .map(|c| c.trim())
            .unwrap_or("");
        // Local model should have called a tool but didn't — likely low quality
        if content.len() < 20 {
            return false;
        }
        // Check for common "I can't do this" patterns from local models
        let refusal_patterns = [
            "I cannot", "I can't", "I'm unable", "I don't have",
            "sorry", "apologize", "as an ai",
            "无法", "抱歉", "不能",
        ];
        let content_lower = content.to_lowercase();
        for pattern in &refusal_patterns {
            if content_lower.contains(pattern) {
                return false;
            }
        }
    }

    // Must have content or tool_calls
    let has_content = choice
        .message
        .content
        .as_ref()
        .is_some_and(|c| c.trim().len() >= 5);

    if !has_content && !has_tool_calls {
        return false;
    }

    true
}

/// Execute a request and track usage + training data.
async fn execute_and_track(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    key_id: String,
    request_model: String,
    model: String,
    task_type: String,
    provider: Arc<dyn crate::providers::Provider>,
    source: &str,
) -> Result<Response, AppError> {
    let source = source.to_string();

    if req.stream {
        // Streaming response
        let start = Instant::now();
        let chunk_stream = provider.stream(&req).await?;

        let db = state.db.clone();
        let collect_data = state.config.distillation.collect_training_data && source == "cloud";
        let req_for_sample = if collect_data { Some(req.clone()) } else { None };
        let task_type_clone = task_type.clone();
        let request_model_clone = request_model.clone();
        let model_clone = model.clone();
        let source_clone = source.clone();

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            let mut prompt_tokens: u32 = 0;
            let mut completion_tokens: u32 = 0;
            let mut accumulated_content = String::new();
            let mut has_tool_calls = false;

            futures::pin_mut!(chunk_stream);
            while let Some(result) = chunk_stream.next().await {
                match result {
                    Ok(chunk) => {
                        if let Some(ref u) = chunk.usage {
                            if u.prompt_tokens > 0 {
                                prompt_tokens = u.prompt_tokens;
                            }
                            if u.completion_tokens > 0 {
                                completion_tokens = u.completion_tokens;
                            }
                        }

                        // Accumulate content for training sample
                        if collect_data {
                            for choice in &chunk.choices {
                                if let Some(ref content) = choice.delta.content {
                                    accumulated_content.push_str(content);
                                }
                                if choice.delta.tool_calls.is_some() {
                                    has_tool_calls = true;
                                }
                            }
                        }

                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let event = Event::default().data(data);
                        if tx.send(Ok::<_, std::convert::Infallible>(event)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {e}");
                        let event = Event::default().data(format!(r#"{{"error": "{e}"}}"#));
                        let _ = tx.send(Ok(event)).await;
                        break;
                    }
                }
            }

            // Send [DONE]
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

            let latency = start.elapsed().as_millis() as i32;
            let total_tokens = prompt_tokens + completion_tokens;
            let (pt, ct, tt) = (
                prompt_tokens as i32,
                completion_tokens as i32,
                total_tokens as i32,
            );

            log_usage(&db, &key_id, &request_model_clone, &model_clone, pt, ct, latency).await;
            log_routing(&db, &task_type_clone, &source_clone, true, false, latency).await;

            // Collect streaming response as training sample
            if collect_data && !accumulated_content.is_empty() {
                if let Some(req_data) = req_for_sample {
                    let response_json = serde_json::json!({
                        "role": "assistant",
                        "content": accumulated_content,
                    });
                    let messages_json =
                        serde_json::to_string(&req_data.messages).unwrap_or_default();
                    let tools_json = req_data.tools.as_ref().map(|t| {
                        serde_json::to_string(t).unwrap_or_default()
                    });

                    let sample = TrainingSampleRow {
                        id: uuid::Uuid::new_v4().to_string(),
                        request_messages: messages_json,
                        request_tools: tools_json,
                        response_content: response_json.to_string(),
                        request_model: request_model_clone,
                        actual_model: model_clone,
                        provider_type: "cloud".to_string(),
                        task_type: task_type_clone,
                        has_tool_calls,
                        is_successful: true,
                        source: "cloud".to_string(),
                        prompt_tokens: pt,
                        completion_tokens: ct,
                        total_tokens: tt,
                        latency_ms: latency,
                        is_exported: false,
                        created_at: chrono::Utc::now().to_rfc3339(),
                    };
                    let _ = db::insert_training_sample(&db, &sample).await;
                }
            }
        });

        let event_stream = ReceiverStream::new(rx);
        Ok(Sse::new(event_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming response
        let start = Instant::now();
        let response = provider.complete(&req).await?;
        let latency = start.elapsed().as_millis() as i32;

        let usage = response.usage.as_ref();
        let pt = usage.map(|u| u.prompt_tokens as i32).unwrap_or(0);
        let ct = usage.map(|u| u.completion_tokens as i32).unwrap_or(0);

        let db = state.db.clone();
        let key_id_clone = key_id.clone();
        let request_model_clone = request_model.clone();
        let model_clone = model.clone();
        let task_type_clone = task_type.clone();
        let source_clone = source.clone();

        // Collect training sample from cloud responses
        let collect_data = state.config.distillation.collect_training_data && source == "cloud";
        let sample_id = if collect_data {
            collect_sample(
                &db,
                &req,
                &response,
                &request_model,
                &model,
                "cloud",
                &task_type,
                &source,
                latency,
            )
            .await
        } else {
            None
        };

        // Update feedback tracker with sample_id for retry detection
        let content_hash = feedback::hash_last_user_message(&req.messages);
        state
            .feedback
            .record_and_check_retry(content_hash, &task_type, sample_id, &source)
            .await;

        tokio::spawn(async move {
            log_usage(
                &db, &key_id_clone, &request_model_clone, &model_clone, pt, ct, latency,
            ).await;
            log_routing(&db, &task_type_clone, &source_clone, true, false, latency).await;
        });

        Ok(Json(response).into_response())
    }
}

/// Log usage to usage_logs table.
async fn log_usage(
    db: &sqlx::SqlitePool,
    key_id: &str,
    request_model: &str,
    model: &str,
    prompt_tokens: i32,
    completion_tokens: i32,
    latency_ms: i32,
) {
    let total_tokens = prompt_tokens + completion_tokens;
    let log = UsageLogRow {
        id: uuid::Uuid::new_v4().to_string(),
        key_id: Some(key_id.to_string()),
        provider_id: None,
        request_model: request_model.to_string(),
        model_id: model.to_string(),
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        status_code: 200,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    let _ = db::insert_usage_log(db, &log).await;
}

/// Log routing decision to routing_history table.
async fn log_routing(
    db: &sqlx::SqlitePool,
    task_type: &str,
    routed_to: &str,
    was_successful: bool,
    was_fallback: bool,
    latency_ms: i32,
) {
    let row = RoutingHistoryRow {
        id: uuid::Uuid::new_v4().to_string(),
        task_type: task_type.to_string(),
        routed_to: routed_to.to_string(),
        was_successful,
        was_fallback,
        latency_ms,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    let _ = db::insert_routing_history(db, &row).await;
}

/// Collect a training sample from a completed request/response pair.
/// Returns the sample ID for feedback tracking.
async fn collect_sample(
    db: &sqlx::SqlitePool,
    req: &ChatCompletionRequest,
    response: &crate::types::chat::ChatCompletionResponse,
    request_model: &str,
    actual_model: &str,
    provider_type: &str,
    task_type: &str,
    source: &str,
    latency_ms: i32,
) -> Option<String> {
    let messages_json = serde_json::to_string(&req.messages).unwrap_or_default();
    let tools_json = req
        .tools
        .as_ref()
        .map(|t| serde_json::to_string(t).unwrap_or_default());

    let has_tool_calls = response
        .choices
        .first()
        .and_then(|c| c.message.tool_calls.as_ref())
        .is_some_and(|tc| !tc.is_empty());

    let response_json = response
        .choices
        .first()
        .map(|c| serde_json::to_string(&c.message).unwrap_or_default())
        .unwrap_or_default();

    let usage = response.usage.as_ref();
    let pt = usage.map(|u| u.prompt_tokens as i32).unwrap_or(0);
    let ct = usage.map(|u| u.completion_tokens as i32).unwrap_or(0);
    let tt = pt + ct;

    let sample_id = uuid::Uuid::new_v4().to_string();
    let sample = TrainingSampleRow {
        id: sample_id.clone(),
        request_messages: messages_json,
        request_tools: tools_json,
        response_content: response_json,
        request_model: request_model.to_string(),
        actual_model: actual_model.to_string(),
        provider_type: provider_type.to_string(),
        task_type: task_type.to_string(),
        has_tool_calls,
        is_successful: true,
        source: source.to_string(),
        prompt_tokens: pt,
        completion_tokens: ct,
        total_tokens: tt,
        latency_ms,
        is_exported: false,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    match db::insert_training_sample(db, &sample).await {
        Ok(_) => Some(sample_id),
        Err(_) => None,
    }
}

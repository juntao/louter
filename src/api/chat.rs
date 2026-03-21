use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::db;
use crate::db::schema::UsageLogRow;
use crate::error::AppError;
use crate::router::smart_router;
use crate::router::static_router::resolve_provider;
use crate::types::chat::ChatCompletionRequest;
use crate::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, AppError> {
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

    // Resolve provider
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
    let request_model = request_model.clone();

    if req.stream {
        // Streaming response
        let start = Instant::now();
        let chunk_stream = provider.stream(&req).await?;

        let db = state.db.clone();

        // Use a channel so we can track usage across the stream lifetime
        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            // Accumulate usage across chunks (Anthropic splits across message_start and message_delta)
            let mut prompt_tokens: u32 = 0;
            let mut completion_tokens: u32 = 0;

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
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let event = Event::default().data(data);
                        if tx.send(Ok::<_, std::convert::Infallible>(event)).await.is_err() {
                            break; // client disconnected
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

            // Log usage with actual token counts
            let latency = start.elapsed().as_millis() as i32;
            let total_tokens = prompt_tokens + completion_tokens;
            let (pt, ct, tt) = (prompt_tokens as i32, completion_tokens as i32, total_tokens as i32);
            let _ = db::insert_usage_log(
                &db,
                &UsageLogRow {
                    id: uuid::Uuid::new_v4().to_string(),
                    key_id: Some(key_id),
                    provider_id: None,
                    request_model,
                    model_id: model,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    total_tokens: tt,
                    latency_ms: latency,
                    status_code: 200,
                    created_at: chrono::Utc::now().to_rfc3339(),
                },
            )
            .await;
        });

        let event_stream = ReceiverStream::new(rx);
        Ok(Sse::new(event_stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Non-streaming response
        let start = Instant::now();
        let response = provider.complete(&req).await?;
        let latency = start.elapsed().as_millis() as i32;

        let usage = response.usage.as_ref();
        let log = UsageLogRow {
            id: uuid::Uuid::new_v4().to_string(),
            key_id: Some(key_id),
            provider_id: None,
            request_model,
            model_id: model,
            prompt_tokens: usage.map(|u| u.prompt_tokens as i32).unwrap_or(0),
            completion_tokens: usage.map(|u| u.completion_tokens as i32).unwrap_or(0),
            total_tokens: usage.map(|u| u.total_tokens as i32).unwrap_or(0),
            latency_ms: latency,
            status_code: 200,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let db = state.db.clone();
        tokio::spawn(async move {
            let _ = db::insert_usage_log(&db, &log).await;
        });

        Ok(Json(response).into_response())
    }
}

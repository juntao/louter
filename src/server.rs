use std::sync::Arc;

use axum::routing::{delete, get, post, put};
use axum::Router;
use tower_http::cors::CorsLayer;

use crate::api;
use crate::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::permissive();

    // OpenAI-compatible API routes
    let v1_routes = Router::new()
        .route("/chat/completions", post(api::chat::chat_completions))
        .route("/models", get(api::models::list_models))
        // Multimodal proxy endpoints (passthrough to provider)
        .route("/images/generations", post(api::proxy::proxy_passthrough))
        .route("/images/edits", post(api::proxy::proxy_passthrough))
        .route("/embeddings", post(api::proxy::proxy_passthrough))
        .route("/audio/speech", post(api::proxy::proxy_passthrough))
        .route("/audio/transcriptions", post(api::proxy::proxy_passthrough))
        .route("/audio/translations", post(api::proxy::proxy_passthrough));

    // Admin API routes
    let admin_routes = Router::new()
        .route("/providers", get(api::admin::list_providers))
        .route("/providers", post(api::admin::create_provider))
        .route("/providers/:id", get(api::admin::get_provider))
        .route("/providers/:id", put(api::admin::update_provider))
        .route("/providers/:id", delete(api::admin::delete_provider))
        .route("/providers/:id/test", post(api::admin::test_provider))
        .route(
            "/providers/auto-configure",
            post(api::auto_configure::auto_configure),
        )
        .route("/keys", get(api::admin::list_keys))
        .route("/keys", post(api::admin::create_key))
        .route("/keys/:id", put(api::admin::update_key))
        .route("/keys/:id", delete(api::admin::delete_key))
        .route("/rules", get(api::admin::list_routing_rules))
        .route("/rules", post(api::admin::create_routing_rule))
        .route("/rules/:id", delete(api::admin::delete_routing_rule))
        .route("/usage", get(api::admin::list_usage_logs))
        .route("/usage/stats", get(api::admin::usage_stats))
        // Distillation / Hybrid Inference endpoints
        .route("/distill/stats", get(api::distill::distill_stats))
        .route("/distill/routing", get(api::distill::routing_stats))
        .route("/distill/export", post(api::distill::export_samples))
        .route("/distill/config", get(api::distill::get_distill_config))
        .route("/distill/config", put(api::distill::update_distill_config));

    Router::new()
        .nest("/v1", v1_routes)
        .nest("/api/admin", admin_routes)
        .route("/api/docs", get(api::docs::docs))
        .fallback(api::static_files::static_handler)
        .layer(cors)
        .with_state(state)
}

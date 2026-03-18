use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::error::AppResult;
use crate::types::chat::{ModelObject, ModelsResponse};
use crate::AppState;

/// GET /v1/models
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> AppResult<Json<ModelsResponse>> {
    let mut models = Vec::new();

    for (_id, provider) in state.registry.all().await {
        let provider_name = provider.name().to_string();
        match provider.list_models().await {
            Ok(model_list) => {
                for model_id in model_list {
                    models.push(ModelObject {
                        id: model_id,
                        object: "model".to_string(),
                        created: 0,
                        owned_by: provider_name.clone(),
                    });
                }
            }
            Err(e) => {
                tracing::warn!("Failed to list models from {}: {e}", provider_name);
            }
        }
    }

    Ok(Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    }))
}

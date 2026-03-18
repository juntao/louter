use async_trait::async_trait;

use crate::error::{AppError, AppResult};
use crate::providers::openai::OpenAIProvider;
use crate::providers::{ChunkStream, Provider};
use crate::types::chat::{ChatCompletionRequest, ChatCompletionResponse};
use crate::types::provider::ProviderType;

/// Ollama exposes an OpenAI-compatible API at /v1/*.
pub struct OllamaProvider {
    inner: OpenAIProvider,
    base_url: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(name: String, base_url: String) -> Self {
        let api_url = format!("{}/v1", base_url.trim_end_matches('/'));
        Self {
            inner: OpenAIProvider::new(name, api_url, String::new()),
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Ollama
    }

    async fn complete(&self, req: &ChatCompletionRequest) -> AppResult<ChatCompletionResponse> {
        self.inner.complete(req).await
    }

    async fn stream(&self, req: &ChatCompletionRequest) -> AppResult<ChunkStream> {
        self.inner.stream(req).await
    }

    async fn list_models(&self) -> AppResult<Vec<String>> {
        // Ollama has a native /api/tags endpoint
        let resp = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let body: serde_json::Value = r.json().await.unwrap_or_default();
                let models = body["models"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(models)
            }
            Ok(r) => Err(AppError::ProviderError(format!(
                "Ollama API error: {}",
                r.status()
            ))),
            Err(e) => Err(AppError::ProviderError(format!(
                "Ollama connection error: {e}"
            ))),
        }
    }
}

use async_trait::async_trait;
use futures::StreamExt;

use crate::error::{AppError, AppResult};
use crate::providers::sse::parse_sse_stream;
use crate::providers::{ChunkStream, Provider};
use crate::types::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
};
use crate::types::provider::ProviderType;

/// Azure OpenAI uses a different URL scheme and auth header.
pub struct AzureProvider {
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    pub api_version: String,
    pub client: reqwest::Client,
}

impl AzureProvider {
    pub fn new(name: String, base_url: String, api_key: String, api_version: Option<String>) -> Self {
        Self {
            provider_name: name,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            api_version: api_version.unwrap_or_else(|| "2024-02-01".to_string()),
            client: reqwest::Client::new(),
        }
    }

    /// Azure URL: {base_url}/openai/deployments/{model}/chat/completions?api-version={ver}
    fn build_url(&self, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.base_url, model, self.api_version
        )
    }
}

#[async_trait]
impl Provider for AzureProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Azure
    }

    async fn complete(&self, req: &ChatCompletionRequest) -> AppResult<ChatCompletionResponse> {
        let url = self.build_url(&req.model);
        let mut req_body = serde_json::to_value(req)
            .map_err(|e| AppError::Internal(format!("Failed to serialize request: {e}")))?;
        req_body["stream"] = serde_json::Value::Bool(false);
        // Azure doesn't use the model field in the body
        req_body.as_object_mut().unwrap().remove("model");

        let resp = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::ProviderError(format!(
                "Azure OpenAI API error ({}): {}",
                status, body
            )));
        }

        let response: ChatCompletionResponse = resp
            .json()
            .await
            .map_err(|e| AppError::ProviderError(format!("Failed to parse response: {e}")))?;

        Ok(response)
    }

    async fn stream(&self, req: &ChatCompletionRequest) -> AppResult<ChunkStream> {
        let url = self.build_url(&req.model);
        let mut req_body = serde_json::to_value(req)
            .map_err(|e| AppError::Internal(format!("Failed to serialize request: {e}")))?;
        req_body["stream"] = serde_json::Value::Bool(true);
        req_body.as_object_mut().unwrap().remove("model");

        let resp = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::ProviderError(format!(
                "Azure OpenAI API error ({}): {}",
                status, body
            )));
        }

        let byte_stream = Box::pin(resp.bytes_stream());
        let sse_stream = parse_sse_stream(byte_stream);

        let chunk_stream = sse_stream.filter_map(|event| async move {
            if event.event == "done" || event.data == "[DONE]" {
                return None;
            }
            match serde_json::from_str::<ChatCompletionChunk>(&event.data) {
                Ok(chunk) => Some(Ok(chunk)),
                Err(e) => Some(Err(format!("Failed to parse chunk: {e}"))),
            }
        });

        Ok(Box::pin(chunk_stream))
    }

    async fn list_models(&self) -> AppResult<Vec<String>> {
        // Azure doesn't have a standard model list endpoint; return empty
        Ok(vec![])
    }
}

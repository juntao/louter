use async_trait::async_trait;
use futures::StreamExt;

use crate::error::{AppError, AppResult};
use crate::providers::sse::parse_sse_stream;
use crate::providers::{ChunkStream, Provider};
use crate::types::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
};
use crate::types::provider::ProviderType;

pub struct OpenAIProvider {
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    pub client: reqwest::Client,
}

impl OpenAIProvider {
    pub fn new(name: String, base_url: String, api_key: String) -> Self {
        Self {
            provider_name: name,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    async fn complete(&self, req: &ChatCompletionRequest) -> AppResult<ChatCompletionResponse> {
        let mut req_body = serde_json::to_value(req)
            .map_err(|e| AppError::Internal(format!("Failed to serialize request: {e}")))?;
        req_body["stream"] = serde_json::Value::Bool(false);

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::ProviderError(format!(
                "OpenAI API error ({}): {}",
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
        let mut req_body = serde_json::to_value(req)
            .map_err(|e| AppError::Internal(format!("Failed to serialize request: {e}")))?;
        req_body["stream"] = serde_json::Value::Bool(true);
        // Request usage stats in the final streaming chunk
        req_body["stream_options"] = serde_json::json!({"include_usage": true});

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::ProviderError(format!(
                "OpenAI API error ({}): {}",
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
        let resp = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !resp.status().is_success() {
            return Ok(vec![]);
        }

        let body: serde_json::Value = resp.json().await.unwrap_or_default();
        let models = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

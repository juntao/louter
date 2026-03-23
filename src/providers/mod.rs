pub mod anthropic;
pub mod azure;
pub mod deepseek;
pub mod ollama;
pub mod openai;
pub mod sse;
pub mod tool_call_normalizer;

use std::sync::Arc;

use crate::db::schema::ProviderRow;
use crate::error::AppResult;
use crate::types::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::types::provider::ProviderType;
use async_trait::async_trait;
use futures::stream::BoxStream;

pub type ChunkStream = BoxStream<'static, Result<ChatCompletionChunk, String>>;

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn provider_type(&self) -> ProviderType;
    async fn complete(&self, req: &ChatCompletionRequest) -> AppResult<ChatCompletionResponse>;
    async fn stream(&self, req: &ChatCompletionRequest) -> AppResult<ChunkStream>;
    async fn list_models(&self) -> AppResult<Vec<String>>;
}

/// Create a Provider instance from a database row.
pub fn create_provider_from_row(row: &ProviderRow) -> Option<Arc<dyn Provider>> {
    let pt = ProviderType::from_str(&row.provider_type)?;
    let provider: Arc<dyn Provider> = match pt {
        ProviderType::OpenAI => Arc::new(openai::OpenAIProvider::new(
            row.name.clone(),
            row.base_url.clone(),
            row.api_key.clone(),
        )),
        ProviderType::Anthropic => Arc::new(anthropic::AnthropicProvider::new(
            row.name.clone(),
            row.base_url.clone(),
            row.api_key.clone(),
        )),
        ProviderType::DeepSeek => Arc::new(deepseek::DeepSeekProvider::new(
            row.name.clone(),
            row.base_url.clone(),
            row.api_key.clone(),
        )),
        ProviderType::Ollama => Arc::new(ollama::OllamaProvider::new(
            row.name.clone(),
            row.base_url.clone(),
        )),
        ProviderType::Azure => {
            let api_version = serde_json::from_str::<serde_json::Value>(&row.config_json)
                .ok()
                .and_then(|v| v.get("api_version")?.as_str().map(|s| s.to_string()));
            Arc::new(azure::AzureProvider::new(
                row.name.clone(),
                row.base_url.clone(),
                row.api_key.clone(),
                api_version,
            ))
        }
        // Custom: any OpenAI-compatible provider (Qwen, Groq, Together, Mistral, vLLM, etc.)
        ProviderType::Custom => Arc::new(openai::OpenAIProvider::new(
            row.name.clone(),
            row.base_url.clone(),
            row.api_key.clone(),
        )),
    };
    Some(provider)
}

use async_trait::async_trait;

use crate::error::AppResult;
use crate::providers::openai::OpenAIProvider;
use crate::providers::{ChunkStream, Provider};
use crate::types::chat::{ChatCompletionRequest, ChatCompletionResponse};
use crate::types::provider::ProviderType;

/// DeepSeek uses an OpenAI-compatible API, so we wrap OpenAIProvider.
pub struct DeepSeekProvider {
    inner: OpenAIProvider,
}

impl DeepSeekProvider {
    pub fn new(name: String, base_url: String, api_key: String) -> Self {
        Self {
            inner: OpenAIProvider::new(name, base_url, api_key),
        }
    }
}

#[async_trait]
impl Provider for DeepSeekProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::DeepSeek
    }

    async fn complete(&self, req: &ChatCompletionRequest) -> AppResult<ChatCompletionResponse> {
        self.inner.complete(req).await
    }

    async fn stream(&self, req: &ChatCompletionRequest) -> AppResult<ChunkStream> {
        self.inner.stream(req).await
    }

    async fn list_models(&self) -> AppResult<Vec<String>> {
        self.inner.list_models().await
    }
}

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    Azure,
    Ollama,
    DeepSeek,
    /// Generic OpenAI-compatible provider (Qwen, Groq, Together, Mistral, vLLM, etc.)
    Custom,
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderType::OpenAI => write!(f, "openai"),
            ProviderType::Anthropic => write!(f, "anthropic"),
            ProviderType::Azure => write!(f, "azure"),
            ProviderType::Ollama => write!(f, "ollama"),
            ProviderType::DeepSeek => write!(f, "deepseek"),
            ProviderType::Custom => write!(f, "custom"),
        }
    }
}

impl ProviderType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(ProviderType::OpenAI),
            "anthropic" => Some(ProviderType::Anthropic),
            "azure" => Some(ProviderType::Azure),
            "ollama" => Some(ProviderType::Ollama),
            "deepseek" => Some(ProviderType::DeepSeek),
            "custom" => Some(ProviderType::Custom),
            _ => None,
        }
    }
}

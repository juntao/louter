use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ProviderRow {
    pub id: String,
    pub name: String,
    pub provider_type: String,
    pub base_url: String,
    pub api_key: String,
    pub is_enabled: bool,
    pub config_json: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ModelRow {
    pub id: String,
    pub provider_id: String,
    pub model_id: String,
    pub is_enabled: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct KeyRow {
    pub id: String,
    pub key_value: String,
    pub name: String,
    pub default_provider_id: Option<String>,
    pub is_enabled: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct RoutingRuleRow {
    pub id: String,
    pub key_id: String,
    pub model_pattern: String,
    pub target_provider_id: String,
    pub priority: i32,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct UsageLogRow {
    pub id: String,
    pub key_id: Option<String>,
    pub provider_id: Option<String>,
    pub request_model: String,
    pub model_id: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    pub latency_ms: i32,
    pub status_code: i32,
    pub created_at: String,
}

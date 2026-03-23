use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub database: DatabaseConfig,
    pub smart_routing: Option<HashMap<String, String>>,
    #[serde(default)]
    pub hybrid: HybridConfig,
    #[serde(default)]
    pub distillation: DistillationConfig,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    6188
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    #[serde(default = "default_db_path")]
    pub path: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: default_db_path(),
        }
    }
}

fn default_db_path() -> String {
    "louter.db".to_string()
}

/// Hybrid inference configuration: local model + cloud fallback
#[derive(Debug, Deserialize)]
pub struct HybridConfig {
    /// Enable hybrid routing (local-first with cloud fallback)
    #[serde(default)]
    pub enabled: bool,

    /// Name of the local provider (must match a registered provider name)
    #[serde(default)]
    pub local_provider: String,

    /// Model name to use on the local provider
    #[serde(default)]
    pub local_model: String,

    /// Name of the cloud provider for fallback
    #[serde(default)]
    pub cloud_provider: String,

    /// Model name to use on the cloud provider
    #[serde(default)]
    pub cloud_model: String,

    /// Minimum historical success rate (0.0-1.0) to route to local model.
    /// Below this threshold, requests go directly to cloud.
    #[serde(default = "default_min_success_rate")]
    pub min_local_success_rate: f64,

    /// Minimum number of samples before trusting the success rate.
    /// Below this, requests go to cloud to collect more data.
    #[serde(default = "default_min_samples")]
    pub min_samples: i64,

    /// Enable try-local-first with fallback to cloud (non-streaming only).
    /// When false, uses pure routing (no retry).
    #[serde(default)]
    pub fallback_enabled: bool,

    /// Task types eligible for local routing. Empty = all types.
    #[serde(default)]
    pub local_task_types: Vec<String>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            local_provider: String::new(),
            local_model: String::new(),
            cloud_provider: String::new(),
            cloud_model: String::new(),
            min_local_success_rate: 0.7,
            min_samples: 20,
            fallback_enabled: false,
            local_task_types: Vec::new(),
        }
    }
}

fn default_min_success_rate() -> f64 {
    0.7
}

fn default_min_samples() -> i64 {
    20
}

/// Distillation data collection configuration
#[derive(Debug, Deserialize)]
pub struct DistillationConfig {
    /// Enable automatic collection of training samples from cloud responses
    #[serde(default = "default_true")]
    pub collect_training_data: bool,

    /// Maximum number of training samples to keep (oldest are pruned)
    #[serde(default = "default_max_samples")]
    pub max_samples: i64,

    /// Only collect samples from successful responses (status 200)
    #[serde(default = "default_true")]
    pub only_successful: bool,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            collect_training_data: true,
            max_samples: 100_000,
            only_successful: true,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_max_samples() -> i64 {
    100_000
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file {}: {e}", path.display()))?;

        let config: AppConfig =
            toml::from_str(&content).map_err(|e| format!("Failed to parse config: {e}"))?;

        Ok(config)
    }
}

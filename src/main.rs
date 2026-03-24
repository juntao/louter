mod api;
mod config;
#[allow(dead_code)]
mod db;
mod error;
mod dynamic_config;
mod feedback;
mod providers;
mod session;
mod router;
mod server;
mod types;

use std::path::PathBuf;
use std::sync::Arc;

use config::AppConfig;
use router::static_router::ProviderRegistry;
use sqlx::SqlitePool;

pub struct AppState {
    pub db: SqlitePool,
    pub registry: ProviderRegistry,
    pub config: AppConfig,
    pub feedback: feedback::FeedbackTracker,
    pub session_router: session::SessionRouter,
    pub dynamic_config: dynamic_config::DynamicHybridConfig,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "louter=info".into()),
        )
        .init();

    // Load config
    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("louter.toml"));

    let config = if config_path.exists() {
        match AppConfig::load(&config_path) {
            Ok(c) => {
                tracing::info!("Loaded config from {}", config_path.display());
                c
            }
            Err(e) => {
                tracing::error!("Failed to load config: {e}");
                std::process::exit(1);
            }
        }
    } else {
        tracing::info!("No config file found, using defaults");
        AppConfig {
            server: Default::default(),
            database: Default::default(),
            smart_routing: None,
            hybrid: Default::default(),
            distillation: Default::default(),
        }
    };

    // Initialize database
    let db_url = format!("sqlite:{}?mode=rwc", config.database.path);
    let pool = match db::init_db(&db_url).await {
        Ok(p) => {
            tracing::info!("Database initialized at {}", config.database.path);
            p
        }
        Err(e) => {
            tracing::error!("Failed to initialize database: {e}");
            std::process::exit(1);
        }
    };

    // Load providers from database
    let registry = ProviderRegistry::new();
    match db::list_providers(&pool).await {
        Ok(rows) => {
            for row in &rows {
                if !row.is_enabled {
                    continue;
                }
                match providers::create_provider_from_row(row) {
                    Some(provider) => {
                        tracing::info!("Loaded provider: {} ({})", row.name, row.provider_type);
                        registry.register(row.id.clone(), provider).await;
                    }
                    None => {
                        tracing::warn!(
                            "Unknown provider type '{}' for '{}', skipping",
                            row.provider_type,
                            row.name
                        );
                    }
                }
            }
        }
        Err(e) => {
            tracing::warn!("Failed to load providers from DB: {e}");
        }
    }

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let state = Arc::new(AppState {
        db: pool,
        registry,
        config,
        feedback: feedback::FeedbackTracker::new(),
        session_router: session::SessionRouter::new(),
        dynamic_config: dynamic_config::DynamicHybridConfig::new(),
    });

    let app = server::build_router(state);

    tracing::info!("Louter listening on http://{addr}");
    tracing::info!("Open http://{addr} to configure providers and keys via Web UI");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app).await.expect("Server error");
}

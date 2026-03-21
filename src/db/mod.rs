pub mod schema;

use crate::error::AppResult;
use schema::*;
use sqlx::SqlitePool;

pub async fn init_db(database_url: &str) -> Result<SqlitePool, String> {
    let pool = SqlitePool::connect(database_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {e}"))?;

    // Run migrations
    let migration_sql = include_str!("../../migrations/001_initial.sql");
    sqlx::raw_sql(migration_sql)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to run migrations: {e}"))?;

    // Run incremental migrations (ignore errors for already-applied columns)
    let m002 = include_str!("../../migrations/002_add_request_model.sql");
    let _ = sqlx::raw_sql(m002).execute(&pool).await;

    Ok(pool)
}

// ── Provider CRUD ──

pub async fn list_providers(pool: &SqlitePool) -> AppResult<Vec<ProviderRow>> {
    let rows = sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers ORDER BY name")
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

pub async fn get_provider(pool: &SqlitePool, id: &str) -> AppResult<Option<ProviderRow>> {
    let row = sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

pub async fn get_provider_by_name(pool: &SqlitePool, name: &str) -> AppResult<Option<ProviderRow>> {
    let row = sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers WHERE name = ?")
        .bind(name)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

pub async fn insert_provider(pool: &SqlitePool, row: &ProviderRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO providers (id, name, provider_type, base_url, api_key, is_enabled, config_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.name)
    .bind(&row.provider_type)
    .bind(&row.base_url)
    .bind(&row.api_key)
    .bind(row.is_enabled)
    .bind(&row.config_json)
    .bind(&row.created_at)
    .bind(&row.updated_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_provider(pool: &SqlitePool, row: &ProviderRow) -> AppResult<()> {
    sqlx::query(
        "UPDATE providers SET name = ?, provider_type = ?, base_url = ?, api_key = ?, is_enabled = ?, config_json = ?, updated_at = datetime('now') WHERE id = ?"
    )
    .bind(&row.name)
    .bind(&row.provider_type)
    .bind(&row.base_url)
    .bind(&row.api_key)
    .bind(row.is_enabled)
    .bind(&row.config_json)
    .bind(&row.id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_provider(pool: &SqlitePool, id: &str) -> AppResult<()> {
    sqlx::query("DELETE FROM providers WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Key CRUD ──

pub async fn list_keys(pool: &SqlitePool) -> AppResult<Vec<KeyRow>> {
    let rows = sqlx::query_as::<_, KeyRow>("SELECT * FROM keys ORDER BY name")
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

pub async fn get_key_by_value(pool: &SqlitePool, key_value: &str) -> AppResult<Option<KeyRow>> {
    let row = sqlx::query_as::<_, KeyRow>("SELECT * FROM keys WHERE key_value = ? AND is_enabled = 1")
        .bind(key_value)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

pub async fn insert_key(pool: &SqlitePool, row: &KeyRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO keys (id, key_value, name, default_provider_id, is_enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.key_value)
    .bind(&row.name)
    .bind(&row.default_provider_id)
    .bind(row.is_enabled)
    .bind(&row.created_at)
    .bind(&row.updated_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_key(pool: &SqlitePool, row: &KeyRow) -> AppResult<()> {
    sqlx::query(
        "UPDATE keys SET name = ?, default_provider_id = ?, is_enabled = ?, updated_at = datetime('now') WHERE id = ?"
    )
    .bind(&row.name)
    .bind(&row.default_provider_id)
    .bind(row.is_enabled)
    .bind(&row.id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_key(pool: &SqlitePool, id: &str) -> AppResult<()> {
    sqlx::query("DELETE FROM keys WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Routing Rules ──

pub async fn get_routing_rules_for_key(pool: &SqlitePool, key_id: &str) -> AppResult<Vec<RoutingRuleRow>> {
    let rows = sqlx::query_as::<_, RoutingRuleRow>(
        "SELECT * FROM routing_rules WHERE key_id = ? ORDER BY priority DESC"
    )
    .bind(key_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn list_routing_rules(pool: &SqlitePool) -> AppResult<Vec<RoutingRuleRow>> {
    let rows = sqlx::query_as::<_, RoutingRuleRow>(
        "SELECT * FROM routing_rules ORDER BY priority DESC"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn insert_routing_rule(pool: &SqlitePool, row: &RoutingRuleRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO routing_rules (id, key_id, model_pattern, target_provider_id, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.key_id)
    .bind(&row.model_pattern)
    .bind(&row.target_provider_id)
    .bind(row.priority)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_routing_rule(pool: &SqlitePool, id: &str) -> AppResult<()> {
    sqlx::query("DELETE FROM routing_rules WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn delete_routing_rule_by_key_and_pattern(
    pool: &SqlitePool,
    key_id: &str,
    model_pattern: &str,
) -> AppResult<()> {
    sqlx::query("DELETE FROM routing_rules WHERE key_id = ? AND model_pattern = ?")
        .bind(key_id)
        .bind(model_pattern)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Usage Logs ──

pub async fn insert_usage_log(pool: &SqlitePool, row: &UsageLogRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO usage_logs (id, key_id, provider_id, request_model, model_id, prompt_tokens, completion_tokens, total_tokens, latency_ms, status_code, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.key_id)
    .bind(&row.provider_id)
    .bind(&row.request_model)
    .bind(&row.model_id)
    .bind(row.prompt_tokens)
    .bind(row.completion_tokens)
    .bind(row.total_tokens)
    .bind(row.latency_ms)
    .bind(row.status_code)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn list_usage_logs(pool: &SqlitePool, limit: i32) -> AppResult<Vec<UsageLogRow>> {
    let rows = sqlx::query_as::<_, UsageLogRow>(
        "SELECT * FROM usage_logs ORDER BY created_at DESC LIMIT ?"
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Daily token usage aggregation
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct DailyUsage {
    pub day: String,
    pub requests: i32,
    pub total_tokens: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub avg_latency_ms: f64,
}

pub async fn get_daily_usage(pool: &SqlitePool, days: i32) -> AppResult<Vec<DailyUsage>> {
    let rows = sqlx::query_as::<_, DailyUsage>(
        "SELECT date(created_at) as day, \
         COUNT(*) as requests, \
         COALESCE(SUM(total_tokens), 0) as total_tokens, \
         COALESCE(SUM(prompt_tokens), 0) as prompt_tokens, \
         COALESCE(SUM(completion_tokens), 0) as completion_tokens, \
         COALESCE(AVG(latency_ms), 0) as avg_latency_ms \
         FROM usage_logs \
         WHERE created_at >= datetime('now', '-' || ? || ' days') \
         GROUP BY date(created_at) \
         ORDER BY day ASC"
    )
    .bind(days)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Per-model token usage aggregation
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct ModelUsage {
    pub model_id: String,
    pub requests: i32,
    pub total_tokens: i64,
}

pub async fn get_model_usage(pool: &SqlitePool, days: i32) -> AppResult<Vec<ModelUsage>> {
    let rows = sqlx::query_as::<_, ModelUsage>(
        "SELECT model_id, COUNT(*) as requests, \
         COALESCE(SUM(total_tokens), 0) as total_tokens \
         FROM usage_logs \
         WHERE created_at >= datetime('now', '-' || ? || ' days') \
         GROUP BY model_id \
         ORDER BY total_tokens DESC"
    )
    .bind(days)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

// ── Models ──

pub async fn list_models_for_provider(pool: &SqlitePool, provider_id: &str) -> AppResult<Vec<ModelRow>> {
    let rows = sqlx::query_as::<_, ModelRow>(
        "SELECT * FROM models WHERE provider_id = ? AND is_enabled = 1 ORDER BY model_id"
    )
    .bind(provider_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn upsert_model(pool: &SqlitePool, row: &ModelRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO models (id, provider_id, model_id, is_enabled, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(provider_id, model_id) DO UPDATE SET is_enabled = excluded.is_enabled"
    )
    .bind(&row.id)
    .bind(&row.provider_id)
    .bind(&row.model_id)
    .bind(row.is_enabled)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

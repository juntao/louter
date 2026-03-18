use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Debug)]
pub enum AppError {
    BadRequest(String),
    Unauthorized(String),
    NotFound(String),
    NoRoute(String),
    ProviderError(String),
    Internal(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::BadRequest(msg) => write!(f, "Bad request: {msg}"),
            AppError::Unauthorized(msg) => write!(f, "Unauthorized: {msg}"),
            AppError::NotFound(msg) => write!(f, "Not found: {msg}"),
            AppError::NoRoute(msg) => write!(f, "No route: {msg}"),
            AppError::ProviderError(msg) => write!(f, "Provider error: {msg}"),
            AppError::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "invalid_request_error", msg.as_str()),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, "authentication_error", msg.as_str()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found_error", msg.as_str()),
            AppError::NoRoute(msg) => (StatusCode::BAD_REQUEST, "routing_error", msg.as_str()),
            AppError::ProviderError(msg) => (StatusCode::BAD_GATEWAY, "upstream_error", msg.as_str()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg.as_str()),
        };

        let body = json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": status.as_u16(),
            }
        });

        (status, Json(body)).into_response()
    }
}

impl From<sqlx::Error> for AppError {
    fn from(e: sqlx::Error) -> Self {
        AppError::Internal(format!("Database error: {e}"))
    }
}

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        AppError::ProviderError(format!("HTTP error: {e}"))
    }
}

pub type AppResult<T> = Result<T, AppError>;

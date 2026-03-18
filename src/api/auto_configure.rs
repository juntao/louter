use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::types::chat::{ChatCompletionRequest, Message, MessageContent};
use crate::AppState;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AutoConfigureRequest {
    /// The documentation URL to analyze
    pub doc_url: String,
    /// The API key for the new provider being configured
    #[serde(default)]
    pub api_key: String,
    /// Which existing provider ID to use as the analyzer LLM (optional, auto-selects first available)
    pub analyzer_provider_id: Option<String>,
    /// Which model to use for analysis (optional, uses a sensible default)
    pub analyzer_model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AutoConfigureResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<SuggestedConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_analysis: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SuggestedConfig {
    pub name: String,
    pub provider_type: String,
    pub base_url: String,
    pub is_openai_compatible: bool,
    pub models: Vec<String>,
    pub auth_header: String,
    pub notes: String,
}

/// POST /api/admin/providers/auto-configure
///
/// Uses an already-configured LLM to analyze API documentation and suggest provider configuration.
pub async fn auto_configure(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AutoConfigureRequest>,
) -> AppResult<Json<AutoConfigureResult>> {
    // 1. Find an analyzer LLM provider
    let all_providers = state.registry.all().await;
    if all_providers.is_empty() {
        return Ok(Json(AutoConfigureResult {
            success: false,
            error: Some("No LLM providers configured yet. Please add at least one provider manually first.".to_string()),
            config: None,
            raw_analysis: None,
        }));
    }

    let analyzer = if let Some(ref id) = req.analyzer_provider_id {
        state.registry.get(id).await
            .ok_or_else(|| AppError::NotFound("Specified analyzer provider not found".to_string()))?
    } else {
        // Auto-select: prefer non-Ollama providers (they're usually smarter)
        let mut selected = None;
        for (_id, p) in &all_providers {
            let pt = p.provider_type().to_string();
            if pt != "ollama" {
                selected = Some(p.clone());
                break;
            }
        }
        selected.unwrap_or_else(|| all_providers[0].1.clone())
    };

    let analyzer_model = req.analyzer_model.clone().unwrap_or_else(|| {
        // Pick a sensible default model based on provider type
        match analyzer.provider_type().to_string().as_str() {
            "openai" => "gpt-4o-mini".to_string(),
            "anthropic" => "claude-sonnet-4-20250514".to_string(),
            "deepseek" => "deepseek-chat".to_string(),
            _ => "gpt-4o-mini".to_string(),
        }
    });

    // 2. Fetch the documentation content
    let doc_content = match fetch_doc_content(&req.doc_url).await {
        Ok(content) => content,
        Err(e) => {
            return Ok(Json(AutoConfigureResult {
                success: false,
                error: Some(format!("Failed to fetch documentation: {e}")),
                config: None,
                raw_analysis: None,
            }));
        }
    };

    // Truncate if too long (keep it within reasonable token limits)
    let doc_content = if doc_content.len() > 30000 {
        format!("{}...\n\n[Content truncated, {} total characters]", &doc_content[..30000], doc_content.len())
    } else {
        doc_content
    };

    // 3. Ask the LLM to analyze the documentation
    let prompt = build_analysis_prompt(&doc_content, &req.doc_url);

    let chat_req = ChatCompletionRequest {
        model: analyzer_model,
        messages: vec![
            Message {
                role: "system".to_string(),
                content: Some(MessageContent::Text(
                    "You are an expert at analyzing API documentation and extracting configuration details. Always respond with valid JSON only, no markdown fences.".to_string()
                )),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text(prompt)),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.0),
        top_p: None,
        max_tokens: Some(2000),
        stream: false,
        stop: None,
        tools: None,
        tool_choice: None,
    };

    let response = match analyzer.complete(&chat_req).await {
        Ok(r) => r,
        Err(e) => {
            return Ok(Json(AutoConfigureResult {
                success: false,
                error: Some(format!("LLM analysis failed: {e}")),
                config: None,
                raw_analysis: None,
            }));
        }
    };

    let raw_text = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
        .cloned()
        .unwrap_or_default();

    // 4. Parse the LLM's JSON response
    let config = parse_llm_response(&raw_text);

    match config {
        Some(mut cfg) => {
            // Fill in the API key from the request
            cfg.notes = format!(
                "Auto-configured from: {}\nAnalyzed by: {}",
                req.doc_url,
                analyzer.name()
            );
            Ok(Json(AutoConfigureResult {
                success: true,
                error: None,
                config: Some(cfg),
                raw_analysis: Some(raw_text),
            }))
        }
        None => Ok(Json(AutoConfigureResult {
            success: false,
            error: Some("Could not parse LLM response into a valid configuration. See raw_analysis for details.".to_string()),
            config: None,
            raw_analysis: Some(raw_text),
        })),
    }
}

/// Fetch documentation content from a URL.
/// Tries OpenAPI spec paths first, then falls back to fetching the page directly.
async fn fetch_doc_content(url: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    // Try to find OpenAPI spec at common paths relative to the base domain
    if let Ok(parsed) = url::Url::parse(url) {
        let origin = format!("{}://{}", parsed.scheme(), parsed.host_str().unwrap_or(""));
        let spec_paths = [
            "/openapi.json",
            "/swagger.json",
            "/api/openapi.json",
            "/v1/openapi.json",
            "/llms.txt",
        ];
        for path in &spec_paths {
            let spec_url = format!("{}{}", origin, path);
            if let Ok(resp) = client.get(&spec_url).send().await {
                if resp.status().is_success() {
                    if let Ok(text) = resp.text().await {
                        if text.len() > 100 && (text.contains("openapi") || text.contains("swagger") || text.contains("paths")) {
                            return Ok(format!("[OpenAPI spec found at {}]\n\n{}", spec_url, text));
                        }
                    }
                }
            }
        }
    }

    // Fetch the actual documentation page
    let resp = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0 (compatible; Louter/0.1)")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch URL: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body = resp.text().await.map_err(|e| format!("Failed to read body: {e}"))?;

    if content_type.contains("json") {
        return Ok(body);
    }

    // Strip HTML tags for a basic text extraction
    Ok(strip_html_tags(&body))
}

/// Very basic HTML tag stripping to extract readable text.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut last_was_space = false;

    let html_lower = html.to_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let chars_lower: Vec<char> = html_lower.chars().collect();

    let mut i = 0;
    while i < chars.len() {
        if !in_tag && i + 7 < chars_lower.len() {
            let ahead: String = chars_lower[i..i + 7].iter().collect();
            if ahead == "<script" || ahead == "<style " || ahead == "<style>" {
                in_script = true;
                in_tag = true;
                i += 1;
                continue;
            }
        }

        if in_script && i + 8 < chars_lower.len() {
            let ahead: String = chars_lower[i..i + 9].iter().collect();
            if ahead.starts_with("</script") || ahead.starts_with("</style>") {
                in_script = false;
                // Skip to end of tag
                while i < chars.len() && chars[i] != '>' {
                    i += 1;
                }
                i += 1;
                continue;
            }
            i += 1;
            continue;
        }

        if chars[i] == '<' {
            in_tag = true;
            i += 1;
            continue;
        }

        if chars[i] == '>' && in_tag {
            in_tag = false;
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
            i += 1;
            continue;
        }

        if !in_tag && !in_script {
            let c = chars[i];
            if c.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(c);
                last_was_space = false;
            }
        }

        i += 1;
    }

    // Decode common HTML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn build_analysis_prompt(doc_content: &str, doc_url: &str) -> String {
    format!(
        r#"Analyze the following API documentation from {doc_url} and extract the configuration needed to use this API as an LLM provider.

Return a JSON object with exactly these fields:
{{
  "name": "short lowercase provider name (e.g. qwen, groq, mistral)",
  "provider_type": "custom",
  "base_url": "the API base URL for chat completions (e.g. https://api.example.com/v1)",
  "is_openai_compatible": true/false (does it follow OpenAI's chat completion API format?),
  "models": ["list", "of", "available", "model", "names"],
  "auth_header": "how to pass the API key (e.g. 'Authorization: Bearer' or 'x-api-key')"
}}

Important rules:
- The base_url should be the URL prefix BEFORE /chat/completions. For example, if the full endpoint is https://api.example.com/v1/chat/completions, the base_url is https://api.example.com/v1
- If the API follows the OpenAI format (same request/response structure for /chat/completions), set is_openai_compatible to true
- List actual model names mentioned in the documentation
- Only return the JSON object, nothing else

Documentation content:
{doc_content}"#
    )
}

fn parse_llm_response(text: &str) -> Option<SuggestedConfig> {
    // Try to parse directly
    if let Ok(config) = serde_json::from_str::<SuggestedConfig>(text) {
        return Some(config);
    }

    // Try to extract JSON from markdown code fences
    let trimmed = text.trim();
    let json_str = if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    serde_json::from_str::<SuggestedConfig>(json_str).ok()
}

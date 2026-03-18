use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::providers::sse::parse_sse_stream;
use crate::providers::{ChunkStream, Provider};
use crate::types::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    Choice, ChoiceMessage, ChunkChoice, ChunkDelta, ChunkFunctionCall, ChunkToolCall,
    ContentPart, FunctionCall, Message, MessageContent, ToolCall, Usage,
};
use crate::types::provider::ProviderType;

pub struct AnthropicProvider {
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    pub client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(name: String, base_url: String, api_key: String) -> Self {
        Self {
            provider_name: name,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

// ── Anthropic native types ──

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicResponseBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── Conversion: OpenAI -> Anthropic ──

fn convert_request(req: &ChatCompletionRequest) -> AnthropicRequest {
    let mut system = None;
    let mut messages = Vec::new();

    for msg in &req.messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(ref content) = msg.content {
                    system = Some(content.as_text());
                }
            }
            "user" => {
                let content = convert_user_content(msg);
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content,
                });
            }
            "assistant" => {
                let content = convert_assistant_content(msg);
                messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content,
                });
            }
            "tool" => {
                let tool_call_id = msg.tool_call_id.clone().unwrap_or_default();
                let text = msg
                    .content
                    .as_ref()
                    .map(|c| c.as_text())
                    .unwrap_or_default();
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Blocks(vec![AnthropicBlock::ToolResult {
                        tool_use_id: tool_call_id,
                        content: text,
                    }]),
                });
            }
            _ => {}
        }
    }

    // Merge consecutive same-role messages (Anthropic requires alternating roles)
    messages = merge_consecutive_messages(messages);

    let tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                input_schema: t
                    .function
                    .parameters
                    .clone()
                    .unwrap_or(serde_json::json!({"type": "object", "properties": {}})),
            })
            .collect()
    });

    AnthropicRequest {
        model: req.model.clone(),
        messages,
        max_tokens: req.max_tokens.unwrap_or(4096),
        system,
        temperature: req.temperature,
        top_p: req.top_p,
        stop_sequences: match &req.stop {
            Some(crate::types::chat::StopCondition::Single(s)) => Some(vec![s.clone()]),
            Some(crate::types::chat::StopCondition::Multiple(v)) => Some(v.clone()),
            None => None,
        },
        tools,
        tool_choice: req.tool_choice.clone(),
        stream: req.stream,
    }
}

fn convert_user_content(msg: &Message) -> AnthropicContent {
    match &msg.content {
        Some(MessageContent::Text(t)) => AnthropicContent::Text(t.clone()),
        Some(MessageContent::Parts(parts)) => {
            let blocks: Vec<AnthropicBlock> = parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => AnthropicBlock::Text { text: text.clone() },
                    ContentPart::ImageUrl { image_url } => {
                        // Try to extract base64 data from data: URLs
                        if let Some(rest) = image_url.url.strip_prefix("data:") {
                            if let Some((media_type, data)) = rest.split_once(";base64,") {
                                return AnthropicBlock::Image {
                                    source: AnthropicImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: media_type.to_string(),
                                        data: data.to_string(),
                                    },
                                };
                            }
                        }
                        // Fallback: treat URL as text
                        AnthropicBlock::Text {
                            text: format!("[Image: {}]", image_url.url),
                        }
                    }
                })
                .collect();
            AnthropicContent::Blocks(blocks)
        }
        None => AnthropicContent::Text(String::new()),
    }
}

fn convert_assistant_content(msg: &Message) -> AnthropicContent {
    let mut blocks = Vec::new();

    if let Some(ref content) = msg.content {
        let text = content.as_text();
        if !text.is_empty() {
            blocks.push(AnthropicBlock::Text { text });
        }
    }

    if let Some(ref tool_calls) = msg.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
            blocks.push(AnthropicBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
    }

    if blocks.is_empty() {
        AnthropicContent::Text(String::new())
    } else if blocks.len() == 1 {
        if let AnthropicBlock::Text { ref text } = blocks[0] {
            return AnthropicContent::Text(text.clone());
        }
        AnthropicContent::Blocks(blocks)
    } else {
        AnthropicContent::Blocks(blocks)
    }
}

fn merge_consecutive_messages(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    let mut merged: Vec<AnthropicMessage> = Vec::new();

    for msg in messages {
        if let Some(last) = merged.last_mut() {
            if last.role == msg.role {
                // Merge content blocks
                let existing_blocks = match &last.content {
                    AnthropicContent::Text(t) => vec![AnthropicBlock::Text { text: t.clone() }],
                    AnthropicContent::Blocks(b) => b.clone(),
                };
                let new_blocks = match msg.content {
                    AnthropicContent::Text(t) => vec![AnthropicBlock::Text { text: t }],
                    AnthropicContent::Blocks(b) => b,
                };
                let mut combined = existing_blocks;
                combined.extend(new_blocks);
                last.content = AnthropicContent::Blocks(combined);
                continue;
            }
        }
        merged.push(msg);
    }

    merged
}

// ── Conversion: Anthropic -> OpenAI ──

fn convert_response(resp: AnthropicResponse) -> ChatCompletionResponse {
    let mut content_text = String::new();
    let mut tool_calls = Vec::new();
    let mut tc_index = 0u32;

    for block in &resp.content {
        match block {
            AnthropicResponseBlock::Text { text } => {
                content_text.push_str(text);
            }
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                });
                tc_index += 1;
            }
        }
    }
    let _ = tc_index;

    let finish_reason = match resp.stop_reason.as_deref() {
        Some("end_turn") | Some("stop") => Some("stop".to_string()),
        Some("tool_use") => Some("tool_calls".to_string()),
        Some("max_tokens") => Some("length".to_string()),
        other => other.map(|s| s.to_string()),
    };

    ChatCompletionResponse {
        id: resp.id,
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: resp.model,
        choices: vec![Choice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".to_string(),
                content: if content_text.is_empty() {
                    None
                } else {
                    Some(content_text)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason,
        }],
        usage: Some(Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
    }
}

// ── Provider implementation ──

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    async fn complete(&self, req: &ChatCompletionRequest) -> AppResult<ChatCompletionResponse> {
        let mut anthropic_req = convert_request(req);
        anthropic_req.stream = false;

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&anthropic_req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::ProviderError(format!(
                "Anthropic API error ({}): {}",
                status, body
            )));
        }

        let anthropic_resp: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| AppError::ProviderError(format!("Failed to parse response: {e}")))?;

        Ok(convert_response(anthropic_resp))
    }

    async fn stream(&self, req: &ChatCompletionRequest) -> AppResult<ChunkStream> {
        let mut anthropic_req = convert_request(req);
        anthropic_req.stream = true;

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&anthropic_req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::ProviderError(format!(
                "Anthropic API error ({}): {}",
                status, body
            )));
        }

        let model = req.model.clone();
        let byte_stream = Box::pin(resp.bytes_stream());
        let sse_stream = parse_sse_stream(byte_stream);

        // Streaming state machine
        let chunk_stream = sse_stream.filter_map(move |event| {
            let model = model.clone();
            async move {
                let data: serde_json::Value = match serde_json::from_str(&event.data) {
                    Ok(v) => v,
                    Err(_) => return None,
                };

                match event.event.as_str() {
                    "message_start" => {
                        let msg_id = data["message"]["id"]
                            .as_str()
                            .unwrap_or("msg_unknown")
                            .to_string();
                        // Capture input_tokens from message_start
                        let usage = data["message"].get("usage").and_then(|u| {
                            let input = u["input_tokens"].as_u64().unwrap_or(0) as u32;
                            Some(Usage {
                                prompt_tokens: input,
                                completion_tokens: 0,
                                total_tokens: input,
                            })
                        });
                        Some(Ok(ChatCompletionChunk {
                            id: msg_id,
                            object: "chat.completion.chunk".to_string(),
                            created: chrono::Utc::now().timestamp(),
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                            usage,
                        }))
                    }
                    "content_block_start" => {
                        let block_type = data["content_block"]["type"].as_str().unwrap_or("");
                        if block_type == "tool_use" {
                            let index = data["index"].as_u64().unwrap_or(0) as u32;
                            let id = data["content_block"]["id"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            let name = data["content_block"]["name"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            Some(Ok(ChatCompletionChunk {
                                id: String::new(),
                                object: "chat.completion.chunk".to_string(),
                                created: chrono::Utc::now().timestamp(),
                                model: model.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta {
                                        role: None,
                                        content: None,
                                        tool_calls: Some(vec![ChunkToolCall {
                                            index,
                                            id: Some(id),
                                            call_type: Some("function".to_string()),
                                            function: Some(ChunkFunctionCall {
                                                name: Some(name),
                                                arguments: None,
                                            }),
                                        }]),
                                    },
                                    finish_reason: None,
                                }],
                                usage: None,
                            }))
                        } else {
                            None
                        }
                    }
                    "content_block_delta" => {
                        let delta_type = data["delta"]["type"].as_str().unwrap_or("");
                        match delta_type {
                            "text_delta" => {
                                let text = data["delta"]["text"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                Some(Ok(ChatCompletionChunk {
                                    id: String::new(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: chrono::Utc::now().timestamp(),
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: Some(text),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                }))
                            }
                            "input_json_delta" => {
                                let index = data["index"].as_u64().unwrap_or(0) as u32;
                                let partial = data["delta"]["partial_json"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                Some(Ok(ChatCompletionChunk {
                                    id: String::new(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: chrono::Utc::now().timestamp(),
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: None,
                                            tool_calls: Some(vec![ChunkToolCall {
                                                index,
                                                id: None,
                                                call_type: None,
                                                function: Some(ChunkFunctionCall {
                                                    name: None,
                                                    arguments: Some(partial),
                                                }),
                                            }]),
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                }))
                            }
                            _ => None,
                        }
                    }
                    "message_delta" => {
                        let stop_reason = data["delta"]["stop_reason"]
                            .as_str()
                            .unwrap_or("stop");
                        let finish_reason = match stop_reason {
                            "end_turn" | "stop" => "stop",
                            "tool_use" => "tool_calls",
                            "max_tokens" => "length",
                            other => other,
                        };
                        // Extract usage from message_delta
                        let usage = data.get("usage").and_then(|u| {
                            let output = u["output_tokens"].as_u64().unwrap_or(0) as u32;
                            let input = u["input_tokens"].as_u64().unwrap_or(0) as u32;
                            Some(Usage {
                                prompt_tokens: input,
                                completion_tokens: output,
                                total_tokens: input + output,
                            })
                        });
                        Some(Ok(ChatCompletionChunk {
                            id: String::new(),
                            object: "chat.completion.chunk".to_string(),
                            created: chrono::Utc::now().timestamp(),
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                    tool_calls: None,
                                },
                                finish_reason: Some(finish_reason.to_string()),
                            }],
                            usage,
                        }))
                    }
                    "message_stop" | "ping" | "content_block_stop" => None,
                    _ => None,
                }
            }
        });

        Ok(Box::pin(chunk_stream))
    }

    async fn list_models(&self) -> AppResult<Vec<String>> {
        // Anthropic doesn't have a models list API; return well-known models
        Ok(vec![
            "claude-sonnet-4-20250514".to_string(),
            "claude-haiku-4-5-20251001".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
        ])
    }
}

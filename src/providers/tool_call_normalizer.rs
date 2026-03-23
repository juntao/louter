//! Tool call response normalizer.
//!
//! Detects tool calls embedded as text in model responses (common with local models
//! like Qwen, Hermes, Llama) and converts them to proper OpenAI-format `tool_calls`.
//!
//! Supported formats:
//! - Hermes: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
//! - Qwen:   `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` (same as Hermes)
//! - Generic JSON: `{"name": "...", "arguments": {...}}` in content
//! - Action/Input: `Action: tool_name\nAction Input: {...}` (ReAct format)

use crate::types::chat::{ChatCompletionResponse, FunctionCall, ToolCall};
use regex::Regex;
use std::sync::LazyLock;

/// Regex for Hermes/Qwen `<tool_call>...</tool_call>` format.
static TOOL_CALL_XML_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>").unwrap()
});

/// Regex for `<|tool_call|>` variant (some Qwen models).
static TOOL_CALL_PIPE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<\|tool_call\|>\s*(\{[\s\S]*?\})\s*(?:<\|/tool_call\|>|$)").unwrap()
});

/// Regex for ReAct format: Action: name\nAction Input: {...}
static REACT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{[\s\S]*?\})\s*(?:\n|$)").unwrap()
});

/// Normalize a ChatCompletionResponse from a local model.
///
/// If the response has text content containing embedded tool calls but no
/// proper `tool_calls` field, extract and convert them.
pub fn normalize_response(mut response: ChatCompletionResponse) -> ChatCompletionResponse {
    for choice in &mut response.choices {
        // Skip if already has proper tool_calls
        if choice
            .message
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty())
        {
            continue;
        }

        let content = match &choice.message.content {
            Some(c) if !c.trim().is_empty() => c.clone(),
            _ => continue,
        };

        // Try to extract tool calls from content
        let (extracted_calls, remaining_content) = extract_tool_calls(&content);

        if !extracted_calls.is_empty() {
            choice.message.tool_calls = Some(extracted_calls);
            // Update content: keep non-tool-call text, or clear if only tool calls
            let remaining = remaining_content.trim();
            if remaining.is_empty() {
                choice.message.content = None;
            } else {
                choice.message.content = Some(remaining.to_string());
            }
            // Fix finish_reason
            if choice.finish_reason.as_deref() != Some("tool_calls") {
                choice.finish_reason = Some("tool_calls".to_string());
            }
        }
    }

    response
}

/// Extract tool calls from text content.
///
/// Returns (extracted_tool_calls, remaining_content_after_extraction).
fn extract_tool_calls(content: &str) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut remaining = content.to_string();

    // 1. Try Hermes/Qwen XML format: <tool_call>{...}</tool_call>
    for cap in TOOL_CALL_XML_RE.captures_iter(content) {
        if let Some(json_str) = cap.get(1) {
            if let Some(call) = parse_tool_call_json(json_str.as_str(), calls.len()) {
                calls.push(call);
            }
        }
    }
    if !calls.is_empty() {
        remaining = TOOL_CALL_XML_RE.replace_all(&remaining, "").to_string();
        return (calls, remaining);
    }

    // 2. Try pipe format: <|tool_call|>{...}<|/tool_call|>
    for cap in TOOL_CALL_PIPE_RE.captures_iter(content) {
        if let Some(json_str) = cap.get(1) {
            if let Some(call) = parse_tool_call_json(json_str.as_str(), calls.len()) {
                calls.push(call);
            }
        }
    }
    if !calls.is_empty() {
        remaining = TOOL_CALL_PIPE_RE.replace_all(&remaining, "").to_string();
        return (calls, remaining);
    }

    // 3. Try ReAct format: Action: name\nAction Input: {...}
    for cap in REACT_RE.captures_iter(content) {
        if let (Some(name), Some(args)) = (cap.get(1), cap.get(2)) {
            let call = ToolCall {
                id: format!("call_{}", calls.len()),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.as_str().to_string(),
                    arguments: args.as_str().to_string(),
                },
            };
            calls.push(call);
        }
    }
    if !calls.is_empty() {
        remaining = REACT_RE.replace_all(&remaining, "").to_string();
        return (calls, remaining);
    }

    // 4. Try bare JSON object with "name" and "arguments" fields
    if let Some(call) = try_parse_bare_json(content) {
        remaining = String::new();
        return (vec![call], remaining);
    }

    (calls, remaining)
}

/// Parse a JSON string into a ToolCall.
fn parse_tool_call_json(json_str: &str, index: usize) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = v.get("name")?.as_str()?;
    let arguments = v.get("arguments")?;

    let args_str = if arguments.is_string() {
        arguments.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(arguments).ok()?
    };

    Some(ToolCall {
        id: format!("call_{index}"),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: args_str,
        },
    })
}

/// Try to parse the entire content as a bare JSON tool call.
fn try_parse_bare_json(content: &str) -> Option<ToolCall> {
    let trimmed = content.trim();
    // Must start with { and end with }
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return None;
    }
    parse_tool_call_json(trimmed, 0)
}

/// Normalize a streaming chunk's content similarly.
/// For streaming, we can only detect complete tool call patterns in
/// accumulated content, so this is primarily for non-streaming use.
/// Streaming tool calls from Ollama should already be in OpenAI format
/// since Ollama's /v1 API handles this.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{ChatCompletionResponse, Choice, ChoiceMessage, Usage};

    fn make_response(content: &str) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".to_string(),
                    content: Some(content.to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        }
    }

    #[test]
    fn test_hermes_format() {
        let resp = make_response(
            r#"<tool_call>{"name": "web_search", "arguments": {"query": "latest news"}}</tool_call>"#,
        );
        let normalized = normalize_response(resp);
        let tc = normalized.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "web_search");
        assert!(tc[0].function.arguments.contains("latest news"));
        assert_eq!(
            normalized.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert!(normalized.choices[0].message.content.is_none());
    }

    #[test]
    fn test_qwen_pipe_format() {
        let resp = make_response(
            r#"<|tool_call|>{"name": "exec", "arguments": {"command": "ls -la"}}<|/tool_call|>"#,
        );
        let normalized = normalize_response(resp);
        let tc = normalized.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "exec");
    }

    #[test]
    fn test_react_format() {
        let resp = make_response(
            "I need to search for this.\nAction: web_search\nAction Input: {\"query\": \"test\"}\n",
        );
        let normalized = normalize_response(resp);
        let tc = normalized.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "web_search");
        // Should preserve the thinking text
        let content = normalized.choices[0].message.content.as_ref().unwrap();
        assert!(content.contains("I need to search"));
    }

    #[test]
    fn test_multiple_tool_calls() {
        let resp = make_response(
            r#"<tool_call>{"name": "web_search", "arguments": {"query": "a"}}</tool_call>
<tool_call>{"name": "web_fetch", "arguments": {"url": "https://example.com"}}</tool_call>"#,
        );
        let normalized = normalize_response(resp);
        let tc = normalized.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 2);
        assert_eq!(tc[0].function.name, "web_search");
        assert_eq!(tc[1].function.name, "web_fetch");
    }

    #[test]
    fn test_no_tool_calls() {
        let resp = make_response("Just a normal text response");
        let normalized = normalize_response(resp);
        assert!(normalized.choices[0].message.tool_calls.is_none());
        assert_eq!(
            normalized.choices[0].message.content.as_deref(),
            Some("Just a normal text response")
        );
    }

    #[test]
    fn test_already_has_tool_calls() {
        let mut resp = make_response("some content");
        resp.choices[0].message.tool_calls = Some(vec![ToolCall {
            id: "existing".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "existing_tool".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        let normalized = normalize_response(resp);
        let tc = normalized.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "existing"); // Should not be modified
    }

    #[test]
    fn test_bare_json() {
        let resp =
            make_response(r#"{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}"#);
        let normalized = normalize_response(resp);
        let tc = normalized.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "read_file");
    }
}

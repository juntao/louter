use std::collections::HashMap;
use std::sync::Arc;

use crate::providers::Provider;
use crate::router::static_router::ProviderRegistry;
use crate::types::chat::Message;

/// Classify the last user message into a category.
pub fn classify(messages: &[Message]) -> &'static str {
    let text = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .map(|c| c.as_text())
        .unwrap_or_default();

    let text_lower = text.to_lowercase();

    // 1. Code detection
    if text.contains("```")
        || has_any_word(&text_lower, &[
            "function", "import ", "const ", "var ", "fn ", "pub ", "async",
            "=>", "compile", "runtime", "debug",
        ])
        || has_any_word(&text, &["def ", "class "])
        || has_any_word(&text_lower, &["error", "bug", "fix"])
    {
        return "code";
    }

    // 2. Math detection
    if has_any_char(&text, &['∑', '∫', '∂', '√'])
        || has_any_word(&text, &["\\frac", "\\sum", "\\int"])
        || (text.contains('$') && text.matches('$').count() >= 2)
        || has_any_word(&text_lower, &[
            "equation", "theorem", "proof", "calculate",
        ])
        || has_any_word(&text, &["公式", "方程", "计算"])
    {
        return "math";
    }

    // 3. Translation detection
    if has_any_word(&text_lower, &["translate", "translation"])
        || text.contains("翻译")
    {
        return "translation";
    }

    // 4. Default
    "general"
}

fn has_any_word(text: &str, words: &[&str]) -> bool {
    words.iter().any(|w| text.contains(w))
}

fn has_any_char(text: &str, chars: &[char]) -> bool {
    chars.iter().any(|c| text.contains(*c))
}

/// Route based on smart classification.
///
/// Returns `(provider, model_name)` if a matching provider is found.
pub async fn smart_route(
    config: &HashMap<String, String>,
    registry: &ProviderRegistry,
    messages: &[Message],
) -> Option<(Arc<dyn Provider>, String)> {
    let category = classify(messages);
    let value = config.get(category).or_else(|| config.get("general"))?;

    let (provider_name, model_name) = value.split_once('/')?;

    let all = registry.all().await;
    let (_, provider) = all
        .iter()
        .find(|(_, p)| p.name().eq_ignore_ascii_case(provider_name))?;

    Some((provider.clone(), model_name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{Message, MessageContent};

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_classify_code() {
        let msgs = vec![msg("user", "Help me write a Python function to sort a list")];
        assert_eq!(classify(&msgs), "code");

        let msgs = vec![msg("user", "```rust\nfn main() {}\n```")];
        assert_eq!(classify(&msgs), "code");

        let msgs = vec![msg("user", "I have a bug in my code")];
        assert_eq!(classify(&msgs), "code");
    }

    #[test]
    fn test_classify_math() {
        let msgs = vec![msg("user", "Solve this equation: x² + 2x + 1 = 0")];
        assert_eq!(classify(&msgs), "math");

        let msgs = vec![msg("user", "请帮我解这个方程")];
        assert_eq!(classify(&msgs), "math");

        let msgs = vec![msg("user", "What is $\\frac{1}{2} + \\frac{1}{3}$?")];
        assert_eq!(classify(&msgs), "math");
    }

    #[test]
    fn test_classify_translation() {
        let msgs = vec![msg("user", "Translate this to Chinese")];
        assert_eq!(classify(&msgs), "translation");

        let msgs = vec![msg("user", "请翻译这段话")];
        assert_eq!(classify(&msgs), "translation");
    }

    #[test]
    fn test_classify_general() {
        let msgs = vec![msg("user", "今天天气怎么样")];
        assert_eq!(classify(&msgs), "general");

        let msgs = vec![msg("user", "Tell me a joke")];
        assert_eq!(classify(&msgs), "general");
    }

    #[test]
    fn test_classify_uses_last_user_message() {
        let msgs = vec![
            msg("user", "Hello"),
            msg("assistant", "Hi there!"),
            msg("user", "Help me debug this error"),
        ];
        assert_eq!(classify(&msgs), "code");
    }
}

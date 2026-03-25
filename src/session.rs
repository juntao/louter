//! Session-level routing — keeps the same model throughout a conversation,
//! with smart escalation to cloud when local model can't handle it.
//!
//! Session detection: hash the conversation's system prompt + first user message
//! to create a fingerprint. Same fingerprint = same session = same model.
//!
//! Escalation: if local model fails mid-session (bad response, retry detected),
//! escalate to cloud for all remaining turns. Never downgrade mid-session.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;
use tokio::sync::Mutex;

use crate::types::chat::Message;

/// Routing decision cached for a session.
#[derive(Clone, Debug)]
pub struct SessionDecision {
    /// "local" or "cloud"
    pub target: String,
    /// Was this session escalated from local to cloud?
    pub escalated: bool,
    /// When this decision was made
    pub created_at: Instant,
    /// How many requests have used this decision
    pub request_count: u32,
}

/// Session-aware router that keeps routing consistent within a conversation.
pub struct SessionRouter {
    /// Map of session fingerprint → routing decision
    sessions: Mutex<HashMap<u64, SessionDecision>>,
    /// Sessions expire after this many seconds of inactivity
    ttl_secs: u64,
    /// Max sessions to cache
    max_sessions: usize,
}

impl SessionRouter {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            ttl_secs: 600, // 10 minutes
            max_sessions: 256,
        }
    }

    /// Check if we have a cached routing decision for this conversation.
    ///
    /// Returns `Some(target)` if a session exists, `None` if this is a new session.
    pub async fn get_session_target(&self, messages: &[Message]) -> Option<String> {
        let fp = compute_session_fingerprint(messages);
        let mut sessions = self.sessions.lock().await;

        // Prune expired sessions
        let now = Instant::now();
        sessions.retain(|_, v| now.duration_since(v.created_at).as_secs() < self.ttl_secs);

        if let Some(decision) = sessions.get_mut(&fp) {
            decision.request_count += 1;
            Some(decision.target.clone())
        } else {
            None
        }
    }

    /// Record a routing decision for a new session.
    pub async fn set_session_target(&self, messages: &[Message], target: &str) {
        let fp = compute_session_fingerprint(messages);
        let mut sessions = self.sessions.lock().await;

        // Evict oldest if at capacity
        if sessions.len() >= self.max_sessions {
            if let Some(oldest_key) = sessions
                .iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| *k)
            {
                sessions.remove(&oldest_key);
            }
        }

        sessions.insert(
            fp,
            SessionDecision {
                target: target.to_string(),
                escalated: false,
                created_at: Instant::now(),
                request_count: 1,
            },
        );
    }

    /// Escalate a session from local to cloud.
    ///
    /// Called when local model fails mid-conversation. All subsequent requests
    /// in this session will be routed to cloud.
    pub async fn escalate_to_cloud(&self, messages: &[Message]) {
        let fp = compute_session_fingerprint(messages);
        let mut sessions = self.sessions.lock().await;

        if let Some(decision) = sessions.get_mut(&fp) {
            if decision.target != "cloud" {
                tracing::info!(
                    "Session escalated: local → cloud (after {} requests)",
                    decision.request_count
                );
                decision.target = "cloud".to_string();
                decision.escalated = true;
            }
        }
    }

    /// Get session stats for the admin API.
    pub async fn get_stats(&self) -> SessionStats {
        let sessions = self.sessions.lock().await;
        let now = Instant::now();

        let active: Vec<&SessionDecision> = sessions
            .values()
            .filter(|s| now.duration_since(s.created_at).as_secs() < self.ttl_secs)
            .collect();

        let local_count = active.iter().filter(|s| s.target == "local").count();
        let cloud_count = active.iter().filter(|s| s.target == "cloud").count();
        let escalated_count = active.iter().filter(|s| s.escalated).count();
        let total_requests: u32 = active.iter().map(|s| s.request_count).sum();

        SessionStats {
            active_sessions: active.len(),
            local_sessions: local_count,
            cloud_sessions: cloud_count,
            escalated_sessions: escalated_count,
            total_requests,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionStats {
    pub active_sessions: usize,
    pub local_sessions: usize,
    pub cloud_sessions: usize,
    pub escalated_sessions: usize,
    pub total_requests: u32,
}

/// Compute a session fingerprint from the conversation.
///
/// Uses system prompt + first user message to identify the session.
/// This stays stable across turns as new messages are appended.
fn compute_session_fingerprint(messages: &[Message]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Hash the system prompt (identifies the agent/config)
    if let Some(sys_msg) = messages.iter().find(|m| m.role == "system") {
        if let Some(ref content) = sys_msg.content {
            let text = content.as_text();
            // Use first 200 chars of system prompt (stable across turns)
            let prefix = if text.len() > 200 { &text[..200] } else { &text };
            prefix.hash(&mut hasher);
        }
    }

    // Hash the first user message (identifies the conversation topic)
    if let Some(first_user) = messages.iter().find(|m| m.role == "user") {
        if let Some(ref content) = first_user.content {
            let text = content.as_text();
            let prefix = if text.len() > 200 { &text[..200] } else { &text };
            prefix.hash(&mut hasher);
        }
    }

    hasher.finish()
}

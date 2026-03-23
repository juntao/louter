//! Implicit feedback detection from request patterns.
//!
//! Detects when an agent retries a request (similar request within a short window),
//! which signals that the previous response was unsatisfactory.
//! This provides free success/failure labels for training samples.

use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::Mutex;

/// A recent request entry for similarity comparison.
#[derive(Clone)]
#[allow(dead_code)]
struct RecentRequest {
    /// Hash of the last user message content (for fast comparison)
    content_hash: u64,
    /// Task type classification
    task_type: String,
    /// Training sample ID (if a sample was saved)
    sample_id: Option<String>,
    /// When this request was received
    timestamp: Instant,
    /// Source: "local" or "cloud"
    source: String,
}

/// Tracks recent requests to detect retry patterns.
pub struct FeedbackTracker {
    /// Recent requests, newest last. Bounded size.
    recent: Mutex<VecDeque<RecentRequest>>,
    /// Window in seconds within which a similar request counts as a retry
    retry_window_secs: u64,
    /// Max entries to keep
    max_entries: usize,
}

impl FeedbackTracker {
    pub fn new() -> Self {
        Self {
            recent: Mutex::new(VecDeque::with_capacity(64)),
            retry_window_secs: 60,
            max_entries: 64,
        }
    }

    /// Record a new request and check if it's a retry of a recent request.
    ///
    /// Returns `Some(sample_id)` if this request is a retry of a previous one
    /// that had a training sample — the caller should mark that sample as failed.
    pub async fn record_and_check_retry(
        &self,
        content_hash: u64,
        task_type: &str,
        sample_id: Option<String>,
        source: &str,
    ) -> Option<String> {
        let mut recent = self.recent.lock().await;
        let now = Instant::now();

        // Prune old entries
        while recent.front().is_some_and(|r| now.duration_since(r.timestamp).as_secs() > 300) {
            recent.pop_front();
        }

        // Check if this is a retry: same content hash within the retry window
        let mut failed_sample_id = None;
        for prev in recent.iter().rev() {
            if now.duration_since(prev.timestamp).as_secs() > self.retry_window_secs {
                break;
            }
            if prev.content_hash == content_hash {
                // This is a retry! The previous response was bad.
                if let Some(ref sid) = prev.sample_id {
                    failed_sample_id = Some(sid.clone());
                    tracing::info!(
                        "Implicit feedback: retry detected for sample {}, marking as failed",
                        sid
                    );
                }
                break;
            }
        }

        // Record this request
        let entry = RecentRequest {
            content_hash,
            task_type: task_type.to_string(),
            sample_id,
            timestamp: now,
            source: source.to_string(),
        };
        recent.push_back(entry);

        // Trim to max size
        while recent.len() > self.max_entries {
            recent.pop_front();
        }

        failed_sample_id
    }
}

/// Compute a simple hash of the user's last message for similarity detection.
pub fn hash_last_user_message(messages: &[crate::types::chat::Message]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .map(|c| c.as_text())
        .unwrap_or_default();

    // Use first 500 chars for hashing (avoid hashing huge contexts)
    let text = if last_user.len() > 500 {
        &last_user[..500]
    } else {
        &last_user
    };

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

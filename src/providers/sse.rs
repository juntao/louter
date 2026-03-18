use bytes::Bytes;
use futures::stream::BoxStream;
use futures::StreamExt;

/// Parse a byte stream into SSE events.
/// Returns a stream of (event_type, data) tuples.
pub fn parse_sse_stream(
    byte_stream: BoxStream<'static, Result<Bytes, reqwest::Error>>,
) -> BoxStream<'static, SseEvent> {
    let mut buffer = String::new();

    let stream = byte_stream.flat_map(move |chunk| {
        let mut events = Vec::new();

        match chunk {
            Ok(bytes) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes));

                // Process complete lines
                while let Some(pos) = buffer.find("\n\n") {
                    let block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    let event = parse_sse_block(&block);
                    if let Some(evt) = event {
                        events.push(evt);
                    }
                }
            }
            Err(e) => {
                events.push(SseEvent {
                    event: "error".to_string(),
                    data: e.to_string(),
                });
            }
        }

        futures::stream::iter(events)
    });

    Box::pin(stream)
}

#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
}

fn parse_sse_block(block: &str) -> Option<SseEvent> {
    let mut event_type = String::new();
    let mut data_lines = Vec::new();

    for line in block.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            event_type = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data:") {
            let value = value.trim();
            if value == "[DONE]" {
                return Some(SseEvent {
                    event: "done".to_string(),
                    data: "[DONE]".to_string(),
                });
            }
            data_lines.push(value.to_string());
        } else if line.starts_with(':') {
            // comment, ignore
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    let data = data_lines.join("\n");

    if event_type.is_empty() {
        event_type = "message".to_string();
    }

    Some(SseEvent { event: event_type, data })
}

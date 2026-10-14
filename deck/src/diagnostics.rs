use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEntry {
    pub ts: DateTime<Utc>,
    pub address: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    errors: VecDeque<ErrorEntry>,
    sent_log: VecDeque<String>,
    max_errors: usize,
    max_sent: usize,
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self {
            errors: VecDeque::new(),
            sent_log: VecDeque::new(),
            max_errors: 100,
            max_sent: 100,
        }
    }
}

impl Diagnostics {
    pub fn add_error(&mut self, address: String, reason: String) {
        self.errors.push_front(ErrorEntry {
            ts: Utc::now(),
            address,
            reason,
        });
        while self.errors.len() > self.max_errors {
            self.errors.pop_back();
        }
    }

    pub fn add_sent(&mut self, msg: String) {
        self.sent_log.push_front(msg);
        while self.sent_log.len() > self.max_sent {
            self.sent_log.pop_back();
        }
    }

    pub fn errors_text(&self) -> String {
        let lines: Vec<String> = self
            .errors
            .iter()
            .map(|e| {
                format!(
                    "[{}] {}: {}",
                    e.ts.format("%H:%M:%S%.3f"),
                    e.address,
                    e.reason
                )
            })
            .collect();
        lines.join("\n")
    }

    pub fn sent_text(&self) -> String {
        self.sent_log.iter().cloned().collect::<Vec<_>>().join("\n")
    }
}

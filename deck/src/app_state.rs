use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnState {
    Disconnected,
    WaitingForHeartbeat,
    Connected,
    Stale,
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub conn_state: ConnState,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub fps: f32,
    pub frames: i32,
    pub output_name: String,
    pub confirmed_input: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: i32,
    pub guidance: f32,
    pub delta: f32,
    pub maxfps: i32,
    pub steps: Vec<i32>,
    pub input_text: String,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            conn_state: ConnState::Disconnected,
            last_heartbeat: None,
            fps: 0.0,
            frames: 0,
            output_name: String::new(),
            confirmed_input: String::new(),
            prompt: String::new(),
            negative_prompt: String::new(),
            seed: -1,
            guidance: 1.0,
            delta: 1.0,
            maxfps: 60,
            steps: vec![17],
            input_text: String::new(),
        }
    }
}

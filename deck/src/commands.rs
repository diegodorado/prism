#[derive(Debug, Clone)]
pub enum Command {
    Play,
    Pause,
    Stop,
    Unload,
    ApplyPrompts {
        prompt: String,
        negative: String,
    },
    ApplyParams {
        seed: i32,
        guidance: f32,
        delta: f32,
        maxfps: i32,
        steps: Vec<i32>,
    },
    ApplyInput {
        name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamKind {
    Seed,
    Guidance,
    Delta,
    Maxfps,
    Steps,
}

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
    ApplyInput {
        name: String,
    },
    /// A slider/text edit produced a new value. Debounced before sending.
    SetParam {
        kind: ParamKind,
        value: f64,
    },
}

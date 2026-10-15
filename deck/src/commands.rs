#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamKind {
    Seed,
    Guidance,
    Delta,
    Maxfps,
    Steps,
}

/// Text fields that are debounced before sending. Prompts can be long, so we
/// do not want to emit an OSC message on every keystroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextKind {
    Prompt,
    Negative,
}

#[derive(Debug, Clone)]
pub enum Command {
    Play,
    Pause,
    Stop,
    Unload,
    ApplyInput {
        name: String,
    },
    /// A control (slider) produced a new value. Debounced before sending.
    SetParam {
        kind: ParamKind,
        value: f64,
    },
    /// A text field (prompt / negative) changed. Debounced before sending.
    SetText {
        kind: TextKind,
        value: String,
    },
}

use std::path::PathBuf;

/// Directory where diagnostics logs are written. On macOS this is
/// `~/Library/Logs/prism-deck`; other platforms fall back to the local data /
/// temp directory. The path is also logged to stderr on startup so users can
/// find it.
pub fn log_dir() -> PathBuf {
    dirs::state_dir()
        .or_else(dirs::data_local_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("prism-deck")
        .join("logs")
}

/// Initialise `tracing` to write to a rolling file and to stderr.
///
/// Returns the guard that flushes the file appender on drop; it must be kept
/// alive for the duration of the program.
pub fn init() -> anyhow::Result<tracing_appender::non_blocking::WorkerGuard> {
    let dir = log_dir();
    std::fs::create_dir_all(&dir)?;
    let appender = tracing_appender::rolling::daily(&dir, "prism-deck.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(appender);

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    eprintln!("prism-deck: logging to {}", dir.display());
    Ok(guard)
}

//! `openjiuwen` CLI binary and run/repl front-end.

fn main() -> std::process::ExitCode {
    println!("openjiuwen {}", env!("CARGO_PKG_VERSION"));
    std::process::ExitCode::SUCCESS
}

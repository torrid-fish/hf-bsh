mod api;
mod fmt;
mod progress;
mod shell;

use std::io::{BufRead, IsTerminal};
use std::path::PathBuf;
use std::process::ExitCode;
use std::rc::Rc;

use rustyline::config::Configurer;
use rustyline::error::ReadlineError;
use rustyline::{CompletionType, Editor};

use crate::api::Client;
use crate::shell::{Shell, ShellHelper};

const USAGE: &str = "\
hf-bsh — interactive bucket shell for the Hugging Face Hub.

Usage:
  hf-bsh [options] [<target>]
  hf bsh [options] [<target>]            (when installed as an `hf` extension)

Target:
  buckets/<ns>/<name>        bucket to open (read/write)

External sources for `cp`:
  hf://buckets/<ns>/<name>/<path>        other bucket (server-side xet copy)
  hf://datasets/<repo>/<path>            dataset file (server-side xet copy)
  hf://models/<repo>/<path>              model file   (server-side xet copy)

Options:
  --endpoint <URL>           override Hub endpoint (or set $HF_ENDPOINT)
  --token <TOKEN>            override auth token (or set $HF_TOKEN)
  -h, --help                 print this help
  -V, --version              print version

Authentication:
  Reads $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN, or the file at
  $HF_TOKEN_PATH / $HF_HOME/token / ~/.cache/huggingface/token.
";

struct CliArgs {
    target: Option<String>,
    endpoint: Option<String>,
    token: Option<String>,
}

fn parse_args() -> Result<CliArgs, String> {
    let mut it = std::env::args().skip(1);
    let mut out = CliArgs { target: None, endpoint: None, token: None };
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{}", USAGE);
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("hf-bsh {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--endpoint" => {
                out.endpoint =
                    Some(it.next().ok_or_else(|| "--endpoint requires a value".to_string())?);
            }
            "--token" => {
                out.token =
                    Some(it.next().ok_or_else(|| "--token requires a value".to_string())?);
            }
            s if s.starts_with("--endpoint=") => {
                out.endpoint = Some(s["--endpoint=".len()..].to_string());
            }
            s if s.starts_with("--token=") => {
                out.token = Some(s["--token=".len()..].to_string());
            }
            s if s.starts_with('-') => return Err(format!("unknown option: {}", s)),
            s => {
                if out.target.is_some() {
                    return Err(format!("unexpected extra argument: {}", s));
                }
                out.target = Some(s.to_string());
            }
        }
    }
    Ok(out)
}

fn history_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".hf-bsh_history"))
}

fn run_batch(mut shell: Shell) {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        match shell.run_line(&line) {
            Ok(true) => break,
            Ok(false) => {}
            Err(e) => eprintln!("hf-bsh: {}", e),
        }
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("hf-bsh: {}\n\n{}", e, USAGE);
            return ExitCode::from(2);
        }
    };

    let client = Client::with_overrides(args.endpoint, args.token);
    let mut shell = Shell::new(client);

    if let Some(target) = args.target {
        if let Err(e) = shell.run_line(&format!("open {}", target)) {
            eprintln!("hf-bsh: {}", e);
        }
    }

    if !std::io::stdin().is_terminal() {
        run_batch(shell);
        return ExitCode::SUCCESS;
    }

    println!(
        "hf-bsh {} — type `help` for commands.\n  open buckets/<ns>/<name>      bucket (read/write)\n  cp hf://datasets/<id>/<path>  <dst>/   pull external data in",
        env!("CARGO_PKG_VERSION")
    );

    let helper = ShellHelper { state: Rc::clone(&shell.state) };
    let mut rl: Editor<ShellHelper, rustyline::history::FileHistory> = match Editor::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("hf-bsh: failed to init readline: {}", e);
            return ExitCode::from(1);
        }
    };
    rl.set_completion_type(CompletionType::List);
    rl.set_helper(Some(helper));

    let hist = history_path();
    if let Some(ref h) = hist {
        let _ = rl.load_history(h);
    }

    loop {
        let prompt = shell.prompt();
        match rl.readline(&prompt) {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                match shell.run_line(&line) {
                    Ok(true) => break,
                    Ok(false) => {}
                    Err(e) => eprintln!("hf-bsh: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("hf-bsh: readline: {}", e);
                break;
            }
        }
    }

    if let Some(ref h) = hist {
        let _ = rl.save_history(h);
    }
    ExitCode::SUCCESS
}

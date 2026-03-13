mod cache;
mod convert;
mod freq;
mod input;
mod layout;
mod output;
mod render_svg;
mod tokenize;

use std::env;
use std::path::{Path, PathBuf};

use input::InputMode;

const DEFAULT_TOP_N: usize = 120;
const DEFAULT_MIN_COUNT: u32 = 2;
const DEFAULT_WIDTH: u32 = 1600;
const DEFAULT_HEIGHT: u32 = 1000;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_CACHE_TTL_HOURS: u64 = 24;

#[derive(Debug, Clone)]
struct AppConfig {
    input_mode: InputMode,
    out_dir: PathBuf,
    top_n: usize,
    min_count: u32,
    width: u32,
    height: u32,
    seed: u64,
    cache_dir: PathBuf,
    cache_ttl_hours: u64,
    stopwords_file: Option<PathBuf>,
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let config = match parse_args(&args)? {
        Some(config) => config,
        None => return Ok(()),
    };

    let stopwords = tokenize::load_stopwords(config.stopwords_file.as_deref())?;

    let source = input::load_source(
        &config.input_mode,
        &config.cache_dir,
        config.cache_ttl_hours,
    )?;

    let token_result = tokenize::tokenize_and_count(&source.markdown, &stopwords, config.min_count);
    if token_result.words.is_empty() {
        return Err("no words left after token filtering; try lowering --min-count or changing input".to_string());
    }

    let words = freq::select_and_scale(&token_result.words, config.top_n, config.seed);
    if words.is_empty() {
        return Err("no words selected for rendering".to_string());
    }

    let placed = layout::place_words(&words, config.width, config.height, config.seed);
    if placed.is_empty() {
        return Err("failed to place any words on canvas".to_string());
    }

    let svg = render_svg::render_svg(config.width, config.height, &placed);

    let stats = output::OutputStats {
        total_tokens: token_result.total_tokens,
        unique_tokens: token_result.unique_tokens,
        kept_tokens: token_result.kept_tokens,
    };
    let snapshot = output::ConfigSnapshot {
        top_n: config.top_n,
        min_count: config.min_count,
        width: config.width,
        height: config.height,
        seed: config.seed,
        cache_ttl_hours: config.cache_ttl_hours,
    };

    output::write_outputs(
        &config.out_dir,
        &source,
        &snapshot,
        &stats,
        &placed,
        &svg,
    )?;

    println!(
        "generated wordcloud: {}",
        config.out_dir.join("wordcloud.svg").display()
    );
    println!(
        "generated metadata: {}",
        config.out_dir.join("wordcloud.json").display()
    );
    println!("saved normalized source: {}", config.out_dir.join("source.md").display());

    Ok(())
}

fn parse_args(args: &[String]) -> Result<Option<AppConfig>, String> {
    if args.len() == 1 {
        print_usage();
        return Ok(None);
    }

    let mut input_file: Option<PathBuf> = None;
    let mut use_stdin = false;
    let mut input_url: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut top_n = DEFAULT_TOP_N;
    let mut min_count = DEFAULT_MIN_COUNT;
    let mut width = DEFAULT_WIDTH;
    let mut height = DEFAULT_HEIGHT;
    let mut seed = DEFAULT_SEED;
    let mut cache_dir: Option<PathBuf> = None;
    let mut cache_ttl_hours = DEFAULT_CACHE_TTL_HOURS;
    let mut stopwords_file: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return Ok(None);
            }
            "--input" => {
                let value = next_value(args, i, "--input")?;
                input_file = Some(PathBuf::from(value));
                i += 2;
            }
            "--stdin" => {
                use_stdin = true;
                i += 1;
            }
            "--url" => {
                let value = next_value(args, i, "--url")?;
                input_url = Some(value.to_string());
                i += 2;
            }
            "--out-dir" => {
                let value = next_value(args, i, "--out-dir")?;
                out_dir = Some(PathBuf::from(value));
                i += 2;
            }
            "--top-n" => {
                let value = next_value(args, i, "--top-n")?;
                top_n = parse_usize(value, "--top-n")?;
                i += 2;
            }
            "--min-count" => {
                let value = next_value(args, i, "--min-count")?;
                min_count = parse_u32(value, "--min-count")?;
                i += 2;
            }
            "--width" => {
                let value = next_value(args, i, "--width")?;
                width = parse_u32(value, "--width")?;
                i += 2;
            }
            "--height" => {
                let value = next_value(args, i, "--height")?;
                height = parse_u32(value, "--height")?;
                i += 2;
            }
            "--seed" => {
                let value = next_value(args, i, "--seed")?;
                seed = parse_u64(value, "--seed")?;
                i += 2;
            }
            "--cache-dir" => {
                let value = next_value(args, i, "--cache-dir")?;
                cache_dir = Some(PathBuf::from(value));
                i += 2;
            }
            "--cache-ttl-hours" => {
                let value = next_value(args, i, "--cache-ttl-hours")?;
                cache_ttl_hours = parse_u64(value, "--cache-ttl-hours")?;
                i += 2;
            }
            "--stopwords" => {
                let value = next_value(args, i, "--stopwords")?;
                stopwords_file = Some(PathBuf::from(value));
                i += 2;
            }
            unknown => {
                return Err(format!("unknown argument: {unknown}"));
            }
        }
    }

    if width == 0 || height == 0 {
        return Err("--width and --height must be greater than 0".to_string());
    }
    if top_n == 0 {
        return Err("--top-n must be greater than 0".to_string());
    }

    let mode_count = usize::from(input_file.is_some()) + usize::from(use_stdin) + usize::from(input_url.is_some());
    if mode_count != 1 {
        return Err("exactly one of --input, --stdin, --url must be provided".to_string());
    }

    let input_mode = if let Some(path) = input_file {
        InputMode::File(path)
    } else if use_stdin {
        InputMode::Stdin
    } else {
        let url = input_url.expect("url checked by mode_count");
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err("--url must start with http:// or https://".to_string());
        }
        InputMode::Url(url)
    };

    let out_dir = out_dir.ok_or_else(|| "--out-dir is required".to_string())?;

    let cache_dir = cache_dir.unwrap_or_else(default_cache_dir);

    Ok(Some(AppConfig {
        input_mode,
        out_dir,
        top_n,
        min_count,
        width,
        height,
        seed,
        cache_dir,
        cache_ttl_hours,
        stopwords_file,
    }))
}

fn next_value<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str, String> {
    if i + 1 >= args.len() {
        return Err(format!("missing value for {flag}"));
    }
    Ok(args[i + 1].as_str())
}

fn parse_usize(raw: &str, flag: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

fn parse_u32(raw: &str, flag: &str) -> Result<u32, String> {
    raw.parse::<u32>()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

fn parse_u64(raw: &str, flag: &str) -> Result<u64, String> {
    raw.parse::<u64>()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

fn default_cache_dir() -> PathBuf {
    if let Some(home) = env::var_os("HOME") {
        return Path::new(&home).join(".cache").join("wordcloud-rs");
    }
    PathBuf::from(".wordcloud-rs-cache")
}

fn print_usage() {
    println!(
        "wordcloud-rs\n\n\
Usage:\n\
  wordcloud-rs --input <file> --out-dir <dir> [options]\n\
  wordcloud-rs --stdin --out-dir <dir> [options]\n\
  wordcloud-rs --url <url> --out-dir <dir> [options]\n\n\
Options:\n\
  --input <file>            Read input text from file\n\
  --stdin                   Read input text from stdin\n\
  --url <url>               Download and process content from URL\n\
  --out-dir <dir>           Output directory (required)\n\
  --top-n <n>               Number of top words to render (default: {DEFAULT_TOP_N})\n\
  --min-count <n>           Minimum token frequency to keep (default: {DEFAULT_MIN_COUNT})\n\
  --width <px>              SVG canvas width (default: {DEFAULT_WIDTH})\n\
  --height <px>             SVG canvas height (default: {DEFAULT_HEIGHT})\n\
  --seed <n>                Deterministic layout seed (default: {DEFAULT_SEED})\n\
  --cache-dir <path>        URL cache directory (default: ~/.cache/wordcloud-rs)\n\
  --cache-ttl-hours <h>     URL cache TTL hours (default: {DEFAULT_CACHE_TTL_HOURS})\n\
  --stopwords <file>        Extra stopwords file (one token per line)\n\
  --help, -h                Show this help"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_requires_single_source() {
        let args = vec![
            "wordcloud-rs".to_string(),
            "--stdin".to_string(),
            "--input".to_string(),
            "a.txt".to_string(),
            "--out-dir".to_string(),
            "out".to_string(),
        ];
        let err = parse_args(&args).unwrap_err();
        assert!(err.contains("exactly one of"));
    }

    #[test]
    fn parse_stdin_ok() {
        let args = vec![
            "wordcloud-rs".to_string(),
            "--stdin".to_string(),
            "--out-dir".to_string(),
            "out".to_string(),
        ];
        let cfg = parse_args(&args).unwrap().unwrap();
        assert!(matches!(cfg.input_mode, InputMode::Stdin));
        assert_eq!(cfg.top_n, DEFAULT_TOP_N);
    }
}

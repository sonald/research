use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::cache::{self, CacheMeta};

#[derive(Debug, Clone)]
pub enum InputMode {
    File(PathBuf),
    Stdin,
    Url(String),
}

#[derive(Debug, Clone)]
pub struct SourceDocument {
    pub markdown: String,
    pub source_kind: String,
    pub source_value: String,
    pub cache_hit: bool,
    pub cache_stale: bool,
    pub fetched_at: u64,
}

pub fn load_source(mode: &InputMode, cache_dir: &Path, cache_ttl_hours: u64) -> Result<SourceDocument, String> {
    match mode {
        InputMode::File(path) => read_file(path),
        InputMode::Stdin => read_stdin(),
        InputMode::Url(url) => read_url(url, cache_dir, cache_ttl_hours),
    }
}

fn read_file(path: &Path) -> Result<SourceDocument, String> {
    let markdown = fs::read_to_string(path)
        .map_err(|e| format!("failed reading input file {}: {e}", path.display()))?;
    Ok(SourceDocument {
        markdown,
        source_kind: "file".to_string(),
        source_value: path.display().to_string(),
        cache_hit: false,
        cache_stale: false,
        fetched_at: cache::now_secs(),
    })
}

fn read_stdin() -> Result<SourceDocument, String> {
    let mut markdown = String::new();
    std::io::stdin()
        .read_to_string(&mut markdown)
        .map_err(|e| format!("failed reading stdin: {e}"))?;
    Ok(SourceDocument {
        markdown,
        source_kind: "stdin".to_string(),
        source_value: "stdin".to_string(),
        cache_hit: false,
        cache_stale: false,
        fetched_at: cache::now_secs(),
    })
}

fn read_url(url: &str, cache_dir: &Path, cache_ttl_hours: u64) -> Result<SourceDocument, String> {
    let now = cache::now_secs();
    let ttl_secs = cache_ttl_hours.saturating_mul(3600);
    let key = cache::cache_key(url);

    if let Some(cached) = cache::load_fresh(cache_dir, &key, ttl_secs, now)? {
        let markdown = crate::convert::to_markdown(url, &cached.body, &cached.meta.content_type)?;
        return Ok(SourceDocument {
            markdown,
            source_kind: "url".to_string(),
            source_value: url.to_string(),
            cache_hit: true,
            cache_stale: false,
            fetched_at: cached.meta.fetched_at,
        });
    }

    match crate::convert::download_url(url) {
        Ok(downloaded) => {
            let markdown = crate::convert::to_markdown(url, &downloaded.body, &downloaded.content_type)?;
            let meta = CacheMeta {
                url: url.to_string(),
                content_type: downloaded.content_type,
                fetched_at: now,
            };
            cache::store(cache_dir, &key, &downloaded.body, &meta)?;
            Ok(SourceDocument {
                markdown,
                source_kind: "url".to_string(),
                source_value: url.to_string(),
                cache_hit: false,
                cache_stale: false,
                fetched_at: now,
            })
        }
        Err(download_err) => {
            if let Some(stale) = cache::load_any(cache_dir, &key)? {
                let markdown = crate::convert::to_markdown(url, &stale.body, &stale.meta.content_type)?;
                Ok(SourceDocument {
                    markdown,
                    source_kind: "url".to_string(),
                    source_value: url.to_string(),
                    cache_hit: true,
                    cache_stale: true,
                    fetched_at: stale.meta.fetched_at,
                })
            } else {
                Err(format!("download failed and no cache available: {download_err}"))
            }
        }
    }
}

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct CacheMeta {
    pub url: String,
    pub content_type: String,
    pub fetched_at: u64,
}

#[derive(Debug, Clone)]
pub struct CachedEntry {
    pub body: Vec<u8>,
    pub meta: CacheMeta,
    pub stale: bool,
}

pub fn fnv1a64_hex(input: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in input {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

pub fn cache_key(url: &str) -> String {
    fnv1a64_hex(url.as_bytes())
}

pub fn load_fresh(cache_dir: &Path, key: &str, ttl_secs: u64, now_secs: u64) -> Result<Option<CachedEntry>, String> {
    let entry = load_internal(cache_dir, key)?;
    let Some(mut cached) = entry else {
        return Ok(None);
    };

    let age = now_secs.saturating_sub(cached.meta.fetched_at);
    if age <= ttl_secs {
        cached.stale = false;
        Ok(Some(cached))
    } else {
        Ok(None)
    }
}

pub fn load_any(cache_dir: &Path, key: &str) -> Result<Option<CachedEntry>, String> {
    let mut entry = load_internal(cache_dir, key)?;
    if let Some(ref mut cached) = entry {
        cached.stale = true;
    }
    Ok(entry)
}

pub fn store(cache_dir: &Path, key: &str, body: &[u8], meta: &CacheMeta) -> Result<(), String> {
    fs::create_dir_all(cache_dir).map_err(|e| format!("failed to create cache dir {}: {e}", cache_dir.display()))?;

    let body_path = body_path(cache_dir, key);
    let meta_path = meta_path(cache_dir, key);

    fs::write(&body_path, body)
        .map_err(|e| format!("failed to write cache body {}: {e}", body_path.display()))?;

    let mut meta_raw = String::new();
    meta_raw.push_str("url=");
    meta_raw.push_str(&escape_meta_field(&meta.url));
    meta_raw.push('\n');
    meta_raw.push_str("content_type=");
    meta_raw.push_str(&escape_meta_field(&meta.content_type));
    meta_raw.push('\n');
    meta_raw.push_str("fetched_at=");
    meta_raw.push_str(&meta.fetched_at.to_string());
    meta_raw.push('\n');

    fs::write(&meta_path, meta_raw)
        .map_err(|e| format!("failed to write cache meta {}: {e}", meta_path.display()))?;

    Ok(())
}

fn load_internal(cache_dir: &Path, key: &str) -> Result<Option<CachedEntry>, String> {
    let body_path = body_path(cache_dir, key);
    let meta_path = meta_path(cache_dir, key);

    if !body_path.exists() || !meta_path.exists() {
        return Ok(None);
    }

    let body = fs::read(&body_path)
        .map_err(|e| format!("failed to read cache body {}: {e}", body_path.display()))?;
    let meta_raw = fs::read_to_string(&meta_path)
        .map_err(|e| format!("failed to read cache meta {}: {e}", meta_path.display()))?;
    let meta = parse_meta(&meta_raw)?;

    Ok(Some(CachedEntry {
        body,
        meta,
        stale: false,
    }))
}

fn parse_meta(raw: &str) -> Result<CacheMeta, String> {
    let mut url: Option<String> = None;
    let mut content_type: Option<String> = None;
    let mut fetched_at: Option<u64> = None;

    for line in raw.lines() {
        if let Some(value) = line.strip_prefix("url=") {
            url = Some(unescape_meta_field(value));
        } else if let Some(value) = line.strip_prefix("content_type=") {
            content_type = Some(unescape_meta_field(value));
        } else if let Some(value) = line.strip_prefix("fetched_at=") {
            fetched_at = Some(
                value
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| format!("invalid fetched_at in cache meta: {value}"))?,
            );
        }
    }

    Ok(CacheMeta {
        url: url.ok_or_else(|| "missing url in cache meta".to_string())?,
        content_type: content_type.unwrap_or_default(),
        fetched_at: fetched_at.ok_or_else(|| "missing fetched_at in cache meta".to_string())?,
    })
}

fn escape_meta_field(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\n', "\\n")
}

fn unescape_meta_field(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => out.push('\n'),
                    '\\' => out.push('\\'),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            } else {
                out.push('\\');
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn body_path(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("{key}.body"))
}

fn meta_path(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("{key}.meta"))
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("wordcloud-rs-cache-test-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn fnv_hash_is_stable() {
        assert_eq!(fnv1a64_hex(b"abc"), "e71fa2190541574b");
    }

    #[test]
    fn cache_roundtrip_and_ttl() {
        let dir = unique_temp_dir();
        let key = "abc";
        let meta = CacheMeta {
            url: "https://example.com".to_string(),
            content_type: "text/plain".to_string(),
            fetched_at: 100,
        };
        store(&dir, key, b"hello", &meta).unwrap();

        let fresh = load_fresh(&dir, key, 10, 105).unwrap().unwrap();
        assert_eq!(fresh.body, b"hello");
        assert!(!fresh.stale);

        let expired = load_fresh(&dir, key, 10, 111).unwrap();
        assert!(expired.is_none());

        let stale = load_any(&dir, key).unwrap().unwrap();
        assert!(stale.stale);

        let _ = fs::remove_dir_all(dir);
    }
}

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct DownloadedContent {
    pub body: Vec<u8>,
    pub content_type: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentKind {
    Pdf,
    Markdown,
    Html,
    Plain,
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn download_url(url: &str) -> Result<DownloadedContent, String> {
    let unique = temp_id();
    let body_path = std::env::temp_dir().join(format!("wordcloud-rs-curl-{unique}.body"));
    let headers_path = std::env::temp_dir().join(format!("wordcloud-rs-curl-{unique}.headers"));

    let output = Command::new("curl")
        .arg("-L")
        .arg("--silent")
        .arg("--show-error")
        .arg("--fail")
        .arg("--max-time")
        .arg("30")
        .arg("-D")
        .arg(&headers_path)
        .arg("-o")
        .arg(&body_path)
        .arg(url)
        .output();

    let run = match output {
        Ok(out) => out,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err("curl is required but not found in PATH".to_string());
        }
        Err(e) => return Err(format!("failed to execute curl: {e}")),
    };

    if !run.status.success() {
        let _ = fs::remove_file(&body_path);
        let _ = fs::remove_file(&headers_path);
        let stderr = String::from_utf8_lossy(&run.stderr);
        return Err(format!("curl download failed: {}", stderr.trim()));
    }

    let body = fs::read(&body_path)
        .map_err(|e| format!("failed reading downloaded body {}: {e}", body_path.display()))?;
    let headers = fs::read_to_string(&headers_path)
        .map_err(|e| format!("failed reading curl headers {}: {e}", headers_path.display()))?;

    let _ = fs::remove_file(&body_path);
    let _ = fs::remove_file(&headers_path);

    let content_type = parse_content_type(&headers);

    Ok(DownloadedContent { body, content_type })
}

pub fn to_markdown(url: &str, body: &[u8], content_type: &str) -> Result<String, String> {
    let kind = detect_kind(url, content_type, body);
    match kind {
        ContentKind::Pdf => pdf_to_text(body),
        ContentKind::Markdown => Ok(String::from_utf8_lossy(body).to_string()),
        ContentKind::Html => match html_to_markdown(body) {
            Ok(markdown) => Ok(markdown),
            Err(_) => Ok(String::from_utf8_lossy(body).to_string()),
        },
        ContentKind::Plain => Ok(String::from_utf8_lossy(body).to_string()),
    }
}

fn detect_kind(url: &str, content_type: &str, body: &[u8]) -> ContentKind {
    let lower_ct = content_type.to_lowercase();
    let lower_path = normalize_url_path(url);

    if lower_ct.contains("application/pdf") || lower_path.ends_with(".pdf") {
        return ContentKind::Pdf;
    }
    if lower_ct.contains("text/markdown")
        || lower_path.ends_with(".md")
        || lower_path.ends_with(".markdown")
    {
        return ContentKind::Markdown;
    }
    if lower_ct.contains("text/html") || lower_ct.contains("application/xhtml+xml") {
        return ContentKind::Html;
    }

    let preview = String::from_utf8_lossy(&body[..body.len().min(512)]).to_lowercase();
    if preview.contains("<html") || preview.contains("<!doctype html") {
        return ContentKind::Html;
    }

    ContentKind::Plain
}

fn normalize_url_path(url: &str) -> String {
    let mut s = url.to_lowercase();
    if let Some(i) = s.find('#') {
        s.truncate(i);
    }
    if let Some(i) = s.find('?') {
        s.truncate(i);
    }
    s
}

fn parse_content_type(headers: &str) -> String {
    let mut ct = String::new();
    for line in headers.lines() {
        let trimmed = line.trim();
        if trimmed.to_ascii_lowercase().starts_with("content-type:") {
            if let Some((_, v)) = trimmed.split_once(':') {
                let v = v.trim();
                let bare = v.split(';').next().unwrap_or(v).trim();
                ct = bare.to_string();
            }
        }
    }
    ct
}

fn html_to_markdown(html: &[u8]) -> Result<String, String> {
    run_command_with_stdin(
        "pandoc",
        &["-f", "html", "-t", "gfm", "--wrap=none"],
        html,
        "pandoc is required to convert HTML to markdown",
    )
}

fn pdf_to_text(body: &[u8]) -> Result<String, String> {
    let unique = temp_id();
    let input_path: PathBuf = std::env::temp_dir().join(format!("wordcloud-rs-pdf-{unique}.pdf"));
    fs::write(&input_path, body)
        .map_err(|e| format!("failed writing temporary pdf {}: {e}", input_path.display()))?;

    let output = Command::new("pdftotext")
        .arg(&input_path)
        .arg("-")
        .output();

    let _ = fs::remove_file(&input_path);

    let out = match output {
        Ok(out) => out,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err("pdftotext is required to process PDF URLs".to_string());
        }
        Err(e) => return Err(format!("failed to execute pdftotext: {e}")),
    };

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("pdftotext failed: {}", stderr.trim()));
    }

    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn run_command_with_stdin(
    program: &str,
    args: &[&str],
    stdin_data: &[u8],
    not_found_msg: &str,
) -> Result<String, String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                not_found_msg.to_string()
            } else {
                format!("failed to execute {program}: {e}")
            }
        })?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| format!("failed to open stdin for {program}"))?;
        stdin
            .write_all(stdin_data)
            .map_err(|e| format!("failed writing stdin to {program}: {e}"))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("failed waiting for {program}: {e}"))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("{program} failed: {}", stderr.trim()));
    }

    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn temp_id() -> String {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos}-{}-{n}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_parser_picks_last() {
        let headers = "HTTP/1.1 301 Moved\r\nContent-Type: text/html\r\n\r\nHTTP/2 200\r\nContent-Type: text/markdown; charset=utf-8\r\n\r\n";
        assert_eq!(parse_content_type(headers), "text/markdown");
    }

    #[test]
    fn detect_markdown_by_extension() {
        let kind = detect_kind("https://a/b/c.md?x=1", "", b"# hi");
        assert_eq!(kind, ContentKind::Markdown);
    }

    #[test]
    fn detect_html_by_body_preview() {
        let kind = detect_kind("https://x/y", "", b"<!doctype html><html><body>hi</body></html>");
        assert_eq!(kind, ContentKind::Html);
    }
}

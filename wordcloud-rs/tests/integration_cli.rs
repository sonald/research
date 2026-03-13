use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

static COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn cli_rejects_multiple_sources() {
    let out_dir = unique_temp_dir("cli-multiple");
    let input_file = out_dir.join("in.md");
    fs::write(&input_file, "alpha beta").unwrap();

    let out = Command::new(bin_path())
        .args([
            "--stdin",
            "--input",
            input_file.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("exactly one of --input, --stdin, --url"));
}

#[test]
fn generates_outputs_from_input_file() {
    let dir = unique_temp_dir("file");
    let out_dir = dir.join("out");
    let input_file = dir.join("input.md");
    fs::write(
        &input_file,
        "Rust rust language systems performance memory safety\n人工智能 人工智能",
    )
    .unwrap();

    let out = Command::new(bin_path())
        .args([
            "--input",
            input_file.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    assert!(out_dir.join("wordcloud.svg").exists());
    assert!(out_dir.join("wordcloud.json").exists());
    assert!(out_dir.join("source.md").exists());

    let json = fs::read_to_string(out_dir.join("wordcloud.json")).unwrap();
    assert!(json.contains("\"kind\": \"file\""));
    assert!(json.contains("\"words\": ["));
}

#[test]
fn generates_outputs_from_stdin() {
    let dir = unique_temp_dir("stdin");
    let out_dir = dir.join("out");

    let mut child = Command::new(bin_path())
        .args(["--stdin", "--out-dir", out_dir.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"Rust rust rust systems programming programming")
        .unwrap();

    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    assert!(out_dir.join("wordcloud.svg").exists());
    let json = fs::read_to_string(out_dir.join("wordcloud.json")).unwrap();
    assert!(json.contains("\"kind\": \"stdin\""));
}

#[test]
fn url_uses_fresh_cache_without_second_network_hit() {
    let server = TestServer::spawn();
    server.set_response(ResponseMode::Markdown(
        "# Demo\nRust rust systems language".to_string(),
        "text/markdown".to_string(),
    ));

    let dir = unique_temp_dir("url-cache-hit");
    let out1 = dir.join("out1");
    let out2 = dir.join("out2");
    let cache_dir = dir.join("cache");

    run_ok(&[
        "--url",
        &server.url(),
        "--out-dir",
        out1.to_str().unwrap(),
        "--cache-dir",
        cache_dir.to_str().unwrap(),
    ]);

    assert_eq!(server.request_count(), 1);

    run_ok(&[
        "--url",
        &server.url(),
        "--out-dir",
        out2.to_str().unwrap(),
        "--cache-dir",
        cache_dir.to_str().unwrap(),
    ]);

    assert_eq!(server.request_count(), 1);
    let json = fs::read_to_string(out2.join("wordcloud.json")).unwrap();
    assert!(json.contains("\"cache_hit\": true"));
    assert!(json.contains("\"cache_stale\": false"));
}

#[test]
fn url_falls_back_to_stale_cache_when_network_fails() {
    let server = TestServer::spawn();
    server.set_response(ResponseMode::Markdown(
        "# Demo\nalpha alpha alpha beta beta".to_string(),
        "text/markdown".to_string(),
    ));

    let dir = unique_temp_dir("url-stale");
    let out1 = dir.join("out1");
    let out2 = dir.join("out2");
    let cache_dir = dir.join("cache");

    run_ok(&[
        "--url",
        &server.url(),
        "--out-dir",
        out1.to_str().unwrap(),
        "--cache-dir",
        cache_dir.to_str().unwrap(),
        "--cache-ttl-hours",
        "0",
    ]);

    assert_eq!(server.request_count(), 1);

    thread::sleep(Duration::from_secs(2));
    server.set_response(ResponseMode::Fail);

    run_ok(&[
        "--url",
        &server.url(),
        "--out-dir",
        out2.to_str().unwrap(),
        "--cache-dir",
        cache_dir.to_str().unwrap(),
        "--cache-ttl-hours",
        "0",
    ]);

    assert_eq!(server.request_count(), 2);
    let json = fs::read_to_string(out2.join("wordcloud.json")).unwrap();
    assert!(json.contains("\"cache_hit\": true"));
    assert!(json.contains("\"cache_stale\": true"));
}

fn run_ok(args: &[&str]) {
    let out = Command::new(bin_path()).args(args).output().unwrap();
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
}

fn bin_path() -> String {
    env!("CARGO_BIN_EXE_wordcloud-rs").to_string()
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("wordcloud-rs-test-{name}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[derive(Clone)]
enum ResponseMode {
    Markdown(String, String),
    Fail,
}

struct TestServer {
    addr: String,
    mode: Arc<Mutex<ResponseMode>>,
    requests: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl TestServer {
    fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();

        let mode = Arc::new(Mutex::new(ResponseMode::Markdown(
            "# default\nalpha alpha".to_string(),
            "text/markdown".to_string(),
        )));
        let requests = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));

        let mode_c = Arc::clone(&mode);
        let requests_c = Arc::clone(&requests);
        let shutdown_c = Arc::clone(&shutdown);

        let handle = thread::spawn(move || {
            loop {
                if shutdown_c.load(Ordering::Relaxed) {
                    break;
                }
                match listener.accept() {
                    Ok((stream, _)) => {
                        requests_c.fetch_add(1, Ordering::Relaxed);
                        let mode = mode_c.lock().unwrap().clone();
                        handle_connection(stream, mode);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            addr: format!("http://{addr}/data"),
            mode,
            requests,
            shutdown,
            handle: Some(handle),
        }
    }

    fn url(&self) -> String {
        self.addr.clone()
    }

    fn set_response(&self, mode: ResponseMode) {
        *self.mode.lock().unwrap() = mode;
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect(self.addr.trim_start_matches("http://").split('/').next().unwrap());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn handle_connection(mut stream: TcpStream, mode: ResponseMode) {
    let mut buf = [0u8; 1024];
    let _ = stream.read(&mut buf);

    match mode {
        ResponseMode::Fail => {
            let resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(resp.as_bytes());
        }
        ResponseMode::Markdown(body, ct) => {
            let bytes = body.into_bytes();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.write_all(&bytes);
        }
    }
}

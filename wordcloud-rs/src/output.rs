use std::fmt::Write;
use std::fs;
use std::path::Path;

use crate::input::SourceDocument;
use crate::layout::PlacedWord;

#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    pub top_n: usize,
    pub min_count: u32,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub cache_ttl_hours: u64,
}

#[derive(Debug, Clone)]
pub struct OutputStats {
    pub total_tokens: u64,
    pub unique_tokens: usize,
    pub kept_tokens: u64,
}

pub fn write_outputs(
    out_dir: &Path,
    source: &SourceDocument,
    config: &ConfigSnapshot,
    stats: &OutputStats,
    words: &[PlacedWord],
    svg: &str,
) -> Result<(), String> {
    fs::create_dir_all(out_dir)
        .map_err(|e| format!("failed creating output directory {}: {e}", out_dir.display()))?;

    let source_path = out_dir.join("source.md");
    let svg_path = out_dir.join("wordcloud.svg");
    let json_path = out_dir.join("wordcloud.json");

    fs::write(&source_path, &source.markdown)
        .map_err(|e| format!("failed writing {}: {e}", source_path.display()))?;
    fs::write(&svg_path, svg).map_err(|e| format!("failed writing {}: {e}", svg_path.display()))?;

    let json = build_json(source, config, stats, words);
    fs::write(&json_path, json).map_err(|e| format!("failed writing {}: {e}", json_path.display()))?;

    Ok(())
}

fn build_json(source: &SourceDocument, config: &ConfigSnapshot, stats: &OutputStats, words: &[PlacedWord]) -> String {
    let mut out = String::new();
    out.push_str("{\n");

    out.push_str("  \"source\": {\n");
    writeln!(
        &mut out,
        "    \"kind\": \"{}\",",
        escape_json(&source.source_kind)
    )
    .unwrap();
    writeln!(
        &mut out,
        "    \"value\": \"{}\",",
        escape_json(&source.source_value)
    )
    .unwrap();
    writeln!(&mut out, "    \"cache_hit\": {},", source.cache_hit).unwrap();
    writeln!(&mut out, "    \"cache_stale\": {},", source.cache_stale).unwrap();
    writeln!(&mut out, "    \"fetched_at\": {}", source.fetched_at).unwrap();
    out.push_str("  },\n");

    out.push_str("  \"config\": {\n");
    writeln!(&mut out, "    \"top_n\": {},", config.top_n).unwrap();
    writeln!(&mut out, "    \"min_count\": {},", config.min_count).unwrap();
    writeln!(&mut out, "    \"width\": {},", config.width).unwrap();
    writeln!(&mut out, "    \"height\": {},", config.height).unwrap();
    writeln!(&mut out, "    \"seed\": {},", config.seed).unwrap();
    writeln!(&mut out, "    \"cache_ttl_hours\": {}", config.cache_ttl_hours).unwrap();
    out.push_str("  },\n");

    out.push_str("  \"stats\": {\n");
    writeln!(&mut out, "    \"total_tokens\": {},", stats.total_tokens).unwrap();
    writeln!(&mut out, "    \"unique_tokens\": {},", stats.unique_tokens).unwrap();
    writeln!(&mut out, "    \"kept_tokens\": {}", stats.kept_tokens).unwrap();
    out.push_str("  },\n");

    out.push_str("  \"words\": [\n");
    for (idx, word) in words.iter().enumerate() {
        out.push_str("    {\n");
        writeln!(&mut out, "      \"text\": \"{}\",", escape_json(&word.text)).unwrap();
        writeln!(&mut out, "      \"count\": {},", word.count).unwrap();
        writeln!(&mut out, "      \"weight\": {:.6},", word.weight).unwrap();
        writeln!(&mut out, "      \"font_size\": {:.3},", word.font_size).unwrap();
        writeln!(&mut out, "      \"x\": {:.3},", word.x).unwrap();
        writeln!(&mut out, "      \"y\": {:.3},", word.y).unwrap();
        writeln!(&mut out, "      \"rotate\": {},", word.rotate).unwrap();
        writeln!(&mut out, "      \"color\": \"{}\"", escape_json(&word.color)).unwrap();
        if idx + 1 == words.len() {
            out.push_str("    }\n");
        } else {
            out.push_str("    },\n");
        }
    }
    out.push_str("  ]\n");

    out.push_str("}\n");
    out
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let code = c as u32;
                let _ = write!(&mut out, "\\u{:04x}", code);
            }
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_basic() {
        assert_eq!(escape_json("a\"b\\c"), "a\\\"b\\\\c");
    }
}

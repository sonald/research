use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TokenCount {
    pub text: String,
    pub count: u32,
    pub is_cjk: bool,
}

#[derive(Debug, Clone)]
pub struct TokenizationResult {
    pub total_tokens: u64,
    pub unique_tokens: usize,
    pub kept_tokens: u64,
    pub words: Vec<TokenCount>,
}

pub fn load_stopwords(extra_file: Option<&Path>) -> Result<HashSet<String>, String> {
    let mut set: HashSet<String> = HashSet::new();

    for word in BUILTIN_EN_STOPWORDS {
        set.insert((*word).to_string());
    }
    for word in BUILTIN_ZH_STOPWORDS {
        set.insert((*word).to_string());
    }

    if let Some(path) = extra_file {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("failed reading stopwords file {}: {e}", path.display()))?;
        for line in content.lines() {
            let word = line.trim();
            if word.is_empty() || word.starts_with('#') {
                continue;
            }
            set.insert(word.to_lowercase());
        }
    }

    Ok(set)
}

pub fn tokenize_and_count(text: &str, stopwords: &HashSet<String>, min_count: u32) -> TokenizationResult {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut total_tokens: u64 = 0;

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if is_ascii_token_char(c) {
            let start = i;
            i += 1;
            while i < chars.len() && is_ascii_token_char(chars[i]) {
                i += 1;
            }
            let mut token: String = chars[start..i].iter().collect();
            token = token.trim_matches('\'').to_lowercase();
            if token.len() >= 3 && !stopwords.contains(&token) {
                total_tokens += 1;
                *counts.entry(token).or_insert(0) += 1;
            }
            continue;
        }

        if is_cjk(c) {
            let start = i;
            i += 1;
            while i < chars.len() && is_cjk(chars[i]) {
                i += 1;
            }
            let seq = &chars[start..i];
            if seq.len() >= 2 {
                for window in seq.windows(2) {
                    let token: String = window.iter().collect();
                    if !stopwords.contains(&token) {
                        total_tokens += 1;
                        *counts.entry(token).or_insert(0) += 1;
                    }
                }
            }
            continue;
        }

        i += 1;
    }

    let unique_tokens = counts.len();
    let mut words = Vec::new();
    for (text, count) in counts {
        if count >= min_count {
            let is_cjk = text.chars().any(is_cjk);
            words.push(TokenCount {
                text,
                count,
                is_cjk,
            });
        }
    }

    words.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.text.cmp(&b.text)));

    let kept_tokens = words.iter().map(|w| u64::from(w.count)).sum();

    TokenizationResult {
        total_tokens,
        unique_tokens,
        kept_tokens,
        words,
    }
}

fn is_ascii_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '\''
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF |
        0x3400..=0x4DBF |
        0xF900..=0xFAFF |
        0x20000..=0x2A6DF |
        0x2A700..=0x2B73F |
        0x2B740..=0x2B81F |
        0x2B820..=0x2CEAF
    )
}

const BUILTIN_EN_STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
];

const BUILTIN_ZH_STOPWORDS: &[&str] = &[
    "的", "了", "和", "是", "在", "就", "都", "而", "及", "与", "着", "或", "一个", "没有", "我们",
    "你们", "他们", "她们", "是否", "不是", "这个", "那个", "这些", "那些", "因为", "所以", "如果",
    "但是", "然后", "并且", "以及", "已经", "可以", "进行", "对于", "通过", "需要", "自己", "不会",
    "不是", "就是", "还有", "一个", "一种", "一样", "一些", "很多", "非常", "可能", "其中", "由于",
    "并", "并不", "让", "把", "被", "给", "向", "从", "到", "对", "于", "上", "下", "中", "里", "后",
    "前", "并没有", "这", "那", "其", "并非", "以及", "并且", "与否", "或者", "而且", "以及",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_tokenization_works() {
        let stopwords = load_stopwords(None).unwrap();
        let r = tokenize_and_count("Rust makes systems programming productive. Rust is fast.", &stopwords, 1);
        let rust = r.words.iter().find(|w| w.text == "rust").unwrap();
        assert_eq!(rust.count, 2);
    }

    #[test]
    fn cjk_bigram_tokenization_works() {
        let stopwords = HashSet::new();
        let r = tokenize_and_count("人工智能", &stopwords, 1);
        let words: Vec<String> = r.words.iter().map(|w| w.text.clone()).collect();
        assert!(words.contains(&"人工".to_string()));
        assert!(words.contains(&"工智".to_string()));
        assert!(words.contains(&"智能".to_string()));
    }

    #[test]
    fn min_count_filtering_works() {
        let stopwords = HashSet::new();
        let r = tokenize_and_count("alpha alpha beta", &stopwords, 2);
        assert_eq!(r.words.len(), 1);
        assert_eq!(r.words[0].text, "alpha");
    }
}

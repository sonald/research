use crate::cache;
use crate::tokenize::TokenCount;

#[derive(Debug, Clone)]
pub struct WordSpec {
    pub text: String,
    pub count: u32,
    pub is_cjk: bool,
    pub weight: f32,
    pub font_size: f32,
    pub rotate: i32,
    pub color: String,
}

pub fn select_and_scale(tokens: &[TokenCount], top_n: usize, seed: u64) -> Vec<WordSpec> {
    if tokens.is_empty() || top_n == 0 {
        return Vec::new();
    }

    let selected = tokens.iter().take(top_n).collect::<Vec<_>>();
    let min_count = selected.iter().map(|w| w.count).min().unwrap_or(1) as f32;
    let max_count = selected.iter().map(|w| w.count).max().unwrap_or(1) as f32;

    selected
        .into_iter()
        .map(|token| {
            let weight = if (max_count - min_count).abs() < f32::EPSILON {
                1.0
            } else {
                let min_ln = min_count.ln();
                let max_ln = max_count.ln();
                let cur_ln = (token.count as f32).ln();
                ((cur_ln - min_ln) / (max_ln - min_ln)).clamp(0.0, 1.0)
            };

            let font_size = 18.0 + weight * (120.0 - 18.0);
            let h = hash_word(seed, &token.text);
            let rotate = if h % 10 < 2 { 90 } else { 0 };
            let color = PALETTE[(h as usize) % PALETTE.len()].to_string();

            WordSpec {
                text: token.text.clone(),
                count: token.count,
                is_cjk: token.is_cjk,
                weight,
                font_size,
                rotate,
                color,
            }
        })
        .collect()
}

fn hash_word(seed: u64, word: &str) -> u64 {
    let mut bytes = seed.to_le_bytes().to_vec();
    bytes.extend_from_slice(word.as_bytes());
    let hex = cache::fnv1a64_hex(&bytes);
    u64::from_str_radix(&hex, 16).unwrap_or(0)
}

const PALETTE: &[&str] = &[
    "#1f2937", "#0f766e", "#1d4ed8", "#b45309", "#be123c", "#4f46e5", "#166534", "#7c2d12",
    "#374151", "#115e59", "#334155", "#0e7490",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_size_is_monotonic() {
        let tokens = vec![
            TokenCount {
                text: "aabb".to_string(),
                count: 2,
                is_cjk: false,
            },
            TokenCount {
                text: "cccc".to_string(),
                count: 20,
                is_cjk: false,
            },
        ];
        let out = select_and_scale(&tokens, 10, 42);
        assert!(out[1].font_size >= out[0].font_size);
    }
}

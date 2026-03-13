use crate::freq::WordSpec;

#[derive(Debug, Clone)]
pub struct PlacedWord {
    pub text: String,
    pub count: u32,
    pub weight: f32,
    pub font_size: f32,
    pub x: f32,
    pub y: f32,
    pub rotate: i32,
    pub color: String,
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

pub fn place_words(words: &[WordSpec], width: u32, height: u32, seed: u64) -> Vec<PlacedWord> {
    let mut placed: Vec<PlacedWord> = Vec::new();
    let mut rects: Vec<Rect> = Vec::new();

    let mut rng = XorShift64::new(seed);
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;

    for word in words {
        let mut font_size = word.font_size;
        let mut inserted = false;

        for _ in 0..4 {
            let start_angle = rng.next_f32() * std::f32::consts::TAU;
            if let Some((x, y, rect)) = find_position(
                &word.text,
                word.is_cjk,
                font_size,
                word.rotate,
                center_x,
                center_y,
                width as f32,
                height as f32,
                start_angle,
                &rects,
            ) {
                rects.push(rect);
                placed.push(PlacedWord {
                    text: word.text.clone(),
                    count: word.count,
                    weight: word.weight,
                    font_size,
                    x,
                    y,
                    rotate: word.rotate,
                    color: word.color.clone(),
                });
                inserted = true;
                break;
            }
            font_size = (font_size * 0.88).max(10.0);
        }

        if !inserted {
            continue;
        }
    }

    placed
}

#[allow(clippy::too_many_arguments)]
fn find_position(
    text: &str,
    is_cjk: bool,
    font_size: f32,
    rotate: i32,
    center_x: f32,
    center_y: f32,
    width: f32,
    height: f32,
    start_angle: f32,
    existing: &[Rect],
) -> Option<(f32, f32, Rect)> {
    let (mut box_w, mut box_h) = estimate_box(text, is_cjk, font_size);
    if rotate != 0 {
        std::mem::swap(&mut box_w, &mut box_h);
    }

    let margin = 4.0;

    for step in 0..7000 {
        let t = step as f32 * 0.20;
        let radius = 2.8 * t;
        let angle = start_angle + t;

        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();

        let rect = Rect {
            left: x - box_w / 2.0,
            top: y - box_h / 2.0,
            right: x + box_w / 2.0,
            bottom: y + box_h / 2.0,
        };

        if rect.left < margin
            || rect.top < margin
            || rect.right > width - margin
            || rect.bottom > height - margin
        {
            continue;
        }

        if existing.iter().all(|other| !rects_overlap(rect, *other, 1.5)) {
            return Some((x, y, rect));
        }
    }

    None
}

fn estimate_box(text: &str, is_cjk: bool, font_size: f32) -> (f32, f32) {
    let count = text.chars().count() as f32;
    let width_factor = if is_cjk { 0.98 } else { 0.58 };
    let w = (count * font_size * width_factor).max(font_size * 0.9);
    let h = font_size * 1.06;
    (w, h)
}

fn rects_overlap(a: Rect, b: Rect, padding: f32) -> bool {
    !(a.right + padding <= b.left
        || a.left >= b.right + padding
        || a.bottom + padding <= b.top
        || a.top >= b.bottom + padding)
}

#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let x = self.next_u64();
        (x as f64 / u64::MAX as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_overlap_for_small_set() {
        let words = vec![
            WordSpec {
                text: "rust".to_string(),
                count: 30,
                is_cjk: false,
                weight: 1.0,
                font_size: 64.0,
                rotate: 0,
                color: "#111".to_string(),
            },
            WordSpec {
                text: "系统".to_string(),
                count: 20,
                is_cjk: true,
                weight: 0.8,
                font_size: 52.0,
                rotate: 90,
                color: "#222".to_string(),
            },
            WordSpec {
                text: "编程".to_string(),
                count: 15,
                is_cjk: true,
                weight: 0.6,
                font_size: 40.0,
                rotate: 0,
                color: "#333".to_string(),
            },
        ];

        let placed = place_words(&words, 1200, 800, 42);
        assert_eq!(placed.len(), 3);

        let mut rects = Vec::new();
        for p in &placed {
            let (mut w, mut h) = estimate_box(&p.text, p.text.chars().any(is_cjk_test), p.font_size);
            if p.rotate != 0 {
                std::mem::swap(&mut w, &mut h);
            }
            let r = Rect {
                left: p.x - w / 2.0,
                top: p.y - h / 2.0,
                right: p.x + w / 2.0,
                bottom: p.y + h / 2.0,
            };
            assert!(rects.iter().all(|o| !rects_overlap(r, *o, 1.5)));
            rects.push(r);
        }
    }

    fn is_cjk_test(c: char) -> bool {
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
}

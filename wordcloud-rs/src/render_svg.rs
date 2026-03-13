use std::fmt::Write;

use crate::layout::PlacedWord;

pub fn render_svg(width: u32, height: u32, words: &[PlacedWord]) -> String {
    let mut out = String::new();
    writeln!(
        &mut out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )
    .unwrap();
    writeln!(&mut out, "  <rect width=\"100%\" height=\"100%\" fill=\"#fcfcfa\" />").unwrap();
    writeln!(
        &mut out,
        "  <g font-family=\"Noto Sans CJK SC, PingFang SC, Arial Unicode MS, sans-serif\" text-anchor=\"middle\" dominant-baseline=\"central\">"
    )
    .unwrap();

    for word in words {
        let text = escape_xml(&word.text);
        if word.rotate == 0 {
            writeln!(
                &mut out,
                "    <text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{:.2}\" fill=\"{}\">{}</text>",
                word.x, word.y, word.font_size, word.color, text
            )
            .unwrap();
        } else {
            writeln!(
                &mut out,
                "    <text x=\"{:.2}\" y=\"{:.2}\" font-size=\"{:.2}\" fill=\"{}\" transform=\"rotate({} {:.2} {:.2})\">{}</text>",
                word.x, word.y, word.font_size, word.color, word.rotate, word.x, word.y, text
            )
            .unwrap();
        }
    }

    writeln!(&mut out, "  </g>").unwrap();
    writeln!(&mut out, "</svg>").unwrap();
    out
}

fn escape_xml(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_escape_works() {
        let s = escape_xml("a&b<'\"");
        assert_eq!(s, "a&amp;b&lt;&apos;&quot;");
    }
}

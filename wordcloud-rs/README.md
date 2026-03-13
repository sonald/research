# wordcloud-rs

A minimal-dependency Rust CLI that generates word cloud visualizations from long text.

## Features

- Input modes: local file, `stdin`, URL
- Output files:
  - `wordcloud.svg`
  - `wordcloud.json`
  - `source.md`
- URL cache with TTL (default 24 hours)
- Mixed English/Chinese tokenization
  - English: alnum + `'`
  - Chinese: bigram sliding window
- Self-implemented word cloud layout (spiral placement + collision check)

## Build

```bash
cargo build --release
```

## Usage

```bash
wordcloud-rs --input <file> --out-dir <dir> [options]
wordcloud-rs --stdin --out-dir <dir> [options]
wordcloud-rs --url <url> --out-dir <dir> [options]
```

### Options

- `--top-n <n>` default `120`
- `--min-count <n>` default `2`
- `--width <px>` default `1600`
- `--height <px>` default `1000`
- `--seed <n>` default `42`
- `--cache-dir <path>` default `~/.cache/wordcloud-rs`
- `--cache-ttl-hours <h>` default `24`
- `--stopwords <file>` optional, one token per line

## Examples

```bash
# file input
cargo run -- --input ./sample.txt --out-dir ./out

# stdin input
cat ./sample.txt | cargo run -- --stdin --out-dir ./out

# url input
cargo run -- --url "https://example.com/article.md" --out-dir ./out
```

## Notes

- PDF URL processing requires `pdftotext` in `PATH`.
- HTML URL conversion to markdown requires `pandoc` in `PATH`.

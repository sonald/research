# codex-autoresearch

`codex-autoresearch` is a Codex-friendly port of [davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch). It keeps the persisted experiment loop and append-only run log, but swaps Pi-only extension hooks for:

- a local Python controller CLI
- a Codex skill that teaches the agent how to drive the loop
- a terminal dashboard instead of embedded widgets

## What ships in this repo

- `codex_autoresearch/` — Python package and `codex-autoresearch` CLI
- `skills/codex-autoresearch/` — installable Codex skill
- `templates/` — visible skeletons for `autoresearch.md` and `autoresearch.sh`

The runtime session files still match the upstream workflow:

- `autoresearch.md`
- `autoresearch.sh`
- `autoresearch.jsonl`
- `autoresearch.ideas.md`

## Install

### 1. Install the controller CLI

From a local checkout:

```bash
uv tool install --from /absolute/path/to/codex-autoresearch codex-autoresearch
```

For active development in this repo:

```bash
uv run --project /absolute/path/to/codex-autoresearch codex-autoresearch --help
```

After publishing the repo, the same pattern works from GitHub:

```bash
uv tool install --from git+https://github.com/<owner>/codex-autoresearch codex-autoresearch
```

### 2. Install the Codex skill

If the repo is published on GitHub, install the skill from:

```text
skills/codex-autoresearch
```

with your usual Codex skill install flow.

For local use, symlink or copy:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s /absolute/path/to/codex-autoresearch/skills/codex-autoresearch \
  "${CODEX_HOME:-$HOME/.codex}/skills/codex-autoresearch"
```

Restart Codex after installing the skill.

## Usage

### Start a session

Initialize the session metadata and create the default files if they do not exist yet:

```bash
codex-autoresearch init \
  --name "Optimize benchmark loop" \
  --metric-name latency_ms \
  --metric-unit ms \
  --direction lower \
  --scope src tests
```

This will:

- ensure you are inside a git repo
- switch to `codex/autoresearch/<slug>-<YYYYMMDD>`
- create `autoresearch.md` and `autoresearch.sh` if they are missing
- append a config header to `autoresearch.jsonl`

### Run one experiment cycle

```bash
codex-autoresearch trial --description "Reduce allocations in parser"
```

`trial` executes `./autoresearch.sh`, parses `METRIC name=value` lines, and then:

- keeps + commits the change if the primary metric improved
- discards the scoped file changes if the metric regressed or stayed flat
- marks the run as `crash` if the benchmark fails or does not emit the primary metric

If you want manual inspection between the benchmark and the logging step:

```bash
codex-autoresearch run
codex-autoresearch log --description "Compare alternative buffer layout"
```

### Monitor progress

Quick summary:

```bash
codex-autoresearch status
```

Full table:

```bash
codex-autoresearch dashboard --tail 20
```

Live refresh:

```bash
codex-autoresearch dashboard --tail 20 --watch 2
```

## Skill workflow

The bundled skill is meant to be used when the user asks Codex to start or resume autoresearch. Its job is to:

1. infer or ask for the goal, metric, scope, off-limits paths, and constraints
2. read the relevant source files first
3. maintain `autoresearch.md` and `autoresearch.ideas.md`
4. drive the loop with `codex-autoresearch init`, `trial`, `status`, and `dashboard`

Codex does **not** have the Pi extension hooks that automatically re-inject prompts or reopen the loop, so this port intentionally defaults to manual resume from the persisted files.

## JSONL compatibility

The reader accepts the upstream append-only layout:

- config records with `type=config`
- run records without a `type`, as long as they contain the normal run fields

This port adds extra config metadata such as `scope`, `off_limits`, `constraints`, and `branch`, but keeps the log readable for older Pi sessions.

## Testing

Run the test suite with:

```bash
uv run --with pytest pytest
```

The integration tests create temporary git repos and exercise:

- `init`
- baseline keep
- worse discard + scoped restore
- crash + scoped restore
- status/dashboard rendering

## Differences from Pi

- No embedded widget or hotkey overlays
- No Pi tool registration or prompt hooks
- Manual resume is the default
- Dashboard output is command-driven text, not an inline TUI panel

## Attribution

This project is inspired by and operationally modeled on `pi-autoresearch` by Tobi Lutke and David Cortés. The repository preserves MIT licensing and documents the upstream source in this README.


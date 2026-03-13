---
name: codex-autoresearch
description: Set up and run an autonomous experiment loop in Codex using the local `codex-autoresearch` controller. Use when asked to run autoresearch, optimize a benchmark in a loop, keep/discard experiments automatically, or resume a persisted experiment session.
---

# Codex Autoresearch

Use this skill when the user wants an autonomous benchmark loop in Codex. This is the Codex port of `pi-autoresearch`: the workflow stays the same, but the implementation uses a local CLI and persisted session files instead of Pi extension APIs.

## Prerequisite check

Before using the workflow, confirm the controller is available:

```bash
command -v codex-autoresearch >/dev/null 2>&1
```

If it is missing and you are inside the project repo, use:

```bash
uv run --project /absolute/path/to/codex-autoresearch codex-autoresearch --help
```

If neither works, stop and tell the user the controller must be installed before the skill can drive the loop.

## Files the loop uses

- `autoresearch.md` — session document and resume source of truth
- `autoresearch.sh` — benchmark script that emits `METRIC name=value`
- `autoresearch.jsonl` — append-only history
- `autoresearch.ideas.md` — deferred ideas backlog

## Workflow

1. Ask or infer the optimization goal, primary metric, direction, scope, off-limits paths, and hard constraints.
2. Read the relevant source files before writing anything.
3. If `autoresearch.md` or `autoresearch.sh` is missing, create or refine them.
4. Initialize the session:

```bash
codex-autoresearch init \
  --name "<goal>" \
  --metric-name "<metric>" \
  --metric-unit "<unit>" \
  --direction lower \
  --scope path/one path/two
```

Add `--off-limit` and `--constraint` flags when needed.

5. Run the baseline and each iteration with:

```bash
codex-autoresearch trial --description "<what this iteration changes>"
```

6. Use `codex-autoresearch status` for a quick summary and `codex-autoresearch dashboard --tail 20` for the full table.

## Resume behavior

- If `autoresearch.md` already exists, read it first.
- Then read `autoresearch.jsonl` or run `codex-autoresearch dashboard` to understand the latest state.
- Continue the loop instead of starting a new session unless the optimization target itself changed.

## Guardrails

- `autoresearch.sh` must emit the primary metric exactly as configured.
- Keep the scope list narrow; discard/crash only reverts files inside scope.
- Update `autoresearch.md` as you learn, especially `What's Been Tried`.
- Put promising but deferred ideas into `autoresearch.ideas.md`.
- Do not ask whether to continue in the middle of an active experiment loop unless the user explicitly interrupts the work.


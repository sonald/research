from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from .git_ops import GitError, checkout_branch, commit_keep, current_branch, ensure_git_repo, head_commit, restore_scope, working_tree_dirty
from .models import ConfigRecord, Direction, LastRun, RunRecord, RunStatus
from .rendering import render_dashboard, render_status
from .session import (
    AUTORESEARCH_SH,
    SessionError,
    append_config,
    append_run,
    choose_status,
    default_branch_name,
    delete_last_run,
    ensure_session_files,
    load_last_run,
    load_session,
    parse_metric_lines,
    session_sh_path,
    write_last_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codex-autoresearch")
    parser.add_argument("--cwd", default=".", help="Working directory containing the target git repo.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize or reinitialize an autoresearch session.")
    init_parser.add_argument("--name", required=True)
    init_parser.add_argument("--metric-name", required=True)
    init_parser.add_argument("--metric-unit", default="")
    init_parser.add_argument("--direction", choices=[item.value for item in Direction], default=Direction.LOWER.value)
    init_parser.add_argument("--scope", nargs="+", required=True)
    init_parser.add_argument("--off-limit", action="append", default=[])
    init_parser.add_argument("--constraint", action="append", default=[])
    init_parser.add_argument("--branch")
    init_parser.add_argument("--allow-dirty", action="store_true")

    run_parser = subparsers.add_parser("run", help="Execute ./autoresearch.sh and cache the result.")
    run_parser.add_argument("--timeout", type=int, default=600)

    log_parser = subparsers.add_parser("log", help="Consume the cached run and append it to autoresearch.jsonl.")
    log_parser.add_argument("--description", required=True)

    trial_parser = subparsers.add_parser("trial", help="Run and log one experiment cycle.")
    trial_parser.add_argument("--description", required=True)
    trial_parser.add_argument("--timeout", type=int, default=600)

    subparsers.add_parser("status", help="Show one-line session summary.")

    dashboard_parser = subparsers.add_parser("dashboard", help="Render a terminal dashboard.")
    dashboard_parser.add_argument("--tail", type=int, default=20)
    dashboard_parser.add_argument("--watch", type=float, default=0.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cwd = Path(args.cwd).resolve()

    try:
        if args.command == "init":
            return cmd_init(cwd, args)
        if args.command == "run":
            return cmd_run(cwd, args.timeout)
        if args.command == "log":
            return cmd_log(cwd, args.description)
        if args.command == "trial":
            run_code = cmd_run(cwd, args.timeout)
            if run_code != 0:
                return run_code
            return cmd_log(cwd, args.description)
        if args.command == "status":
            return cmd_status(cwd)
        if args.command == "dashboard":
            return cmd_dashboard(cwd, args.tail, args.watch)
    except (SessionError, GitError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unsupported command {args.command!r}")
    return 2


def cmd_init(cwd: Path, args: argparse.Namespace) -> int:
    ensure_git_repo(cwd)
    if not args.allow_dirty and working_tree_dirty(cwd):
        raise GitError("Working tree is dirty. Commit or stash changes first, or pass --allow-dirty.")

    state = load_session(cwd)
    segment = state.current_segment + 1
    branch = args.branch or default_branch_name(args.name)
    if current_branch(cwd) != branch:
        checkout_branch(cwd, branch)

    config = ConfigRecord(
        name=args.name,
        metric_name=args.metric_name,
        metric_unit=args.metric_unit,
        direction=Direction(args.direction),
        scope=args.scope,
        off_limits=args.off_limit,
        constraints=args.constraint,
        branch=branch,
        segment=segment,
    )
    ensure_session_files(cwd, config)
    append_config(cwd, config)

    print(
        textwrap.dedent(
            f"""\
            Initialized session "{config.name}"
            metric: {config.metric_name} ({config.metric_unit or 'unitless'}, {config.direction.value} is better)
            branch: {config.branch}
            scope: {', '.join(config.scope)}
            """
        ).strip()
    )
    return 0


def cmd_run(cwd: Path, timeout: int) -> int:
    state = load_session(cwd)
    config = state.current_config
    if config is None:
        raise SessionError("No config found. Run `codex-autoresearch init` first.")

    script_path = session_sh_path(cwd)
    if not script_path.exists():
        raise SessionError(f"{AUTORESEARCH_SH} does not exist.")

    started_at = time.time()
    timed_out = False
    parse_error = None

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        exit_code: int | None = result.returncode
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        combined = "\n".join(part for part in ((exc.stdout or ""), (exc.stderr or "")) if part).strip()

    duration_seconds = time.time() - started_at
    parsed_metrics = parse_metric_lines(combined)
    primary_metric = parsed_metrics.pop(config.metric_name, None)
    if not timed_out and exit_code == 0 and primary_metric is None:
        parse_error = f"Primary metric {config.metric_name!r} was not emitted by {AUTORESEARCH_SH}."

    tail_lines = combined.splitlines()[-80:]
    last_run = LastRun(
        command=f"bash {AUTORESEARCH_SH}",
        duration_seconds=duration_seconds,
        exit_code=exit_code,
        timed_out=timed_out,
        parse_error=parse_error,
        primary_metric=primary_metric,
        metrics=parsed_metrics,
        tail_output="\n".join(tail_lines),
        timestamp=int(time.time() * 1000),
    )
    write_last_run(cwd, last_run)

    status = "TIMEOUT" if timed_out else "PASSED" if last_run.succeeded else "FAILED"
    metric_display = "n/a" if primary_metric is None else f"{primary_metric:g}{config.metric_unit}"
    print(f"{status} in {duration_seconds:.2f}s | {config.metric_name}={metric_display}")
    if last_run.tail_output:
        print()
        print(last_run.tail_output)
    return 0


def cmd_log(cwd: Path, description: str) -> int:
    state = load_session(cwd)
    config = state.current_config
    if config is None:
        raise SessionError("No config found. Run `codex-autoresearch init` first.")

    last_run = load_last_run(cwd)
    status = choose_status(config, state, last_run)

    if status is RunStatus.KEEP:
        assert last_run.primary_metric is not None
        commit, commit_note = commit_keep(
            cwd=cwd,
            description=description,
            primary_metric_name=config.metric_name,
            metric=last_run.primary_metric,
            metrics=last_run.metrics,
            scope=config.scope,
        )
    else:
        restore_scope(cwd, config.scope)
        commit = head_commit(cwd)
        commit_note = "restored scope"

    metric_value = last_run.primary_metric if last_run.primary_metric is not None else 0.0
    run = RunRecord(
        run=state.next_run_number,
        segment=config.segment,
        metric=metric_value,
        metrics=last_run.metrics,
        status=status,
        description=description,
        timestamp=last_run.timestamp,
        commit=commit,
        duration_seconds=last_run.duration_seconds,
        exit_code=last_run.exit_code,
    )
    append_run(cwd, run)
    delete_last_run(cwd)

    baseline = load_session(cwd).baseline_run
    baseline_text = f"{baseline.metric:g}{config.metric_unit}" if baseline else "n/a"
    print(
        f"logged run #{run.run}: {run.status.value} | metric={metric_value:g}{config.metric_unit} | "
        f"baseline={baseline_text} | git={commit_note}"
    )
    return 0


def cmd_status(cwd: Path) -> int:
    print(render_status(load_session(cwd)))
    return 0


def cmd_dashboard(cwd: Path, tail: int, watch: float) -> int:
    if watch <= 0:
        print(render_dashboard(load_session(cwd), tail=tail, cwd=cwd))
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")
            print(render_dashboard(load_session(cwd), tail=tail, cwd=cwd))
            time.sleep(watch)
    except KeyboardInterrupt:
        return 0


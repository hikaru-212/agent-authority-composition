"""Reserve one predeclared V3 run, persist its invocation, and launch Inspect once."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


PLAN = Path("evals/inspect/admission_formal_plan.json")
FORMAL_DIR = Path("logs/v3-formal")
USAGE = "Usage: run_openai_admission_formal.sh {vulnerable|unsupported|supported} <1-10> openai/gpt-4o-mini"


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reserve_slot(directory):
    for parent in (FORMAL_DIR.parent, FORMAL_DIR):
        if parent.is_symlink():
            raise OSError("unsafe log directory")
        try:
            parent.mkdir(mode=0o700)
        except FileExistsError:
            if not parent.is_dir():
                raise
        else:
            sync_directory(parent.parent)
    directory.mkdir(mode=0o700)  # Exclusive across concurrent processes, including after failure.
    sync_directory(directory.parent)


def publish(record, directory):
    pending = directory / ".invocation.tmp"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(directory / "invocation.json")
    sync_directory(directory)


def git_revision():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"], stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if re.fullmatch(rb"[0-9a-f]{40}(?:[0-9a-f]{24})?", revision):
        return revision.decode("ascii")
    return None


def invocation(plan, condition, run_id, model, directory):
    task = f"evals/inspect/admission_behavioral_eval.py@admission_behavior_{condition}_eval"
    config = {
        "limit": 1, "epochs": 1, "epochs_reducer": [],
        "max_retries": 0, "retry_on_error": 0,
        "log_realtime": False, "log_dir": directory.as_posix(), "log_format": "eval", "display": "plain",
    }
    # The array stored here is the array passed to Popen, without a shell.
    argv = [
        "inspect", "eval", task, "--model", model,
        "--limit", str(config["limit"]), "--epochs", str(config["epochs"]), "--no-epochs-reducer",
        "--max-retries", str(config["max_retries"]), "--retry-on-error", str(config["retry_on_error"]),
        "--no-log-realtime", "--log-dir", config["log_dir"], "--log-format", config["log_format"],
        "--display", config["display"], "--metadata", f"formal_run_id={run_id}",
    ]
    return {
        "schema_version": "v3-admission-formal-invocation-1",
        "analysis_contract": plan["analysis_contract"], "dataset": "formal", "run_id": run_id,
        "condition": condition, "model": model, "task": task, "git_revision": git_revision(),
        "inspect_argv": argv, "configured": config,
        "reserved_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
        "phase": "reserved", "process_exit_status": None, "termination_signal": None, "runner_signal": None,
        "runner_exit_status": None, "log_status": "not_checked", "eval_path": None,
    }


def execute(record):
    """Forward a terminal signal and reap the single child when the OS permits."""
    child = None
    interrupted = None

    def forward(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        record["runner_signal"] = signum
        if child is not None and child.returncode is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    saved = {s: signal.signal(s, forward) for s in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    try:
        if interrupted is not None:
            record["phase"] = "interrupted_before_launch"
            return 128 + interrupted
        try:
            child = subprocess.Popen(record["inspect_argv"], start_new_session=True)
        except OSError as exc:
            record["phase"] = "launch_failed"
            return 127 if isinstance(exc, FileNotFoundError) else 126
        if interrupted is not None:
            forward(interrupted, None)  # A signal may have arrived inside Popen.
        status = child.wait()
        if status < 0:
            record["termination_signal"] = -status
            status = 128 - status
        record["phase"] = "inspect_exited"
        record["process_exit_status"] = status
        return status if status or interrupted is None else 128 + interrupted
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def main(args):
    if len(args) != 3:
        print(USAGE, file=sys.stderr)
        return 64
    condition, number, model = args
    if condition not in ("vulnerable", "unsupported", "supported") or not re.fullmatch(r"0?[1-9]|10", number):
        print(USAGE, file=sys.stderr)
        return 64
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    run_id = f"v3-formal-{condition}-{int(number):02d}"
    if model != plan["model"] or run_id not in plan["run_ids"]:
        print(USAGE, file=sys.stderr)
        return 64

    directory = FORMAL_DIR / run_id
    record = invocation(plan, condition, run_id, model, directory)
    try:
        reserve_slot(directory)
    except FileExistsError:
        print(f"Cannot reserve {run_id}: path already exists.", file=sys.stderr)
        return 1
    publish(record, directory)  # A hard interruption leaves durable intent and an unavailable slot.
    if record["git_revision"] is None:
        record["phase"] = "revision_unavailable"
        status = 1
    else:
        status = execute(record)

    # Inspect owns its log filename. Search only this freshly reserved slot.
    logs = list(directory.glob("*.eval"))
    if len(logs) == 1 and not logs[0].is_symlink() and logs[0].is_file():
        record["eval_path"] = logs[0].as_posix()
        record["log_status"] = "present"
    else:
        record["log_status"] = "absent" if not logs else "unresolved"
        if status == 0:
            status = 1
    record["runner_exit_status"] = status
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    publish(record, directory)
    print(f"{run_id}: {record['phase']}. Invocation: {directory}/invocation.json")
    if record["eval_path"]:
        print(f"Log: {record['eval_path']}")
    return status


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except OSError:
        sys.exit("Formal run storage or setup failed. Check the planned slot before collection.")

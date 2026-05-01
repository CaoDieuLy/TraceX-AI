"""
Tracking progress monitor for the VPS + LightningAI pipeline.

Examples:
  .\.venv\Scripts\python.exe scripts\progress.py --once
  .\.venv\Scripts\python.exe scripts\progress.py --watch --interval 30 --output tracking_progress.log
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import paramiko


DEFAULT_VPS_HOST = os.getenv("TRACKING_MONITOR_VPS_HOST", "103.72.56.139")
DEFAULT_VPS_USER = os.getenv("TRACKING_MONITOR_VPS_USER", "root")
DEFAULT_VPS_PASSWORD = os.getenv("TRACKING_MONITOR_VPS_PASSWORD", "xVrqU6Dg@VhAcj4")
DEFAULT_CLOUD_URL = os.getenv(
    "TRACKING_MONITOR_CLOUD_URL",
    "https://8000-01kpr5td5by2gc0hjq4s6zwsbt.cloudspaces.litng.ai",
)
DEFAULT_AUTH_TOKEN = os.getenv("TRACKING_MONITOR_AUTH_TOKEN", "abc123")
DEFAULT_OUTPUT = os.getenv("TRACKING_MONITOR_OUTPUT", "tracking_progress.log")

LOG_CMD = (
    "docker logs mcpt-backend --since 2026-05-01T00:00:00 2>&1 | "
    "grep -E \"Async ingestion job queued|Ingestion job .* status=|Ingestion job .* completed people=|"
    "Saved to DB:|Queue sync iteration failed|Requesting tracking ingestion\""
)
DB_CMD = (
    "docker exec mcpt-postgres psql -U mcpt_user -d video_tracking "
    "-t -A -F'|' -c \"SELECT camera_id, COUNT(*) AS count "
    "FROM person_candidates GROUP BY camera_id ORDER BY camera_id;\" 2>&1"
)
TIME_CMD = "date -u +%Y-%m-%dT%H:%M:%SZ"

QUEUED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?Async ingestion job queued "
    r"job_id=(?P<job>[0-9a-f-]+) source_filename=(?P<file>\S+) camera_id=(?P<camera>\S+)"
)
STATUS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?Ingestion job "
    r"(?P<job>[0-9a-f-]+) status=(?P<status>\w+) source_filename=(?P<file>\S+)"
)
COMPLETED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?Ingestion job "
    r"(?P<job>[0-9a-f-]+) completed people=(?P<people>\d+)"
)
SAVED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?Saved to DB: "
    r"(?P<file>\S+) people=(?P<people>\d+)"
)
REQUEST_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?Requesting tracking ingestion "
    r"source_filename=(?P<file>\S+) camera_id=(?P<camera>\S+)"
)


@dataclass
class JobRecord:
    job_id: str
    camera_id: str = ""
    source_filename: str = ""
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    saved_at: datetime | None = None
    status: str = "unknown"
    people: int | None = None


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def elapsed_str(started_at: datetime | None, now_utc: datetime) -> str:
    if not started_at:
        return "-"
    seconds = max(0.0, (now_utc - started_at).total_seconds())
    return f"{seconds / 60:.1f}m"


def duration_str(started_at: datetime | None, finished_at: datetime | None) -> str:
    if not started_at or not finished_at:
        return "-"
    seconds = max(0.0, (finished_at - started_at).total_seconds())
    return f"{seconds / 60:.1f}m"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(none)"
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    fmt = " | ".join("{:<" + str(width) + "}" for width in widths)
    sep = "-+-".join("-" * width for width in widths)
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def run_remote(client: paramiko.SSHClient, cmd: str) -> str:
    _stdin, stdout, stderr = client.exec_command(cmd)
    output = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    return output + err


def fetch_job_status(
    client: paramiko.SSHClient,
    cloud_url: str,
    auth_token: str,
    job_id: str,
) -> dict | None:
    cmd = (
        f"curl -s --max-time 20 -H \"Authorization: Bearer {auth_token}\" "
        f"{cloud_url.rstrip('/')}/api/v1/ingestion/status/{job_id}"
    )
    raw = run_remote(client, cmd).strip()
    if not raw or "Ingestion job not found" in raw:
        return None
    try:
        import json

        return json.loads(raw)
    except Exception:
        return None


def parse_jobs(log_text: str) -> tuple[dict[str, JobRecord], list[str]]:
    jobs: dict[str, JobRecord] = {}
    recent_events: list[str] = []
    file_to_camera: dict[str, str] = {}

    for raw_line in log_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        recent_events.append(line)
        recent_events = recent_events[-20:]

        match = REQUEST_RE.search(line)
        if match:
            file_to_camera[match.group("file")] = match.group("camera")
            continue

        match = QUEUED_RE.search(line)
        if match:
            job_id = match.group("job")
            job = jobs.setdefault(job_id, JobRecord(job_id=job_id))
            job.camera_id = match.group("camera")
            job.source_filename = match.group("file")
            job.queued_at = parse_ts(match.group("ts"))
            job.status = "queued"
            continue

        match = STATUS_RE.search(line)
        if match:
            job_id = match.group("job")
            job = jobs.setdefault(job_id, JobRecord(job_id=job_id))
            job.source_filename = job.source_filename or match.group("file")
            job.camera_id = job.camera_id or file_to_camera.get(job.source_filename, "")
            job.status = match.group("status")
            if job.status == "processing" and job.started_at is None:
                job.started_at = parse_ts(match.group("ts"))
            continue

        match = COMPLETED_RE.search(line)
        if match:
            job_id = match.group("job")
            job = jobs.setdefault(job_id, JobRecord(job_id=job_id))
            job.completed_at = parse_ts(match.group("ts"))
            job.status = "done"
            job.people = int(match.group("people"))
            continue

        match = SAVED_RE.search(line)
        if match:
            filename = match.group("file")
            ts = parse_ts(match.group("ts"))
            people = int(match.group("people"))
            for job in jobs.values():
                if job.source_filename == filename and job.saved_at is None:
                    job.saved_at = ts
                    job.people = people
                    if job.status not in {"processing", "queued"}:
                        job.status = "saved"
            continue

    for job in jobs.values():
        if not job.camera_id:
            job.camera_id = file_to_camera.get(job.source_filename, "")

    return jobs, recent_events


def build_snapshot(
    *,
    client: paramiko.SSHClient,
    cloud_url: str,
    auth_token: str,
) -> str:
    log_text = run_remote(client, LOG_CMD)
    db_text = run_remote(client, DB_CMD)
    now_raw = run_remote(client, TIME_CMD).strip()
    now_utc = parse_iso_utc(now_raw) or datetime.now(timezone.utc)

    jobs, recent_events = parse_jobs(log_text)

    active_jobs = [job for job in jobs.values() if job.status in {"queued", "processing"}]
    for job in active_jobs:
        live = fetch_job_status(client, cloud_url, auth_token, job.job_id)
        if not live:
            job.status = "stale"
            continue
        job.status = str(live.get("status") or job.status)
        job.queued_at = parse_iso_utc(str(live.get("queued_at") or "")) or job.queued_at
        job.started_at = parse_iso_utc(str(live.get("started_at") or "")) or job.started_at
        job.source_filename = str(live.get("source_filename") or job.source_filename)
        job.camera_id = str(live.get("camera_id") or job.camera_id)

    done_rows: list[list[str]] = []
    running_rows: list[list[str]] = []
    failed_rows: list[list[str]] = []
    stale_rows: list[list[str]] = []

    for job in sorted(jobs.values(), key=lambda item: item.queued_at or datetime.min.replace(tzinfo=timezone.utc)):
        if job.status in {"done", "saved"}:
            finished_at = job.saved_at or job.completed_at
            done_rows.append(
                [
                    job.camera_id or "?",
                    job.source_filename or "?",
                    (job.started_at or job.queued_at).strftime("%H:%M:%S") if (job.started_at or job.queued_at) else "-",
                    finished_at.strftime("%H:%M:%S") if finished_at else "-",
                    duration_str(job.started_at or job.queued_at, finished_at),
                    str(job.people or 0),
                ]
            )
        elif job.status in {"queued", "processing"}:
            running_rows.append(
                [
                    job.camera_id or "?",
                    job.source_filename or "?",
                    job.status,
                    (job.started_at or job.queued_at).strftime("%H:%M:%S") if (job.started_at or job.queued_at) else "-",
                    elapsed_str(job.started_at or job.queued_at, now_utc),
                    job.job_id,
                ]
            )
        elif job.status == "stale":
            stale_rows.append(
                [
                    job.camera_id or "?",
                    job.source_filename or "?",
                    (job.started_at or job.queued_at).strftime("%H:%M:%S") if (job.started_at or job.queued_at) else "-",
                    job.job_id,
                ]
            )
        elif job.status == "failed":
            failed_rows.append(
                [
                    job.camera_id or "?",
                    job.source_filename or "?",
                    (job.started_at or job.queued_at).strftime("%H:%M:%S") if (job.started_at or job.queued_at) else "-",
                    job.job_id,
                ]
            )

    db_rows: list[list[str]] = []
    for line in db_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("("):
            continue
        parts = [part.strip() for part in stripped.split("|")]
        if len(parts) == 2 and parts[0]:
            db_rows.append([parts[0], parts[1]])

    lines: list[str] = []
    lines.append(f"=== SNAPSHOT {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ===")
    lines.append(f"DONE: {len(done_rows)}")
    lines.append(f"RUNNING: {len(running_rows)}")
    lines.append(f"FAILED: {len(failed_rows)}")
    lines.append("")
    lines.append("=== DONE ===")
    lines.append(
        format_table(
            ["camera", "file", "start_utc", "end_utc", "duration", "people"],
            done_rows,
        )
    )
    lines.append("")
    lines.append("=== RUNNING ===")
    lines.append(
        format_table(
            ["camera", "file", "status", "start_utc", "elapsed", "job_id"],
            running_rows,
        )
    )
    lines.append("")
    lines.append("=== FAILED ===")
    lines.append(
        format_table(
            ["camera", "file", "last_seen_utc", "job_id"],
            failed_rows,
        )
    )
    lines.append("")
    lines.append("=== STALE ===")
    lines.append(
        format_table(
            ["camera", "file", "last_seen_utc", "job_id"],
            stale_rows,
        )
    )
    lines.append("")
    lines.append("=== DB ===")
    lines.append(format_table(["camera", "count"], db_rows))
    lines.append("")
    lines.append("=== RECENT EVENTS ===")
    lines.extend(recent_events or ["(none)"])
    lines.append("")
    return "\n".join(lines)


def write_snapshot(output_path: Path, content: str, append: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor tracking progress and write snapshots to a log file.")
    parser.add_argument("--host", default=DEFAULT_VPS_HOST)
    parser.add_argument("--user", default=DEFAULT_VPS_USER)
    parser.add_argument("--password", default=DEFAULT_VPS_PASSWORD)
    parser.add_argument("--cloud-url", default=DEFAULT_CLOUD_URL)
    parser.add_argument("--auth-token", default=DEFAULT_AUTH_TOKEN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file on each snapshot instead of appending.",
    )
    return parser.parse_args()


def connect_client(host: str, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=20, allow_agent=False, look_for_keys=False)
    return client


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    append = not args.overwrite

    while True:
        try:
            client = connect_client(args.host, args.user, args.password)
            try:
                snapshot = build_snapshot(
                    client=client,
                    cloud_url=args.cloud_url,
                    auth_token=args.auth_token,
                )
            finally:
                client.close()
            write_snapshot(output_path, snapshot, append=append)
            print(f"[progress] wrote snapshot to {output_path}")
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            error_block = (
                f"=== SNAPSHOT ERROR {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ===\n"
                f"{type(exc).__name__}: {exc}\n"
            )
            write_snapshot(output_path, error_block, append=True)
            print(f"[progress] error: {exc}", file=sys.stderr)

        if args.once:
            return 0
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())

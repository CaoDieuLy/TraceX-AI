#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared_secret_runtime import canonical_secrets_root, ensure_canonical_secret_dirs, load_runtime_env

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "docs",
    "node_modules",
    ".next",
    "dist",
    "build",
    "secret_backups",
    "storage",
    "postgres_data",
}

TEXT_SUFFIXES = {
    ".env",
    ".py",
    ".sh",
    ".yml",
    ".yaml",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".tsx",
    ".ts",
    ".js",
}

SENSITIVE_FILE_FINDINGS = [
    ("secrets/oauth/oauth2_credentials.json", "OAuth client secret", "OAuth", "high"),
    ("secrets/oauth/oauth2_token.pickle", "OAuth refresh token", "Token", "high"),
    ("oauth2_credentials.json", "Legacy OAuth client secret still exists outside canonical secrets folder", "OAuth", "high"),
    ("oauth2_token.pickle", "Legacy OAuth refresh token still exists outside canonical secrets folder", "Token", "high"),
    (
        "Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/tracking-service/.env",
        "Legacy service env still exists outside canonical secrets folder",
        "Hardcoded config",
        "high",
    ),
    (
        "Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/tracking-service/credentials/mcpt-tracker-sa.json",
        "Legacy Google Drive auth file still exists outside canonical secrets folder",
        "OAuth",
        "high",
    ),
]

LINE_PATTERNS = [
    (re.compile(r"^\s*LIGHTNING_API_TOKEN\s*=\s*(?!\s*$)(?!your_|<|change-|sk-...)", re.IGNORECASE), "API token", "API key", "high"),
    (re.compile(r"^\s*POSTGRES_PASSWORD\s*=\s*(?!\s*$)(?!your_|change-|example)", re.IGNORECASE), "Database password", "DB config", "high"),
    (re.compile(r"^\s*JWT_SECRET_KEY\s*=\s*(?!\s*$)(?!your_|change-)", re.IGNORECASE), "JWT secret", "Token", "high"),
    (re.compile(r"^\s*GOOGLE_DRIVE_(ROOT|VINUNI)_FOLDER_ID\s*=\s*(?!\s*$)", re.IGNORECASE), "Drive folder identifier", "Hardcoded URL / endpoint", "medium"),
    (re.compile(r"/teamspace/studios/this_studio|/workspace/project/Multi-Camera-Person-Tracking-and-Re-Identification"), "Machine-specific absolute path", "Hardcoded config", "medium"),
    (re.compile(r"mcpt_password|change-me-postgres-password|change-me-in-production|change-this-jwt-secret"), "Unsafe default secret placeholder", "Hardcoded config", "medium"),
    (re.compile(r"google-drive/drive-sa\.json"), "Deprecated service-account path reference", "OAuth", "medium"),
]

TRACKED_SECRET_PATTERNS = [
    "secrets/shared.env",
    "oauth2_credentials.json",
    "oauth2_token.pickle",
    "secrets/oauth/oauth2_credentials.json",
    "secrets/oauth/oauth2_token.pickle",
]

LEFTOVER_ARTIFACT_PATTERNS = [
    "Multi-Camera-Person-Tracking-and-Re-Identification/create_test_video.py",
    "Multi-Camera-Person-Tracking-and-Re-Identification/create_long_test_video.py",
    "Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/demo.py",
    "Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/torchreid/metrics/rank_cylib/test_cython.py",
    "Multi-Camera-Person-Tracking-and-Re-Identification/test_videos/test_sample.mp4",
    "Multi-Camera-Person-Tracking-and-Re-Identification/test_videos/test_sample.h265",
    "Multi-Camera-Person-Tracking-and-Re-Identification/test_videos/test_sample_long.mp4",
    "Multi-Camera-Person-Tracking-and-Re-Identification/test_videos/test_sample_long.h265",
]


@dataclass
class Finding:
    path: str
    line: int
    info_type: str
    risk: str
    detail: str


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDED_DIRS:
                continue
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.endswith(".env"):
            yield path


def scan_line_patterns() -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_text_files(REPO_ROOT):
        relpath = path.relative_to(REPO_ROOT).as_posix()
        if path.name == "README.md" or relpath == "scripts/check_secret_readiness.py":
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        is_env_like = path.suffix.lower() == ".env" or path.name.endswith(".env") or path.name == "shared.env"
        for index, line in enumerate(content, start=1):
            for pattern, detail, info_type, risk in LINE_PATTERNS:
                if not pattern.search(line):
                    continue
                if detail in {"API token", "Database password", "JWT secret"} and not is_env_like:
                    continue
                if relpath.endswith(".example") and risk == "high":
                    risk = "low"
                    detail = f"{detail} in example/template file"
                findings.append(Finding(relpath, index, info_type, risk, detail))
                break
    return findings


def scan_sensitive_files() -> list[Finding]:
    findings: list[Finding] = []
    for relpath, detail, info_type, risk in SENSITIVE_FILE_FINDINGS:
        path = REPO_ROOT / relpath
        if path.exists():
            findings.append(Finding(relpath, 1, info_type, risk, detail))
    return findings


def scan_tracked_secret_files() -> list[Finding]:
    findings: list[Finding] = []
    result = subprocess.run(
        ["git", "ls-files", *TRACKED_SECRET_PATTERNS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if line.strip():
            findings.append(Finding(line.strip(), 1, "Hardcoded config", "high", "Secret file is still tracked by git"))
    return findings


def scan_leftover_artifacts() -> list[Finding]:
    findings: list[Finding] = []
    for relpath in LEFTOVER_ARTIFACT_PATTERNS:
        if (REPO_ROOT / relpath).exists():
            findings.append(Finding(relpath, 1, "Hardcoded config", "medium", "Leftover test/mock/sample artifact"))
    return findings


def collect_findings() -> list[Finding]:
    findings = scan_sensitive_files() + scan_tracked_secret_files() + scan_leftover_artifacts() + scan_line_patterns()
    deduped: dict[tuple[str, int, str, str], Finding] = {}
    for finding in findings:
        key = (finding.path, finding.line, finding.info_type, finding.detail)
        existing = deduped.get(key)
        if existing is None or (existing.risk == "low" and finding.risk in {"medium", "high"}):
            deduped[key] = finding
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(deduped.values(), key=lambda item: (order[item.risk], item.path, item.line))


def render_markdown(findings: list[Finding]) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        f"# Secret Audit Report ({now})",
        "",
        "## Scope",
        "",
        f"- Repo: `{REPO_ROOT}`",
        f"- Canonical secrets root: `{canonical_secrets_root()}`",
        f"- Findings: `{len(findings)}`",
        "",
        "## Findings",
        "",
        "| Risk | Type | File | Line | Detail |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for finding in findings:
        lines.append(
            f"| {finding.risk} | {finding.info_type} | `{finding.path}` | {finding.line} | {finding.detail} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Canonical Structure",
            "",
            "```text",
            "secrets/",
            "  shared.env",
            "  env/",
            "  db/",
            "  api/",
            "  oauth/",
            "    oauth2_credentials.json",
            "    oauth2_token.pickle",
            "  deploy/",
            "  docker/",
            "  gpu/",
            "```",
            "",
            "## Portable Run Rule",
            "",
            "1. Clone repo.",
            "2. Copy back the `secrets/` folder or decrypt the encrypted bundle.",
            "3. Run `python scripts/init_project.py`.",
            "4. Start the app stack.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit secret readiness and portability.")
    parser.add_argument("--format", choices={"markdown", "json"}, default="markdown")
    parser.add_argument("--write-report", help="Optional path to write the report to disk.")
    args = parser.parse_args()

    ensure_canonical_secret_dirs()
    load_runtime_env(include_tracking_service_env=True, override=True)
    findings = collect_findings()

    if args.format == "json":
        output = json.dumps([asdict(item) for item in findings], indent=2)
    else:
        output = render_markdown(findings)

    if args.write_report:
        report_path = Path(args.write_report)
        if not report_path.is_absolute():
            report_path = REPO_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output, encoding="utf-8")

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

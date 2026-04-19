#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared_secret_runtime import ensure_canonical_secret_dirs, load_runtime_env


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize portable secret-backed project runtime.")
    parser.add_argument("--bundle", help="Optional encrypted secrets bundle to import first.")
    parser.add_argument("--skip-migrate", action="store_true", help="Skip migration from legacy secret locations.")
    parser.add_argument("--skip-check", action="store_true", help="Skip readiness validation after init.")
    args = parser.parse_args()

    ensure_canonical_secret_dirs()

    if args.bundle:
        run(["bash", "scripts/import_secrets_bundle.sh", args.bundle])

    if not args.skip_migrate:
        run(["python", "scripts/migrate_secrets_to_canonical_layout.py"])

    loaded = load_runtime_env(include_tracking_service_env=True, override=True)
    print("Loaded env files:")
    if loaded:
        for env_file in loaded:
            print(f"  - {env_file}")
    else:
        print("  - none found")

    if not args.skip_check:
        run(["python", "scripts/check_secret_readiness.py"])

    print("Project secret initialization completed.")
    print("Suggested next steps:")
    print("  - Root scripts: python upload_oauth2.py")
    print("  - Compose stack: cd Multi-Camera-Person-Tracking-and-Re-Identification && docker compose --env-file ../secrets/shared.env up --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

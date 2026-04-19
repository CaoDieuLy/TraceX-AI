#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared_secret_runtime import (  # noqa: E402
    APP_ROOT,
    canonical_secrets_root,
    ensure_canonical_secret_dirs,
    resolve_google_drive_service_account_path,
    resolve_oauth2_credentials_path,
    resolve_oauth2_token_path,
    shared_env_file,
    tracking_service_env_file,
)


def copy_if_present(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return True


def merge_env(source_file: Path, destination_file: Path) -> None:
    merged: dict[str, str] = {}
    if destination_file.exists():
        merged.update({k: v for k, v in dotenv_values(destination_file).items() if v is not None})
    if source_file.exists():
        merged.update({k: v for k, v in dotenv_values(source_file).items() if v is not None})
    merged["MCPT_SECRETS_ROOT"] = str(canonical_secrets_root())
    merged["MCPT_SHARED_ENV_FILE"] = str(shared_env_file())
    merged["MCPT_OAUTH2_CREDENTIALS_FILE"] = str(canonical_secrets_root() / "oauth" / "oauth2_credentials.json")
    merged["MCPT_OAUTH2_TOKEN_FILE"] = str(canonical_secrets_root() / "oauth" / "oauth2_token.pickle")
    merged["GOOGLE_DRIVE_CREDENTIALS_FILE"] = str(canonical_secrets_root() / "google-drive" / "drive-sa.json")
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in sorted(merged.items())]
    destination_file.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> int:
    paths = ensure_canonical_secret_dirs()
    shared_env = shared_env_file()
    tracking_env = tracking_service_env_file()
    oauth_credentials = canonical_secrets_root() / "oauth" / "oauth2_credentials.json"
    oauth_token = canonical_secrets_root() / "oauth" / "oauth2_token.pickle"
    drive_service_account = canonical_secrets_root() / "google-drive" / "drive-sa.json"

    print(f"Canonical secrets root: {paths['secrets_root']}")

    merge_env(tracking_env, shared_env)
    print(f"Merged env into: {shared_env}")

    copied_oauth_credentials = copy_if_present(resolve_oauth2_credentials_path(), oauth_credentials)
    copied_oauth_token = copy_if_present(resolve_oauth2_token_path(), oauth_token)

    legacy_drive_sa = APP_ROOT / "backend" / "services" / "tracking-service" / "credentials" / "mcpt-tracker-sa.json"
    copied_drive_sa = copy_if_present(resolve_google_drive_service_account_path(), drive_service_account) or copy_if_present(
        legacy_drive_sa,
        drive_service_account,
    )

    print(f"OAuth2 credentials copied: {'yes' if copied_oauth_credentials else 'no'} -> {oauth_credentials}")
    print(f"OAuth2 token copied: {'yes' if copied_oauth_token else 'no'} -> {oauth_token}")
    print(f"Drive service account copied: {'yes' if copied_drive_sa else 'no'} -> {drive_service_account}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

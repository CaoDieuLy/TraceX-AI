#!/usr/bin/env python3
"""
Quick validation: verify canonical OAuth2 Google Drive credentials resolve and basic API access works.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent / "services" / "tracking-service"))

from shared_secret_runtime import build_google_drive_oauth_service, load_runtime_env, resolve_oauth2_credentials_path, resolve_oauth2_token_path, shared_env_file  # noqa: E402

print("=" * 70)
print("GOOGLE DRIVE CONNECTION TEST")
print("=" * 70)

loaded = load_runtime_env(include_tracking_service_env=True)
print("[1] Loaded env files:")
for env_path in loaded:
    print(f"    - {env_path}")
if not loaded:
    print(f"    - none found (expected shared env at {shared_env_file()})")

from app.config import settings  # noqa: E402

print("[2] Settings loaded:")
print(f"    GOOGLE_DRIVE_ENABLED = {settings.google_drive_enabled}")
print(f"    OAUTH2_CREDENTIALS_FILE = {resolve_oauth2_credentials_path()}")
print(f"    OAUTH2_TOKEN_FILE = {resolve_oauth2_token_path()}")
print(f"    MAKE_PUBLIC = {settings.google_drive_make_public}")
print(f"    ROOT_FOLDER_ID = {getattr(settings, 'google_drive_root_folder_id', 'NOT SET')}")

cred_path = resolve_oauth2_credentials_path().expanduser()
token_path = resolve_oauth2_token_path().expanduser()
print("\n[3] OAuth files:")
if not cred_path.exists():
    print(f"    ❌ Missing credentials: {cred_path}")
    sys.exit(1)
if not token_path.exists():
    print(f"    ❌ Missing token: {token_path}")
    sys.exit(1)
print(f"    ✅ Credentials: {cred_path}")
print(f"    ✅ Token: {token_path}")

print("\n[4] Building Drive service...")
try:
    drive_service = build_google_drive_oauth_service()
    print("    ✅ Drive service created successfully")

    print("\n[5] API call...")
    results = drive_service.files().list(pageSize=1, fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        print(f"    ✅ API works. First file: {files[0]['name']} (id: {files[0]['id']})")
    else:
        print("    ✅ API works. Drive is reachable but currently empty.")
except Exception as exc:
    print(f"    ❌ Error: {exc}")
    sys.exit(1)

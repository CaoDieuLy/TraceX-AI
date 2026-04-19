#!/usr/bin/env python3
"""
Quick validation: verify the canonical Google Drive credential resolves and basic API access works.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent / "services" / "tracking-service"))

from shared_secret_runtime import load_runtime_env, shared_env_file  # noqa: E402

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
print(f"    CREDENTIALS_FILE = {settings.google_drive_credentials_file}")
print(f"    MAKE_PUBLIC = {settings.google_drive_make_public}")
print(f"    ROOT_FOLDER_ID = {getattr(settings, 'google_drive_root_folder_id', 'NOT SET')}")

cred_path = Path(settings.google_drive_credentials_file).expanduser()
print("\n[3] Credential file:")
if not cred_path.exists():
    print(f"    ❌ Missing: {cred_path}")
    sys.exit(1)
print(f"    ✅ Exists: {cred_path}")
print(f"    Size: {cred_path.stat().st_size} bytes")

print("\n[4] Building Drive service...")
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        str(cred_path),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
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

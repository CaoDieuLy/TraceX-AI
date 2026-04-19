#!/usr/bin/env python3
"""
Detailed validation: verify canonical Google Drive config, credentials, and API reachability.
Usage: python test_gdrive_connection.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent.parent))
sys.path.insert(0, str(project_root / "services" / "tracking-service"))

from shared_secret_runtime import load_runtime_env, shared_env_file  # noqa: E402

loaded = load_runtime_env(include_tracking_service_env=True)

print("=" * 70)
print("GOOGLE DRIVE CONNECTION TEST")
print("=" * 70)
print("Loaded env files:")
for env_file in loaded:
    print(f"  - {env_file}")
if not loaded:
    print(f"  - none found (expected shared env at {shared_env_file()})")

print("\n[1/5] Checking configuration...")
try:
    from app.config import settings

    print(f"   GOOGLE_DRIVE_ENABLED: {settings.google_drive_enabled}")
    print(f"   CREDENTIALS_FILE: {settings.google_drive_credentials_file}")
    print(f"   MAKE_PUBLIC: {settings.google_drive_make_public}")
except Exception as exc:
    print(f"   ❌ Config error: {exc}")
    sys.exit(1)

print("\n[2/5] Checking credentials file...")
cred_path = Path(settings.google_drive_credentials_file).expanduser()
if cred_path.exists():
    print(f"   ✅ File exists: {cred_path}")
    print(f"   File size: {cred_path.stat().st_size} bytes")
else:
    print(f"   ❌ File NOT found: {cred_path}")
    print("   → Restore or update GOOGLE_DRIVE_CREDENTIALS_FILE in secrets/shared.env")
    sys.exit(1)

print("\n[3/5] Checking Google API libraries...")
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    print("   ✅ google-auth, google-api-python-client installed")
except ImportError as exc:
    print(f"   ❌ Missing library: {exc}")
    sys.exit(1)

print("\n[4/5] Building Drive service...")
try:
    credentials = service_account.Credentials.from_service_account_file(
        str(cred_path),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    print("   ✅ Drive service created")
except Exception as exc:
    print(f"   ❌ Failed to build service: {exc}")
    sys.exit(1)

print("\n[5/5] Testing Drive API (list files)...")
try:
    results = drive_service.files().list(pageSize=5, fields="files(id, name)").execute()
    files = results.get("files", [])
    print(f"   ✅ API call successful! Found {len(files)} files")
    if files:
        print("   First 3 files:")
        for item in files[:3]:
            print(f"     - {item['name']} (id: {item['id']})")
except Exception as exc:
    print(f"   ❌ API call failed: {exc}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ GOOGLE DRIVE CONNECTED SUCCESSFULLY!")
print("=" * 70)

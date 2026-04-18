#!/usr/bin/env python3
"""
Quick test: Check if credentials file is valid and can connect.
"""

import sys
import os
from pathlib import Path

# Setup path - import trực tiếp từ tracking-service
sys.path.insert(0, str(Path(__file__).parent / "services" / "tracking-service"))

print("=" * 70)
print("GOOGLE DRIVE CONNECTION TEST")
print("=" * 70)

# Load .env từ tracking-service folder
from dotenv import load_dotenv
env_path = Path(__file__).parent / "services" / "tracking-service" / ".env"
print(f"[1] Loading .env from: {env_path}")
load_dotenv(env_path, override=True)

# Now import settings
from app.config import settings
print(f"[2] Settings loaded:")
print(f"    GOOGLE_DRIVE_ENABLED = {settings.google_drive_enabled}")
print(f"    CREDENTIALS_FILE = {settings.google_drive_credentials_file}")
print(f"    MAKE_PUBLIC = {settings.google_drive_make_public}")
print(f"    ROOT_FOLDER_ID = {getattr(settings, 'google_drive_root_folder_id', 'NOT SET')}")
print(f"\n[Config]")
print(f"  GOOGLE_DRIVE_ENABLED: {settings.google_drive_enabled}")
print(f"  CREDENTIALS_FILE: {settings.google_drive_credentials_file}")

# Check file
cred_path = Path(settings.google_drive_credentials_file).expanduser()
print(f"\n[File Check]")
if cred_path.exists():
    print(f"  ✅ File exists: {cred_path}")
    print(f"  Size: {cred_path.stat().st_size} bytes")
else:
    print(f"  ❌ File NOT found: {cred_path}")
    sys.exit(1)

# Try to build Drive service
print(f"\n[Building Drive Service]")
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        str(cred_path),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    print("  ✅ Drive service created successfully!")

    # Quick API test: list 1 file
    print(f"\n[API Test]")
    results = drive_service.files().list(pageSize=1, fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        print(f"  ✅ API works! First file: {files[0]['name']} (id: {files[0]['id']})")
    else:
        print(f"  ✅ API works! (No files in Drive)")

    print("\n" + "=" * 70)
    print("✅ GOOGLE DRIVE CONNECTED SUCCESSFULLY!")
    print("=" * 70)

except Exception as e:
    print(f"  ❌ Error: {e}")
    print("\n" + "=" * 70)
    print("❌ CONNECTION FAILED")
    print("=" * 70)
    print("\nPossible reasons:")
    print("  1. Invalid credentials (JSON malformed)")
    print("  2. Service Account disabled/deleted")
    print("  3. Key revoked (needs new key)")
    print("  4. Drive API not enabled in GCP project")
    print("\nTo fix:")
    print("  1. Go to Google Cloud Console")
    print("  2. IAM & Admin → Service Accounts")
    print("  3. Find: drive-uploader@ambient-fuze-493617-t9.iam.gserviceaccount.com")
    print("  4. Delete old key, create new key")
    print("  5. Download JSON and replace mcpt-tracker-sa.json")
    sys.exit(1)

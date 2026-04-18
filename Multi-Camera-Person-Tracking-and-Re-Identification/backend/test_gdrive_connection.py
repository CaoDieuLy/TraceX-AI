#!/usr/bin/env python3
"""
Test script: Verify Google Drive connection for tracking-service.
Usage: python test_gdrive_connection.py
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "backend" / "services" / "tracking-service"))

print("=" * 70)
print("GOOGLE DRIVE CONNECTION TEST")
print("=" * 70)

# Test 1: Check config
print("\n[1/5] Checking configuration...")
try:
    from app.config import settings
    print(f"   GOOGLE_DRIVE_ENABLED: {settings.google_drive_enabled}")
    print(f"   CREDENTIALS_FILE: {settings.google_drive_credentials_file}")
    print(f"   MAKE_PUBLIC: {settings.google_drive_make_public}")
except Exception as e:
    print(f"   ❌ Config error: {e}")
    sys.exit(1)

# Test 2: Check credentials file
print("\n[2/5] Checking credentials file...")
cred_path = Path(settings.google_drive_credentials_file).expanduser()
if cred_path.exists():
    print(f"   ✅ File exists: {cred_path}")
    print(f"   File size: {cred_path.stat().st_size} bytes")
else:
    print(f"   ❌ File NOT found: {cred_path}")
    print(f"   → Create file or update GOOGLE_DRIVE_CREDENTIALS_FILE in .env")

# Test 3: Import Google libraries
print("\n[3/5] Checking Google API libraries...")
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    print("   ✅ google-auth, google-api-python-client installed")
except ImportError as e:
    print(f"   ❌ Missing library: {e}")
    print("   → pip install google-auth google-api-python-client")
    sys.exit(1)

# Test 4: Build Drive service (if file exists)
if cred_path.exists():
    print("\n[4/5] Building Drive service...")
    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(cred_path),
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        print("   ✅ Drive service created")
    except Exception as e:
        print(f"   ❌ Failed to build service: {e}")
        print("   → Check JSON format, Service Account permissions")
        sys.exit(1)

    # Test 5: Test API call
    print("\n[5/5] Testing Drive API (list files)...")
    try:
        results = drive_service.files().list(pageSize=5, fields="files(id, name)").execute()
        files = results.get("files", [])
        print(f"   ✅ API call successful! Found {len(files)} files")
        if files:
            print("   First 3 files:")
            for f in files[:3]:
                print(f"     - {f['name']} (id: {f['id']})")
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        print("   → Check Service Account has Drive API enabled")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ GOOGLE DRIVE CONNECTED SUCCESSFULLY!")
    print("=" * 70)
else:
    print("\n[4-5/5] Skipped (credentials file missing)")
    print("\n" + "=" * 70)
    print("⚠️  GOOGLE DRIVE NOT CONNECTED (missing credentials)")
    print("=" * 70)
    print("\nTo fix:")
    print("1. Create Service Account at https://console.cloud.google.com/")
    print("2. Download JSON key")
    print(f"3. Save to: {cred_path}")
    print("4. Share Drive folder with service-account email")
    print("5. Re-run this script")

#!/usr/bin/env python3
"""
Grant Service Account Editor access to VinUni folder in My Drive.
This allows Service Account to upload files to My Drive.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "services" / "tracking-service"))

from app.config import settings
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

print("=" * 80)
print("SHARE VINUNI FOLDER WITH SERVICE ACCOUNT")
print("=" * 80)

# Build Drive service
credentials_path = Path(settings.google_drive_credentials_file).expanduser()
credentials = service_account.Credentials.from_service_account_file(
    str(credentials_path),
    scopes=["https://www.googleapis.com/auth/drive"]
)
drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
print("✅ Drive service connected")

# Service Account email
sa_email = "drive-uploader@ambient-fuze-493617-t9.iam.gserviceaccount.com"
print(f"\nService Account: {sa_email}")

# Get VinUni folder
root_folder_id = settings.google_drive_root_folder_id or settings.google_drive_vinuni_folder_id
print(f"VinUni folder ID: {root_folder_id}")

try:
    folder = drive.files().get(fileId=root_folder_id, fields="name,mimeType,driveId").execute()
    print(f"Folder: '{folder['name']}' (type: {folder['mimeType']})")

    drive_id = folder.get("driveId")
    if drive_id:
        print(f"📁 This is a Shared Drive (driveId: {drive_id})")
        print("   Service Account needs to be added as member of the Shared Drive.")
    else:
        print(f"📁 This is My Drive (personal)")

except HttpError as e:
    print(f"❌ Cannot access folder: {e}")
    sys.exit(1)

# Grant Editor permission to Service Account
print(f"\n[1] Granting 'Editor' permission to {sa_email}...")

try:
    # Check existing permissions
    perms = drive.permissions().list(
        fileId=root_folder_id,
        fields="permissions(id,emailAddress,role,type)",
        supportsAllDrives=True
    ).execute()
    existing = perms.get("permissions", [])
    print(f"   Current permissions: {len(existing)}")
    for p in existing:
        if p.get("emailAddress") == sa_email:
            print(f"   ✅ Already has permission: {p['role']}")
            break
    else:
        # Add permission
        perm_body = {
            "type": "user",
            "role": "writer",  # Editor = writer
            "emailAddress": sa_email
        }
        new_perm = drive.permissions().create(
            fileId=root_folder_id,
            body=perm_body,
            fields="id,role",
            supportsAllDrives=True,
            sendNotificationEmail=False
        ).execute()
        print(f"   ✅ Granted: {new_perm['role']} (permission ID: {new_perm['id']})")

except HttpError as e:
    error_content = str(e)
    if "cannot share" in error_content.lower() or "insufficient" in error_content.lower():
        print(f"❌ Cannot grant permission via API (expected for My Drive).")
        print(f"\nMANUAL ACTION REQUIRED:")
        print(f"1. Go to Google Drive: https://drive.google.com/drive/folders/{root_folder_id}")
        print(f"2. Right-click folder '{folder['name']}' → Share")
        print(f"3. Add: {sa_email}")
        print(f"4. Permission: Editor (or Content manager)")
        print(f"5. Click 'Send'")
        sys.exit(1)
    else:
        print(f"⚠️  Error: {e}")

# Verify
print("\n[2] Verifying permission...")
try:
    perms = drive.permissions().list(
        fileId=root_folder_id,
        fields="permissions(id,emailAddress,role)",
        supportsAllDrives=True
    ).execute()
    for p in perms.get("permissions", []):
        if p.get("emailAddress") == sa_email:
            print(f"   ✅ Confirmed: {sa_email} → {p['role']}")
            break
    else:
        print(f"   ⚠️  Permission not found. Please share manually.")
        sys.exit(1)
except Exception as e:
    print(f"   ⚠️  Could not verify: {e}")

print("\n" + "=" * 80)
print("SETUP COMPLETE!")
print("=" * 80)
print("\n✅ VinUni folder shared with Service Account")
print("\nNow you can run exchange.py WITH Drive upload:")
print("  Note: exchange.py is a test-only exception and still accepts .mp4 input.")
print(f"  python exchange.py --mp4-file videos/Camera_01.mp4 --camera-id CAM_01")
print("\nNote: If you still get 'storageQuotaExceeded', you need Shared Drive instead of My Drive.")

#!/usr/bin/env python3
"""
Create a Shared Drive for tracking-service and grant Service Account access.
Run this once to set up proper Drive infrastructure.
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
print("SHARED DRIVE SETUP FOR TRACKING SERVICE")
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

# Step 1: Find or create Shared Drive
print("\n[1] Looking for existing Shared Drive...")
try:
    results = drive.drives().list(pageSize=20).execute()
    drives = results.get("drives", [])
    print(f"   Found {len(drives)} Shared Drives:")
    for d in drives:
        print(f"     - {d['name']} (ID: {d['id']})")

    # Look for VinUni-related drive
    shared_drive_id = None
    for d in drives:
        if "vinuni" in d["name"].lower() or "tracking" in d["name"].lower():
            shared_drive_id = d["id"]
            print(f"✅ Using Shared Drive: '{d['name']}' (ID: {shared_drive_id})")
            break

    if not shared_drive_id:
        print("\n❌ No Shared Drive found with 'VinUni' or 'tracking' in name.")
        print("   Please create a Shared Drive in Google Drive with:")
        print("   Name: 'VinUni Tracking' (or similar)")
        print("   Settings: Data residency: Global, Access: Restricted")
        print("\n   Then share it with:")
        print(f"   {sa_email} → 'Content manager' permission")
        sys.exit(1)

except HttpError as e:
    print(f"❌ Error listing Shared Drives: {e}")
    print("   Service Account may not have access to list Shared Drives.")
    print("   Please manually create Shared Drive and share with Service Account.")
    sys.exit(1)

# Step 2: Share Shared Drive with Service Account
print(f"\n[2] Granting Service Account access to Shared Drive...")
print(f"    Adding '{sa_email}' as 'Content manager'...")

try:
    # Add permission at Shared Drive level
    permission = {
        "type": "user",
        "role": "contentManager",  # Can upload, edit, delete files
        "emailAddress": sa_email
    }
    drive.permissions().create(
        fileId=shared_drive_id,
        body=permission,
        fields="id",
        supportsAllDrives=True,
        sendNotificationEmail=False
    ).execute()
    print(f"✅ Permission granted: {sa_email} can upload to Shared Drive")
except HttpError as e:
    if "already exists" in str(e).lower():
        print(f"✅ Permission already exists")
    else:
        print(f"⚠️  Could not auto-grant permission: {e}")
        print(f"   MANUAL: Go to Shared Drive '{shared_drive_id}' → Share → Add '{sa_email}' as 'Content manager'")

# Step 3: Create folder structure inside Shared Drive
print(f"\n[3] Creating folder structure inside Shared Drive...")

folders = [
    ("Queue", "Queue - videos awaiting processing"),
    (".h265", "H.265 compressed videos"),
    ("Metadata", "Metadata JSON files"),
    ("Import_New", "Import - new videos to ingest"),
]

folder_ids = {}
parent_id = shared_drive_id

for folder_name, desc in folders:
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = drive.files().list(q=query, fields="files(id, name)").execute()
    existing = results.get("files", [])

    if existing:
        folder_id = existing[0]["id"]
        print(f"    ✅ {folder_name}/ exists: {folder_id}")
    else:
        folder_meta = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = drive.files().create(
            body=folder_meta,
            fields="id, name",
            supportsAllDrives=True,
        ).execute()
        folder_id = folder["id"]
        print(f"    ✅ Created {folder_name}/: {folder_id}")

    folder_ids[folder_name] = folder_id

# Step 4: Update .env with Shared Drive ID
print(f"\n[4] Updating .env with Shared Drive ID...")
env_file = PROJECT_ROOT / "backend" / "services" / "tracking-service" / ".env"
if env_file.exists():
    env_content = env_file.read_text()
    if "GOOGLE_DRIVE_ROOT_FOLDER_ID" not in env_content:
        env_content += f"\nGOOGLE_DRIVE_ROOT_FOLDER_ID={shared_drive_id}\n"
        env_content += f"GOOGLE_DRIVE_VINUNI_FOLDER_ID={shared_drive_id}\n"
        env_file.write_text(env_content)
        print(f"✅ Updated .env with Shared Drive ID: {shared_drive_id}")
    else:
        print(f"✅ .env already contains GOOGLE_DRIVE_ROOT_FOLDER_ID")
        # Update to Shared Drive ID
        lines = env_content.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("GOOGLE_DRIVE_ROOT_FOLDER_ID="):
                new_lines.append(f"GOOGLE_DRIVE_ROOT_FOLDER_ID={shared_drive_id}")
            elif line.startswith("GOOGLE_DRIVE_VINUNI_FOLDER_ID="):
                new_lines.append(f"GOOGLE_DRIVE_VINUNI_FOLDER_ID={shared_drive_id}")
            else:
                new_lines.append(line)
        env_file.write_text("\n".join(new_lines) + "\n")
        print(f"✅ Updated .env with Shared Drive ID: {shared_drive_id}")

# Summary
print("\n" + "=" * 80)
print("SETUP COMPLETE!")
print("=" * 80)
print(f"\nShared Drive ID: {shared_drive_id}")
print(f"Service Account: {sa_email}")
print(f"Permission: Content Manager (granted)")
print("\nFolder structure:")
for folder_name, folder_id in folder_ids.items():
    print(f"  📁 {folder_name}/ → {folder_id}")

print("\n✅ You can now run exchange.py WITHOUT --no-drive!")
print("\nTest command:")
print(f"  python exchange.py --mp4-file videos/Camera_01.mp4 --camera-id CAM_01")
print("\nNote: First upload may take time as models download to cache.")

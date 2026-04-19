#!/usr/bin/env python3
"""
Setup Google Drive folder structure using OAuth2 user credentials.
Creates: VinUni/Queue/.h265, VinUni/Queue/Metadata, VinUni/Import_New, etc.
"""

import sys
from pathlib import Path

# Add to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "services" / "tracking-service"))

from app.config import settings
from shared_secret_runtime import build_google_drive_oauth_service

print("=" * 70)
print("GOOGLE DRIVE FOLDER SETUP")
print("=" * 70)

drive = build_google_drive_oauth_service()
print("✅ Drive service connected")

# Config
root_folder_id = settings.google_drive_root_folder_id or settings.google_drive_vinuni_folder_id
print(f"\nRoot folder ID: {root_folder_id}")

# Get root folder name
root_meta = drive.files().get(fileId=root_folder_id, fields="name,driveId,mimeType").execute()
print(f"Root folder: '{root_meta['name']}' (type: {root_meta['mimeType']})")

drive_id = root_meta.get("driveId")
if drive_id:
    print(f"📁 This is a Shared Drive (driveId: {drive_id})")
else:
    print(f"📁 This is My Drive (personal)")

# Folder structure cần tạo
folders_to_create = [
    (settings.google_drive_queue_folder_name, "Queue - videos awaiting processing"),
    (settings.google_drive_import_folder_name, "Import - new videos to ingest"),
    (settings.google_drive_h265_folder_name, "H.265 compressed videos"),
    (settings.google_drive_metadata_folder_name, "Metadata JSON files"),
]

created = []
for folder_name, description in folders_to_create:
    print(f"\n[1] Checking/Creating folder: '{folder_name}'")

    # Check if exists
    query = f"name='{folder_name}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if drive_id:
        query += f" and driveId='{drive_id}'"

    results = drive.files().list(q=query, fields="files(id, name)").execute()
    existing = results.get("files", [])

    if existing:
        folder_id = existing[0]["id"]
        print(f"    ✅ Already exists: {folder_id}")
    else:
        # Create folder
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [root_folder_id],
        }
        if drive_id:
            file_metadata["driveId"] = drive_id

        folder = drive.files().create(
            body=file_metadata,
            fields="id, name",
            supportsAllDrives=True,
        ).execute()
        folder_id = folder["id"]
        print(f"    ✅ Created: {folder_id}")

    created.append((folder_name, folder_id, description))

    # Create subfolder .h265 within Queue
    if folder_name == settings.google_drive_queue_folder_name:
        h265_name = settings.google_drive_h265_folder_name
        query_h265 = f"name='{h265_name}' and '{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_h265 = drive.files().list(q=query_h265, fields="files(id, name)").execute()
        if results_h265.get("files"):
            print(f"    ✅ Subfolder '{h265_name}' already exists")
        else:
            h265_folder = drive.files().create(
                body={
                    "name": h265_name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [folder_id],
                },
                fields="id",
                supportsAllDrives=True,
            ).execute()
            print(f"    ✅ Created subfolder '{h265_name}': {h265_folder['id']}")

# Summary
print("\n" + "=" * 70)
print("FOLDER STRUCTURE:")
print("=" * 70)
for folder_name, folder_id, desc in created:
    print(f"  📁 {folder_name}/")
    print(f"     ID: {folder_id}")
    print(f"     Desc: {desc}")
    if folder_name == settings.google_drive_queue_folder_name:
        print(f"     └── {settings.google_drive_h265_folder_name}/")
        print(f"         ID: (created above)")

# OAuth note
print("\n" + "=" * 70)
print("OAUTH INFO:")
print("=" * 70)
print(f"  Root folder: {root_folder_id}")
print("\n🔑 Auth mode: OAuth2 user token from secrets/oauth/")

print("\n✅ Setup complete!")

"""
Google Drive Upload - OAuth2 (User Account)
File: upload_oauth2.py

Ưu điểm:
- Dùng quota của tài khoản user (duonghiepthongminh@gmail.com)
- Upload vào My Drive được
- Chỉ cần login 1 lần, sau đó tự động

Cách chạy lần đầu:
1. Chạy: python upload_oauth2.py
2. Browser mở ra → Login Google → Click "Allow"
3. Token được lưu, xong!
4. Các lần sau: chạy bình thường, không cần login
"""

import os
import sys
import pickle
from pathlib import Path
from datetime import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from shared_secret_runtime import (  # noqa: E402
    ensure_canonical_secret_dirs,
    load_runtime_env,
    resolve_oauth2_credentials_path,
    resolve_oauth2_token_path,
)

load_runtime_env(include_tracking_service_env=False)
ensure_canonical_secret_dirs()

# ========== CONFIGURATION ==========
# Folder ID comes from the shared secret env so it stays portable across machines.
FOLDER_ID = os.getenv("GOOGLE_DRIVE_VINUNI_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID", "")

# OAuth2 credentials file (bạn cần tải từ Google Cloud Console)
# Download: https://console.cloud.google.com/apis/credentials
OAUTH2_CREDENTIALS = str(resolve_oauth2_credentials_path())

# Token file (auto-generated sau khi login lần đầu)
TOKEN_FILE = str(resolve_oauth2_token_path())

# Scopes - yêu cầu quyền Drive
SCOPES = ["https://www.googleapis.com/auth/drive"]

# File test để upload
TEST_FILE = "test_upload_oauth2.txt"
# ====================================


def get_credentials():
    """Lấy credentials - tự động login lần đầu"""
    creds = None

    # Đọc token đã lưu
    if os.path.exists(TOKEN_FILE):
        print(f"[1] Found existing token: {TOKEN_FILE}")
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
        print(f"   ✅ Token loaded")

    # Kiểm tra token hợp lệ
    if creds and creds.valid:
        print(f"[2] Token is valid")
        return creds

    # Token hết hạn nhưng có refresh_token
    if creds and creds.expired and creds.refresh_token:
        print(f"[2] Token expired, refreshing...")
        try:
            creds.refresh(Request())
            # Lưu token mới
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)
            print(f"   ✅ Token refreshed and saved")
            return creds
        except Exception as e:
            print(f"   ❌ Refresh failed: {e}")
            print(f"   → Need to re-authenticate")
            creds = None

    # Không có token hoặc refresh thất bại → Cần login
    if not os.path.exists(OAUTH2_CREDENTIALS):
        print(f"\n{'='*60}")
        print(f"❌ OAUTH2 CREDENTIALS FILE NOT FOUND!")
        print(f"{'='*60}")
        print(f"   Bạn cần tạo OAuth2 Client ID:")
        print(f"   1. Go to: https://console.cloud.google.com/apis/credentials")
        print(f"   2. Select project: ambient-fuze-493617-t9")
        print(f"   3. Click 'Create Credentials' → 'OAuth client ID'")
        print(f"   4. Application type: 'Desktop app'")
        print(f"   5. Download JSON → Save as: {OAUTH2_CREDENTIALS}")
        print(f"   6. Run this script again")
        print(f"{'='*60}\n")
        sys.exit(1)

    print(f"[2] No valid token found")
    print(f"[3] Starting OAuth2 authentication...")
    print(f"   → Browser will open for login")
    print(f"   → Please login and click 'Allow'")

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            OAUTH2_CREDENTIALS,
            SCOPES
        )
        # run_local_server: tự động mở browser và nhận callback
        creds = flow.run_local_server(
            port=0,
            prompt="consent",
            access_type="offline",
            include_granted_scopes=False
        )
        
        # Lưu token
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
        
        print(f"\n   ✅ Authentication successful!")
        print(f"   ✅ Token saved to: {TOKEN_FILE}")
        print(f"   ✅ Refresh token: {'Yes' if creds.refresh_token else 'No'}")
        
        return creds

    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        sys.exit(1)


def build_drive_service(creds):
    """Khởi tạo Drive service"""
    print(f"[4] Building Drive service...")
    try:
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        print(f"   ✅ Drive service built")
        return drive
    except Exception as e:
        print(f"   ❌ Failed to build Drive service: {e}")
        sys.exit(1)


def get_user_info(drive):
    """Lấy thông tin tài khoản"""
    print(f"\n[INFO] Account info:")
    try:
        about = drive.about().get(fields="user(emailAddress,displayName),storageQuota(limit,usage,usedInTrash)").execute()
        user = about.get("user", {})
        quota = about.get("storageQuota", {})
        
        print(f"   User: {user.get('displayName', 'N/A')}")
        print(f"   Email: {user.get('emailAddress', 'N/A')}")
        print(f"   Storage Limit: {format_bytes(quota.get('limit', 0))}")
        print(f"   Storage Used: {format_bytes(quota.get('usageInDrive', 0))}")
        
        return about
    except Exception as e:
        print(f"   ⚠️  Could not get account info: {e}")
        return None


def check_folder_access(drive, folder_id):
    """Kiểm tra quyền truy cập folder"""
    print(f"\n[CHECK] Verifying folder access...")
    print(f"       Folder ID: {folder_id}")

    try:
        folder = drive.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,owners"
        ).execute()

        print(f"   📁 Folder: {folder.get('name')}")
        print(f"   🏠 Owner: {[o.get('emailAddress') for o in folder.get('owners', [])]}")
        print(f"   ✅ Access verified")
        return True

    except HttpError as e:
        print(f"   ❌ Folder access denied: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def upload_file(drive, file_path, folder_id, make_public=True):
    """Upload file vào folder"""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    print(f"\n[UPLOAD] Uploading: {file_path.name}")
    print(f"         Target: {folder_id}")

    try:
        file_metadata = {
            "name": file_path.name,
            "parents": [folder_id],
        }

        media = MediaFileUpload(
            str(file_path),
            resumable=True
        )

        print(f"[UPLOAD] Uploading...")
        result = drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id,name,mimeType,webViewLink,owners"
        ).execute()

        print(f"\n   ✅ UPLOAD SUCCESS!")
        print(f"   📄 File ID: {result.get('id')}")
        print(f"   📝 Name: {result.get('name')}")
        print(f"   🔗 View: {result.get('webViewLink')}")
        print(f"   👤 Owner: {[o.get('emailAddress') for o in result.get('owners', [])]}")

        # Set public
        if make_public:
            try:
                drive.permissions().create(
                    fileId=result.get('id'),
                    body={"type": "anyone", "role": "reader"}
                ).execute()
                print(f"   🌐 ✅ Set public access")
            except:
                pass

        return result

    except HttpError as e:
        error_details = e.error_details[0] if e.error_details else {}
        print(f"\n   ❌ UPLOAD FAILED!")
        print(f"      Status: {e.status_code}")
        print(f"      Reason: {error_details.get('reason', 'unknown')}")
        print(f"      Message: {error_details.get('message', str(e))}")
        return None


def list_files(drive, folder_id):
    """Liệt kê files trong folder"""
    print(f"\n[TEST] Listing files in folder...")

    try:
        results = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=10,
            fields="files(id,name,mimeType)"
        ).execute()

        files = results.get('files', [])
        print(f"   Found {len(files)} files/folders")
        for f in files[:5]:
            print(f"   - {f.get('name')}")

        return files
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


def format_bytes(bytes_val):
    """Format bytes to human readable"""
    if not bytes_val:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def main():
    print("="*60)
    print("GOOGLE DRIVE UPLOAD - OAUTH2 (USER ACCOUNT)")
    print("="*60)
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Get credentials (login if needed)
    creds = get_credentials()

    # Step 2: Build Drive service
    drive = build_drive_service(creds)

    # Step 3: Get user info & quota
    get_user_info(drive)

    # Step 4: Check folder access
    if not check_folder_access(drive, FOLDER_ID):
        sys.exit(1)

    # Step 5: List existing files
    list_files(drive, FOLDER_ID)

    # Step 6: Create test file
    print("\n" + "-"*60)
    print(f"[SETUP] Creating test file: {TEST_FILE}")
    with open(TEST_FILE, 'w') as f:
        f.write(f"OAuth2 Upload Test\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Account: duonghiepthongminh@gmail.com\n")
    print(f"   ✅ Test file created")

    # Step 7: Upload
    print("\n" + "-"*60)
    result = upload_file(drive, TEST_FILE, FOLDER_ID, make_public=True)

    # Summary
    print("\n" + "="*60)
    if result:
        print("✅ STATUS: UPLOAD SUCCESS")
        print(f"   View: {result.get('webViewLink')}")
        print()
        print("💡 Lưu ý:")
        print("   - Token đã được lưu trong oauth2_token.pickle")
        print("   - Lần sau chạy sẽ tự động, không cần login")
        print("   - Token sẽ tự refresh khi hết hạn")
    else:
        print("❌ STATUS: UPLOAD FAILED")
        print("   Check error message above")
    print("="*60)

    return result is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

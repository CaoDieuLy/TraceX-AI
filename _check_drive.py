"""List Drive Storage and Temp folder contents."""
import pickle, sys
from pathlib import Path
from googleapiclient.discovery import build

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
token = Path(r"D:\workspace\project\secrets\oauth\oauth2_token.pickle")
with open(token, "rb") as f:
    creds = pickle.load(f)
drive = build("drive", "v3", credentials=creds, cache_discovery=False)

def ls(folder_id, label, depth=0):
    res = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,mimeType,size)",
        pageSize=50,
    ).execute()
    items = res.get("files", [])
    indent = "  " * depth
    print(f"{indent}[{label}] {len(items)} items")
    for f in items[:20]:
        sz = f.get("size", "-")
        is_folder = f["mimeType"] == "application/vnd.google-apps.folder"
        print(f"{indent}  {'📁' if is_folder else '📄'} {f['name']} ({sz})")
        if is_folder and depth < 2:
            ls(f["id"], f["name"], depth + 1)

STORAGE_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
TEMP_ID    = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"

ls(STORAGE_ID, "Storage")
print()
ls(TEMP_ID, "Temp")

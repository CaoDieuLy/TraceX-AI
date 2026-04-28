"""Set all files in Drive Storage/ folder to public (anyone with link can download)."""
import pickle, sys
from pathlib import Path
from googleapiclient.discovery import build

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
token = Path(r"D:\workspace\project\secrets\oauth\oauth2_token.pickle")
with open(token, "rb") as f:
    creds = pickle.load(f)
drive = build("drive", "v3", credentials=creds, cache_discovery=False)

STORAGE_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
FOLDER_MIME = "application/vnd.google-apps.folder"


def list_all(folder_id):
    res = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,mimeType)",
        pageSize=100,
    ).execute()
    for f in res.get("files", []):
        if f["mimeType"] == FOLDER_MIME:
            yield from list_all(f["id"])
        elif f["name"].lower().endswith(".mp4"):
            yield f


def set_public(file_id, name):
    try:
        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
            supportsAllDrives=True,
        ).execute()
        print(f"  public: {name}")
    except Exception as e:
        print(f"  WARN {name}: {e}")


print("Setting all .mp4 files in Storage/ to public...")
count = 0
for f in list_all(STORAGE_ID):
    set_public(f["id"], f["name"])
    count += 1

print(f"\nDone: {count} files set to public.")

"""
progress.py — Xem tiến độ xử lý video trên VPS.
Chạy: python3 scripts/progress.py
"""
import re
from datetime import datetime, timezone

import paramiko

VPS_HOST = "103.72.56.139"
VPS_USER = "root"
VPS_PASS = "xVrqU6Dg@VhAcj4"


def run(client, cmd):
    _, stdout, _ = client.exec_command(cmd)
    return stdout.read().decode()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=15)

    # Completed videos from DB
    db_out = run(client, """docker exec mcpt-postgres psql -U mcpt_user -d mcpt_video_tracking_full -t -A -F'|' -c \
"SELECT video_id, TO_CHAR(updated_at AT TIME ZONE 'UTC', 'HH24:MI:SS') FROM queue_video_assets ORDER BY updated_at DESC;" """)

    completed = {}
    for line in db_out.strip().splitlines():
        parts = line.split("|")
        if len(parts) == 2:
            completed[parts[0].strip()] = parts[1].strip()

    # In-progress from backend logs
    logs = run(client, "docker logs mcpt-backend --tail=1000 2>&1")
    in_progress = {}
    for line in logs.splitlines():
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Requesting.*source_filename=(\S+)", line)
        if m:
            ts_str, fname = m.group(1), m.group(2)
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            elapsed = round((datetime.now(timezone.utc) - ts).total_seconds() / 60, 1)
            if fname not in completed:
                in_progress[fname] = f"[processing] {elapsed} min elapsed"

    client.close()

    all_videos = {
        **{v: f"[done] completed at {t}" for v, t in completed.items()},
        **in_progress,
    }

    print(f"\n{'Video':<45} {'Status'}")
    print("-" * 80)
    if all_videos:
        for video, status in sorted(all_videos.items()):
            print(f"{video:<45} {status}")
    else:
        print("  (no videos found yet)")
    print(f"\nTotal: {len(completed)} completed, {len(in_progress)} in-progress\n")


if __name__ == "__main__":
    main()

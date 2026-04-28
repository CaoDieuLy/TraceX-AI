"""Sync changed files to VPS, rebuild mcpt-backend, trigger queue."""
import io, sys, paramiko
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST, USER, PASSWORD = "103.72.56.139", "root", "xVrqU6Dg@VhAcj4"
REMOTE = "/opt/mcpt/A20-App-119"
LOCAL = Path(r"D:\python ky 9\A20-App-119")

SYNC = [
    "infra/env/backend.env",
    "backend/services/tracking-service/requirements.txt",
    "backend/services/tracking-service/app/local_ingestion_pipeline.py",
    "backend/services/tracking-service/app/model_adapters.py",
    "backend/services/tracking-service/app/tracklet_feature_pipeline.py",
]


def run(client, cmd, timeout=180):
    sys.stdout.write(f"$ {cmd[:110]}\n"); sys.stdout.flush()
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    if out.strip(): sys.stdout.write(out.rstrip() + "\n")
    if err.strip(): sys.stdout.write("ERR: " + err.rstrip() + "\n")
    sys.stdout.flush()
    return out


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, username=USER, password=PASSWORD, timeout=20)
    sftp = client.open_sftp()

    print("\n=== Sync files ===")
    for rel in SYNC:
        local = LOCAL / rel
        remote = f"{REMOTE}/{rel}"
        if not local.exists():
            print(f"  SKIP (missing): {rel}"); continue
        run(client, f"mkdir -p {remote.rsplit('/', 1)[0]}")
        sftp.put(str(local), remote)
        print(f"  synced: {rel}")

    sftp.close()

    print("\n=== Rebuild mcpt-backend ===")
    run(client,
        f"cd {REMOTE} && docker compose -f infra/docker-compose.yml "
        "--env-file infra/env/backend.env up -d --build --no-deps mcpt-backend 2>&1 | tail -25",
        timeout=360)

    print("\n=== Health check ===")
    run(client, "sleep 10 && docker ps --format '{{.Names}}\\t{{.Status}}' | grep mcpt-backend")
    run(client, "docker logs mcpt-backend --tail 30 2>&1")

    print("\n=== Verify env ===")
    run(client, f"docker exec mcpt-backend env | grep -E 'STORAGE_INGEST|GOOGLE_DRIVE'")

    print("\n=== Trigger queue ===")
    run(client,
        "curl -s -X POST http://localhost:8000/api/v1/queue/process-storage "
        "-H 'Content-Type: application/json' -d '{}' 2>&1",
        timeout=30)

    print("\n=== Queue logs (30s) ===")
    run(client, "docker logs mcpt-backend --tail 60 2>&1", timeout=60)

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

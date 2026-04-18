#!/usr/bin/env python3
"""
Quick start: Start PostgreSQL (via Docker) if not running.
"""

import subprocess
import sys
import time
from pathlib import Path

print("=" * 70)
print("POSTGRESQL QUICK START")
print("=" * 70)

# Check if PostgreSQL is running
try:
    import psycopg2
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres"
    )
    conn.close()
    print("✅ PostgreSQL is already running on localhost:5432")
    sys.exit(0)
except Exception:
    print("⚠️  PostgreSQL not running on localhost:5432")

# Try Docker
print("\n[1] Checking Docker...")
try:
    subprocess.run(["docker", "--version"], capture_output=True, check=True)
    print("    ✅ Docker is installed")
except Exception:
    print("    ❌ Docker not found. Install Docker first.")
    sys.exit(1)

# Check if postgres container exists
print("\n[2] Checking for existing postgres container...")
result = subprocess.run(["docker", "ps", "-a", "--filter", "name=postgres", "--format", "{{.Names}}"], capture_output=True, text=True)
containers = result.stdout.strip().split('\n')
postgres_container = None
for c in containers:
    if 'postgres' in c.lower():
        postgres_container = c
        break

if postgres_container:
    print(f"    Found container: {postgres_container}")
    status = subprocess.run(["docker", "ps", "--filter", f"name={postgres_container}", "--format", "{{.Status}}"], capture_output=True, text=True)
    if "Up" in status.stdout:
        print(f"    ✅ Container is running")
        sys.exit(0)
    else:
        print(f"    ⚠️  Container exists but not running. Starting...")
        subprocess.run(["docker", "start", postgres_container], check=True)
        print(f"    ✅ Container started")
        time.sleep(3)
        sys.exit(0)
else:
    print("    No postgres container found. Creating new one...")

# Create new container
print("\n[3] Creating PostgreSQL container...")
project_root = Path(__file__).parent
pg_data = project_root / "postgres_data"

cmd = [
    "docker", "run", "-d",
    "--name", "postgres-mcpt",
    "-e", "POSTGRES_USER=mcpt_user",
    "-e", "POSTGRES_PASSWORD=Mcpt@2026!Secure",
    "-e", "POSTGRES_DB=video_tracking",
    "-p", "5432:5432",
    "-v", f"{pg_data}:/var/lib/postgresql/data",
    "postgres:16-alpine"
]

print(f"    Running: {' '.join(cmd)}")
subprocess.run(cmd, check=True)
print("    ✅ Container created")

# Wait for PostgreSQL to be ready
print("\n[4] Waiting for PostgreSQL to be ready...")
for i in range(30):
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="video_tracking",
            user="mcpt_user",
            password="Mcpt@2026!Secure"
        )
        conn.close()
        print("    ✅ PostgreSQL is ready!")
        break
    except Exception:
        print(f"    Waiting... ({i+1}/30)")
        time.sleep(1)
else:
    print("    ❌ PostgreSQL did not start in time")
    print("    Check logs: docker logs postgres-mcpt")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ POSTGRESQL IS RUNNING")
print("=" * 70)
print("\nConnection details:")
print("  Host: localhost")
print("  Port: 5432")
print("  Database: video_tracking")
print("  User: mcpt_user")
print("  Password: Mcpt@2026!Secure")
print("\nYou can now run exchange.py!")

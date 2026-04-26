from __future__ import annotations

import argparse
import logging

from .config import settings
from .database import Base, SessionLocal, engine
from .queue_runtime import QueueSyncService, run_queue_sync_forever


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Drive queue worker for MCPT videos")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap the queue from an explicit local source directory")
    parser.add_argument("--source-dir", type=str, default=None, help="Local directory containing source videos for one-off bootstrap")
    parser.add_argument("--limit", type=int, default=31, help="Maximum number of bootstrap videos to process")
    parser.add_argument("--keep-remote", action="store_true", help="Do not clear remote Queue before bootstrap")
    parser.add_argument("--delete-source", action="store_true", help="Delete each source file after it has been processed successfully")
    parser.add_argument("--once", action="store_true", help="Process Import_New a single time and exit")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    service = QueueSyncService()

    if args.bootstrap:
        if not args.source_dir:
            raise RuntimeError("--source-dir is required when using --bootstrap")
        session = SessionLocal()
        try:
            result = service.bootstrap_from_source_dir(
                session,
                source_dir=args.source_dir,
                limit=args.limit,
                reset_remote_queue=not args.keep_remote,
                delete_source_after_import=args.delete_source,
            )
            logging.getLogger(__name__).info("Bootstrap finished: %s", result)
        finally:
            session.close()

    if args.once:
        session = SessionLocal()
        try:
            result = service.process_import_queue(session)
            logging.getLogger(__name__).info("Import sync finished: %s", result)
        finally:
            session.close()
        return

    if not settings.google_drive_enabled:
        raise RuntimeError("GOOGLE_DRIVE_ENABLED is false. Enable it before running the queue worker.")

    run_queue_sync_forever(SessionLocal)


if __name__ == "__main__":
    main()

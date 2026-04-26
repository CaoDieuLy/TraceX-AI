from __future__ import annotations

import argparse
import logging

from .config import settings
from .database import Base, SessionLocal, engine
from .queue_runtime import QueueSyncService, run_queue_sync_forever


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Storage queue worker for MCPT videos")
    parser.add_argument("--once", action="store_true", help="Process new storage videos a single time and exit")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    service = QueueSyncService()

    if args.once:
        session = SessionLocal()
        try:
            result = service.process_storage_queue(session)
            logging.getLogger(__name__).info("Storage ingest finished: %s", result)
        finally:
            session.close()
        return

    if not settings.storage_ingest_enabled:
        raise RuntimeError("Enable STORAGE_INGEST_ENABLED before running the queue worker.")

    run_queue_sync_forever(SessionLocal)


if __name__ == "__main__":
    main()

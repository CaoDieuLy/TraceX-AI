#!/usr/bin/env python3
"""
Watch Import_New folder for new MP4 uploads.
Automatically converts to H.265 + generates metadata,
moves to Queue, and updates PostgreSQL database.

Run as background service in Lightning Studio.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.queue_manager import get_queue_manager
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("watch_import_new.log"),
    ],
)
logger = logging.getLogger(__name__)


class ImportWatcher:
    """Watches Import_New folder and processes new MP4 files."""

    def __init__(
        self,
        poll_interval: int = 30,
        import_new_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize watcher.

        Args:
            poll_interval: Check every N seconds (default: 30)
            import_new_dir: Override Import_New directory path
        """
        self.poll_interval = poll_interval
        self.manager = get_queue_manager()

        if import_new_dir:
            self.manager.import_new_dir = Path(import_new_dir)
            self.manager._ensure_directories()

        self.seen_files: set[str] = set()
        self._initialize_seen_files()

    def _initialize_seen_files(self) -> None:
        """Initialize set of already-processed files."""
        import_new = self.manager.import_new_dir
        if import_new.exists():
            self.seen_files = {f.name for f in import_new.glob("*.mp4")}
            logger.info(f"Initialized with {len(self.seen_files)} existing MP4 files in Import_New")

    def run(self) -> None:
        """Start watching loop."""
        logger.info("=" * 60)
        logger.info("IMPORT_NEW WATCHER STARTED")
        logger.info(f"  Poll interval: {self.poll_interval}s")
        logger.info(f"  Import_New: {self.manager.import_new_dir}")
        logger.info(f"  Queue:      {self.manager.queue_dir}")
        logger.info("=" * 60)

        try:
            while True:
                self._check_and_process()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Watcher stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Watcher crashed: {e}", exc_info=True)
            raise

    def _check_and_process(self) -> None:
        """Check for new files and process them."""
        import_new = self.manager.import_new_dir

        if not import_new.exists():
            logger.warning(f"Import_New directory not found: {import_new}")
            return

        current_files = {f.name for f in import_new.glob("*.mp4")}
        new_files = current_files - self.seen_files

        if new_files:
            logger.info(f"Detected {len(new_files)} new file(s): {list(new_files)}")
            try:
                result = self.manager.enqueue_new_uploads()
                logger.info(
                    f"Processing complete: {result['processed']} processed, "
                    f"{result['failed']} failed, {result['skipped']} skipped"
                )
            except Exception as e:
                logger.error(f"Failed to process new files: {e}", exc_info=True)

        self.seen_files = current_files


# ==================== Main ====================

def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Watch Import_New folder for video uploads")
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--import-dir",
        type=str,
        default=None,
        help="Override Import_New directory path",
    )
    args = parser.parse_args()

    watcher = ImportWatcher(
        poll_interval=args.interval,
        import_new_dir=args.import_dir,
    )
    watcher.run()


if __name__ == "__main__":
    main()
